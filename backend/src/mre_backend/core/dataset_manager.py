from __future__ import annotations

import csv
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.cluster import KMeans

from mre_backend.adapters.text_transformer import TextTransformerAdapter
from mre_backend.adapters.vision_resnet import VisionResNetAdapter
from mre_backend.api.schemas import DatasetRunRequest
from mre_backend.core.artifacts import ArtifactStore
from mre_backend.core.model_cache import INFERENCE_LOCK, CachedModel, ModelCache
from mre_backend.core.utils import (
    atomic_json,
    require_safe_path,
    resolve_model_id,
    safe_run_id,
    set_seed,
    utc_now_iso,
)


@dataclass
class DatasetRunRecord:
    dataset_run_id: str
    status: str
    request: DatasetRunRequest
    created_at: str
    updated_at: str
    summary: dict[str, Any] | None = None
    examples: list[dict[str, Any]] | None = None
    error: str | None = None


class DatasetManager:
    def __init__(self, settings: Any, artifact_store: ArtifactStore) -> None:
        self.settings = settings
        self.artifact_store = artifact_store
        self.model_cache = ModelCache()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self._runs: dict[str, DatasetRunRecord] = {}
        self._lock = threading.Lock()

    def start(self, request: DatasetRunRequest, model_id: str, task_type: str) -> str:
        resolve_model_id(model_id, self.settings)
        expected = "text_classification" if request.kind == "text" else "image_classification"
        if task_type != expected:
            raise ValueError("Dataset kind and task type must agree")
        dataset_run_id = safe_run_id(prefix="dataset")
        now = utc_now_iso()
        record = DatasetRunRecord(
            dataset_run_id=dataset_run_id,
            status="queued",
            request=request,
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            self._runs[dataset_run_id] = record
        self.executor.submit(self._run_task, record, model_id, task_type)
        return dataset_run_id

    def get(self, dataset_run_id: str) -> DatasetRunRecord | None:
        record = self._runs.get(dataset_run_id)
        if record is not None:
            return record
        path = self.artifact_store.get_artifact_path(dataset_run_id, "dataset.json")
        if not path.is_file():
            return None
        saved = json.loads(path.read_text(encoding="utf-8"))
        saved["request"] = DatasetRunRequest(**saved["request"])
        return DatasetRunRecord(**saved)

    def _run_task(self, record: DatasetRunRecord, model_id: str, task_type: str) -> None:
        with INFERENCE_LOCK:
            self._run_task_serial(record, model_id, task_type)

    def _run_task_serial(self, record: DatasetRunRecord, model_id: str, task_type: str) -> None:
        try:
            record.status = "running"
            record.updated_at = utc_now_iso()
            seed = self.settings.execution.seed
            set_seed(seed)

            device = torch.device("cpu")
            if self.settings.execution.allow_gpu and torch.cuda.is_available():
                device = torch.device("cuda")

            if task_type in {"text_lm", "text_classification"}:
                adapter = TextTransformerAdapter(task_type)
            else:
                adapter = VisionResNetAdapter()

            cache_key = self.model_cache.make_key(task_type, model_id)
            cached = self.model_cache.get(cache_key)
            if cached:
                model = cached.model
                tokenizer = cached.tokenizer
            else:
                model, tokenizer = adapter.load(resolve_model_id(model_id, self.settings), device)
                self.model_cache.set(
                    cache_key,
                    CachedModel(
                        model=model, tokenizer=tokenizer, task_type=task_type, model_id=model_id
                    ),
                )

            if record.request.kind == "text":
                summary, examples = self._run_text(
                    record, adapter, model, tokenizer, device, model_id
                )
            else:
                summary, examples = self._run_images(record, adapter, model, device, model_id)

            record.status = "completed"
            record.summary = summary
            record.examples = examples
            record.updated_at = utc_now_iso()
        except Exception as exc:
            record.status = "failed"
            record.error = str(exc)
            record.updated_at = utc_now_iso()
        finally:
            artifacts = self.artifact_store.create(record.dataset_run_id)
            atomic_json(
                artifacts.run_dir / "dataset.json",
                {
                    "dataset_run_id": record.dataset_run_id,
                    "status": record.status,
                    "request": record.request.model_dump(),
                    "created_at": record.created_at,
                    "updated_at": record.updated_at,
                    "summary": record.summary,
                    "examples": record.examples,
                    "error": record.error,
                },
            )

    def _run_text(
        self,
        record: DatasetRunRecord,
        adapter: Any,
        model: Any,
        tokenizer: Any,
        device: torch.device,
        model_id: str,
    ):
        path = Path(record.request.path or "")
        if not path.is_absolute():
            path = self.settings.paths.safe_data_dir / path
        require_safe_path(path, self.settings.paths.safe_data_dir)

        if not path.is_file() or path.stat().st_size > self.settings.limits.max_upload_mb * 1024**2:
            raise ValueError("Dataset file missing or exceeds the upload limit")
        rows = []
        with path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            required = {record.request.text_column, record.request.label_column}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError("CSV must contain the configured text and label columns")
            for row in reader:
                if (
                    not row.get(record.request.text_column, "").strip()
                    or not row.get(record.request.label_column, "").strip()
                ):
                    raise ValueError("Text and label values cannot be blank")
                rows.append(row)
                if len(rows) >= record.request.max_samples:
                    break

        if not rows:
            raise ValueError("Dataset contains no examples")
        labels = list(getattr(model.config, "id2label", {}).values())
        canonical = {str(label).casefold(): str(label) for label in labels}
        if len(canonical) != len(labels):
            raise ValueError("Model labels are ambiguous under case-insensitive matching")
        for row in rows:
            key = row[record.request.label_column].strip().casefold()
            if key not in canonical:
                raise ValueError("CSV labels must match this model's label names")
            row[record.request.label_column] = canonical[key]

        examples = []
        y_true = []
        y_pred = []
        confidences = []
        embeddings = []

        for row in rows:
            text = row.get(record.request.text_column, "")
            label = row.get(record.request.label_column, "")
            inputs = adapter.prepare_inputs(tokenizer, {"text": text}, device)
            outputs = adapter.forward(model, inputs, {"hidden_states": True, "attentions": False})
            probs = torch.softmax(outputs.logits, dim=-1)[0]
            pred = int(probs.argmax().item())
            conf = float(probs.max().item())
            y_true.append(label)
            if outputs.hidden_states:
                last = outputs.hidden_states[-1].mean(dim=1).squeeze(0).detach().cpu().numpy()
                embeddings.append(last)

            run_id = safe_run_id(prefix="sample")
            artifacts = self.artifact_store.create(run_id)
            metadata = {
                "run_id": run_id,
                "task_type": "text_classification",
                "model_id": model_id,
                "input_text": text,
                "status": "completed",
                "created_at": utc_now_iso(),
            }
            outputs_summary = adapter.postprocess(outputs, tokenizer, top_k=3)
            self.artifact_store.save_outputs(artifacts, outputs_summary)
            self.artifact_store.save_summaries(artifacts, {})
            self.artifact_store.save_metadata(artifacts, metadata)

            pred_label = outputs_summary.get("prediction", str(pred))
            y_pred.append(pred_label)
            confidences.append(conf)
            examples.append(
                {
                    "text": text,
                    "label": label,
                    "prediction": pred_label,
                    "confidence": round(conf, 4),
                    "run_id": run_id,
                }
            )

        labels = sorted(set(y_true + y_pred))
        label_to_idx = {label: idx for idx, label in enumerate(labels)}
        matrix = [[0 for _ in labels] for _ in labels]
        correct = 0
        for true, pred in zip(y_true, y_pred):
            i = label_to_idx.get(true)
            j = label_to_idx.get(pred)
            if i is not None and j is not None:
                matrix[i][j] += 1
            if true == pred:
                correct += 1

        accuracy = correct / len(y_true) if y_true else 0.0

        # calibration bins
        bins = [i / 10 for i in range(11)]
        bin_counts = [0 for _ in range(10)]
        bin_conf = [0.0 for _ in range(10)]
        bin_acc = [0.0 for _ in range(10)]
        for true, pred, conf in zip(y_true, y_pred, confidences):
            idx = min(9, int(conf * 10))
            bin_counts[idx] += 1
            bin_conf[idx] += conf
            bin_acc[idx] += 1.0 if true == pred else 0.0
        avg_conf = []
        avg_acc = []
        for i in range(10):
            if bin_counts[i] == 0:
                avg_conf.append(0.0)
                avg_acc.append(0.0)
            else:
                avg_conf.append(bin_conf[i] / bin_counts[i])
                avg_acc.append(bin_acc[i] / bin_counts[i])

        # error slices by label
        error_by_label = []
        for label in labels:
            total = sum(1 for t in y_true if t == label)
            wrong = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
            error_by_label.append(
                {"label": label, "error_rate": (wrong / total) if total else 0.0, "count": total}
            )

        high_conf_errors = [
            ex
            for ex in examples
            if ex.get("label") != ex.get("prediction") and ex.get("confidence", 0) > 0.8
        ][:10]

        summary = {
            "count": len(rows),
            "labels": labels,
            "accuracy": accuracy,
            "confusion_matrix": {"labels": labels, "matrix": matrix},
            "calibration": {
                "bins": bins,
                "avg_confidence": avg_conf,
                "avg_accuracy": avg_acc,
                "counts": bin_counts,
            },
            "error_slices": {
                "by_label": error_by_label,
                "high_confidence_errors": high_conf_errors,
            },
        }
        if embeddings:
            X = np.stack(embeddings)
            if not np.isfinite(X).all():
                raise ValueError("Model returned non-finite dataset embeddings")
            k = min(4, len(np.unique(X, axis=0)))
            if k >= 2:
                km = KMeans(n_clusters=k, n_init=5, random_state=self.settings.execution.seed)
                clusters = km.fit_predict(X)
                for ex, cid in zip(examples, clusters.tolist()):
                    ex["cluster"] = int(cid)
                summary["clusters"] = int(k)
                cluster_summary = []
                for cid in range(k):
                    cluster_examples = [ex for ex in examples if ex.get("cluster") == cid][:3]
                    cluster_summary.append(
                        {
                            "cluster": cid,
                            "count": sum(1 for ex in examples if ex.get("cluster") == cid),
                            "examples": cluster_examples,
                        }
                    )
                summary["cluster_summary"] = cluster_summary

        return summary, examples

    def _run_images(
        self,
        record: DatasetRunRecord,
        adapter: Any,
        model: Any,
        device: torch.device,
        model_id: str,
    ):
        path = Path(record.request.path or "")
        if not path.is_absolute():
            path = self.settings.paths.safe_data_dir / path
        require_safe_path(path, self.settings.paths.safe_data_dir)

        if not path.is_dir():
            raise ValueError("Image dataset path must be a directory")
        images = sorted(
            file
            for file in path.rglob("*")
            if file.suffix.lower() in {".ppm", ".png", ".jpg", ".jpeg"}
        )
        images = images[: record.request.max_samples]
        if not images:
            raise ValueError("Image dataset contains no supported images")

        examples = []
        for img in images:
            require_safe_path(img, self.settings.paths.safe_data_dir)
            if img.stat().st_size > self.settings.limits.max_upload_mb * 1024**2:
                raise ValueError("Image file exceeds the upload limit")
            inputs = adapter.prepare_inputs(None, {"image_path": img}, device)
            outputs = adapter.forward(model, inputs, {})
            probs = torch.softmax(outputs.logits, dim=-1)[0]
            pred = int(probs.argmax().item())
            conf = float(probs.max().item())

            run_id = safe_run_id(prefix="sample")
            artifacts = self.artifact_store.create(run_id)
            metadata = {
                "run_id": run_id,
                "task_type": "image_classification",
                "model_id": model_id,
                "input_image_path": str(img),
                "status": "completed",
                "created_at": utc_now_iso(),
            }
            outputs_summary = adapter.postprocess(outputs, None, top_k=3)
            self.artifact_store.save_outputs(artifacts, outputs_summary)
            self.artifact_store.save_summaries(artifacts, {})
            self.artifact_store.save_metadata(artifacts, metadata)

            examples.append(
                {
                    "image": str(img),
                    "prediction": outputs_summary.get("prediction", str(pred)),
                    "confidence": round(conf, 4),
                    "run_id": run_id,
                }
            )

        summary = {"count": len(examples)}
        return summary, examples
