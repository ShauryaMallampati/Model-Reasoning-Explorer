from __future__ import annotations

import csv
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
from mre_backend.core.model_cache import CachedModel, ModelCache
from mre_backend.core.utils import require_safe_path, safe_run_id, set_seed, utc_now_iso


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
        return self._runs.get(dataset_run_id)

    def _run_task(self, record: DatasetRunRecord, model_id: str, task_type: str) -> None:
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
                model, tokenizer = adapter.load(model_id, device)
                self.model_cache.set(
                    cache_key,
                    CachedModel(model=model, tokenizer=tokenizer, task_type=task_type, model_id=model_id),
                )

            if record.request.kind == "text":
                summary, examples = self._run_text(record, adapter, model, tokenizer, device, model_id)
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

        rows = []
        with path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
                if len(rows) >= record.request.max_samples:
                    break

        examples = []
        y_true = []
        embeddings = []

        for row in rows:
            text = row.get(record.request.text_column, "")
            label = row.get(record.request.label_column, "")
            inputs = adapter.prepare_inputs(tokenizer, {"text": text}, device)
            outputs = adapter.forward(model, inputs, {"hidden_states": True, "attentions": False})
            pred = int(outputs.logits.argmax(dim=-1).item())
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
            self.artifact_store.save_metadata(artifacts, metadata)
            self.artifact_store.save_outputs(artifacts, outputs_summary)
            self.artifact_store.save_summaries(artifacts, {})

            examples.append(
                {
                    "text": text,
                    "label": label,
                    "prediction": str(pred),
                    "run_id": run_id,
                }
            )

        summary = {
            "count": len(rows),
            "labels": sorted(set(y_true)),
        }
        if embeddings:
            X = np.stack(embeddings)
            k = min(4, len(X))
            if k >= 2:
                km = KMeans(n_clusters=k, n_init=5, random_state=self.settings.execution.seed)
                clusters = km.fit_predict(X)
                for ex, cid in zip(examples, clusters.tolist()):
                    ex["cluster"] = int(cid)
                summary["clusters"] = int(k)

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

        images = list(path.glob("**/*.ppm")) + list(path.glob("**/*.png"))
        images = images[: record.request.max_samples]

        examples = []
        for img in images:
            inputs = adapter.prepare_inputs(None, {"image_path": img}, device)
            outputs = adapter.forward(model, inputs, {})
            pred = int(outputs.logits.argmax(dim=-1).item())

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
            self.artifact_store.save_metadata(artifacts, metadata)
            self.artifact_store.save_outputs(artifacts, outputs_summary)
            self.artifact_store.save_summaries(artifacts, {})

            examples.append({"image": str(img), "prediction": str(pred), "run_id": run_id})

        summary = {"count": len(examples)}
        return summary, examples
