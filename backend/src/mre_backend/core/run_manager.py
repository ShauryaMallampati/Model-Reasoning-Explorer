from __future__ import annotations

import base64
import io
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from mre_backend.adapters.text_transformer import TextTransformerAdapter
from mre_backend.adapters.vision_resnet import VisionResNetAdapter
from mre_backend.analyzers import (
    ActivationPatchingAnalyzer,
    AttentionRolloutAnalyzer,
    CounterfactualSearchAnalyzer,
    GradCamAnalyzer,
    IntegratedGradientsAnalyzer,
    LogitLensAnalyzer,
    OcclusionAnalyzer,
)
from mre_backend.analyzers.logit_lens import project_gpt2_states
from mre_backend.api.schemas import RunRequest
from mre_backend.capture.hooks import CaptureStore, HookManager
from mre_backend.core.artifacts import ArtifactStore
from mre_backend.core.model_cache import INFERENCE_LOCK, CachedModel, ModelCache
from mre_backend.core.utils import (
    require_safe_path,
    resolve_model_id,
    safe_run_id,
    set_seed,
    utc_now_iso,
)


@dataclass
class RunRecord:
    run_id: str
    status: str
    request: RunRequest
    created_at: str
    updated_at: str
    error: str | None = None
    outputs: dict[str, Any] | None = None
    summaries: dict[str, Any] | None = None
    artifacts: list[str] | None = None
    cancel_event: threading.Event | None = None


@dataclass
class RunContext:
    run_id: str
    task_type: str
    model_id: str
    model: Any
    tokenizer: Any | None
    inputs: dict[str, Any]
    outputs: Any
    capture: Any
    request: RunRequest
    device: torch.device


class RunManager:
    def __init__(self, settings: Any, artifact_store: ArtifactStore, ws_manager: Any) -> None:
        self.settings = settings
        self.artifact_store = artifact_store
        self.ws_manager = ws_manager
        self.model_cache = ModelCache()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self._runs: dict[str, RunRecord] = {}
        self._lock = threading.Lock()
        self._analyzers = [
            AttentionRolloutAnalyzer(),
            LogitLensAnalyzer(),
            IntegratedGradientsAnalyzer(),
            GradCamAnalyzer(),
            OcclusionAnalyzer(),
            ActivationPatchingAnalyzer(),
            CounterfactualSearchAnalyzer(),
        ]

    def start_run(self, request: RunRequest) -> str:
        run_id = safe_run_id()
        now = utc_now_iso()
        record = RunRecord(
            run_id=run_id,
            status="queued",
            request=request,
            created_at=now,
            updated_at=now,
            cancel_event=threading.Event(),
        )
        with self._lock:
            self._runs[run_id] = record
        self.executor.submit(self._run_task, record)
        return run_id

    def run_sync(self, request: RunRequest) -> RunRecord:
        run_id = safe_run_id()
        now = utc_now_iso()
        record = RunRecord(
            run_id=run_id,
            status="queued",
            request=request,
            created_at=now,
            updated_at=now,
            cancel_event=threading.Event(),
        )
        self._runs[run_id] = record
        self._run_task(record)
        return record

    def get_run(self, run_id: str) -> RunRecord | None:
        return self._runs.get(run_id)

    def cancel(self, run_id: str) -> bool:
        record = self._runs.get(run_id)
        if not record or not record.cancel_event:
            return False
        record.cancel_event.set()
        return True

    def _select_adapter(self, task_type: str):
        if task_type in {"text_lm", "text_classification"}:
            return TextTransformerAdapter(task_type)
        if task_type == "image_classification":
            return VisionResNetAdapter()
        raise ValueError(f"Unknown task type: {task_type}")

    def _validate_model(self, model_id: str) -> str:
        return resolve_model_id(model_id, self.settings)

    def _decode_image(self, request: RunRequest) -> tuple[Image.Image, str | None]:
        if request.input_image_base64:
            payload = request.input_image_base64
            if "," in payload:
                payload = payload.split(",", 1)[1]
            max_bytes = self.settings.limits.max_upload_mb * 1024 * 1024
            estimated = (len(payload) * 3) // 4
            if estimated > max_bytes:
                raise ValueError("Image payload too large")
            data = base64.b64decode(payload, validate=True)
            if len(data) > max_bytes:
                raise ValueError("Image payload too large")
            with Image.open(io.BytesIO(data)) as source:
                if source.width * source.height > 16_000_000:
                    raise ValueError("Image resolution exceeds 16 million pixels")
                image = source.convert("RGB")
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            return image, base64.b64encode(buffer.getvalue()).decode("ascii")
        if request.input_image_path:
            path = Path(request.input_image_path)
            if not path.is_absolute():
                path = self.settings.paths.safe_data_dir / path
            path = require_safe_path(path, self.settings.paths.safe_data_dir)
            if path.stat().st_size > self.settings.limits.max_upload_mb * 1024**2:
                raise ValueError("Image file too large")
            with Image.open(path) as source:
                if source.width * source.height > 16_000_000:
                    raise ValueError("Image resolution exceeds 16 million pixels")
                image = source.convert("RGB")
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            preview = base64.b64encode(buffered.getvalue()).decode("utf-8")
            return image, preview
        raise ValueError("Image input required")

    def _broadcast(self, run_id: str, event: dict[str, Any]) -> None:
        self.ws_manager.publish(run_id, event)

    def _run_task(self, record: RunRecord) -> None:
        with INFERENCE_LOCK:
            self._run_task_serial(record)

    def _run_task_serial(self, record: RunRecord) -> None:
        run_id = record.run_id
        artifacts = self.artifact_store.create(run_id)
        hook_manager = None
        model = None

        def log(message: str) -> None:
            self.artifact_store.append_log(artifacts, message)
            self._broadcast(run_id, {"type": "log", "message": message})

        def progress(value: int, stage: str) -> None:
            self._broadcast(run_id, {"type": "progress", "value": value, "stage": stage})

        def check_cancel() -> None:
            if record.cancel_event and record.cancel_event.is_set():
                raise RuntimeError("Run cancelled")

        try:
            record.status = "running"
            record.updated_at = utc_now_iso()

            check_cancel()
            resolved_model = self._validate_model(record.request.model_id)
            seed = record.request.options.seed
            if seed is None:
                seed = self.settings.execution.seed
            set_seed(seed)
            log("Validated request and set seed")
            progress(10, "validated")

            device = torch.device("cpu")
            if self.settings.execution.allow_gpu and torch.cuda.is_available():
                device = torch.device("cuda")

            adapter = self._select_adapter(record.request.task_type)
            log("Loading model")
            cache_key = self.model_cache.make_key(record.request.task_type, record.request.model_id)
            cached = self.model_cache.get(cache_key)
            if cached:
                model = cached.model
                tokenizer = cached.tokenizer
            else:
                model, tokenizer = adapter.load(resolved_model, device)
                self.model_cache.set(
                    cache_key,
                    CachedModel(
                        model=model,
                        tokenizer=tokenizer,
                        task_type=record.request.task_type,
                        model_id=record.request.model_id,
                    ),
                )
            model.zero_grad(set_to_none=True)
            progress(20, "model_loaded")

            image_preview = None
            raw_input: dict[str, Any] = {}
            if record.request.task_type.startswith("text"):
                raw_input["text"] = record.request.input_text
            else:
                image, image_preview = self._decode_image(record.request)
                raw_input["image"] = image

            log("Preparing inputs")
            inputs = adapter.prepare_inputs(tokenizer, raw_input, device)
            check_cancel()
            progress(30, "inputs_ready")

            capture_opts = {
                "gradients": record.request.options.capture.gradients,
                "hidden_states": record.request.options.capture.hidden_states,
                "attentions": record.request.options.capture.attentions,
            }
            layers = adapter.select_layers(
                model, record.request.options.capture.layers.model_dump()
            )
            if len(layers) > self.settings.limits.max_layers:
                raise ValueError("Selected layers exceed the configured capture limit")
            capture_store = CaptureStore()
            if (
                record.request.options.capture.activations
                or record.request.options.capture.gradients
            ):
                hook_manager = HookManager(model, layers, record.request.options.capture.gradients)
                capture_store = hook_manager.attach()

            log("Running forward pass")
            outputs = adapter.forward(model, inputs, capture_opts)
            check_cancel()
            progress(50, "forward_done")

            if record.request.options.capture.gradients:
                if record.request.task_type == "text_lm":
                    target_logit = outputs.logits[:, -1, :].max()
                else:
                    target_logit = outputs.logits.max()
                target_logit.backward()

            if hook_manager:
                hook_manager.clear()

            log("Post-processing outputs")
            output_summary = adapter.postprocess(outputs, tokenizer, record.request.options.top_k)

            if record.request.task_type == "text_lm" and tokenizer is not None:
                with torch.no_grad():
                    gen_ids = model.generate(
                        **inputs,
                        max_new_tokens=record.request.options.max_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                output_summary["generated_text"] = tokenizer.decode(gen_ids[0])

            log("Running analyzers")
            summaries: dict[str, Any] = {}
            arrays_saved: list[str] = []
            context = RunContext(
                run_id=run_id,
                task_type=record.request.task_type,
                model_id=record.request.model_id,
                model=model,
                tokenizer=tokenizer,
                inputs=inputs,
                outputs=outputs,
                capture=capture_store,
                request=record.request,
                device=device,
            )

            analyzer_ids = record.request.options.analyzers
            if not analyzer_ids:
                if record.request.task_type == "text_lm":
                    analyzer_ids = ["attention_rollout", "logit_lens", "integrated_gradients"]
                elif record.request.task_type == "text_classification":
                    analyzer_ids = ["integrated_gradients", "counterfactual_search"]
                else:
                    analyzer_ids = [
                        "grad_cam",
                        "occlusion",
                        "integrated_gradients",
                        "counterfactual_search",
                    ]

            supported = {
                item.id for item in self._analyzers if item.supports(record.request.task_type)
            }
            if set(analyzer_ids) - supported:
                raise ValueError("Unknown or unsupported analyzer requested")
            check_cancel()
            for analyzer in self._analyzers:
                if analyzer.id in analyzer_ids and analyzer.supports(record.request.task_type):
                    log(f"Running analyzer: {analyzer.id}")
                    output = analyzer.run(context)
                    summaries[analyzer.id] = output.summary
                    if output.arrays:
                        path = self.artifact_store.save_arrays(
                            artifacts, analyzer.id, output.arrays
                        )
                        arrays_saved.append(path.name)
            progress(75, "analyzers_done")

            log("Saving artifacts")
            if (
                record.request.options.capture.logits
                and outputs.hidden_states
                and hasattr(model, "lm_head")
            ):
                values_list = []
                indices_list = []
                for last_logits in project_gpt2_states(model, outputs.hidden_states):
                    vals, idxs = torch.topk(
                        last_logits,
                        k=min(record.request.options.top_k, last_logits.shape[-1]),
                        dim=-1,
                    )
                    values_list.append(vals.detach().cpu().numpy())
                    indices_list.append(idxs.detach().cpu().numpy())
                arrays = {
                    "values": np.stack(values_list),
                    "indices": np.stack(indices_list),
                }
                path = self.artifact_store.save_arrays(artifacts, "logits_topk", arrays)
                arrays_saved.append(path.name)
            if record.request.options.capture.hidden_states and outputs.hidden_states:
                hs_arrays = {
                    f"layer_{i}": h.detach().cpu().numpy()
                    for i, h in enumerate(outputs.hidden_states)
                }
                if hs_arrays:
                    path = self.artifact_store.save_arrays(artifacts, "hidden_states", hs_arrays)
                    arrays_saved.append(path.name)
            if record.request.options.capture.attentions and outputs.attentions:
                att_arrays = {
                    f"layer_{i}": a.detach().cpu().numpy() for i, a in enumerate(outputs.attentions)
                }
                if att_arrays:
                    path = self.artifact_store.save_arrays(artifacts, "attentions", att_arrays)
                    arrays_saved.append(path.name)

            if record.request.options.capture.activations:
                act_arrays = {
                    k.replace(".", "_"): v.numpy() for k, v in capture_store.activations.items()
                }
                if act_arrays:
                    path = self.artifact_store.save_arrays(artifacts, "activations", act_arrays)
                    arrays_saved.append(path.name)
                    stats = []
                    for name, tensor in capture_store.activations.items():
                        stats.append(
                            {
                                "layer": name,
                                "mean_abs": float(tensor.abs().mean().item()),
                            }
                        )
                    stats_sorted = sorted(stats, key=lambda x: x["mean_abs"], reverse=True)
                    summaries["activation_stats"] = {
                        "layers": stats,
                        "top_layers": stats_sorted[:10],
                    }
            if record.request.options.capture.gradients:
                grad_arrays = {
                    k.replace(".", "_"): v.numpy() for k, v in capture_store.gradients.items()
                }
                if grad_arrays:
                    path = self.artifact_store.save_arrays(artifacts, "gradients", grad_arrays)
                    arrays_saved.append(path.name)

            metadata = {
                "run_id": run_id,
                "task_type": record.request.task_type,
                "model_id": record.request.model_id,
                "created_at": record.created_at,
                "updated_at": utc_now_iso(),
                "seed": seed,
                "status": "completed",
                "input_text": record.request.input_text,
                "input_tokens": (
                    [tokenizer.decode([token]) for token in inputs["input_ids"][0].tolist()]
                    if tokenizer is not None
                    else []
                ),
                "model_revision": getattr(getattr(model, "config", None), "_commit_hash", None),
                "options": record.request.options.model_dump(),
                "input_image_path": record.request.input_image_path,
                "input_image_preview": image_preview,
            }
            self.artifact_store.save_outputs(artifacts, output_summary)
            self.artifact_store.save_summaries(artifacts, summaries)
            # Publish completed metadata last, after all artifacts are readable.
            self.artifact_store.save_metadata(artifacts, metadata)
            progress(95, "artifacts_saved")

            record.status = "completed"
            record.outputs = output_summary
            record.summaries = summaries
            record.artifacts = arrays_saved + [
                "metadata.json",
                "outputs.json",
                "summaries.json",
                "logs.txt",
            ]
            record.updated_at = utc_now_iso()

            log("Run completed")
            self._broadcast(run_id, {"type": "status", "status": "completed", "run_id": run_id})
            progress(100, "completed")

        except Exception as exc:
            record.status = "failed"
            record.error = str(exc)
            record.updated_at = utc_now_iso()
            log(f"Run failed: {exc}")
            self.artifact_store.save_metadata(
                artifacts,
                {
                    "run_id": run_id,
                    "status": "failed",
                    "error": record.error,
                    "task_type": record.request.task_type,
                    "model_id": record.request.model_id,
                    "created_at": record.created_at,
                    "updated_at": record.updated_at,
                },
            )
            self._broadcast(run_id, {"type": "status", "status": "failed", "run_id": run_id})
        finally:
            if hook_manager is not None:
                hook_manager.clear()
            if model is not None:
                model.zero_grad(set_to_none=True)
