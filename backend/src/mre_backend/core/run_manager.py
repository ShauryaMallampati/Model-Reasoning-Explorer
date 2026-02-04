from __future__ import annotations

import asyncio
import base64
import io
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
from mre_backend.api.schemas import RunRequest
from mre_backend.capture.hooks import CaptureStore, HookManager
from mre_backend.core.artifacts import ArtifactStore
from mre_backend.core.model_cache import CachedModel, ModelCache
from mre_backend.core.utils import require_safe_path, safe_run_id, set_seed, utc_now_iso


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
        self.executor = ThreadPoolExecutor(max_workers=2)
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

    def _validate_model(self, model_id: str) -> None:
        allowlist = set(self.settings.models.allowlist)
        if model_id in allowlist:
            return
        model_path = Path(model_id)
        if not model_path.is_absolute():
            model_path = self.settings.paths.safe_model_dir / model_path
        require_safe_path(model_path, self.settings.paths.safe_model_dir)
        if not model_path.exists():
            raise ValueError("Model path not found")

    def _decode_image(self, request: RunRequest) -> tuple[Image.Image, str | None]:
        if request.input_image_base64:
            data = base64.b64decode(request.input_image_base64)
            image = Image.open(io.BytesIO(data)).convert("RGB")
            return image, request.input_image_base64
        if request.input_image_path:
            path = Path(request.input_image_path)
            if not path.is_absolute():
                path = self.settings.paths.safe_data_dir / path
            require_safe_path(path, self.settings.paths.safe_data_dir)
            image = Image.open(path).convert("RGB")
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            preview = base64.b64encode(buffered.getvalue()).decode("utf-8")
            return image, preview
        raise ValueError("Image input required")

    def _broadcast(self, run_id: str, event: dict[str, Any]) -> None:
        try:
            asyncio.run(self.ws_manager.broadcast(run_id, event))
        except RuntimeError:
            pass

    def _run_task(self, record: RunRecord) -> None:
        run_id = record.run_id
        artifacts = self.artifact_store.create(run_id)

        def log(message: str) -> None:
            self.artifact_store.append_log(artifacts, message)
            self._broadcast(run_id, {"type": "log", "message": message})

        def check_cancel() -> None:
            if record.cancel_event and record.cancel_event.is_set():
                raise RuntimeError("Run cancelled")

        try:
            record.status = "running"
            record.updated_at = utc_now_iso()

            self._validate_model(record.request.model_id)
            seed = record.request.options.seed or self.settings.execution.seed
            set_seed(seed)
            log("Validated request and set seed")

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
                model, tokenizer = adapter.load(record.request.model_id, device)
                self.model_cache.set(
                    cache_key,
                    CachedModel(
                        model=model,
                        tokenizer=tokenizer,
                        task_type=record.request.task_type,
                        model_id=record.request.model_id,
                    ),
                )

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

            capture_opts = {
                "gradients": record.request.options.capture.gradients,
                "hidden_states": record.request.options.capture.hidden_states,
                "attentions": record.request.options.capture.attentions,
            }
            layers = adapter.select_layers(model, record.request.options.capture.layers.model_dump())
            capture_store = CaptureStore()
            hook_manager = None
            if record.request.options.capture.activations or record.request.options.capture.gradients:
                hook_manager = HookManager(model, layers, record.request.options.capture.gradients)
                capture_store = hook_manager.attach()

            log("Running forward pass")
            outputs = adapter.forward(model, inputs, capture_opts)
            check_cancel()

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
                try:
                    max_new = max(1, int(record.request.options.max_tokens))
                    gen_ids = model.generate(**inputs, max_new_tokens=max_new)
                    output_summary["generated_text"] = tokenizer.decode(gen_ids[0])
                except Exception:
                    pass

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

            check_cancel()
            for analyzer in self._analyzers:
                if analyzer.id in analyzer_ids and analyzer.supports(record.request.task_type):
                    log(f"Running analyzer: {analyzer.id}")
                    output = analyzer.run(context)
                    summaries[analyzer.id] = output.summary
                    if output.arrays:
                        path = self.artifact_store.save_arrays(artifacts, analyzer.id, output.arrays)
                        arrays_saved.append(path.name)

            log("Saving artifacts")
            if record.request.options.capture.hidden_states and outputs.hidden_states:
                hs_arrays = {
                    f"layer_{i}": h.detach().cpu().numpy() for i, h in enumerate(outputs.hidden_states)
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
                "input_image_path": record.request.input_image_path,
                "input_image_preview": image_preview,
            }
            self.artifact_store.save_metadata(artifacts, metadata)
            self.artifact_store.save_outputs(artifacts, output_summary)
            self.artifact_store.save_summaries(artifacts, summaries)

            record.status = "completed"
            record.outputs = output_summary
            record.summaries = summaries
            record.artifacts = arrays_saved + ["metadata.json", "outputs.json", "summaries.json", "logs.txt"]
            record.updated_at = utc_now_iso()

            log("Run completed")
            self._broadcast(run_id, {"type": "status", "status": "completed", "run_id": run_id})

        except Exception as exc:
            record.status = "failed"
            record.error = str(exc)
            record.updated_at = utc_now_iso()
            log(f"Run failed: {exc}")
