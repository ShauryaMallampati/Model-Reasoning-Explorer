from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from mre_backend.api.schemas import (
    CompareRequest,
    CompareResponse,
    DatasetRunRequest,
    DatasetRunResponse,
    DatasetRunStatus,
    RunRequest,
    RunResponse,
    RunStatusResponse,
)
from mre_backend.core.comparison import compare_artifacts
from mre_backend.core.model_registry import DEMO_MODELS
from mre_backend.core.utils import require_safe_path, resolve_model_id, safe_component
from mre_backend.export_report import export_report

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/api/models")
def list_models(request: Request) -> dict[str, Any]:
    allowed = request.app.state.settings.models.allowlist
    return {"models": [model for model in DEMO_MODELS if model["id"] in allowed]}


@router.post("/api/run", response_model=RunResponse)
def start_run(request: Request, payload: RunRequest) -> RunResponse:
    resolve_model_id(payload.model_id, request.app.state.settings)
    run_id = request.app.state.run_manager.start_run(payload)
    return RunResponse(run_id=run_id)


@router.post("/api/run/{run_id}/cancel")
def cancel_run(request: Request, run_id: str) -> dict[str, Any]:
    safe_component(run_id)
    return {"cancelled": request.app.state.run_manager.cancel(run_id)}


@router.get("/api/run/{run_id}", response_model=RunStatusResponse)
def get_run(request: Request, run_id: str) -> RunStatusResponse:
    store = request.app.state.artifact_store
    record = request.app.state.run_manager.get_run(run_id)
    root = store.get_run_dir(run_id)
    if record is None and not root.is_dir():
        raise HTTPException(404, "Run not found")
    values = {}
    for name in ("metadata", "outputs", "summaries"):
        path = store.get_artifact_path(run_id, f"{name}.json")
        values[name] = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    metadata = values["metadata"] or {}
    return RunStatusResponse(
        run_id=run_id,
        status=record.status if record else metadata.get("status", "queued"),
        error=record.error if record else metadata.get("error"),
        artifacts=(
            sorted(
                path.name
                for path in root.iterdir()
                if path.is_file() and not path.name.startswith(".")
            )
            if root.is_dir()
            else []
        ),
        **values,
    )


@router.get("/api/run/{run_id}/artifact/{name}")
def get_artifact(request: Request, run_id: str, name: str):
    path = request.app.state.artifact_store.get_artifact_path(run_id, name)
    if not path.is_file():
        raise HTTPException(404, "Artifact not found")
    return FileResponse(path, filename=path.name)


@router.post("/api/compare", response_model=CompareResponse)
def compare_runs(request: Request, payload: CompareRequest) -> CompareResponse:
    try:
        summary = compare_artifacts(request.app.state.artifact_store, payload.run_a, payload.run_b)
    except FileNotFoundError as exc:
        raise HTTPException(404, "Run or required artifact not found") from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return CompareResponse(summary=summary)


@router.post("/api/dataset/run", response_model=DatasetRunResponse)
async def start_dataset_run(
    request: Request,
    kind: Literal["text", "image"] = Form(...),
    model_id: str = Form("distilbert-base-uncased-finetuned-sst-2-english"),
    task_type: Literal["text_classification", "image_classification"] = Form("text_classification"),
    path: str | None = Form(None),
    text_column: str = Form("text"),
    label_column: str = Form("label"),
    max_samples: int = Form(50, ge=1, le=200),
    file: UploadFile | None = File(None),
) -> DatasetRunResponse:
    settings = request.app.state.settings
    resolve_model_id(model_id, settings)
    if (kind == "text") != (task_type == "text_classification"):
        raise HTTPException(422, "Dataset kind and model task must agree")
    if file is not None:
        if kind != "text" or not (file.filename or "").lower().endswith(".csv"):
            raise HTTPException(422, "Only text CSV uploads are supported")
        max_bytes = settings.limits.max_upload_mb * 1024 * 1024
        try:
            data = await file.read(max_bytes + 1)
        finally:
            await file.close()
        if len(data) > max_bytes:
            raise HTTPException(413, "File too large")
        # Never use a caller-supplied filename as a filesystem path.
        upload_dir = require_safe_path(
            settings.paths.safe_data_dir / "uploads", settings.paths.safe_data_dir
        )
        upload_dir.mkdir(parents=True, exist_ok=True)
        destination = upload_dir / f"dataset_{uuid4().hex}.csv"
        destination.write_bytes(data)
        path = str(destination)
    if not path:
        raise HTTPException(400, "Dataset path or file required")
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = settings.paths.safe_data_dir / candidate
    require_safe_path(candidate, settings.paths.safe_data_dir)
    payload = DatasetRunRequest(
        kind=kind,
        path=str(candidate),
        text_column=text_column,
        label_column=label_column,
        max_samples=max_samples,
    )
    run_id = request.app.state.dataset_manager.start(payload, model_id, task_type)
    return DatasetRunResponse(dataset_run_id=run_id)


@router.get("/api/dataset/{dataset_run_id}", response_model=DatasetRunStatus)
def get_dataset_run(request: Request, dataset_run_id: str) -> DatasetRunStatus:
    safe_component(dataset_run_id)
    record = request.app.state.dataset_manager.get(dataset_run_id)
    if record is None:
        raise HTTPException(404, "Dataset run not found")
    return DatasetRunStatus(
        dataset_run_id=dataset_run_id,
        status=record.status,
        error=record.error,
        summary=record.summary,
        examples=record.examples or [],
    )


@router.post("/api/run/{run_id}/export")
def export_run(request: Request, run_id: str) -> dict[str, str]:
    settings = request.app.state.settings
    try:
        export_report(run_id, settings.paths.runs_dir, settings.paths.reports_dir)
    except FileNotFoundError as exc:
        raise HTTPException(404, "Run not found") from exc
    return {"report_url": f"/api/run/{safe_component(run_id)}/report"}


@router.get("/api/run/{run_id}/report")
def get_report(request: Request, run_id: str):
    root = request.app.state.settings.paths.reports_dir
    path = require_safe_path(root / safe_component(run_id) / "index.html", root)
    if not path.is_file():
        raise HTTPException(404, "Export the report first")
    return FileResponse(
        path,
        media_type="text/html",
        headers={
            "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'",
            "X-Content-Type-Options": "nosniff",
        },
    )
