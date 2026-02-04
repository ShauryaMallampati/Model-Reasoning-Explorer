from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from mre_backend.api.schemas import (
    CompareRequest,
    CompareResponse,
    DatasetRunResponse,
    DatasetRunStatus,
    RunRequest,
    RunResponse,
    RunStatusResponse,
)
from mre_backend.core.model_registry import DEMO_MODELS
from mre_backend.export_report import export_report

router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/api/models")
async def list_models() -> dict[str, Any]:
    return {"models": DEMO_MODELS}


@router.post("/api/run", response_model=RunResponse)
async def start_run(request: Request, payload: RunRequest) -> RunResponse:
    run_manager = request.app.state.run_manager
    run_id = run_manager.start_run(payload)
    return RunResponse(run_id=run_id)


@router.post("/api/run/{run_id}/cancel")
async def cancel_run(request: Request, run_id: str) -> dict[str, Any]:
    run_manager = request.app.state.run_manager
    cancelled = run_manager.cancel(run_id)
    return {"cancelled": cancelled}


@router.get("/api/run/{run_id}", response_model=RunStatusResponse)
async def get_run(request: Request, run_id: str) -> RunStatusResponse:
    run_manager = request.app.state.run_manager
    artifact_store = request.app.state.artifact_store

    record = run_manager.get_run(run_id)
    metadata = None
    outputs = None
    summaries = None
    artifacts = []

    run_dir = artifact_store.get_run_dir(run_id)
    if run_dir.exists():
        metadata_path = run_dir / "metadata.json"
        outputs_path = run_dir / "outputs.json"
        summaries_path = run_dir / "summaries.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
        if outputs_path.exists():
            outputs = json.loads(outputs_path.read_text())
        if summaries_path.exists():
            summaries = json.loads(summaries_path.read_text())
        artifacts = [p.name for p in run_dir.iterdir() if p.is_file()]

    status = record.status if record else (metadata.get("status") if metadata else "unknown")
    return RunStatusResponse(
        run_id=run_id,
        status=status,
        metadata=metadata,
        outputs=outputs,
        summaries=summaries,
        artifacts=artifacts,
    )


@router.get("/api/run/{run_id}/artifact/{name}")
async def get_artifact(request: Request, run_id: str, name: str):
    artifact_store = request.app.state.artifact_store
    path = artifact_store.get_artifact_path(run_id, name)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(path)


@router.post("/api/compare", response_model=CompareResponse)
async def compare_runs(request: Request, payload: CompareRequest) -> CompareResponse:
    artifact_store = request.app.state.artifact_store

    def load(run_id: str) -> dict[str, Any]:
        run_dir = artifact_store.get_run_dir(run_id)
        if not run_dir.exists():
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
        outputs = json.loads((run_dir / "outputs.json").read_text())
        summaries = {}
        summaries_path = run_dir / "summaries.json"
        if summaries_path.exists():
            summaries = json.loads(summaries_path.read_text())
        return {"outputs": outputs, "summaries": summaries}

    a = load(payload.run_a)
    b = load(payload.run_b)

    summary = {
        "run_a": payload.run_a,
        "run_b": payload.run_b,
        "prediction_a": a["outputs"].get("prediction"),
        "prediction_b": b["outputs"].get("prediction"),
        "top_k_a": a["outputs"].get("top_k", []),
        "top_k_b": b["outputs"].get("top_k", []),
    }

    logit_lens_a = a["summaries"].get("logit_lens")
    logit_lens_b = b["summaries"].get("logit_lens")
    if logit_lens_a and logit_lens_b:
        layers_a = logit_lens_a.get("layers", [])
        layers_b = logit_lens_b.get("layers", [])
        count = min(len(layers_a), len(layers_b))
        deltas = []
        for i in range(count):
            pa = layers_a[i].get("top_k", [{}])[0].get("prob", 0)
            pb = layers_b[i].get("top_k", [{}])[0].get("prob", 0)
            deltas.append(pa - pb)
        if deltas:
            max_idx = max(range(len(deltas)), key=lambda i: abs(deltas[i]))
        else:
            max_idx = None
        summary["logit_lens_delta"] = {
            "layers": count,
            "deltas": deltas,
            "pinpoint_layer": max_idx,
        }

    return CompareResponse(summary=summary)


@router.post("/api/dataset/run", response_model=DatasetRunResponse)
async def start_dataset_run(
    request: Request,
    kind: str = Form(...),
    model_id: str = Form("distilbert-base-uncased-finetuned-sst-2-english"),
    task_type: str = Form("text_classification"),
    path: str | None = Form(None),
    text_column: str = Form("text"),
    label_column: str = Form("label"),
    max_samples: int = Form(50),
    file: UploadFile | None = File(None),
) -> DatasetRunResponse:
    settings = request.app.state.settings
    dataset_manager = request.app.state.dataset_manager

    if file is not None:
        upload_dir = settings.paths.safe_data_dir / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        dest = upload_dir / file.filename
        data = await file.read()
        max_bytes = settings.limits.max_upload_mb * 1024 * 1024
        if len(data) > max_bytes:
            raise HTTPException(status_code=413, detail="File too large")
        dest.write_bytes(data)
        path = str(dest)

    if not path:
        raise HTTPException(status_code=400, detail="Dataset path or file required")

    from mre_backend.api.schemas import DatasetRunRequest

    req = DatasetRunRequest(
        kind=kind,
        path=path,
        text_column=text_column,
        label_column=label_column,
        max_samples=max_samples,
    )
    dataset_run_id = dataset_manager.start(req, model_id, task_type)
    return DatasetRunResponse(dataset_run_id=dataset_run_id)


@router.get("/api/dataset/{dataset_run_id}", response_model=DatasetRunStatus)
async def get_dataset_run(request: Request, dataset_run_id: str) -> DatasetRunStatus:
    dataset_manager = request.app.state.dataset_manager
    record = dataset_manager.get(dataset_run_id)
    if not record:
        raise HTTPException(status_code=404, detail="Dataset run not found")
    return DatasetRunStatus(
        dataset_run_id=dataset_run_id,
        status=record.status,
        summary=record.summary,
        examples=record.examples or [],
    )


@router.post("/api/run/{run_id}/export")
async def export_run(request: Request, run_id: str) -> dict[str, str]:
    settings = request.app.state.settings
    path = export_report(run_id, settings.paths.runs_dir, settings.paths.reports_dir)
    return {"report_path": str(path)}
