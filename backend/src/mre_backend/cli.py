from __future__ import annotations

import typer
import uvicorn

from mre_backend.api.schemas import RunRequest, RunOptions
from mre_backend.core.config import load_settings
from mre_backend.core.artifacts import ArtifactStore
from mre_backend.core.run_manager import RunManager
from mre_backend.core.ws import WsManager
from mre_backend.export_report import export_report

app = typer.Typer(add_completion=False)


@app.command()
def serve(host: str = "0.0.0.0", port: int = 8000):
    """Start the FastAPI server."""
    uvicorn.run("mre_backend.main:app", host=host, port=port, reload=False)


@app.command()
def export(run_id: str = typer.Option(..., "--run-id")):
    """Export a static HTML report for a run."""
    settings_bundle = load_settings()
    settings = settings_bundle.settings
    path = export_report(run_id, settings.paths.runs_dir, settings.paths.reports_dir)
    typer.echo(f"Report written to {path}")


@app.command("demo-text")
def demo_text(prompt: str = "The capital of France is", max_tokens: int = 12):
    """Run a demo text LM inference and save artifacts."""
    settings_bundle = load_settings()
    settings = settings_bundle.settings
    artifact_store = ArtifactStore(settings.paths.runs_dir)
    ws_manager = WsManager()
    run_manager = RunManager(settings, artifact_store, ws_manager)

    options = RunOptions(max_tokens=max_tokens)
    request = RunRequest(
        task_type="text_lm",
        model_id="sshleifer/tiny-gpt2",
        input_text=prompt,
        options=options,
    )
    record = run_manager.run_sync(request)
    typer.echo(f"Run complete: {record.run_id}")


@app.command("demo-vision")
def demo_vision(image: str):
    """Run a demo image classification inference and save artifacts."""
    settings_bundle = load_settings()
    settings = settings_bundle.settings
    artifact_store = ArtifactStore(settings.paths.runs_dir)
    ws_manager = WsManager()
    run_manager = RunManager(settings, artifact_store, ws_manager)

    request = RunRequest(
        task_type="image_classification",
        model_id="resnet18",
        input_image_path=image,
    )
    record = run_manager.run_sync(request)
    typer.echo(f"Run complete: {record.run_id}")


@app.command("list-models")
def list_models():
    from mre_backend.core.model_registry import DEMO_MODELS

    for model in DEMO_MODELS:
        typer.echo(f"{model['id']} ({model['task_type']})")


if __name__ == "__main__":
    app()
