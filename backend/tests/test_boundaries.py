import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from mre_backend.api.schemas import RunOptions, RunRequest
from mre_backend.core.artifacts import ArtifactStore
from mre_backend.core.config import load_settings
from mre_backend.core.utils import resolve_model_id, safe_component, safe_run_id, set_seed
from mre_backend.export_report import export_report
from mre_backend.main import create_app


@pytest.mark.parametrize(
    "name", ["..", "../outside", "/absolute", "a/b", "a\\b", ".hidden", "a..b"]
)
def test_identifiers_cannot_be_paths(name):
    with pytest.raises(ValueError):
        safe_component(name)


def test_symlink_artifact_escape_is_rejected(tmp_path):
    store = ArtifactStore(tmp_path / "runs")
    artifacts = store.create("safe_run")
    outside = tmp_path / "private.txt"
    outside.write_text("private")
    (artifacts.run_dir / "link.txt").symlink_to(outside)
    with pytest.raises(ValueError):
        store.get_artifact_path("safe_run", "link.txt")


def test_model_ids_are_resolved_inside_the_allowed_directory(settings):
    local = settings.paths.safe_model_dir / "tiny"
    local.mkdir()
    assert resolve_model_id("tiny", settings) == str(local.resolve())
    with pytest.raises(ValueError):
        resolve_model_id("../data", settings)
    with pytest.raises(ValueError):
        resolve_model_id("unapproved/hub-model", settings)


def test_seeded_runs_do_not_reuse_artifact_ids():
    set_seed(0)
    first = safe_run_id()
    set_seed(0)
    assert safe_run_id() != first


@pytest.mark.parametrize(
    "options",
    [
        {"top_k": 0},
        {"max_tokens": 129},
        {"seed": -1},
        {"layers": {"mode": "every_n", "stride": 0}},
        {"capture": {"layers": {"mode": "every_n", "stride": 0}}},
    ],
)
def test_invalid_execution_options_fail_before_scheduling(options):
    with pytest.raises(ValidationError):
        RunOptions(**options)


def test_empty_text_and_ambiguous_image_input_are_rejected():
    with pytest.raises(ValidationError):
        RunRequest(task_type="text_lm", model_id="sshleifer/tiny-gpt2", input_text="  ")
    with pytest.raises(ValidationError):
        RunRequest(
            task_type="image_classification",
            model_id="resnet18",
            input_image_path="image.png",
            input_image_base64="abc",
        )


def test_default_config_path_is_a_file(monkeypatch):
    monkeypatch.delenv("MRE_CONFIG", raising=False)
    assert load_settings().config_path.is_file()
    monkeypatch.setenv("MRE_CONFIG", "definitely-missing.toml")
    with pytest.raises(ValueError, match="Configuration"):
        load_settings()


def test_index_updates_are_not_lost_between_workers(tmp_path):
    store = ArtifactStore(tmp_path / "runs")

    def save(index):
        artifacts = store.create(f"run_{index}")
        store.save_metadata(artifacts, {"run_id": f"run_{index}", "status": "completed"})

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(save, range(20)))
    assert len(store.list_runs()) == 20
    json.loads(store.index_path.read_text())


def test_static_report_escapes_untrusted_model_input(tmp_path):
    store = ArtifactStore(tmp_path / "runs")
    artifacts = store.create("run_report")
    store.save_metadata(
        artifacts,
        {
            "run_id": "run_report",
            "status": "completed",
            "input_text": "</script><script>alert('x')</script>",
        },
    )
    store.save_outputs(artifacts, {"prediction": "<img src=x onerror=alert(1)>"})
    store.save_summaries(artifacts, {})
    path = export_report("run_report", store.runs_dir, tmp_path / "reports")
    text = path.read_text()
    assert "<script>" not in text
    assert "<img src=x" not in text
    assert "&lt;script&gt;" in text
    assert "Content-Security-Policy" in text


def test_http_errors_upload_scope_and_origin_controls(settings, monkeypatch):
    app = create_app(settings)
    submitted = []

    def capture(request, model_id, task_type):
        submitted.append((request, model_id, task_type))
        return "dataset_test"

    monkeypatch.setattr(app.state.dataset_manager, "start", capture)
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/api/run/missing").status_code == 404
        response = client.post(
            "/api/run",
            json={
                "task_type": "text_lm",
                "model_id": "unapproved",
                "input_text": "Hi",
            },
        )
        assert response.status_code == 400
        assert (
            client.post(
                "/api/run", json={}, headers={"origin": "https://untrusted.example"}
            ).status_code
            == 403
        )
        data = {
            "kind": "text",
            "model_id": "sshleifer/tiny-gpt2",
            "task_type": "text_classification",
        }
        response = client.post(
            "/api/dataset/run",
            data=data,
            files={"file": ("../../escape.csv", b"text,label\nhello,positive\n", "text/csv")},
        )
        assert response.status_code == 200
        saved = Path(submitted[0][0].path)
        assert saved.parent == settings.paths.safe_data_dir / "uploads"
        assert saved.name.startswith("dataset_")
        assert not (settings.paths.base_dir / "escape.csv").exists()
        assert client.post("/api/dataset/run", data={**data, "max_samples": 0}).status_code == 422
        assert client.post("/api/dataset/run", data={**data, "kind": "image"}).status_code == 422
        response = client.post(
            "/api/dataset/run",
            data=data,
            files={"file": ("large.csv", b"x" * (1024**2 + 1), "text/csv")},
        )
        assert response.status_code == 413
        assert len(submitted) == 1
