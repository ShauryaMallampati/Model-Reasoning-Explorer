from pathlib import Path

import pytest
import torch

from mre_backend.core.config import Settings


@pytest.fixture
def settings(tmp_path: Path):
    paths = {
        "base_dir": tmp_path,
        "runs_dir": tmp_path / "runs",
        "reports_dir": tmp_path / "reports",
        "safe_model_dir": tmp_path / "models",
        "safe_data_dir": tmp_path / "data",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return Settings(
        paths=paths,
        limits={"max_upload_mb": 1},
        execution={"seed": 0},
        models={"allowlist": ["sshleifer/tiny-gpt2", "resnet18"]},
    )


@pytest.fixture(scope="session", autouse=True)
def bounded_torch_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)
