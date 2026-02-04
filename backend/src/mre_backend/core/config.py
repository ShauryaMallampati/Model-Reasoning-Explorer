from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class PathConfig(BaseModel):
    base_dir: Path
    runs_dir: Path
    reports_dir: Path
    safe_model_dir: Path
    safe_data_dir: Path


class LimitsConfig(BaseModel):
    max_upload_mb: int = 20
    max_batch_size: int = 16
    max_layers: int = 128


class ExecutionConfig(BaseModel):
    allow_gpu: bool = False
    seed: int = 42


class ModelsConfig(BaseModel):
    allowlist: list[str] = []


class Settings(BaseModel):
    paths: PathConfig
    limits: LimitsConfig
    execution: ExecutionConfig
    models: ModelsConfig


@dataclass(frozen=True)
class SettingsBundle:
    settings: Settings
    config_path: Path


def _resolve_paths(raw: dict[str, Any], root: Path) -> dict[str, Any]:
    resolved = dict(raw)
    for key, value in raw.items():
        if isinstance(value, str):
            resolved[key] = (root / value).resolve()
    return resolved


def load_settings() -> SettingsBundle:
    config_path = Path(os.environ.get("MRE_CONFIG", ""))
    if not config_path:
        # Default to backend/mre_config.toml
        config_path = Path(__file__).resolve().parents[3] / "mre_config.toml"

    config_data: dict[str, Any] = {}
    if config_path.exists():
        with config_path.open("rb") as f:
            config_data = tomllib.load(f)

    # Resolve paths relative to the config file location
    base_dir_value = config_data.get("paths", {}).get("base_dir", "../")
    base_dir_path = Path(base_dir_value)
    if base_dir_path.is_absolute():
        base_dir = base_dir_path
    else:
        base_dir = (config_path.parent / base_dir_path).resolve()

    paths = _resolve_paths(
        config_data.get("paths",
            {
                "base_dir": str(base_dir),
                "runs_dir": "backend/runs",
                "reports_dir": "backend/reports",
                "safe_model_dir": "models",
                "safe_data_dir": "sample_data",
            },
        ),
        config_path.parent,
    )
    paths["base_dir"] = base_dir

    limits = config_data.get("limits", {})
    execution = config_data.get("execution", {})
    models = config_data.get("models", {})

    settings = Settings(
        paths=PathConfig(**paths),
        limits=LimitsConfig(**limits),
        execution=ExecutionConfig(**execution),
        models=ModelsConfig(**models),
    )

    settings.paths.runs_dir.mkdir(parents=True, exist_ok=True)
    settings.paths.reports_dir.mkdir(parents=True, exist_ok=True)
    settings.paths.safe_model_dir.mkdir(parents=True, exist_ok=True)
    settings.paths.safe_data_dir.mkdir(parents=True, exist_ok=True)

    return SettingsBundle(settings=settings, config_path=config_path)
