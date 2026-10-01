from __future__ import annotations

import json
import os
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import torch


def utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def json_dumps(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True, default=str, allow_nan=False)


def safe_run_id(prefix: str = "run") -> str:
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    # Model seeding must never reset the run identifier sequence.
    suffix = uuid4().hex
    return f"{prefix}_{ts}_{suffix}"


def safe_component(value: str) -> str:
    """Accept one portable filename component, never a relative or absolute path."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}", value) or ".." in value:
        raise ValueError("Invalid artifact identifier")
    return value


def atomic_json(path: Path, data: Any) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json_dumps(data), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def resolve_model_id(model_id: str, settings: Any) -> str:
    if model_id in settings.models.allowlist:
        return model_id
    path = Path(model_id)
    if not path.is_absolute():
        path = settings.paths.safe_model_dir / path
    path = require_safe_path(path, settings.paths.safe_model_dir).resolve()
    if not path.is_dir():
        raise ValueError("Model directory not found")
    return str(path)


def is_safe_path(path: Path, base_dir: Path) -> bool:
    try:
        path_resolved = path.resolve()
        base_resolved = base_dir.resolve()
    except FileNotFoundError:
        path_resolved = path.absolute()
        base_resolved = base_dir.absolute()
    return base_resolved in path_resolved.parents or path_resolved == base_resolved


def require_safe_path(path: Path, base_dir: Path) -> Path:
    if not is_safe_path(path, base_dir):
        raise ValueError("Path is outside the configured safe directory")
    return path.resolve()


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
