from __future__ import annotations

import json
import os
import random
import string
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
    return json.dumps(data, indent=2, sort_keys=True, default=str)


def safe_run_id(prefix: str = "run") -> str:
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return f"{prefix}_{ts}_{suffix}"


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
        raise ValueError(f"Path {path} is outside safe directory {base_dir}")
    return path


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
