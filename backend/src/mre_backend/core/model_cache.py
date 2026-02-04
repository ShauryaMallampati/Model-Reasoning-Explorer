from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CachedModel:
    model: Any
    tokenizer: Any | None
    task_type: str
    model_id: str


class ModelCache:
    def __init__(self) -> None:
        self._cache: dict[str, CachedModel] = {}

    def get(self, key: str) -> CachedModel | None:
        return self._cache.get(key)

    def set(self, key: str, model: CachedModel) -> None:
        self._cache[key] = model

    def make_key(self, task_type: str, model_id: str) -> str:
        return f"{task_type}:{model_id}"
