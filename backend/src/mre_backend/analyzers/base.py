from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class AnalyzerOutput:
    summary: dict[str, Any]
    arrays: dict[str, np.ndarray] | None = None


class BaseAnalyzer(ABC):
    id: str

    @abstractmethod
    def supports(self, task_type: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def run(self, context: Any) -> AnalyzerOutput:
        raise NotImplementedError
