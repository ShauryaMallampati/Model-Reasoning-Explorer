from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class AdapterOutputs:
    logits: torch.Tensor
    hidden_states: tuple[torch.Tensor, ...] | None = None
    attentions: tuple[torch.Tensor, ...] | None = None
    extra: dict[str, Any] | None = None


class BaseAdapter(ABC):
    task_type: str

    @abstractmethod
    def load(self, model_id: str, device: torch.device) -> tuple[Any, Any | None]:
        raise NotImplementedError

    @abstractmethod
    def prepare_inputs(self, tokenizer: Any | None, raw: dict[str, Any], device: torch.device) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def forward(self, model: Any, inputs: dict[str, Any], capture: dict[str, Any]) -> AdapterOutputs:
        raise NotImplementedError

    @abstractmethod
    def postprocess(self, outputs: AdapterOutputs, tokenizer: Any | None, top_k: int) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def list_layers(self, model: Any) -> list[str]:
        raise NotImplementedError

    def select_layers(self, model: Any, selection: dict[str, Any]) -> list[str]:
        all_layers = self.list_layers(model)
        mode = selection.get("mode", "all")
        if mode == "all":
            return all_layers
        if mode == "every_n":
            stride = int(selection.get("stride", 1))
            return [layer for idx, layer in enumerate(all_layers) if idx % stride == 0]
        if mode == "custom":
            chosen = selection.get("layers", [])
            return [layer for layer in all_layers if layer in chosen]
        return all_layers
