from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn


@dataclass
class CaptureStore:
    activations: dict[str, torch.Tensor] = field(default_factory=dict)
    gradients: dict[str, torch.Tensor] = field(default_factory=dict)


class HookManager:
    def __init__(self, model: nn.Module, layers: list[str], capture_gradients: bool) -> None:
        self.model = model
        self.layers = layers
        self.capture_gradients = capture_gradients
        self.store = CaptureStore()
        self._handles: list[Any] = []

    def attach(self) -> CaptureStore:
        for name, module in self.model.named_modules():
            if name in self.layers:
                self._handles.append(module.register_forward_hook(self._make_forward_hook(name)))
                if self.capture_gradients:
                    self._handles.append(module.register_full_backward_hook(self._make_backward_hook(name)))
        return self.store

    def clear(self) -> None:
        for handle in self._handles:
            try:
                handle.remove()
            except Exception:
                pass
        self._handles.clear()

    def _make_forward_hook(self, name: str):
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            if isinstance(output, torch.Tensor):
                self.store.activations[name] = output.detach().cpu()

        return hook

    def _make_backward_hook(self, name: str):
        def hook(_module: nn.Module, _grad_input: tuple[Any, ...], grad_output: tuple[Any, ...]) -> None:
            if grad_output and isinstance(grad_output[0], torch.Tensor):
                self.store.gradients[name] = grad_output[0].detach().cpu()

        return hook
