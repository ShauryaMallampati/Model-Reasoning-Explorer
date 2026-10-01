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
        return self.store

    def clear(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _make_forward_hook(self, name: str):
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            tensor = output
            if isinstance(output, (tuple, list)):
                # GPT-2 returns hidden states first; attention-enabled DistilBERT
                # returns (attention, hidden states). Both hidden outputs have
                # shape (batch, tokens, width); attention matrices have rank four.
                tensor = next(
                    (
                        value
                        for value in output
                        if isinstance(value, torch.Tensor) and value.ndim == 3
                    ),
                    output[0] if output else None,
                )
            if not isinstance(tensor, torch.Tensor):
                return
            # Clone so later in-place activations cannot change recorded values.
            self.store.activations[name] = tensor.detach().cpu().clone()
            if self.capture_gradients and tensor.requires_grad:
                # Tensor hooks avoid full-backward-hook views, which conflict
                # with the in-place ReLUs used by torchvision ResNet.
                def capture_gradient(gradient: torch.Tensor) -> None:
                    self.store.gradients[name] = gradient.detach().cpu().clone()

                self._handles.append(tensor.register_hook(capture_gradient))

        return hook
