from __future__ import annotations

from typing import Any

import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

from .base import AdapterOutputs, BaseAdapter


class VisionResNetAdapter(BaseAdapter):
    task_type = "image_classification"

    def __init__(self) -> None:
        # These are needed even when the model is reused from the cache.
        weights = ResNet18_Weights.DEFAULT
        self._preprocess = weights.transforms()
        self._labels = weights.meta.get("categories", [])

    def load(self, model_id: str, device: torch.device) -> tuple[Any, Any | None]:
        if model_id != "resnet18":
            raise ValueError("Only resnet18 is supported for vision demo")
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
        model.to(device)
        model.eval()
        self._preprocess = weights.transforms()
        self._labels = weights.meta.get("categories", [])
        return model, None

    def prepare_inputs(
        self, tokenizer: Any | None, raw: dict[str, Any], device: torch.device
    ) -> dict[str, Any]:
        path = raw.get("image_path")
        image = raw.get("image")
        if image is None:
            if not path:
                raise ValueError("Image path or image bytes required")
            image = Image.open(path).convert("RGB")
        if self._preprocess is None:
            self._preprocess = transforms.Compose(
                [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]
            )
        tensor = self._preprocess(image).unsqueeze(0).to(device)
        return {"pixel_values": tensor}

    def forward(
        self, model: Any, inputs: dict[str, Any], capture: dict[str, Any]
    ) -> AdapterOutputs:
        with torch.set_grad_enabled(bool(capture.get("gradients"))):
            outputs = model(inputs["pixel_values"])
        return AdapterOutputs(logits=outputs)

    def postprocess(
        self, outputs: AdapterOutputs, tokenizer: Any | None, top_k: int
    ) -> dict[str, Any]:
        probs = torch.softmax(outputs.logits, dim=-1)
        values, indices = torch.topk(probs, k=min(top_k, probs.size(-1)), dim=-1)
        top = []
        for idx, val in zip(indices[0].tolist(), values[0].tolist()):
            label = self._labels[idx] if self._labels and idx < len(self._labels) else str(idx)
            top.append({"label": label, "prob": float(val)})
        return {"prediction": top[0]["label"], "top_k": top}

    def list_layers(self, model: Any) -> list[str]:
        names = []
        for name, _ in model.named_modules():
            if name.startswith("layer") or name in {"conv1", "bn1", "fc"}:
                names.append(name)
        return names
