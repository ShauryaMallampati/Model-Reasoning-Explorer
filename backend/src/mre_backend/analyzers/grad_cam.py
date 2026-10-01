from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .base import AnalyzerOutput, BaseAnalyzer


class GradCamAnalyzer(BaseAnalyzer):
    id = "grad_cam"

    def supports(self, task_type: str) -> bool:
        return task_type == "image_classification"

    def run(self, context: Any) -> AnalyzerOutput:
        activations = context.capture.activations
        gradients = context.capture.gradients

        # Pick the last conv-like layer available
        chosen = None
        for name in reversed(list(activations.keys())):
            act = activations[name]
            grad = gradients.get(name)
            if act.ndim == 4 and grad is not None and grad.ndim == 4:
                chosen = (name, act, grad)
                break
        if chosen is None:
            return AnalyzerOutput(summary={"message": "No conv activations for Grad-CAM"})

        name, act, grad = chosen
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * act).sum(dim=1, keepdim=False))
        cam_np = cam[0].cpu().numpy()
        cam_np = cam_np - cam_np.min()
        if cam_np.max() > 0:
            cam_np = cam_np / cam_np.max()

        preview = cam_np[:: max(1, cam_np.shape[0] // 16), :: max(1, cam_np.shape[1] // 16)]

        return AnalyzerOutput(
            summary={
                "layer": name,
                "shape": list(cam_np.shape),
                "heatmap_preview": preview.tolist(),
            },
            arrays={"heatmap": cam_np.astype(np.float32)},
        )
