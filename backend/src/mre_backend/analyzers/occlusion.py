from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .base import AnalyzerOutput, BaseAnalyzer


class OcclusionAnalyzer(BaseAnalyzer):
    id = "occlusion"

    def supports(self, task_type: str) -> bool:
        return task_type == "image_classification"

    def run(self, context: Any) -> AnalyzerOutput:
        inputs = context.inputs
        model = context.model
        device = context.device

        x = inputs["pixel_values"]
        target = int(context.outputs.logits.argmax(dim=-1).item())
        baseline = torch.softmax(context.outputs.logits, dim=-1)[0, target].item()

        _, _, h, w = x.shape
        grid = 8
        step_h = h // grid
        step_w = w // grid
        heatmap = np.zeros((grid, grid), dtype=np.float32)
        mean_val = float(x.mean().item())

        for i in range(grid):
            for j in range(grid):
                x_masked = x.clone()
                h0, h1 = i * step_h, min(h, (i + 1) * step_h)
                w0, w1 = j * step_w, min(w, (j + 1) * step_w)
                x_masked[:, :, h0:h1, w0:w1] = mean_val
                with torch.no_grad():
                    logits = model(x_masked)
                    prob = torch.softmax(logits, dim=-1)[0, target].item()
                heatmap[i, j] = max(0.0, baseline - prob)

        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        return AnalyzerOutput(
            summary={"target": target, "grid": grid, "heatmap_preview": heatmap.tolist()},
            arrays={"heatmap": heatmap},
        )
