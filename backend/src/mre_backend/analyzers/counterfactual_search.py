from __future__ import annotations

from typing import Any

import torch

from .base import AnalyzerOutput, BaseAnalyzer


class CounterfactualSearchAnalyzer(BaseAnalyzer):
    id = "counterfactual_search"

    def supports(self, task_type: str) -> bool:
        return task_type in {"text_classification", "image_classification"}

    def run(self, context: Any) -> AnalyzerOutput:
        if context.task_type == "text_classification":
            return self._text_search(context)
        return self._vision_search(context)

    def _text_search(self, context: Any) -> AnalyzerOutput:
        tokenizer = context.tokenizer
        model = context.model
        inputs = context.inputs
        device = context.device

        baseline = int(context.outputs.logits.argmax(dim=-1).item())
        input_ids = inputs["input_ids"][0].tolist()

        best = None
        for idx in range(len(input_ids)):
            if input_ids[idx] in tokenizer.all_special_ids:
                continue
            modified = input_ids[:idx] + input_ids[idx + 1 :]
            if not modified:
                continue
            mod_tensor = torch.tensor([modified], device=device)
            attn = torch.ones_like(mod_tensor)
            with torch.no_grad():
                logits = model(input_ids=mod_tensor, attention_mask=attn).logits
            pred = int(logits.argmax(dim=-1).item())
            if pred != baseline:
                best = {
                    "operation": "delete",
                    "position": idx,
                    "new_prediction": pred,
                    "text": tokenizer.decode(modified),
                }
                break

        if best is None:
            best = {"message": "No single-token deletion flipped prediction"}

        return AnalyzerOutput(summary={"baseline": baseline, "result": best})

    def _vision_search(self, context: Any) -> AnalyzerOutput:
        model = context.model
        inputs = context.inputs

        x = inputs["pixel_values"]
        baseline = int(context.outputs.logits.argmax(dim=-1).item())
        _, _, h, w = x.shape
        grid = 6
        step_h = h // grid
        step_w = w // grid
        mean_val = float(x.mean().item())

        best = None
        for i in range(grid):
            for j in range(grid):
                x_masked = x.clone()
                h0, h1 = i * step_h, min(h, (i + 1) * step_h)
                w0, w1 = j * step_w, min(w, (j + 1) * step_w)
                x_masked[:, :, h0:h1, w0:w1] = mean_val
                with torch.no_grad():
                    logits = model(x_masked)
                pred = int(logits.argmax(dim=-1).item())
                if pred != baseline:
                    best = {"grid": grid, "i": i, "j": j, "new_prediction": pred}
                    break
            if best:
                break

        if best is None:
            best = {"message": "No single patch flip found"}

        return AnalyzerOutput(summary={"baseline": baseline, "result": best})
