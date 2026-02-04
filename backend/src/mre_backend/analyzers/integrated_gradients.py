from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .base import AnalyzerOutput, BaseAnalyzer

try:
    from captum.attr import IntegratedGradients, LayerIntegratedGradients
except Exception:  # pragma: no cover - optional
    IntegratedGradients = None
    LayerIntegratedGradients = None


class IntegratedGradientsAnalyzer(BaseAnalyzer):
    id = "integrated_gradients"

    def supports(self, task_type: str) -> bool:
        return task_type in {"text_lm", "text_classification", "image_classification"}

    def run(self, context: Any) -> AnalyzerOutput:
        if IntegratedGradients is None:
            return AnalyzerOutput(summary={"message": "captum not installed"})

        model = context.model
        outputs = context.outputs
        inputs = context.inputs
        if context.task_type == "image_classification":
            target = int(outputs.logits.argmax(dim=-1).item())

            def forward_fn(x):
                return model(x)

            ig = IntegratedGradients(forward_fn)
            attributions = ig.attribute(inputs["pixel_values"], target=target)
            attr = attributions[0].detach().cpu().numpy()
            attr = attr.mean(axis=0)
            attr = attr - attr.min()
            if attr.max() > 0:
                attr = attr / attr.max()
            preview = attr[:: max(1, attr.shape[0] // 16), :: max(1, attr.shape[1] // 16)]
            return AnalyzerOutput(
                summary={"target": target, "shape": list(attr.shape), "heatmap_preview": preview.tolist()},
                arrays={"attribution": attr.astype(np.float32)},
            )

        if context.task_type == "text_classification":
            target = int(outputs.logits.argmax(dim=-1).item())
            embed = model.get_input_embeddings()

            def forward_fn(input_ids, attention_mask):
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                return out.logits

            lig = LayerIntegratedGradients(forward_fn, embed)
            attributions, _ = lig.attribute(
                inputs["input_ids"],
                additional_forward_args=(inputs.get("attention_mask"),),
                target=target,
                return_convergence_delta=True,
            )
            attr = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()
            preview = attr[: min(64, len(attr))]
            return AnalyzerOutput(
                summary={"target": target, "length": int(attr.shape[0]), "attribution_preview": preview.tolist()},
                arrays={"attribution": attr.astype(np.float32)},
            )

        # text_lm: use last token logit
        target = int(outputs.logits[:, -1, :].argmax(dim=-1).item())
        embed = model.get_input_embeddings()

        def forward_fn(input_ids, attention_mask):
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            return out.logits[:, -1, :]

        lig = LayerIntegratedGradients(forward_fn, embed)
        attributions, _ = lig.attribute(
            inputs["input_ids"],
            additional_forward_args=(inputs.get("attention_mask"),),
            target=target,
            return_convergence_delta=True,
        )
        attr = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()
        preview = attr[: min(64, len(attr))]
        return AnalyzerOutput(
            summary={"target": target, "length": int(attr.shape[0]), "attribution_preview": preview.tolist()},
            arrays={"attribution": attr.astype(np.float32)},
        )
