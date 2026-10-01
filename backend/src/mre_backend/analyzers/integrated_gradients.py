from __future__ import annotations

from typing import Any

import numpy as np
from captum.attr import IntegratedGradients, LayerIntegratedGradients

from .base import AnalyzerOutput, BaseAnalyzer


class IntegratedGradientsAnalyzer(BaseAnalyzer):
    id = "integrated_gradients"

    def supports(self, task_type: str) -> bool:
        return task_type in {"text_lm", "text_classification", "image_classification"}

    def run(self, context: Any) -> AnalyzerOutput:
        model, inputs = context.model, context.inputs
        if context.task_type == "image_classification":
            target = int(context.outputs.logits.argmax(dim=-1).item())
            ig = IntegratedGradients(model)
            attributions, delta = ig.attribute(
                inputs["pixel_values"],
                target=target,
                internal_batch_size=4,
                return_convergence_delta=True,
            )
            # Keep the signed attribution in the artifact; normalize only its preview.
            signed = attributions[0].detach().cpu().numpy().sum(axis=0)
            preview = signed - signed.min()
            if preview.max() > 0:
                preview = preview / preview.max()
            preview = preview[:: max(1, preview.shape[0] // 16), :: max(1, preview.shape[1] // 16)]
            return AnalyzerOutput(
                summary={
                    "target": target,
                    "shape": list(signed.shape),
                    "heatmap_preview": preview.tolist(),
                    "baseline": "zero in normalized image space",
                    "convergence_delta": float(delta.detach().cpu().abs().max()),
                    "preview_normalization": "min-max; raw signed sums stored separately",
                },
                arrays={"attribution": signed.astype(np.float32)},
            )

        is_lm = context.task_type == "text_lm"
        logits = context.outputs.logits[:, -1, :] if is_lm else context.outputs.logits
        target = int(logits.argmax(dim=-1).item())

        def forward_fn(input_ids, attention_mask):
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            return output.logits[:, -1, :] if is_lm else output.logits

        lig = LayerIntegratedGradients(forward_fn, model.get_input_embeddings())
        attributions, delta = lig.attribute(
            inputs["input_ids"],
            additional_forward_args=(inputs.get("attention_mask"),),
            target=target,
            internal_batch_size=4,
            return_convergence_delta=True,
        )
        attribution = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()
        return AnalyzerOutput(
            summary={
                "target": target,
                "length": len(attribution),
                "attribution_preview": attribution[:64].tolist(),
                "baseline": "token ID zero (not a neutral-text guarantee)",
                "convergence_delta": float(delta.detach().cpu().abs().max()),
            },
            arrays={"attribution": attribution.astype(np.float32)},
        )
