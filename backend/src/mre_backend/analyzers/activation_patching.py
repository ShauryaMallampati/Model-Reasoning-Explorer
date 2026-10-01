from __future__ import annotations

from typing import Any

import torch

from .base import AnalyzerOutput, BaseAnalyzer


class ActivationPatchingAnalyzer(BaseAnalyzer):
    id = "activation_patching"

    def supports(self, task_type: str) -> bool:
        return task_type == "text_lm"

    def run(self, context: Any) -> AnalyzerOutput:
        text = context.request.options.counterfactual_text
        if not text:
            return AnalyzerOutput(summary={"message": "No counterfactual_text provided"})
        model = context.model
        if getattr(model.config, "model_type", None) != "gpt2":
            return AnalyzerOutput(summary={"message": "Patching is verified only for GPT-2 blocks"})
        inputs = context.tokenizer(text, return_tensors="pt")
        inputs = {key: value.to(context.device) for key, value in inputs.items()}
        if inputs["input_ids"].shape != context.inputs["input_ids"].shape:
            raise ValueError(
                "Patching requires equal token counts and explicit positional alignment"
            )
        blocks = list(model.transformer.h)
        selected = context.request.options.patch_layers
        if selected is None:
            selected = list(range(max(0, len(blocks) - 4), len(blocks)))
        if not selected or any(index < 0 or index >= len(blocks) for index in selected):
            raise ValueError("Patch layer is outside the model")
        selected = list(dict.fromkeys(selected))
        captured = {}
        handles = []

        def capture(index):
            def hook(_module, _inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                captured[index] = hidden.detach().clone()

            return hook

        try:
            for index in selected:
                handles.append(blocks[index].register_forward_hook(capture(index)))
            with torch.no_grad():
                model(**inputs)
        finally:
            for handle in handles:
                handle.remove()

        baseline_logits = context.outputs.logits[:, -1, :]
        token = int(baseline_logits.argmax(dim=-1).item())
        baseline_probability = torch.softmax(baseline_logits, dim=-1)[0, token].item()
        results = []
        for index in selected:

            def replace(_module, _inputs, output):
                hidden = captured[index]
                return (hidden,) + output[1:] if isinstance(output, tuple) else hidden

            handle = blocks[index].register_forward_hook(replace)
            try:
                with torch.no_grad():
                    logits = model(**context.inputs).logits[:, -1, :]
            finally:
                handle.remove()
            probability = torch.softmax(logits, dim=-1)[0, token].item()
            results.append({"layer": index, "delta_prob": probability - baseline_probability})
        return AnalyzerOutput(
            summary={
                "baseline_token": token,
                "results": results,
                "alignment": "same token count, position by position; not semantic alignment",
            }
        )
