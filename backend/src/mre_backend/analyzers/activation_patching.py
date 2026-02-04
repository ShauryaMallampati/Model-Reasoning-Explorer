from __future__ import annotations

from typing import Any
import torch

from .base import AnalyzerOutput, BaseAnalyzer


class ActivationPatchingAnalyzer(BaseAnalyzer):
    id = "activation_patching"

    def supports(self, task_type: str) -> bool:
        return task_type == "text_lm"

    def run(self, context: Any) -> AnalyzerOutput:
        counterfactual = context.request.options.counterfactual_text
        if not counterfactual:
            return AnalyzerOutput(summary={"message": "No counterfactual_text provided"})

        model = context.model
        tokenizer = context.tokenizer
        device = context.device

        if not hasattr(model, "transformer") or not hasattr(model.transformer, "h"):
            return AnalyzerOutput(summary={"message": "Model does not support patching"})

        # Run counterfactual to collect hidden states
        cf_inputs = tokenizer(counterfactual, return_tensors="pt")
        cf_inputs = {k: v.to(device) for k, v in cf_inputs.items()}
        with torch.no_grad():
            cf_outputs = model(**cf_inputs, output_hidden_states=True)
        cf_hidden = cf_outputs.hidden_states

        if cf_hidden is None:
            return AnalyzerOutput(summary={"message": "No hidden states for counterfactual"})

        baseline_logits = context.outputs.logits[:, -1, :]
        baseline_token = int(baseline_logits.argmax(dim=-1).item())

        layer_modules = list(model.transformer.h)
        total_layers = len(layer_modules)
        selection = context.request.options.patch_layers
        if not selection:
            selection = list(range(max(0, total_layers - 4), total_layers))

        results = []
        for layer_idx in selection:
            if layer_idx + 1 >= len(cf_hidden):
                continue

            def hook(_module, _inputs, output):
                hidden = cf_hidden[layer_idx + 1]
                if isinstance(output, tuple):
                    return (hidden,) + output[1:]
                return hidden

            handle = layer_modules[layer_idx].register_forward_hook(hook)
            with torch.no_grad():
                patched = model(**context.inputs)
            handle.remove()

            patched_logits = patched.logits[:, -1, :]
            patched_prob = torch.softmax(patched_logits, dim=-1)[0, baseline_token].item()
            base_prob = torch.softmax(baseline_logits, dim=-1)[0, baseline_token].item()
            results.append(
                {
                    "layer": layer_idx,
                    "delta_prob": float(patched_prob - base_prob),
                }
            )

        return AnalyzerOutput(summary={"baseline_token": baseline_token, "results": results})
