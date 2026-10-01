from __future__ import annotations

from typing import Any

import torch

from .base import AnalyzerOutput, BaseAnalyzer


@torch.no_grad()
def project_gpt2_states(model: Any, hidden_states: tuple[torch.Tensor, ...]):
    """Read out GPT-2 states, normalizing intermediates but not the final state twice."""
    if getattr(model.config, "model_type", None) != "gpt2":
        raise ValueError("Normalized logit-lens readout is verified only for GPT-2 models")
    for index, hidden in enumerate(hidden_states):
        last = hidden[:, -1, :]
        if index < len(hidden_states) - 1:
            last = model.transformer.ln_f(last)
        yield model.lm_head(last)


class LogitLensAnalyzer(BaseAnalyzer):
    id = "logit_lens"

    def supports(self, task_type: str) -> bool:
        return task_type == "text_lm"

    def run(self, context: Any) -> AnalyzerOutput:
        hidden_states = context.outputs.hidden_states
        if not hidden_states:
            return AnalyzerOutput(summary={"message": "No hidden states captured"})
        if getattr(context.model.config, "model_type", None) != "gpt2":
            return AnalyzerOutput(
                summary={"message": "Logit-lens normalization supports GPT-2 only"}
            )
        layers = []
        for index, logits in enumerate(project_gpt2_states(context.model, hidden_states)):
            probabilities = torch.softmax(logits, dim=-1)
            values, indices = torch.topk(
                probabilities, k=min(context.request.options.top_k, probabilities.shape[-1]), dim=-1
            )
            tokens = [context.tokenizer.decode([item]) for item in indices[0].tolist()]
            layers.append(
                {
                    "layer": index,
                    "top_k": [
                        {"token": token, "prob": probability}
                        for token, probability in zip(tokens, values[0].tolist(), strict=True)
                    ],
                }
            )
        return AnalyzerOutput(
            summary={
                "layers": layers,
                "method": "GPT-2 final-layer-normalized output-head projections",
                "limitation": (
                    "Intermediate readouts are probes, not predictions actually made by the model"
                ),
            }
        )
