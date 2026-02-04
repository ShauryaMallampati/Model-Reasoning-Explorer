from __future__ import annotations

from typing import Any

import torch

from .base import AnalyzerOutput, BaseAnalyzer


class LogitLensAnalyzer(BaseAnalyzer):
    id = "logit_lens"

    def supports(self, task_type: str) -> bool:
        return task_type == "text_lm"

    def run(self, context: Any) -> AnalyzerOutput:
        hidden_states = context.outputs.hidden_states
        if hidden_states is None:
            return AnalyzerOutput(summary={"message": "No hidden states captured"})

        model = context.model
        tokenizer = context.tokenizer
        if not hasattr(model, "lm_head"):
            return AnalyzerOutput(summary={"message": "Model missing lm_head"})

        top_k = context.request.options.get("top_k", 5)
        layer_summaries = []
        for idx, hidden in enumerate(hidden_states):
            logits = model.lm_head(hidden)
            last_logits = logits[:, -1, :]
            probs = torch.softmax(last_logits, dim=-1)
            values, indices = torch.topk(probs, k=top_k, dim=-1)
            tokens = [tokenizer.decode([i]) for i in indices[0].tolist()]
            layer_summaries.append(
                {
                    "layer": idx,
                    "top_k": [
                        {"token": t, "prob": float(p)} for t, p in zip(tokens, values[0])
                    ],
                }
            )

        return AnalyzerOutput(summary={"layers": layer_summaries})
