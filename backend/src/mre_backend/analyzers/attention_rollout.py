from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .base import AnalyzerOutput, BaseAnalyzer


class AttentionRolloutAnalyzer(BaseAnalyzer):
    id = "attention_rollout"

    def supports(self, task_type: str) -> bool:
        return task_type in {"text_lm", "text_classification"}

    def run(self, context: Any) -> AnalyzerOutput:
        attentions = context.outputs.attentions
        if not attentions or any(item is None for item in attentions):
            return AnalyzerOutput(summary={"message": "No attentions captured"})

        # attentions: tuple[num_layers] of (batch, heads, seq, seq)
        attn = torch.stack(attentions, dim=0)  # layers, batch, heads, seq, seq
        attn = attn.mean(dim=2)  # average heads
        batch_size = attn.size(1)
        seq_len = attn.size(-1)

        rollout = torch.eye(seq_len, device=attn.device).unsqueeze(0).repeat(batch_size, 1, 1)
        # Residual-adjusted, row-normalized rollout: each later layer maps
        # through the earlier layers. This is descriptive attention flow.
        identity = torch.eye(seq_len, device=attn.device)
        for layer in range(attn.size(0)):
            transition = attn[layer] + identity
            transition = transition / transition.sum(dim=-1, keepdim=True)
            rollout = transition @ rollout

        rollout_np = rollout[0].detach().cpu().numpy()
        query = -1 if context.task_type == "text_lm" else 0
        last_token = rollout_np[query]

        preview = last_token[: min(64, len(last_token))]
        return AnalyzerOutput(
            summary={
                "sequence_length": int(seq_len),
                "rollout_preview": preview.tolist(),
                "query_token": query,
                "method": "residual-adjusted head-mean rollout",
                "reference": "https://aclanthology.org/2020.acl-main.385/",
            },
            arrays={"rollout": last_token.astype(np.float32)},
        )
