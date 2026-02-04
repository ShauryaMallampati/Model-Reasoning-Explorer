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
        if attentions is None:
            return AnalyzerOutput(summary={"message": "No attentions captured"})

        # attentions: tuple[num_layers] of (batch, heads, seq, seq)
        attn = torch.stack(attentions, dim=0)  # layers, batch, heads, seq, seq
        attn = attn.mean(dim=2)  # average heads
        batch_size = attn.size(1)
        seq_len = attn.size(-1)

        rollout = torch.eye(seq_len, device=attn.device).unsqueeze(0).repeat(batch_size, 1, 1)
        for layer in range(attn.size(0)):
            rollout = rollout @ attn[layer]

        rollout_np = rollout[0].detach().cpu().numpy()
        last_token = rollout_np[-1]
        last_token = last_token / (last_token.max() if last_token.max() > 0 else 1.0)

        preview = last_token[: min(64, len(last_token))]
        return AnalyzerOutput(
            summary={"sequence_length": int(seq_len), "rollout_preview": preview.tolist()},
            arrays={"rollout": last_token.astype(np.float32)},
        )
