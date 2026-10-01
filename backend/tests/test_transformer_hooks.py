"""Capture hidden states, not attention matrices, from supported model blocks."""

import pytest
import torch
from transformers import (
    DistilBertConfig,
    DistilBertForSequenceClassification,
    GPT2Config,
    GPT2LMHeadModel,
)

from mre_backend.capture.hooks import HookManager


@pytest.mark.parametrize("kind", ["distilbert", "gpt2"])
def test_block_activation_and_gradient_match_hidden_output(kind):
    torch.manual_seed(42)
    if kind == "distilbert":
        config = DistilBertConfig(
            vocab_size=31,
            max_position_embeddings=32,
            n_layers=1,
            n_heads=2,
            dim=8,
            hidden_dim=16,
            num_labels=2,
            attn_implementation="eager",
        )
        model = DistilBertForSequenceClassification(config)
        layer = "distilbert.transformer.layer.0"
    else:
        config = GPT2Config(vocab_size=31, n_positions=32, n_layer=1, n_head=2, n_embd=8)
        model = GPT2LMHeadModel(config)
        layer = "transformer.h.0"
    model.eval()
    manager = HookManager(model, [layer], capture_gradients=True)
    captured = manager.attach()
    try:
        output = model(
            input_ids=torch.tensor([[2, 5, 7]]),
            output_attentions=True,
            output_hidden_states=True,
        )
        assert captured.activations[layer].shape == (1, 3, 8)
        if kind == "distilbert":
            # DistilBERT puts attention BEFORE the block's hidden output.
            torch.testing.assert_close(captured.activations[layer], output.hidden_states[-1])
        output.logits.sum().backward()
        assert captured.gradients[layer].shape == (1, 3, 8)
        assert torch.isfinite(captured.gradients[layer]).all()
    finally:
        manager.clear()
    assert all(not module._forward_hooks for module in model.modules())
