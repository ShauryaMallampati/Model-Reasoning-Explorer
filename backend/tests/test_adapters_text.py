import torch
from mre_backend.adapters.text_transformer import TextTransformerAdapter


def test_text_transformer_adapter_forward():
    adapter = TextTransformerAdapter("text_lm")
    model, tokenizer = adapter.load("sshleifer/tiny-gpt2", torch.device("cpu"))
    inputs = adapter.prepare_inputs(tokenizer, {"text": "Hello"}, torch.device("cpu"))
    outputs = adapter.forward(model, inputs, {"hidden_states": True, "attentions": False})
    assert outputs.logits is not None
    assert outputs.logits.ndim == 3
