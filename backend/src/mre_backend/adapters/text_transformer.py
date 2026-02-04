from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from .base import AdapterOutputs, BaseAdapter


class TextTransformerAdapter(BaseAdapter):
    def __init__(self, task_type: str) -> None:
        self.task_type = task_type
        self._id2label: dict[int, str] = {}

    def load(self, model_id: str, device: torch.device) -> tuple[Any, Any]:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        if self.task_type == "text_lm":
            model = AutoModelForCausalLM.from_pretrained(model_id)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(model_id)
            self._id2label = getattr(model.config, "id2label", {}) or {}
        model.to(device)
        model.eval()
        return model, tokenizer

    def prepare_inputs(
        self, tokenizer: Any | None, raw: dict[str, Any], device: torch.device
    ) -> dict[str, Any]:
        if tokenizer is None:
            raise ValueError("Tokenizer required for text tasks")
        text = raw.get("text")
        if not text:
            raise ValueError("Input text is required")
        inputs = tokenizer(text, return_tensors="pt")
        return {k: v.to(device) for k, v in inputs.items()}

    def forward(self, model: Any, inputs: dict[str, Any], capture: dict[str, Any]) -> AdapterOutputs:
        with torch.set_grad_enabled(bool(capture.get("gradients"))):
            outputs = model(
                **inputs,
                output_hidden_states=bool(capture.get("hidden_states")),
                output_attentions=bool(capture.get("attentions")),
            )
        hidden_states = outputs.hidden_states if hasattr(outputs, "hidden_states") else None
        attentions = outputs.attentions if hasattr(outputs, "attentions") else None
        logits = outputs.logits
        return AdapterOutputs(logits=logits, hidden_states=hidden_states, attentions=attentions)

    def postprocess(self, outputs: AdapterOutputs, tokenizer: Any | None, top_k: int) -> dict[str, Any]:
        if tokenizer is None:
            return {}
        if self.task_type == "text_lm":
            last_logits = outputs.logits[:, -1, :]
            probs = torch.softmax(last_logits, dim=-1)
            values, indices = torch.topk(probs, k=top_k, dim=-1)
            tokens = [tokenizer.decode([idx]) for idx in indices[0].tolist()]
            return {
                "prediction": tokens[0],
                "top_k": [{"token": t, "prob": float(p)} for t, p in zip(tokens, values[0])],
            }

        probs = torch.softmax(outputs.logits, dim=-1)
        values, indices = torch.topk(probs, k=min(top_k, probs.size(-1)), dim=-1)
        id2label = self._id2label or {}
        top = []
        for idx, val in zip(indices[0].tolist(), values[0].tolist()):
            label = id2label.get(idx, str(idx))
            top.append({"label": label, "prob": float(val)})
        return {"prediction": top[0]["label"], "top_k": top}

    def list_layers(self, model: Any) -> list[str]:
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            return [f"transformer.h.{i}" for i in range(len(model.transformer.h))]
        if hasattr(model, "transformer") and hasattr(model.transformer, "layer"):
            return [f"transformer.layer.{i}" for i in range(len(model.transformer.layer))]
        if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
            return [f"encoder.layer.{i}" for i in range(len(model.encoder.layer))]
        return [name for name, _ in model.named_modules() if name]
