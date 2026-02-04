from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class LayerSelection(BaseModel):
    mode: Literal["all", "every_n", "custom"] = "all"
    stride: int = 1
    layers: list[str] = Field(default_factory=list)


class CaptureOptions(BaseModel):
    activations: bool = True
    gradients: bool = False
    attentions: bool = True
    hidden_states: bool = True
    logits: bool = False
    layers: LayerSelection = Field(default_factory=LayerSelection)


class RunOptions(BaseModel):
    top_k: int = 5
    max_tokens: int = 32
    seed: int | None = None
    capture: CaptureOptions = Field(default_factory=CaptureOptions)
    analyzers: list[str] = Field(default_factory=list)
    counterfactual_text: str | None = None
    patch_layers: list[int] | None = None


class RunRequest(BaseModel):
    task_type: Literal["text_lm", "text_classification", "image_classification"]
    model_id: str
    input_text: str | None = None
    input_image_path: str | None = None
    input_image_base64: str | None = None
    options: RunOptions = Field(default_factory=RunOptions)


class RunResponse(BaseModel):
    run_id: str


class RunStatusResponse(BaseModel):
    run_id: str
    status: str
    metadata: dict[str, Any] | None = None
    outputs: dict[str, Any] | None = None
    summaries: dict[str, Any] | None = None
    artifacts: list[str] = Field(default_factory=list)


class CompareRequest(BaseModel):
    run_a: str
    run_b: str


class CompareResponse(BaseModel):
    summary: dict[str, Any]


class DatasetRunRequest(BaseModel):
    kind: Literal["text", "image"]
    path: str | None = None
    text_column: str = "text"
    label_column: str = "label"
    max_samples: int = 50
    batch_size: int = 8
    layer_for_embedding: str | None = None


class DatasetRunResponse(BaseModel):
    dataset_run_id: str


class DatasetRunStatus(BaseModel):
    dataset_run_id: str
    status: str
    summary: dict[str, Any] | None = None
    examples: list[dict[str, Any]] = Field(default_factory=list)
