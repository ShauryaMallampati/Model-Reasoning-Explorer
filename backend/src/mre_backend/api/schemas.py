from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class LayerSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["all", "every_n", "custom"] = "all"
    stride: int = Field(default=1, ge=1, le=128)
    layers: list[str] = Field(default_factory=list, max_length=128)


class CaptureOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    activations: bool = True
    gradients: bool = False
    attentions: bool = True
    hidden_states: bool = True
    logits: bool = False
    layers: LayerSelection = Field(default_factory=LayerSelection)


class RunOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    top_k: int = Field(default=5, ge=1, le=50)
    max_tokens: int = Field(default=32, ge=1, le=128)
    seed: int | None = Field(default=None, ge=0, le=2**32 - 1)
    capture: CaptureOptions = Field(default_factory=CaptureOptions)
    analyzers: list[str] = Field(default_factory=list, max_length=16)
    counterfactual_text: str | None = Field(default=None, max_length=20000)
    patch_layers: list[int] | None = Field(default=None, max_length=128)


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_type: Literal["text_lm", "text_classification", "image_classification"]
    model_id: str = Field(min_length=1, max_length=512)
    input_text: str | None = Field(default=None, max_length=20000)
    input_image_path: str | None = Field(default=None, max_length=1024)
    input_image_base64: str | None = Field(default=None, max_length=28_000_000)
    options: RunOptions = Field(default_factory=RunOptions)

    @model_validator(mode="after")
    def validate_input(self):
        if self.task_type.startswith("text"):
            if not self.input_text or not self.input_text.strip():
                raise ValueError("Text input is required")
        elif bool(self.input_image_path) == bool(self.input_image_base64):
            raise ValueError("Provide exactly one image input")
        return self


class RunResponse(BaseModel):
    run_id: str


class RunStatusResponse(BaseModel):
    run_id: str
    status: str
    error: str | None = None
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
    max_samples: int = Field(default=50, ge=1, le=200)
    batch_size: int = Field(default=8, ge=1, le=16)
    layer_for_embedding: str | None = None


class DatasetRunResponse(BaseModel):
    dataset_run_id: str


class DatasetRunStatus(BaseModel):
    dataset_run_id: str
    status: str
    error: str | None = None
    summary: dict[str, Any] | None = None
    examples: list[dict[str, Any]] = Field(default_factory=list)
