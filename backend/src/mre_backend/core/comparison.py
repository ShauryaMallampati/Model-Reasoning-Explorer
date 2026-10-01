from __future__ import annotations

import json
from typing import Any

import numpy as np

from .artifacts import ArtifactStore


def compare_artifacts(store: ArtifactStore, run_a: str, run_b: str) -> dict[str, Any]:
    def load(run_id):
        root = store.get_run_dir(run_id)
        if not root.is_dir():
            raise FileNotFoundError("Run not found")
        metadata_path = store.get_artifact_path(run_id, "metadata.json")
        if not metadata_path.is_file():
            raise ValueError("Both runs must be complete before comparison")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("status") != "completed":
            raise ValueError("Both runs must be complete before comparison")
        values = {"metadata": metadata}
        for name in ("outputs", "summaries"):
            path = store.get_artifact_path(run_id, f"{name}.json")
            values[name] = json.loads(path.read_text(encoding="utf-8"))
        return values

    a, b = load(run_a), load(run_b)
    summary: dict[str, Any] = {"run_a": run_a, "run_b": run_b, "notes": []}
    for suffix, run in (("a", a), ("b", b)):
        summary[f"prediction_{suffix}"] = run["outputs"].get("prediction")
        summary[f"generated_{suffix}"] = run["outputs"].get("generated_text")
        summary[f"top_k_{suffix}"] = run["outputs"].get("top_k", [])

    metadata_a, metadata_b = a["metadata"], b["metadata"]
    same_model = all(
        metadata_a.get(key) == metadata_b.get(key)
        for key in ("model_id", "model_revision", "task_type")
    )
    same_tokens = bool(metadata_a.get("input_tokens")) and metadata_a.get(
        "input_tokens"
    ) == metadata_b.get("input_tokens")
    same_input = same_tokens or (
        bool(metadata_a.get("input_image_preview"))
        and metadata_a.get("input_image_preview") == metadata_b.get("input_image_preview")
    )
    if not same_model or not same_input:
        summary["notes"].append(
            "Elementwise tensor differences require the same model and aligned input. "
            "Output predictions are still compared; no token or layer alignment was invented."
        )
        return summary

    def arrays(run_id, name):
        path = store.get_artifact_path(run_id, f"arrays_{name}.npz")
        if not path.is_file():
            return {}
        with np.load(path, allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}

    for name, destination in (
        ("hidden_states", "layer_similarity"),
        ("attentions", "attention_delta"),
    ):
        left, right = arrays(run_a, name), arrays(run_b, name)
        layers, values = [], []
        for layer in sorted(
            set(left) & set(right), key=lambda value: int(value.rsplit("_", 1)[-1])
        ):
            x, y = left[layer], right[layer]
            if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
                summary["notes"].append(f"Skipped incompatible {name}/{layer}")
                continue
            if name == "hidden_states":
                x = x.mean(axis=tuple(range(x.ndim - 1)))
                y = y.mean(axis=tuple(range(y.ndim - 1)))
                denominator = np.linalg.norm(x) * np.linalg.norm(y)
                if denominator == 0:
                    continue
                value = float(np.dot(x, y) / denominator)
            else:
                value = float(np.mean(np.abs(x - y)))
            layers.append(layer)
            values.append(value)
        if layers:
            summary[destination] = {"layers": layers, "values": values}

    left = arrays(run_a, "integrated_gradients").get("attribution")
    right = arrays(run_b, "integrated_gradients").get("attribution")
    if left is not None and right is not None and left.shape == right.shape:
        delta = left - right
        if delta.ndim == 1:
            summary["attribution_diff"] = {
                "type": "text",
                "preview": delta[:64].tolist(),
                "tokens": metadata_a["input_tokens"][:64],
            }
        elif delta.ndim == 2:
            summary["attribution_diff"] = {
                "type": "image",
                "preview": delta[
                    :: max(1, delta.shape[0] // 16), :: max(1, delta.shape[1] // 16)
                ].tolist(),
            }
    summary["notes"].append(
        "Similarity and attribution differences are descriptive, not causal proof."
    )
    return summary
