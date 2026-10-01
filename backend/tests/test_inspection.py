import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from torch import nn

from mre_backend.adapters.text_transformer import TextTransformerAdapter
from mre_backend.analyzers.activation_patching import ActivationPatchingAnalyzer
from mre_backend.analyzers.attention_rollout import AttentionRolloutAnalyzer
from mre_backend.analyzers.grad_cam import GradCamAnalyzer
from mre_backend.analyzers.logit_lens import project_gpt2_states
from mre_backend.api.schemas import RunRequest
from mre_backend.capture.hooks import HookManager
from mre_backend.core.artifacts import ArtifactStore
from mre_backend.core.comparison import compare_artifacts
from mre_backend.core.run_manager import RunManager
from mre_backend.core.ws import WsManager
from mre_backend.main import create_app


@pytest.fixture(scope="module")
def tiny_lm():
    adapter = TextTransformerAdapter("text_lm")
    model, tokenizer = adapter.load("sshleifer/tiny-gpt2", torch.device("cpu"))
    return adapter, model, tokenizer


def test_real_text_run_attribution_storage_retrieval_and_repeat(settings):
    store = ArtifactStore(settings.paths.runs_dir)
    manager = RunManager(settings, store, WsManager())
    request = RunRequest(
        task_type="text_lm",
        model_id="sshleifer/tiny-gpt2",
        input_text="A small model can be inspected.",
        options={"max_tokens": 2, "seed": 0, "capture": {"gradients": True}},
    )
    try:
        first = manager.run_sync(request)
        second = manager.run_sync(request)
        assert first.status == second.status == "completed", (first.error, second.error)
        assert first.run_id != second.run_id
        assert first.outputs == second.outputs
        metadata = json.loads(store.get_artifact_path(first.run_id, "metadata.json").read_text())
        assert metadata["options"]["seed"] == 0
        assert metadata["model_revision"]
        scores = first.summaries["integrated_gradients"]["attribution_preview"]
        assert len(scores) == len(metadata["input_tokens"])
        assert np.isfinite(scores).all()
        assert first.summaries["integrated_gradients"]["convergence_delta"] >= 0
        result = compare_artifacts(store, first.run_id, second.run_id)
        np.testing.assert_allclose(result["layer_similarity"]["values"], 1, atol=1e-6)
        np.testing.assert_allclose(result["attribution_diff"]["preview"], 0, atol=1e-7)
        app = create_app(settings)
        with TestClient(app) as client:
            response = client.get(f"/api/run/{first.run_id}")
            assert response.status_code == 200
            assert response.json()["status"] == "completed"
            assert client.post(f"/api/run/{first.run_id}/export").status_code == 200
            report = client.get(f"/api/run/{first.run_id}/report")
            assert report.status_code == 200
            assert "Content-Security-Policy" in report.headers
    finally:
        manager.executor.shutdown(wait=True)


def test_failed_run_is_persisted_and_hooks_are_removed(settings, tiny_lm, monkeypatch):
    adapter, model, tokenizer = tiny_lm
    store = ArtifactStore(settings.paths.runs_dir)
    manager = RunManager(settings, store, WsManager())
    monkeypatch.setattr(adapter, "load", lambda *_: (model, tokenizer))
    monkeypatch.setattr(manager, "_select_adapter", lambda _: adapter)
    original_hooks = sum(len(module._forward_hooks) for module in model.modules())
    monkeypatch.setattr(
        adapter, "forward", lambda *_: (_ for _ in ()).throw(ValueError("test failure"))
    )
    try:
        record = manager.run_sync(
            RunRequest(
                task_type="text_lm", model_id="sshleifer/tiny-gpt2", input_text="Failure fixture"
            )
        )
        assert record.status == "failed"
        assert sum(len(module._forward_hooks) for module in model.modules()) == original_hooks
        metadata = json.loads(store.get_artifact_path(record.run_id, "metadata.json").read_text())
        assert metadata["error"] == "test failure"
    finally:
        manager.executor.shutdown(wait=True)


def test_logit_lens_final_projection_matches_actual_logits(tiny_lm):
    adapter, model, tokenizer = tiny_lm
    inputs = adapter.prepare_inputs(tokenizer, {"text": "Inspect this"}, torch.device("cpu"))
    outputs = adapter.forward(model, inputs, {"hidden_states": True})
    projected = list(project_gpt2_states(model, outputs.hidden_states))
    torch.testing.assert_close(projected[-1], outputs.logits[:, -1, :])


def test_same_input_activation_patch_has_zero_effect_and_cleans_hooks(tiny_lm):
    adapter, model, tokenizer = tiny_lm
    inputs = adapter.prepare_inputs(tokenizer, {"text": "A small test"}, torch.device("cpu"))
    outputs = adapter.forward(model, inputs, {})
    request = RunRequest(
        task_type="text_lm",
        model_id="sshleifer/tiny-gpt2",
        input_text="A small test",
        options={"counterfactual_text": "A small test", "patch_layers": [0, 1]},
    )
    context = SimpleNamespace(
        model=model,
        tokenizer=tokenizer,
        device=torch.device("cpu"),
        inputs=inputs,
        outputs=outputs,
        request=request,
    )
    baseline = sum(len(module._forward_hooks) for module in model.modules())
    result = ActivationPatchingAnalyzer().run(context)
    np.testing.assert_allclose(
        [row["delta_prob"] for row in result.summary["results"]], 0, atol=1e-7
    )
    assert sum(len(module._forward_hooks) for module in model.modules()) == baseline
    request.options.counterfactual_text = "This input is deliberately much longer than the first."
    with pytest.raises(ValueError, match="equal token counts"):
        ActivationPatchingAnalyzer().run(context)


def test_rollout_composes_later_attention_leftwards_with_residuals():
    first = torch.tensor([[1.0, 0.0], [0.2, 0.8]])
    second = torch.tensor([[0.7, 0.3], [0.6, 0.4]])
    outputs = SimpleNamespace(attentions=(first[None, None], second[None, None]))
    expected = ((second + torch.eye(2)) / 2) @ ((first + torch.eye(2)) / 2)
    for task, query in (("text_lm", -1), ("text_classification", 0)):
        result = AttentionRolloutAnalyzer().run(SimpleNamespace(outputs=outputs, task_type=task))
        np.testing.assert_allclose(result.arrays["rollout"], expected[query].numpy(), atol=1e-7)
        assert sum(result.arrays["rollout"]) == pytest.approx(1)


def test_real_backward_hooks_and_grad_cam_match_manual_calculation():
    model = nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(4, 2),
    )
    hooks = HookManager(model, ["1"], capture_gradients=True)
    store = hooks.attach()
    try:
        logits = model(torch.ones(1, 3, 8, 8))
        logits[0, 0].backward()
        assert set(store.activations) == set(store.gradients) == {"1"}
        result = GradCamAnalyzer().run(SimpleNamespace(capture=store))
        weights = store.gradients["1"].mean(dim=(2, 3), keepdim=True)
        expected = (weights * store.activations["1"]).sum(dim=1).relu()[0].numpy()
        expected -= expected.min()
        if expected.max() > 0:
            expected /= expected.max()
        np.testing.assert_allclose(result.arrays["heatmap"], expected, atol=1e-7)
    finally:
        hooks.clear()
    assert not model[1]._forward_hooks


def test_misaligned_comparison_does_not_invent_tensor_differences(tmp_path):
    store = ArtifactStore(tmp_path / "runs")
    for run_id, model in (("a", "first"), ("b", "second")):
        artifacts = store.create(run_id)
        store.save_metadata(
            artifacts,
            {
                "run_id": run_id,
                "status": "completed",
                "model_id": model,
                "task_type": "text_lm",
                "input_tokens": ["hello"],
            },
        )
        store.save_outputs(artifacts, {"generated_text": "hello"})
        store.save_summaries(artifacts, {})
    result = compare_artifacts(store, "a", "b")
    assert "attribution_diff" not in result
    assert "layer_similarity" not in result
    assert "alignment" in result["notes"][0]
