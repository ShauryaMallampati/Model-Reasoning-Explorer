from mre_backend.core.artifacts import ArtifactStore


def test_artifact_store_roundtrip(tmp_path):
    store = ArtifactStore(tmp_path)
    artifacts = store.create("run_test")
    store.save_metadata(artifacts, {"run_id": "run_test"})
    store.save_outputs(artifacts, {"prediction": "ok"})
    store.save_summaries(artifacts, {"summary": True})

    assert (tmp_path / "run_test" / "metadata.json").exists()
    assert (tmp_path / "run_test" / "outputs.json").exists()
    assert (tmp_path / "run_test" / "summaries.json").exists()
