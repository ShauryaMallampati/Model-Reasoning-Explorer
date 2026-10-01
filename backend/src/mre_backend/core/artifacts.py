from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .utils import atomic_json, require_safe_path, safe_component, utc_now_iso


@dataclass
class RunArtifacts:
    run_id: str
    run_dir: Path

    @property
    def metadata_path(self) -> Path:
        return self.run_dir / "metadata.json"

    @property
    def outputs_path(self) -> Path:
        return self.run_dir / "outputs.json"

    @property
    def summaries_path(self) -> Path:
        return self.run_dir / "summaries.json"

    @property
    def logs_path(self) -> Path:
        return self.run_dir / "logs.txt"


class ArtifactStore:
    def __init__(self, runs_dir: Path) -> None:
        self.runs_dir = runs_dir.resolve()
        self.index_path = self.runs_dir / "index.json"
        self._index_lock = threading.RLock()
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        if not self.index_path.exists():
            atomic_json(self.index_path, {"runs": []})

    def create(self, run_id: str) -> RunArtifacts:
        run_dir = self.get_run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        return RunArtifacts(run_id=run_id, run_dir=run_dir)

    def save_metadata(self, artifacts: RunArtifacts, metadata: dict[str, Any]) -> None:
        metadata = dict(metadata)
        metadata.setdefault("saved_at", utc_now_iso())
        atomic_json(artifacts.metadata_path, metadata)
        self._update_index(metadata)

    def save_outputs(self, artifacts: RunArtifacts, outputs: dict[str, Any]) -> None:
        atomic_json(artifacts.outputs_path, outputs)

    def save_summaries(self, artifacts: RunArtifacts, summaries: dict[str, Any]) -> None:
        atomic_json(artifacts.summaries_path, summaries)

    def save_arrays(
        self, artifacts: RunArtifacts, name: str, arrays: dict[str, np.ndarray]
    ) -> Path:
        path = self.get_artifact_path(artifacts.run_id, f"arrays_{safe_component(name)}.npz")
        np.savez_compressed(path, **arrays)
        return path

    def append_log(self, artifacts: RunArtifacts, line: str) -> None:
        with artifacts.logs_path.open("a", encoding="utf-8") as handle:
            handle.write(line.rstrip() + "\n")

    def list_runs(self) -> list[dict[str, Any]]:
        with self._index_lock:
            return json.loads(self.index_path.read_text(encoding="utf-8")).get("runs", [])

    def get_run_dir(self, run_id: str) -> Path:
        path = self.runs_dir / safe_component(run_id)
        return require_safe_path(path, self.runs_dir)

    def get_artifact_path(self, run_id: str, name: str) -> Path:
        root = self.get_run_dir(run_id)
        return require_safe_path(root / safe_component(name), root)

    def _update_index(self, metadata: dict[str, Any]) -> None:
        with self._index_lock:
            runs = [run for run in self.list_runs() if run.get("run_id") != metadata.get("run_id")]
            runs.insert(0, metadata)
            atomic_json(self.index_path, {"runs": runs[:200]})
