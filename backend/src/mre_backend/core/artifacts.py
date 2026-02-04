from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .utils import json_dumps, utc_now_iso


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
        self.runs_dir = runs_dir
        self.index_path = self.runs_dir / "index.json"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        if not self.index_path.exists():
            self.index_path.write_text(json_dumps({"runs": []}))

    def create(self, run_id: str) -> RunArtifacts:
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return RunArtifacts(run_id=run_id, run_dir=run_dir)

    def save_metadata(self, artifacts: RunArtifacts, metadata: dict[str, Any]) -> None:
        metadata = dict(metadata)
        metadata.setdefault("saved_at", utc_now_iso())
        artifacts.metadata_path.write_text(json_dumps(metadata))
        self._update_index(metadata)

    def save_outputs(self, artifacts: RunArtifacts, outputs: dict[str, Any]) -> None:
        artifacts.outputs_path.write_text(json_dumps(outputs))

    def save_summaries(self, artifacts: RunArtifacts, summaries: dict[str, Any]) -> None:
        artifacts.summaries_path.write_text(json_dumps(summaries))

    def save_arrays(self, artifacts: RunArtifacts, name: str, arrays: dict[str, np.ndarray]) -> Path:
        path = artifacts.run_dir / f"arrays_{name}.npz"
        np.savez_compressed(path, **arrays)
        return path

    def append_log(self, artifacts: RunArtifacts, line: str) -> None:
        with artifacts.logs_path.open("a") as f:
            f.write(line.rstrip() + "\n")

    def list_runs(self) -> list[dict[str, Any]]:
        if not self.index_path.exists():
            return []
        return json.loads(self.index_path.read_text()).get("runs", [])

    def get_run_dir(self, run_id: str) -> Path:
        return self.runs_dir / run_id

    def get_artifact_path(self, run_id: str, name: str) -> Path:
        return self.get_run_dir(run_id) / name

    def _update_index(self, metadata: dict[str, Any]) -> None:
        data = {"runs": []}
        if self.index_path.exists():
            data = json.loads(self.index_path.read_text())
        runs = data.get("runs", [])
        runs = [r for r in runs if r.get("run_id") != metadata.get("run_id")]
        runs.insert(0, metadata)
        data["runs"] = runs[:200]
        self.index_path.write_text(json.dumps(data, indent=2))
