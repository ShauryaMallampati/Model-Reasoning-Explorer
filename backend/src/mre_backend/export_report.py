from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mre_backend.core.utils import json_dumps


def export_report(run_id: str, runs_dir: Path, reports_dir: Path) -> Path:
    run_dir = runs_dir / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run {run_id} not found")

    metadata = _read_json(run_dir / "metadata.json")
    outputs = _read_json(run_dir / "outputs.json")
    summaries = _read_json(run_dir / "summaries.json")

    report_dir = reports_dir / run_id
    report_dir.mkdir(parents=True, exist_ok=True)
    html_path = report_dir / "index.html"

    payload = json.dumps({"metadata": metadata, "outputs": outputs, "summaries": summaries})

    html = f"""<!doctype html>
<html>
<head>
  <meta charset='utf-8'/>
  <title>MRE Report {run_id}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    pre {{ background: #f6f6f6; padding: 12px; border-radius: 6px; }}
  </style>
</head>
<body>
  <h1>Model Reasoning Explorer Report</h1>
  <p><strong>Run ID:</strong> {run_id}</p>
  <h2>Metadata</h2>
  <pre id="metadata"></pre>
  <h2>Outputs</h2>
  <pre id="outputs"></pre>
  <h2>Summaries</h2>
  <pre id="summaries"></pre>

  <script>
    const payload = {payload};
    document.getElementById('metadata').textContent = JSON.stringify(payload.metadata, null, 2);
    document.getElementById('outputs').textContent = JSON.stringify(payload.outputs, null, 2);
    document.getElementById('summaries').textContent = JSON.stringify(payload.summaries, null, 2);
  </script>
</body>
</html>"""

    html_path.write_text(html)
    return html_path


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())
