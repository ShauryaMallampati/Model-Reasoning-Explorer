from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from mre_backend.core.utils import require_safe_path, safe_component


def export_report(run_id: str, runs_dir: Path, reports_dir: Path) -> Path:
    safe_component(run_id)
    run_dir = require_safe_path(runs_dir / run_id, runs_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError("Run not found")
    metadata = _read_json(run_dir / "metadata.json")
    if metadata.get("status") != "completed":
        raise ValueError("Only completed runs can be exported")
    sections = {
        "Metadata": metadata,
        "Outputs": _read_json(run_dir / "outputs.json"),
        "Summaries": _read_json(run_dir / "summaries.json"),
    }
    report_dir = require_safe_path(reports_dir / run_id, reports_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    html_path = report_dir / "index.html"
    # Static escaped text avoids executing input that contains HTML or script tags.
    content = "\n".join(
        f"<h2>{title}</h2><pre>{html.escape(json.dumps(value, indent=2))}</pre>"
        for title, value in sections.items()
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>MRE Report {html.escape(run_id)}</title>
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'">
<style>body{{font-family:system-ui;margin:24px}}pre{{white-space:pre-wrap;background:#f6f6f6;
padding:12px;border-radius:6px;overflow-wrap:anywhere}}</style></head>
<body><h1>Model Reasoning Explorer Report</h1><p>Run: {html.escape(run_id)}</p>
{content}</body></html>"""
    html_path.write_text(document, encoding="utf-8")
    return html_path


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
