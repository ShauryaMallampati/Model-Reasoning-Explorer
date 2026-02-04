import React, { useState } from "react";
import { getDatasetRun, startDatasetRun } from "../api/client";
import DatasetTable from "../components/dataset/DatasetTable";

const DatasetExplorer: React.FC = () => {
  const [kind, setKind] = useState("text");
  const [file, setFile] = useState<File | null>(null);
  const [path, setPath] = useState("sample_data/text/sample_text.csv");
  const [datasetRunId, setDatasetRunId] = useState<string | null>(null);
  const [summary, setSummary] = useState<any>(null);
  const [examples, setExamples] = useState<any[]>([]);

  React.useEffect(() => {
    if (kind === "text") {
      setPath("sample_data/text/sample_text.csv");
    } else {
      setPath("sample_data/images");
    }
  }, [kind]);

  const runDataset = async () => {
    const form = new FormData();
    form.append("kind", kind);
    form.append("model_id", "distilbert-base-uncased-finetuned-sst-2-english");
    form.append("task_type", kind === "text" ? "text_classification" : "image_classification");
    if (path) form.append("path", path);
    if (file) form.append("file", file);
    const res = await startDatasetRun(form);
    setDatasetRunId(res.dataset_run_id);
    poll(res.dataset_run_id);
  };

  const poll = async (id: string) => {
    const res = await getDatasetRun(id);
    setSummary(res.summary);
    setExamples(res.examples || []);
    if (res.status !== "completed") {
      setTimeout(() => poll(id), 1000);
    }
  };

  return (
    <div className="page">
      <section className="panel">
        <h2>Dataset Explorer</h2>
        <div className="grid two">
          <div>
            <label>Dataset Type</label>
            <select value={kind} onChange={(e) => setKind(e.target.value)}>
              <option value="text">Text (CSV)</option>
              <option value="image">Images (Folder)</option>
            </select>
          </div>
          <div>
            <label>Path (server-side)</label>
            <input value={path} onChange={(e) => setPath(e.target.value)} placeholder="sample_data/text/sample_text.csv" />
          </div>
        </div>
        {kind === "text" && (
          <div>
            <label>Upload CSV</label>
            <input type="file" accept=".csv" onChange={(e) => setFile(e.target.files?.[0] || null)} />
          </div>
        )}
        <button className="primary" onClick={runDataset}>Run Dataset</button>
        {datasetRunId && <span className="hint">Dataset run: {datasetRunId}</span>}
      </section>

      <section className="panel">
        <h3>Summary</h3>
        <pre className="code-block">{JSON.stringify(summary ?? {}, null, 2)}</pre>
      </section>

      <DatasetTable examples={examples} />
    </div>
  );
};

export default DatasetExplorer;
