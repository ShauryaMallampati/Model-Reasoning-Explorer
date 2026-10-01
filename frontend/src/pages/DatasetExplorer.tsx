import React, { useState } from "react";
import { getDatasetRun, startDatasetRun } from "../api/client";
import DatasetTable from "../components/dataset/DatasetTable";
import Chart from "../components/common/Chart";

const DatasetExplorer: React.FC = () => {
  const [kind, setKind] = useState("text");
  const [file, setFile] = useState<File | null>(null);
  const [path, setPath] = useState("text/sample_text.csv");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [datasetRunId, setDatasetRunId] = useState<string | null>(null);
  const [summary, setSummary] = useState<any>(null);
  const [examples, setExamples] = useState<any[]>([]);

  React.useEffect(() => {
    if (kind === "text") {
      setPath("text/sample_text.csv");
    } else {
      setPath("images");
    }
  }, [kind]);

  const runDataset = async () => {
    const form = new FormData();
    form.append("kind", kind);
    form.append("model_id", kind === "text" ? "distilbert-base-uncased-finetuned-sst-2-english" : "resnet18");
    form.append("task_type", kind === "text" ? "text_classification" : "image_classification");
    if (path) form.append("path", path);
    if (file && kind === "text") form.append("file", file);
    setBusy(true);
    setError("");
    setSummary(null);
    setExamples([]);
    setDatasetRunId(null);
    try {
      const res = await startDatasetRun(form);
      setDatasetRunId(res.dataset_run_id);
    } catch (error) {
      setError(error instanceof Error ? error.message : "Dataset request failed");
      setBusy(false);
    }
  };

  React.useEffect(() => {
    if (!datasetRunId) return;
    let active = true;
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      try {
        const res = await getDatasetRun(datasetRunId);
        if (!active) return;
        setSummary(res.summary);
        setExamples(res.examples || []);
        if (res.status === "failed") {
          setError(res.error || "Dataset run failed");
          setBusy(false);
        } else if (res.status === "completed") {
          setBusy(false);
        } else {
          timer = setTimeout(poll, 1000);
        }
      } catch (error) {
        if (!active) return;
        setError(error instanceof Error ? error.message : "Dataset status unavailable");
        setBusy(false);
      }
    };
    void poll();
    return () => { active = false; clearTimeout(timer); };
  }, [datasetRunId]);

  return (
    <div className="page">
      <section className="panel">
        <h2>Dataset Explorer</h2>
        {error && <p role="alert">{error}</p>}
        <div className="grid two">
          <div>
            <label>Dataset Type</label>
            <select aria-label="Dataset Type" disabled={busy} value={kind} onChange={(e) => setKind(e.target.value)}>
              <option value="text">Text (CSV)</option>
              <option value="image">Images (Folder)</option>
            </select>
          </div>
          <div>
            <label>Path (server-side)</label>
            <input aria-label="Dataset Path" value={path} onChange={(e) => setPath(e.target.value)} placeholder="text/sample_text.csv" />
          </div>
        </div>
        {kind === "text" && (
          <div>
            <label>Upload CSV</label>
            <input type="file" accept=".csv" onChange={(e) => setFile(e.target.files?.[0] || null)} />
          </div>
        )}
        <button className="primary" onClick={runDataset} disabled={busy}>{busy ? "Running..." : "Run Dataset"}</button>
        {datasetRunId && <span className="hint">Dataset run: {datasetRunId}</span>}
      </section>

      <section className="panel">
        <h3>Summary</h3>
        <pre className="code-block">{JSON.stringify(summary ?? {}, null, 2)}</pre>
      </section>

      {summary?.confusion_matrix && (
        <section className="panel">
          <h3>Confusion Matrix</h3>
          <table className="table">
            <thead>
              <tr>
                <th>True \\ Pred</th>
                {summary.confusion_matrix.labels.map((label: string) => (
                  <th key={label}>{label}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {summary.confusion_matrix.matrix.map((row: number[], idx: number) => (
                <tr key={idx}>
                  <td>{summary.confusion_matrix.labels[idx]}</td>
                  {row.map((val, j) => (
                    <td key={j}>{val}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}

      {summary?.calibration && (
        <section className="panel">
          <h3>Empirical Confidence Bins</h3>
          <p className="hint">Descriptive statistics for the selected sample, not evidence of general calibration.</p>
          <Chart
            option={{
              xAxis: { type: "category", data: summary.calibration.bins.slice(0, -1) },
              yAxis: { type: "value" },
              series: [
                { name: "Avg Confidence", data: summary.calibration.avg_confidence, type: "line" },
                { name: "Avg Accuracy", data: summary.calibration.avg_accuracy, type: "line" }
              ],
              tooltip: { trigger: "axis" }
            }}
          />
        </section>
      )}

      {summary?.error_slices && (
        <section className="panel">
          <h3>Error Slices</h3>
          <ul className="list">
            {summary.error_slices.by_label?.map((item: any) => (
              <li key={item.label}>
                <span>{item.label}</span>
                <span>{(item.error_rate * 100).toFixed(1)}%</span>
              </li>
            ))}
          </ul>
          <h4>High Confidence Errors</h4>
          <pre className="code-block">{JSON.stringify(summary.error_slices.high_confidence_errors ?? [], null, 2)}</pre>
        </section>
      )}

      {summary?.cluster_summary && (
        <section className="panel">
          <h3>Clusters</h3>
          <pre className="code-block">{JSON.stringify(summary.cluster_summary ?? [], null, 2)}</pre>
        </section>
      )}

      <DatasetTable examples={examples} />
    </div>
  );
};

export default DatasetExplorer;
