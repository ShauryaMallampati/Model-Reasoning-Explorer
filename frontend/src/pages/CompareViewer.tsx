import React, { useState } from "react";
import { compareRuns } from "../api/client";
import type { ComparisonSummary } from "../api/client";
import DiffPanels from "../components/compare/DiffPanels";

const CompareViewer: React.FC = () => {
  const [runA, setRunA] = useState("");
  const [runB, setRunB] = useState("");
  const [summary, setSummary] = useState<ComparisonSummary>();
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const runCompare = async () => {
    setBusy(true);
    setError("");
    setSummary(undefined);
    try {
      const res = await compareRuns(runA.trim(), runB.trim());
      setSummary(res.summary);
    } catch (error) {
      setError(error instanceof Error ? error.message : "Comparison failed");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="page">
      <section className="panel">
        <h2>Compare Runs</h2>
        {error && <p role="alert">{error}</p>}
        <div className="grid two">
          <div>
            <label>Run A</label>
            <input aria-label="Run A" value={runA} onChange={(e) => setRunA(e.target.value)} />
          </div>
          <div>
            <label>Run B</label>
            <input aria-label="Run B" value={runB} onChange={(e) => setRunB(e.target.value)} />
          </div>
        </div>
        <button className="primary" onClick={runCompare} disabled={busy || !runA.trim() || !runB.trim()}>
          {busy ? "Comparing..." : "Compare"}
        </button>
      </section>

      <DiffPanels summary={summary} />
    </div>
  );
};

export default CompareViewer;
