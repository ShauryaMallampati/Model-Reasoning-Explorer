import React, { useState } from "react";
import { compareRuns } from "../api/client";
import DiffPanels from "../components/compare/DiffPanels";

const CompareViewer: React.FC = () => {
  const [runA, setRunA] = useState("");
  const [runB, setRunB] = useState("");
  const [summary, setSummary] = useState<any>(null);

  const runCompare = async () => {
    const res = await compareRuns(runA, runB);
    setSummary(res.summary);
  };

  return (
    <div className="page">
      <section className="panel">
        <h2>Compare Runs</h2>
        <div className="grid two">
          <div>
            <label>Run A</label>
            <input value={runA} onChange={(e) => setRunA(e.target.value)} />
          </div>
          <div>
            <label>Run B</label>
            <input value={runB} onChange={(e) => setRunB(e.target.value)} />
          </div>
        </div>
        <button className="primary" onClick={runCompare}>Compare</button>
      </section>

      <DiffPanels summary={summary ?? {}} />
    </div>
  );
};

export default CompareViewer;
