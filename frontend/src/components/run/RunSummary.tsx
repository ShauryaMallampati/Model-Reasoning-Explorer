import React from "react";

type RunSummaryProps = {
  outputs?: { prediction?: string; top_k?: { label?: string; token?: string; prob: number }[] };
  metadata?: Record<string, unknown>;
};

const RunSummary: React.FC<RunSummaryProps> = ({ outputs, metadata }) => {
  return (
    <section className="panel">
      <h3>Overview</h3>
      <div className="grid two">
        <div>
          <h4>Prediction</h4>
          <div className="pill">{outputs?.prediction ?? "-"}</div>
          {outputs?.generated_text && (
            <p className="hint">Generated: {outputs.generated_text}</p>
          )}
        </div>
        <div>
          <h4>Metadata</h4>
          <pre className="code-block">{JSON.stringify(metadata ?? {}, null, 2)}</pre>
        </div>
      </div>
      <h4>Top-K</h4>
      <ul className="list">
        {(outputs?.top_k ?? []).map((item, idx) => (
          <li key={idx}>
            <span>{item.label ?? item.token}</span>
            <span>{item.prob.toFixed(4)}</span>
          </li>
        ))}
      </ul>
    </section>
  );
};

export default RunSummary;
