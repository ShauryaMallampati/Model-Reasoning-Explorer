import React from "react";
import type { RunOutputs } from "../../api/client";

type RunSummaryProps = {
  outputs?: RunOutputs;
  metadata?: Record<string, unknown>;
};

const RunSummary: React.FC<RunSummaryProps> = ({ outputs, metadata }) => {
  return (
    <section className="panel">
      <h3>Overview</h3>
      <div className="grid two">
        <div>
          <h4>Prediction</h4>
          <div className="pill">{outputs?.prediction ?? outputs?.top_k?.[0]?.token ?? "-"}</div>
          {outputs?.generated_text && (
            <p className="hint">Generated: {outputs.generated_text}</p>
          )}
        </div>
        <div>
          <h4>Run Details</h4>
          <p><strong>Model:</strong> {String(metadata?.model_id ?? "Unknown")}</p>
          <p><strong>Task:</strong> {String(metadata?.task_type ?? "Unknown")}</p>
          <p><strong>Seed:</strong> {String(metadata?.seed ?? "Not recorded")}</p>
          <p><strong>Input:</strong> {String(metadata?.input_text ?? "Image input")}</p>
          <details><summary>Recorded metadata</summary>
            <pre className="code-block">{JSON.stringify(metadata ?? {}, (key, value) =>
              key === "input_image_preview" && value ? "[Image preview stored with run]" : value, 2)}</pre>
          </details>
        </div>
      </div>
      <h4>Top-K</h4>
      <ul className="list">
        {(outputs?.top_k ?? []).map((item, idx) => (
          <li key={idx}>
            <span>{item.label ?? item.token}</span>
            <span>{item.prob.toPrecision(4)}</span>
          </li>
        ))}
      </ul>
    </section>
  );
};

export default RunSummary;
