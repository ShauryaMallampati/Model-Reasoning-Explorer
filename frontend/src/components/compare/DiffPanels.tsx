import React from "react";
import type { ComparisonSummary } from "../../api/client";
import Chart from "../common/Chart";
import AttributionText from "../run/AttributionText";

const DiffPanels: React.FC<{ summary?: ComparisonSummary }> = ({ summary }) => {
  if (!summary) return <p className="hint">Choose two completed runs to compare.</p>;
  const similarity = summary.layer_similarity;
  const attention = summary.attention_delta;
  const attribution = summary.attribution_diff;
  return (
    <div className="grid two">
      <section className="panel">
        <h3>Output Comparison</h3>
        <div className="grid two">
          {(["a", "b"] as const).map((side) => (
            <div key={side}>
              <h4>Run {side.toUpperCase()}</h4>
              <div className="pill">{summary[`prediction_${side}`] ?? "Next-token model"}</div>
              <p>{summary[`generated_${side}`] || "No generated text"}</p>
            </div>
          ))}
        </div>
        {summary.notes?.map((note) => <p className="hint" key={note}>{note}</p>)}
      </section>
      {similarity && <section className="panel">
        <h3>Mean Hidden-State Cosine Similarity</h3>
        <Chart option={{ xAxis: { type: "category", data: similarity.layers },
          yAxis: { type: "value" }, series: [{ data: similarity.values, type: "line" }] }} />
      </section>}
      {attention && <section className="panel">
        <h3>Mean Absolute Attention Difference</h3>
        <Chart option={{ xAxis: { type: "category", data: attention.layers },
          yAxis: { type: "value" }, series: [{ data: attention.values, type: "bar" }] }} />
      </section>}
      {attribution && <section className="panel">
        <h3>Attribution Difference</h3>
        {attribution.type === "text" ?
          <AttributionText tokens={attribution.tokens || []} scores={attribution.preview} /> :
          <pre className="code-block">{JSON.stringify(attribution.preview, null, 2)}</pre>}
      </section>}
    </div>
  );
};
export default DiffPanels;
