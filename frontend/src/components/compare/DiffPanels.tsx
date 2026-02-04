import React from "react";
import Chart from "../common/Chart";
import AttributionText from "../run/AttributionText";

type DiffPanelsProps = {
  summary?: Record<string, unknown>;
};

const DiffPanels: React.FC<DiffPanelsProps> = ({ summary }) => {
  const deltas = (summary?.logit_lens_delta?.deltas as number[]) || [];
  const layerSimilarity = (summary?.layer_similarity?.values as number[]) || [];
  const attentionDelta = (summary?.attention_delta?.values as number[]) || [];
  const attributionDiff = summary?.attribution_diff;
  const tokensA = (summary?.generated_a || "").split(/\s+/).filter(Boolean);
  const tokensB = (summary?.generated_b || "").split(/\s+/).filter(Boolean);
  const maxLen = Math.max(tokensA.length, tokensB.length);
  const diffRows = Array.from({ length: maxLen }).map((_, idx) => ({
    a: tokensA[idx] || "",
    b: tokensB[idx] || "",
    same: tokensA[idx] === tokensB[idx]
  }));

  const deltaOption = {
    xAxis: { type: "category", data: deltas.map((_, i) => i) },
    yAxis: { type: "value" },
    series: [{ data: deltas, type: "bar" }]
  };

  const simOption = {
    xAxis: { type: "category", data: layerSimilarity.map((_, i) => i) },
    yAxis: { type: "value" },
    series: [{ data: layerSimilarity, type: "line", smooth: true }]
  };

  const attOption = {
    xAxis: { type: "category", data: attentionDelta.map((_, i) => i) },
    yAxis: { type: "value" },
    series: [{ data: attentionDelta, type: "bar" }]
  };

  return (
    <div className="grid two">
      <section className="panel">
        <h3>Prediction Diff</h3>
        <div className="grid two">
          <div>
            <h4>Run A</h4>
            <div className="pill">{summary?.prediction_a ?? "-"}</div>
            <p className="hint">{summary?.generated_a}</p>
          </div>
          <div>
            <h4>Run B</h4>
            <div className="pill">{summary?.prediction_b ?? "-"}</div>
            <p className="hint">{summary?.generated_b}</p>
          </div>
        </div>
        {summary?.logit_lens_delta?.pinpoint_layer !== undefined && (
          <p className="hint">Pinpoint layer: {summary.logit_lens_delta.pinpoint_layer}</p>
        )}
      </section>
      <section className="panel">
        <h3>Layerwise Similarity</h3>
        <Chart option={simOption} />
      </section>
      <section className="panel">
        <h3>Logit Lens Delta</h3>
        <Chart option={deltaOption} />
      </section>
      <section className="panel">
        <h3>Attention Delta</h3>
        <Chart option={attOption} />
      </section>
      <section className="panel">
        <h3>Attribution Diff</h3>
        {attributionDiff?.type === "text" ? (
          <AttributionText
            tokens={(summary?.generated_a || "").split(/\s+/).slice(0, (attributionDiff.preview || []).length)}
            scores={attributionDiff.preview || []}
          />
        ) : (
          <pre className="code-block">{JSON.stringify(attributionDiff ?? {}, null, 2)}</pre>
        )}
      </section>
      <section className="panel">
        <h3>Diff View</h3>
        <div className="diff-view">
          {diffRows.map((row, idx) => (
            <div key={idx} className={row.same ? "diff-row" : "diff-row diff-row-changed"}>
              <span className="diff-cell">{row.a}</span>
              <span className="diff-cell">{row.b}</span>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
};

export default DiffPanels;
