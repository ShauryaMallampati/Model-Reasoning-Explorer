import React from "react";
import Chart from "../common/Chart";

type DiffPanelsProps = {
  summary?: Record<string, unknown>;
};

const DiffPanels: React.FC<DiffPanelsProps> = ({ summary }) => {
  const deltas = (summary?.logit_lens_delta?.deltas as number[]) || Array.from({ length: 8 }).map((_, i) => Math.sin(i / 2) * 0.2);
  const option = {
    xAxis: { type: "category", data: deltas.map((_, i) => i) },
    yAxis: { type: "value" },
    series: [{ data: deltas, type: "bar" }]
  };

  return (
    <div className="grid two">
      <section className="panel">
        <h3>Prediction Diff</h3>
        <pre className="code-block">{JSON.stringify(summary ?? {}, null, 2)}</pre>
        {summary?.logit_lens_delta?.pinpoint_layer !== undefined && (
          <p className="hint">Pinpoint layer: {summary.logit_lens_delta.pinpoint_layer}</p>
        )}
      </section>
      <section className="panel">
        <h3>Layerwise Similarity</h3>
        <Chart option={option} />
      </section>
    </div>
  );
};

export default DiffPanels;
