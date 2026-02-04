import React from "react";
import Chart from "../common/Chart";

type LayerTimelineProps = {
  values: number[];
  label?: string;
};

const LayerTimeline: React.FC<LayerTimelineProps> = ({ values, label = "Layer Metric" }) => {
  const option = {
    xAxis: {
      type: "category",
      data: values.map((_, idx) => idx)
    },
    yAxis: {
      type: "value"
    },
    series: [
      {
        data: values,
        type: "line",
        smooth: true,
        areaStyle: {}
      }
    ],
    tooltip: {
      trigger: "axis"
    }
  };

  return (
    <section className="panel">
      <h3>{label}</h3>
      <Chart option={option} />
    </section>
  );
};

export default LayerTimeline;
