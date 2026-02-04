import React from "react";
import ReactECharts from "echarts-for-react";

type ChartProps = {
  option: Record<string, unknown>;
  height?: number;
};

const Chart: React.FC<ChartProps> = ({ option, height = 260 }) => {
  return <ReactECharts style={{ height }} option={option} />;
};

export default Chart;
