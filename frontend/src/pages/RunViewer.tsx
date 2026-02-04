import React, { useEffect, useMemo, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { exportRun, getRun } from "../api/client";
import RunSummary from "../components/run/RunSummary";
import AttributionText from "../components/run/AttributionText";
import AttributionImage from "../components/run/AttributionImage";
import LayerTimeline from "../components/run/LayerTimeline";
import NeuronExplorer from "../components/run/NeuronExplorer";
import CounterfactualPanel from "../components/run/CounterfactualPanel";
import Spinner from "../components/common/Spinner";

const RunViewer: React.FC = () => {
  const { runId } = useParams();
  const navigate = useNavigate();
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [tab, setTab] = useState("overview");
  const [reportPath, setReportPath] = useState<string | null>(null);
  const [attrMethod, setAttrMethod] = useState("integrated_gradients");

  useEffect(() => {
    if (!runId) return;
    setLoading(true);
    getRun(runId).then((res) => {
      setData(res);
      setLoading(false);
    });
  }, [runId]);

  useEffect(() => {
    if (!data?.metadata?.task_type) return;
    if (data.metadata.task_type.startsWith("text")) {
      setAttrMethod("integrated_gradients");
    } else {
      setAttrMethod("grad_cam");
    }
  }, [data?.metadata?.task_type]);

  useEffect(() => {
    if (!runId) return;
    if (!data || data.status === "completed" || data.status === "failed") return;
    const id = setInterval(() => {
      getRun(runId).then((res) => setData(res));
    }, 1500);
    return () => clearInterval(id);
  }, [runId, data]);


  const exportReport = async () => {
    if (!runId) return;
    const res = await exportRun(runId);
    setReportPath(res.report_path);
  };

  const shareLink = () => {
    if (!runId) return;
    const url = `${window.location.origin}/runs/${runId}`;
    navigator.clipboard.writeText(url);
  };

  const tokens = useMemo(() => {
    const text = data?.metadata?.input_text as string | undefined;
    if (!text) return [];
    return text.split(/\s+/);
  }, [data, attrMethod]);

  const attributionScores = useMemo(() => {
    if (!data) return [];
    if (attrMethod === "attention_rollout") {
      return data?.summaries?.attention_rollout?.rollout_preview ?? [];
    }
    return data?.summaries?.integrated_gradients?.attribution_preview ?? [];
  }, [data]);

  const imageHeatmap = useMemo(() => {
    if (!data) return undefined;
    if (attrMethod === "grad_cam") return data?.summaries?.grad_cam?.heatmap_preview;
    if (attrMethod === "occlusion") return data?.summaries?.occlusion?.heatmap_preview;
    return data?.summaries?.integrated_gradients?.heatmap_preview;
  }, [data, attrMethod]);

  const layerValues = useMemo(() => {
    const logitLens = data?.summaries?.logit_lens?.layers || [];
    if (logitLens.length > 0) {
      return logitLens.map((layer: any) => layer.top_k?.[0]?.prob ?? 0);
    }
    const stats = data?.summaries?.activation_stats?.layers || [];
    return stats.map((layer: any) => layer.mean_abs ?? 0);
  }, [data]);

  const layerLabel = useMemo(() => {
    const logitLens = data?.summaries?.logit_lens?.layers || [];
    return logitLens.length > 0 ? "Logit Lens Top-1" : "Activation Norms";
  }, [data]);

  const neuronList = useMemo(() => {
    const patch = data?.summaries?.activation_patching?.results || [];
    if (patch.length > 0) {
      return patch.map((r: any) => ({ name: `layer_${r.layer}`, value: r.delta_prob ?? 0 }));
    }
    const stats = data?.summaries?.activation_stats?.top_layers || [];
    return stats.map((s: any) => ({ name: s.layer, value: s.mean_abs ?? 0 }));
  }, [data]);

  if (loading) {
    return <Spinner />;
  }

  if (!data) {
    return <div className="panel">Run not found</div>;
  }

  return (
    <div className="page">
      <div className="row space-between">
        <h2>Run Viewer</h2>
        <div className="row">
          <button className="secondary" onClick={shareLink}>Copy Share Link</button>
          <button className="secondary" onClick={() => navigate("/")}>New Run</button>
        </div>
      </div>

      <div className="tabs">
        {[
          ["overview", "Overview"],
          ["attribution", "Attribution"],
          ["layers", "Layer Timeline"],
          ["neurons", "Neuron Explorer"],
          ["counterfactual", "Counterfactual"],
          ["artifacts", "Artifacts"]
        ].map(([id, label]) => (
          <button key={id} className={tab === id ? "active" : ""} onClick={() => setTab(id)}>
            {label}
          </button>
        ))}
      </div>

      {tab === "overview" && (
        <RunSummary outputs={data.outputs} metadata={data.metadata} />
      )}

      {tab === "attribution" && (
        <div className="grid two">
          {data.metadata?.task_type?.startsWith("text") ? (
            <section className="panel">
              <h3>Token Attribution</h3>
              <div className="row">
                <label className="hint">Method</label>
                <select value={attrMethod} onChange={(e) => setAttrMethod(e.target.value)}>
                  <option value="integrated_gradients">Integrated Gradients</option>
                  <option value="attention_rollout">Attention Rollout</option>
                </select>
              </div>
              <AttributionText tokens={tokens} scores={attributionScores} />
            </section>
          ) : (
            <section className="panel">
              <h3>Image Attribution</h3>
              <div className="row">
                <label className="hint">Method</label>
                <select value={attrMethod} onChange={(e) => setAttrMethod(e.target.value)}>
                  <option value="grad_cam">Grad-CAM</option>
                  <option value="occlusion">Occlusion</option>
                  <option value="integrated_gradients">Integrated Gradients</option>
                </select>
              </div>
              <AttributionImage
                imageUrl={data.metadata?.input_image_preview ? `data:image/png;base64,${data.metadata.input_image_preview}` : undefined}
                heatmap={imageHeatmap}
              />
            </section>
          )}
          <section className="panel">
            <h3>Analyzer Summary</h3>
            <pre className="code-block">{JSON.stringify(data.summaries ?? {}, null, 2)}</pre>
          </section>
        </div>
      )}

      {tab === "layers" && (
        <LayerTimeline values={layerValues} label={layerLabel} />
      )}

      {tab === "neurons" && (
        <NeuronExplorer activations={neuronList} />
      )}

      {tab === "counterfactual" && (
        <CounterfactualPanel
          originalText={data.metadata?.input_text}
          modelId={data.metadata?.model_id}
          taskType={data.metadata?.task_type}
          imagePreview={data.metadata?.input_image_preview}
          onNewRun={(id) => navigate(`/runs/${id}`)}
        />
      )}

      {tab === "artifacts" && (
        <section className="panel">
          <h3>Artifacts</h3>
          <button className="secondary" onClick={exportReport}>Export Report</button>
          {reportPath && <p className="hint">Report: {reportPath}</p>}
          <ul className="list">
            {(data.artifacts || []).map((name: string) => (
              <li key={name}>{name}</li>
            ))}
          </ul>
        </section>
      )}
    </div>
  );
};

export default RunViewer;
