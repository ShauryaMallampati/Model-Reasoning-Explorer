import React, { useEffect, useMemo, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { exportRun, getRun, getArtifactUrl, resourceUrl } from "../api/client";
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
  const [error, setError] = useState("");
  const [tab, setTab] = useState("overview");
  const [reportPath, setReportPath] = useState<string | null>(null);
  const [attrMethod, setAttrMethod] = useState("integrated_gradients");

  useEffect(() => {
    if (!runId) return;
    setLoading(true);
    setError("");
    let active = true;
    getRun(runId).then((res) => {
      if (active) setData(res);
    }).catch((error: Error) => {
      if (active) setError(error.message);
    }).finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
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
      getRun(runId).then((res) => setData(res)).catch((error: Error) => {
        setError(error.message);
        clearInterval(id);
      });
    }, 1500);
    return () => clearInterval(id);
  }, [runId, data]);


  const exportReport = async () => {
    if (!runId) return;
    try {
      const res = await exportRun(runId);
      setReportPath(resourceUrl(res.report_url));
    } catch (error) {
      setError(error instanceof Error ? error.message : "Export failed");
    }
  };

  const shareLink = () => {
    if (!runId) return;
    const url = `${window.location.origin}/runs/${runId}`;
    navigator.clipboard.writeText(url).catch(() => setError("Clipboard unavailable; copy the address bar URL."));
  };

  const tokens = useMemo(() => {
    return (data?.metadata?.input_tokens as string[] | undefined)?.slice(0, 64) || [];
  }, [data]);

  const attributionScores = useMemo(() => {
    if (!data) return [];
    if (attrMethod === "attention_rollout") {
      return data?.summaries?.attention_rollout?.rollout_preview ?? [];
    }
    return data?.summaries?.integrated_gradients?.attribution_preview ?? [];
  }, [data, attrMethod]);

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
    return <div className="panel" role="alert">{error || "Run not found"}</div>;
  }

  return (
    <div className="page">
      {(error || data.error) && <p className="panel" role="alert">{error || data.error}</p>}
      <p className="hint">Status: {data.status}</p>
      <div className="row space-between">
        <h2>Run Viewer</h2>
        <div className="row">
          <button className="secondary" onClick={shareLink}>Copy Local Run Link</button>
          <button className="secondary" onClick={() => navigate("/")}>New Run</button>
        </div>
      </div>

      <div className="tabs">
        {[
          ["overview", "Overview"],
          ["attribution", "Attribution"],
          ["layers", "Layer Timeline"],
          ["neurons", "Layer Statistics"],
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
              {attributionScores.length ?
                <AttributionText tokens={tokens.slice(0, attributionScores.length)} scores={attributionScores} /> :
                <p className="hint">No attribution was recorded for this method.</p>}
              <p className="hint">{attrMethod === "integrated_gradients"
                ? "Orange and blue show positive and negative contributions relative to the recorded baseline. Color intensity is scaled within this input."
                : "Attention flow is descriptive, not a causal explanation of the prediction."}</p>
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
            <h3>Recorded Analysis</h3>
            <p className="hint">{data.metadata?.model_id}</p>
            <h4>Model output</h4>
            <div className="pill">{data.outputs?.prediction ?? data.outputs?.top_k?.[0]?.token ?? "Not available"}</div>
            <p>Seed {data.metadata?.seed ?? "not recorded"} · {tokens.length || "Image"} {tokens.length ? "model tokens" : "input"}</p>
            <pre className="code-block">{JSON.stringify(data.summaries?.[attrMethod] ?? { message: "Not captured" },
              (key, value) => key.endsWith("_preview") ? "Available in the saved artifact" : value, 2)}</pre>
            <details><summary>All analyzer metadata</summary>
              <pre className="code-block">{JSON.stringify(data.summaries ?? {}, null, 2)}</pre>
            </details>
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
          <button className="secondary" onClick={exportReport} disabled={data.status !== "completed"}>Export Report</button>
          {reportPath && <p><a href={reportPath} target="_blank" rel="noreferrer">Open exported report</a></p>}
          <ul className="list">
            {(data.artifacts || []).map((name: string) => (
              <li key={name}><a href={getArtifactUrl(runId!, name)} download>{name}</a></li>
            ))}
          </ul>
        </section>
      )}
    </div>
  );
};

export default RunViewer;
