import React, { useEffect, useState } from "react";
import Editor from "react-simple-code-editor";
import Prism from "prismjs";
import "prismjs/components/prism-markdown";
import { cancelRun, listModels, startRun } from "../api/client";
import LogsPanel from "../components/run/LogsPanel";
import Spinner from "../components/common/Spinner";
import { useRunStore } from "../state/store";

const Home: React.FC = () => {
  const [models, setModels] = useState<any[]>([]);
  const [taskType, setTaskType] = useState("text_lm");
  const [modelId, setModelId] = useState("");
  const [prompt, setPrompt] = useState("The capital of France is");
  const [imageBase64, setImageBase64] = useState<string | undefined>();
  const [loading, setLoading] = useState(false);
  const [captureActivations, setCaptureActivations] = useState(true);
  const [captureGradients, setCaptureGradients] = useState(false);
  const [captureAttention, setCaptureAttention] = useState(true);
  const [captureLogits, setCaptureLogits] = useState(false);
  const [layerMode, setLayerMode] = useState("all");
  const [layerStride, setLayerStride] = useState(2);
  const [layerList, setLayerList] = useState("");

  const runId = useRunStore((s) => s.runId);
  const status = useRunStore((s) => s.status);
  const setRunId = useRunStore((s) => s.setRunId);
  const clear = useRunStore((s) => s.clear);

  useEffect(() => {
    listModels().then((data) => {
      setModels(data.models || []);
      if (data.models?.length) {
        setModelId(data.models[0].id);
        setTaskType(data.models[0].task_type);
      }
    });
  }, []);

  useEffect(() => {
    const match = models.find((m) => m.task_type === taskType);
    if (match) setModelId(match.id);
  }, [taskType, models]);

  useEffect(() => {
    if (taskType === "image_classification") {
      setCaptureGradients(true);
    }
  }, [taskType]);

  const handleImageUpload = (file: File) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result as string;
      const base64 = result.split(",")[1];
      setImageBase64(base64);
    };
    reader.readAsDataURL(file);
  };

  const run = async () => {
    clear();
    setLoading(true);
    const payload: any = {
      task_type: taskType,
      model_id: modelId,
      options: {
        capture: {
          activations: captureActivations,
          gradients: captureGradients,
          attentions: captureAttention,
          logits: captureLogits,
          layers: {
            mode: layerMode,
            stride: layerStride,
            layers: layerList.split(",").map((s) => s.trim()).filter(Boolean)
          }
        }
      }
    };
    if (taskType.startsWith("text")) {
      payload.input_text = prompt;
    } else {
      payload.input_image_base64 = imageBase64;
    }
    const res = await startRun(payload);
    setRunId(res.run_id);
    setLoading(false);
  };

  const cancel = async () => {
    if (!runId) return;
    await cancelRun(runId);
  };

  return (
    <div className="page">
      <section className="panel">
        <h2>Run Launcher</h2>
        <div className="grid two">
          <div>
            <label>Task Type</label>
            <select value={taskType} onChange={(e) => setTaskType(e.target.value)}>
              <option value="text_lm">Text LM (Next Token)</option>
              <option value="text_classification">Text Classification</option>
              <option value="image_classification">Image Classification</option>
            </select>
          </div>
          <div>
            <label>Model</label>
            <select value={modelId} onChange={(e) => setModelId(e.target.value)}>
              {models
                .filter((m) => m.task_type === taskType)
                .map((model) => (
                  <option key={model.id} value={model.id}>
                    {model.id}
                  </option>
                ))}
            </select>
          </div>
        </div>

        {taskType.startsWith("text") && (
          <div>
            <label>Prompt</label>
            <Editor
              value={prompt}
              onValueChange={setPrompt}
              highlight={(code) => Prism.highlight(code, Prism.languages.markdown, "markdown")}
              padding={12}
              className="code-editor"
            />
          </div>
        )}

        {taskType === "image_classification" && (
          <div>
            <label>Image Upload</label>
            <input
              type="file"
              accept="image/*,.ppm"
              onChange={(e) => e.target.files && handleImageUpload(e.target.files[0])}
            />
          </div>
        )}

        <div className="grid three">
          <label className="checkbox">
            <input
              type="checkbox"
              checked={captureActivations}
              onChange={(e) => setCaptureActivations(e.target.checked)}
            />
            Capture activations
          </label>
          <label className="checkbox">
            <input
              type="checkbox"
              checked={captureGradients}
              onChange={(e) => setCaptureGradients(e.target.checked)}
            />
            Capture gradients (required for Grad-CAM)
          </label>
          <label className="checkbox">
            <input
              type="checkbox"
              checked={captureAttention}
              onChange={(e) => setCaptureAttention(e.target.checked)}
            />
            Capture attention
          </label>
          <label className="checkbox">
            <input
              type="checkbox"
              checked={captureLogits}
              onChange={(e) => setCaptureLogits(e.target.checked)}
            />
            Capture logits per layer
          </label>
        </div>
        {taskType === "image_classification" && !captureGradients && (
          <p className="hint">Grad-CAM requires gradients to be enabled.</p>
        )}


        <div className="panel" style={{ marginTop: 16 }}>
          <h3>Layer Selection</h3>
          <div className="grid three">
            <div>
              <label>Mode</label>
              <select value={layerMode} onChange={(e) => setLayerMode(e.target.value)}>
                <option value="all">All Layers</option>
                <option value="every_n">Every N</option>
                <option value="custom">Custom List</option>
              </select>
            </div>
            <div>
              <label>Stride</label>
              <input
                type="number"
                value={layerStride}
                onChange={(e) => setLayerStride(Number(e.target.value))}
                disabled={layerMode !== "every_n"}
              />
            </div>
            <div>
              <label>Custom Layers</label>
              <input
                value={layerList}
                onChange={(e) => setLayerList(e.target.value)}
                placeholder="transformer.h.0,transformer.h.4"
                disabled={layerMode !== "custom"}
              />
            </div>
          </div>
        </div>
        <div className="row">
          <button className="primary" onClick={run} disabled={loading}>
            {loading ? "Running..." : "Run"}
          </button>
          {runId && status === "running" && (
            <button className="secondary" onClick={cancel}>
              Cancel
            </button>
          )}
          {loading && <Spinner />}
          {runId && <span className="hint">Run ID: {runId}</span>}
        </div>
      </section>

      <LogsPanel runId={runId} />
    </div>
  );
};

export default Home;
