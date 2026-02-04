import React, { useState } from "react";
import { startRun } from "../../api/client";

type CounterfactualPanelProps = {
  originalText?: string;
  modelId?: string;
  taskType?: "text_lm" | "text_classification" | "image_classification";
  onNewRun?: (runId: string) => void;
};

const CounterfactualPanel: React.FC<CounterfactualPanelProps> = ({
  originalText,
  modelId,
  taskType,
  onNewRun
}) => {
  const [text, setText] = useState(originalText || "");
  const [busy, setBusy] = useState(false);

  const rerun = async () => {
    if (!modelId || !taskType) return;
    setBusy(true);
    const payload = {
      task_type: taskType,
      model_id: modelId,
      input_text: text,
      options: {
        analyzers: [],
        counterfactual_text: text
      }
    };
    const res = await startRun(payload);
    setBusy(false);
    onNewRun?.(res.run_id);
  };

  const searchFlip = async () => {
    if (!modelId || !taskType) return;
    setBusy(true);
    const payload = {
      task_type: taskType,
      model_id: modelId,
      input_text: text,
      options: {
        analyzers: ["counterfactual_search"],
        counterfactual_text: text
      }
    };
    const res = await startRun(payload);
    setBusy(false);
    onNewRun?.(res.run_id);
  };

  return (
    <section className="panel">
      <h3>Counterfactual Sandbox</h3>
      <textarea
        className="textarea"
        rows={4}
        value={text}
        onChange={(e) => setText(e.target.value)}
      />
      <div className="row">
        <button className="primary" onClick={rerun} disabled={busy}>
          {busy ? "Running..." : "Rerun"}
        </button>
        <button className="secondary" onClick={searchFlip} disabled={busy}>
          Minimal Flip Search
        </button>
        <span className="hint">Edit input and rerun to validate changes</span>
      </div>
    </section>
  );
};

export default CounterfactualPanel;
