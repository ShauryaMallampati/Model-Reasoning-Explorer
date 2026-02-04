import React, { useEffect, useRef, useState } from "react";
import { startRun } from "../../api/client";

type CounterfactualPanelProps = {
  originalText?: string;
  modelId?: string;
  taskType?: "text_lm" | "text_classification" | "image_classification";
  imagePreview?: string;
  onNewRun?: (runId: string) => void;
};

const CounterfactualPanel: React.FC<CounterfactualPanelProps> = ({
  originalText,
  modelId,
  taskType,
  imagePreview,
  onNewRun
}) => {
  const [text, setText] = useState(originalText || "");
  const [busy, setBusy] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [rect, setRect] = useState<{ x: number; y: number; w: number; h: number } | null>(null);
  const [dragging, setDragging] = useState(false);
  const [origin, setOrigin] = useState<{ x: number; y: number } | null>(null);

  useEffect(() => {
    if (!imagePreview || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const img = new Image();
    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);
      if (rect) {
        ctx.strokeStyle = "#f97316";
        ctx.lineWidth = 2;
        ctx.strokeRect(rect.x, rect.y, rect.w, rect.h);
      }
    };
    img.src = `data:image/png;base64,${imagePreview}`;
  }, [imagePreview, rect]);

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

  const occludeAndRerun = async () => {
    if (!modelId || !taskType || !imagePreview || !rect) return;
    setBusy(true);
    const img = new Image();
    img.onload = async () => {
      const canvas = document.createElement("canvas");
      canvas.width = img.width;
      canvas.height = img.height;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.drawImage(img, 0, 0);
      ctx.fillStyle = "rgba(0,0,0,0.6)";
      ctx.fillRect(rect.x, rect.y, rect.w, rect.h);
      const dataUrl = canvas.toDataURL("image/png");
      const base64 = dataUrl.split(",")[1];
      const payload = {
        task_type: taskType,
        model_id: modelId,
        input_image_base64: base64,
        options: {
          analyzers: ["occlusion", "grad_cam", "integrated_gradients"]
        }
      };
      const res = await startRun(payload);
      setBusy(false);
      onNewRun?.(res.run_id);
    };
    img.src = `data:image/png;base64,${imagePreview}`;
  };

  const onMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!canvasRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const scaleX = canvasRef.current.width / rect.width;
    const scaleY = canvasRef.current.height / rect.height;
    setOrigin({
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY
    });
    setDragging(true);
  };

  const onMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!dragging || !origin || !canvasRef.current) return;
    const bounds = canvasRef.current.getBoundingClientRect();
    const scaleX = canvasRef.current.width / bounds.width;
    const scaleY = canvasRef.current.height / bounds.height;
    const x = (e.clientX - bounds.left) * scaleX;
    const y = (e.clientY - bounds.top) * scaleY;
    const w = x - origin.x;
    const h = y - origin.y;
    setRect({
      x: Math.min(origin.x, x),
      y: Math.min(origin.y, y),
      w: Math.abs(w),
      h: Math.abs(h)
    });
  };

  const onMouseUp = () => {
    setDragging(false);
    setOrigin(null);
  };

  return (
    <section className="panel">
      <h3>Counterfactual Sandbox</h3>
      {taskType !== "image_classification" && (
        <>
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
        </>
      )}
      {taskType === "image_classification" && imagePreview && (
        <>
          <p className="hint">Drag to select an occlusion region, then rerun.</p>
          <canvas
            ref={canvasRef}
            className="occlusion-canvas"
            onMouseDown={onMouseDown}
            onMouseMove={onMouseMove}
            onMouseUp={onMouseUp}
          />
          <div className="row">
            <button className="primary" onClick={occludeAndRerun} disabled={busy || !rect}>
              {busy ? "Running..." : "Occlude and Rerun"}
            </button>
            <button className="secondary" onClick={() => setRect(null)} disabled={busy}>
              Reset
            </button>
          </div>
        </>
      )}
    </section>
  );
};

export default CounterfactualPanel;
