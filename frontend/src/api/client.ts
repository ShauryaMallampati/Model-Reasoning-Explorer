const API_BASE = import.meta.env.VITE_API_BASE || "";

export type RunRequest = {
  task_type: "text_lm" | "text_classification" | "image_classification";
  model_id: string;
  input_text?: string;
  input_image_path?: string;
  input_image_base64?: string;
  options?: Record<string, unknown>;
};

export async function listModels() {
  const res = await fetch(`${API_BASE}/api/models`);
  return res.json();
}

export async function startRun(payload: RunRequest) {
  const res = await fetch(`${API_BASE}/api/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  return res.json();
}

export async function getRun(runId: string) {
  const res = await fetch(`${API_BASE}/api/run/${runId}`);
  return res.json();
}

export async function cancelRun(runId: string) {
  const res = await fetch(`${API_BASE}/api/run/${runId}/cancel`, { method: "POST" });
  return res.json();
}

export async function compareRuns(runA: string, runB: string) {
  const res = await fetch(`${API_BASE}/api/compare`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ run_a: runA, run_b: runB })
  });
  return res.json();
}

export async function startDatasetRun(formData: FormData) {
  const res = await fetch(`${API_BASE}/api/dataset/run`, {
    method: "POST",
    body: formData
  });
  return res.json();
}

export async function getDatasetRun(datasetRunId: string) {
  const res = await fetch(`${API_BASE}/api/dataset/${datasetRunId}`);
  return res.json();
}

export function getArtifactUrl(runId: string, name: string) {
  return `${API_BASE}/api/run/${runId}/artifact/${name}`;
}

export function getWsUrl(runId: string) {
  const base = API_BASE.replace(/^http/, "ws");
  return `${base}/ws/runs/${runId}`;
}

export async function exportRun(runId: string) {
  const res = await fetch(`${API_BASE}/api/run/${runId}/export`, { method: "POST" });
  return res.json();
}
