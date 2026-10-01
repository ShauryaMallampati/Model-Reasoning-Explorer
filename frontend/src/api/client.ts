const API_BASE = (import.meta.env.VITE_API_BASE || "").replace(/\/$/, "");

export type TaskType = "text_lm" | "text_classification" | "image_classification";
export type RunRequest = {
  task_type: TaskType;
  model_id: string;
  input_text?: string;
  input_image_path?: string;
  input_image_base64?: string;
  options?: Record<string, unknown>;
};
export type ModelOption = { id: string; task_type: TaskType };
export type RunOutputs = {
  prediction?: string;
  generated_text?: string;
  top_k?: { label?: string; token?: string; prob: number }[];
};
export type ComparisonSummary = {
  prediction_a?: string;
  prediction_b?: string;
  generated_a?: string;
  generated_b?: string;
  layer_similarity?: { layers: string[]; values: number[] };
  attention_delta?: { layers: string[]; values: number[] };
  attribution_diff?: { type: "text" | "image"; preview: number[]; tokens?: string[] };
  notes?: string[];
};

async function requestJson<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, options);
  let data;
  try {
    data = await response.json();
  } catch {
    throw new Error(`Server returned an unreadable response (${response.status})`);
  }
  if (!response.ok) {
    const detail = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail);
    throw new Error(detail || `Request failed (${response.status})`);
  }
  return data as T;
}

export const listModels = () => requestJson<{ models: ModelOption[] }>("/api/models");
export const startRun = (payload: RunRequest) => requestJson<{ run_id: string }>("/api/run", {
  method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload)
});
export const getRun = (runId: string) => requestJson<Record<string, any>>(
  `/api/run/${encodeURIComponent(runId)}`
);
export const cancelRun = (runId: string) => requestJson<{ cancelled: boolean }>(
  `/api/run/${encodeURIComponent(runId)}/cancel`, { method: "POST" }
);
export const compareRuns = (runA: string, runB: string) => requestJson<{ summary: ComparisonSummary }>(
  "/api/compare", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ run_a: runA, run_b: runB })
  }
);
export const startDatasetRun = (body: FormData) => requestJson<{ dataset_run_id: string }>(
  "/api/dataset/run", { method: "POST", body }
);
export const getDatasetRun = (id: string) => requestJson<Record<string, any>>(
  `/api/dataset/${encodeURIComponent(id)}`
);
export const getArtifactUrl = (runId: string, name: string) =>
  `${API_BASE}/api/run/${encodeURIComponent(runId)}/artifact/${encodeURIComponent(name)}`;
export function getWsUrl(runId: string) {
  const url = new URL(`${API_BASE}/ws/runs/${encodeURIComponent(runId)}`, window.location.origin);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  return url.toString();
}
export const exportRun = (runId: string) => requestJson<{ report_url: string }>(
  `/api/run/${encodeURIComponent(runId)}/export`, { method: "POST" }
);
export const resourceUrl = (path: string) => `${API_BASE}${path}`;
