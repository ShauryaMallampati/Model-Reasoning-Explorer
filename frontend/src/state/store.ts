import { create } from "zustand";

export type RunState = {
  runId: string | null;
  logs: string[];
  status: string;
  progress: number;
  stage: string;
  setRunId: (id: string | null) => void;
  addLog: (log: string) => void;
  setStatus: (status: string) => void;
  setProgress: (progress: number, stage: string) => void;
  clear: () => void;
};

export const useRunStore = create<RunState>((set) => ({
  runId: null,
  logs: [],
  status: "idle",
  progress: 0,
  stage: "",
  setRunId: (runId) => set({ runId }),
  addLog: (log) => set((state) => ({ logs: [...state.logs, log] })),
  setStatus: (status) => set({ status }),
  setProgress: (progress, stage) => set({ progress, stage }),
  clear: () => set({ runId: null, logs: [], status: "idle", progress: 0, stage: "" })
}));
