import { create } from "zustand";

export type RunState = {
  runId: string | null;
  logs: string[];
  status: string;
  setRunId: (id: string | null) => void;
  addLog: (log: string) => void;
  setStatus: (status: string) => void;
  clear: () => void;
};

export const useRunStore = create<RunState>((set) => ({
  runId: null,
  logs: [],
  status: "idle",
  setRunId: (runId) => set({ runId }),
  addLog: (log) => set((state) => ({ logs: [...state.logs, log] })),
  setStatus: (status) => set({ status }),
  clear: () => set({ runId: null, logs: [], status: "idle" })
}));
