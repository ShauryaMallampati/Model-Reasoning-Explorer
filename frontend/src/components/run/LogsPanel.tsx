import React, { useEffect } from "react";
import { getWsUrl } from "../../api/client";
import { useRunStore } from "../../state/store";

const LogsPanel: React.FC<{ runId?: string | null }> = ({ runId }) => {
  const logs = useRunStore((s) => s.logs);
  const addLog = useRunStore((s) => s.addLog);
  const setStatus = useRunStore((s) => s.setStatus);
  const progress = useRunStore((s) => s.progress);
  const stage = useRunStore((s) => s.stage);
  const setProgress = useRunStore((s) => s.setProgress);

  useEffect(() => {
    if (!runId) return;
    const ws = new WebSocket(getWsUrl(runId));
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === "log") {
          addLog(data.message);
        }
        if (data.type === "status") {
          setStatus(data.status);
        }
        if (data.type === "progress") {
          setProgress(data.value ?? 0, data.stage ?? "");
        }
      } catch {
        addLog(event.data);
      }
    };
    return () => ws.close();
  }, [runId, addLog, setStatus]);

  return (
    <section className="panel logs">
      <h3>Live Logs</h3>
      <div className="progress">
        <div className="progress-track">
          <div className="progress-fill" style={{ width: `${progress}%` }} />
        </div>
        <span className="hint">{stage}</span>
      </div>
      <div className="logs-body">
        {logs.length === 0 ? <p>No logs yet</p> : logs.map((l, i) => <div key={i}>{l}</div>)}
      </div>
    </section>
  );
};

export default LogsPanel;
