import React, { useEffect } from "react";
import { getWsUrl } from "../../api/client";
import { useRunStore } from "../../state/store";

const LogsPanel: React.FC<{ runId?: string | null }> = ({ runId }) => {
  const logs = useRunStore((s) => s.logs);
  const addLog = useRunStore((s) => s.addLog);
  const setStatus = useRunStore((s) => s.setStatus);

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
      } catch {
        addLog(event.data);
      }
    };
    return () => ws.close();
  }, [runId, addLog, setStatus]);

  return (
    <section className="panel logs">
      <h3>Live Logs</h3>
      <div className="logs-body">
        {logs.length === 0 ? <p>No logs yet</p> : logs.map((l, i) => <div key={i}>{l}</div>)}
      </div>
    </section>
  );
};

export default LogsPanel;
