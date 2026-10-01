import React, { useEffect } from "react";
import { Link } from "react-router-dom";
import { getRun, getWsUrl } from "../../api/client";
import { useRunStore } from "../../state/store";

const LogsPanel: React.FC<{ runId?: string | null }> = ({ runId }) => {
  const { logs, addLog, setStatus, progress, stage, setProgress } = useRunStore();
  useEffect(() => {
    if (!runId) return;
    let active = true;
    const socket = new WebSocket(getWsUrl(runId));
    socket.onopen = () => { if (!active) socket.close(); };
    socket.onmessage = (event) => {
      if (!active) return;
      try {
        const data = JSON.parse(event.data);
        if (data.type === "log") addLog(data.message);
        if (data.type === "status") setStatus(data.status);
        if (data.type === "progress") setProgress(data.value ?? 0, data.stage ?? "");
      } catch {
        addLog("Unrecognized log message");
      }
    };
    // Runs can finish before the WebSocket connects; recover terminal state by polling.
    const poll = async () => {
      try {
        const result = await getRun(runId);
        if (!active) return;
        setStatus(result.status);
        if (["completed", "failed"].includes(result.status)) {
          setProgress(result.status === "completed" ? 100 : useRunStore.getState().progress, result.status);
          if (result.error) addLog(result.error);
          clearInterval(timer);
        }
      } catch (error) {
        if (active) addLog(error instanceof Error ? error.message : "Status unavailable");
        clearInterval(timer);
      }
    };
    const timer = setInterval(poll, 1500);
    void poll();
    return () => {
      active = false;
      clearInterval(timer);
      if (socket.readyState === WebSocket.OPEN) socket.close();
    };
  }, [runId, addLog, setStatus, setProgress]);

  return <section className="panel logs">
    <div className="row space-between"><h3>Live Logs</h3>
      {runId && <Link to={`/runs/${runId}`}>Open Run Viewer</Link>}
    </div>
    <div className="progress">
      <div className="progress-track"><div className="progress-fill" style={{ width: `${progress}%` }} /></div>
      <span className="hint">{stage}</span>
    </div>
    <div className="logs-body">
      {logs.length ? logs.map((line, index) => <div key={index}>{line}</div>) : <p>No logs yet</p>}
    </div>
  </section>;
};
export default LogsPanel;
