import { Activity, BrainCircuit, Clock3, Radio, Wifi, WifiOff } from "lucide-react";

import type { ConnectionStatus, ConsoleSession, RunStatus } from "../types";

interface StatusHeaderProps {
  connection: ConnectionStatus;
  session: ConsoleSession | null;
  runStatus: RunStatus;
  gameLoop?: number;
  plannerRunning: boolean;
  plannerElapsedSeconds?: number;
}

const connectionLabels: Record<ConnectionStatus, string> = {
  connecting: "Connecting",
  live: "Live",
  reconnecting: "Reconnecting",
  disconnected: "Disconnected",
};

function StatusItem({ label, value, mono = false }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="status-item">
      <span className="status-label">{label}</span>
      <span className={mono ? "status-value mono" : "status-value"}>{value}</span>
    </div>
  );
}

export function StatusHeader({
  connection,
  session,
  runStatus,
  gameLoop,
  plannerRunning,
  plannerElapsedSeconds,
}: StatusHeaderProps) {
  const ConnectionIcon = connection === "live" ? Wifi : WifiOff;
  return (
    <header className="status-header">
      <div className="brand-block">
        <div className="brand-mark" aria-hidden="true">
          <Activity size={20} />
        </div>
        <div>
          <p className="eyebrow">Agent observability</p>
          <h1>RTSCortex</h1>
        </div>
      </div>

      <div className="connection-block" aria-live="polite">
        <span className={`connection-pill connection-${connection}`}>
          <ConnectionIcon size={14} aria-hidden="true" />
          {connectionLabels[connection]}
        </span>
        <span className="run-state">
          <Radio size={13} aria-hidden="true" />
          {runStatus}
        </span>
      </div>

      <div className="status-grid">
        <StatusItem label="Run" value={session?.run_id ?? "Waiting"} mono />
        <StatusItem label="Map / seed" value={`${session?.map_name ?? session?.scenario ?? "—"} / ${session?.seed ?? "—"}`} />
        <StatusItem label="Game loop" value={gameLoop?.toLocaleString() ?? "—"} mono />
        <StatusItem label="Model" value={session?.model ?? "—"} />
        <div className="status-item planner-status">
          <span className="status-label">
            <BrainCircuit size={12} aria-hidden="true" /> Planner
          </span>
          <span className={plannerRunning ? "status-value planner-active" : "status-value"}>
            {plannerRunning ? (
              <>
                <Clock3 size={13} aria-hidden="true" /> Running {plannerElapsedSeconds ?? 0}s
              </>
            ) : (
              "Idle"
            )}
          </span>
        </div>
      </div>
    </header>
  );
}
