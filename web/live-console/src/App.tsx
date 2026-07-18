import { AlertTriangle, LockKeyhole } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";

import { DecisionRail } from "./components/DecisionRail";
import { EventDetailDrawer } from "./components/EventDetailDrawer";
import { EventTimeline } from "./components/EventTimeline";
import { FramePanel } from "./components/FramePanel";
import { StatusHeader } from "./components/StatusHeader";
import { findLatest, moduleName } from "./data";
import {
  decisionRailProjection,
  modelTelemetryProjection,
  plannerProjection,
} from "./projection";
import type { StoredEvent } from "./types";
import { useConsole } from "./useConsole";

function useElapsed(started: StoredEvent | undefined, running: boolean): number | undefined {
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  useEffect(() => {
    if (!running) return;
    const startedAt = started?.created_at ? Date.parse(started.created_at) : Number.NaN;
    const timer = window.setInterval(() => {
      setElapsedSeconds((previous) =>
        Number.isFinite(startedAt) ? Math.max(0, Math.floor((Date.now() - startedAt) / 1_000)) : previous + 1,
      );
    }, 1_000);
    return () => window.clearInterval(timer);
  }, [running, started?.created_at, started?.event_id]);
  return running ? elapsedSeconds : undefined;
}

export function App() {
  const { state, dispatch } = useConsole();
  const events = state.events;
  const { started: plannerStarted, running: plannerRunning } = useMemo(
    () => plannerProjection(events),
    [events],
  );
  const plannerElapsedSeconds = useElapsed(plannerStarted, plannerRunning);

  const moduleStarted = findLatest(events, (event) => event.event_type === "module_started");
  const moduleTerminal = findLatest(
    events,
    (event) => event.event_type === "module_result" || event.event_type === "module_failed",
  );
  const moduleRunning = Boolean(moduleStarted && (!moduleTerminal || moduleStarted.event_id > moduleTerminal.event_id));
  const moduleElapsedSeconds = useElapsed(moduleStarted, moduleRunning);

  const projection = useMemo(() => decisionRailProjection(events), [events]);
  const modelTelemetry = useMemo(() => modelTelemetryProjection(events), [events]);

  const selectedEvent =
    state.selectedEventId === null ? null : events.find((event) => event.event_id === state.selectedEventId) ?? null;
  const closeDrawer = useCallback(() => dispatch({ type: "select_event", eventId: null }), [dispatch]);
  const gameLoop = Math.max(
    state.session?.game_loop ?? 0,
    projection.observation?.game_loop ?? 0,
    state.frames.screen?.game_loop ?? 0,
  );

  return (
    <div className="app-shell">
      <StatusHeader
        connection={state.connection}
        session={state.session}
        runStatus={state.runStatus}
        gameLoop={gameLoop || undefined}
        plannerRunning={plannerRunning}
        plannerElapsedSeconds={plannerElapsedSeconds}
      />

      {state.error && (
        <div className="error-banner" role="status">
          <AlertTriangle size={16} aria-hidden="true" />
          <span><strong>Console channel issue:</strong> {state.error}</span>
          <button type="button" onClick={() => dispatch({ type: "error", error: null })}>Dismiss</button>
        </div>
      )}

      <main className="workspace-grid">
        <FramePanel
          screen={state.frames.screen}
          minimap={state.frames.minimap}
          connection={state.connection}
          historical={state.runStatus === "historical"}
          staleAfterSeconds={state.session?.stale_after_seconds ?? 2}
        />
        <DecisionRail
          raceProfile={projection.raceProfile}
          situation={projection.situation}
          tacticalShadow={projection.tacticalShadow}
          roleIntent={projection.roleIntent}
          arbitration={projection.arbitration}
          playbookRule={projection.playbookRule}
          reflection={projection.reflection}
          goalProgress={projection.goalProgress}
          plan={projection.plan}
          candidateActions={projection.candidateActions}
          context={projection.context}
          decision={projection.decision}
          execution={projection.execution}
          plannerRunning={plannerRunning}
          activeModule={moduleRunning && moduleStarted ? moduleName(moduleStarted) : undefined}
          moduleElapsedSeconds={moduleElapsedSeconds}
          modelTelemetry={modelTelemetry}
        />
      </main>

      <EventTimeline
        events={events}
        onSelect={(eventId) => dispatch({ type: "select_event", eventId })}
      />

      <footer className="console-footer">
        <span><LockKeyhole size={13} aria-hidden="true" /> Read-only console</span>
        <span>Protocol {state.session?.protocol_version ?? "1.1"}</span>
        <span className="mono">{events.length.toLocaleString()} / {state.eventLimit.toLocaleString()} events retained</span>
      </footer>

      <EventDetailDrawer
        event={selectedEvent}
        events={events}
        onClose={closeDrawer}
        onSelect={(eventId) => dispatch({ type: "select_event", eventId })}
      />
    </div>
  );
}
