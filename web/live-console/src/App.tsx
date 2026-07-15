import { AlertTriangle, LockKeyhole } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";

import { DecisionRail } from "./components/DecisionRail";
import { EventDetailDrawer } from "./components/EventDetailDrawer";
import { EventTimeline } from "./components/EventTimeline";
import { FramePanel } from "./components/FramePanel";
import { StatusHeader } from "./components/StatusHeader";
import { findLatest, moduleName, readNumber } from "./data";
import type { StoredEvent } from "./types";
import { useConsole } from "./useConsole";

const plannerTerminalEvents = new Set([
  "planner_cycle",
  "planner_timeout",
  "planner_error",
  "plan_accepted",
  "module_failed",
]);

function latestModule(events: StoredEvent[], name: string): StoredEvent | undefined {
  return findLatest(
    events,
    (event) => event.event_type === "module_result" && moduleName(event)?.toLowerCase() === name,
  );
}

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
  const plannerStarted = findLatest(events, (event) => event.event_type === "planner_started");
  const plannerTerminal = findLatest(events, (event) => plannerTerminalEvents.has(event.event_type));
  const plannerRunning = Boolean(plannerStarted && (!plannerTerminal || plannerStarted.event_id > plannerTerminal.event_id));
  const plannerElapsedSeconds = useElapsed(plannerStarted, plannerRunning);

  const moduleStarted = findLatest(events, (event) => event.event_type === "module_started");
  const moduleTerminal = findLatest(
    events,
    (event) => event.event_type === "module_result" || event.event_type === "module_failed",
  );
  const moduleRunning = Boolean(moduleStarted && (!moduleTerminal || moduleStarted.event_id > moduleTerminal.event_id));
  const moduleElapsedSeconds = useElapsed(moduleStarted, moduleRunning);

  const projection = useMemo(
    () => ({
      reflection: latestModule(events, "reflection"),
      plan: latestModule(events, "planning"),
      action: latestModule(events, "action"),
      context: findLatest(events, (event) => event.event_type === "context_prepared"),
      decision: findLatest(events, (event) => event.event_type === "decision"),
      execution: findLatest(events, (event) => event.event_type === "execution"),
      observation: findLatest(events, (event) => event.event_type === "observation"),
    }),
    [events],
  );

  const modelTelemetry = useMemo(() => {
    const calls = events.filter(
      (event) => event.event_type === "module_result" && event.payload.model_call === true,
    );
    const totals = calls.reduce(
      (accumulator, event) => {
        const usage = event.payload.usage;
        if (typeof usage !== "object" || usage === null || Array.isArray(usage)) return accumulator;
        accumulator.prompt_tokens += readNumber(usage, "prompt_tokens") ?? 0;
        accumulator.completion_tokens += readNumber(usage, "completion_tokens") ?? 0;
        accumulator.total_tokens += readNumber(usage, "total_tokens") ?? 0;
        return accumulator;
      },
      { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
    );
    const latest = calls.at(-1);
    return {
      retained_request_count: calls.length,
      latest_latency_ms: latest ? Math.round(readNumber(latest.payload, "latency_ms") ?? 0) : 0,
      ...totals,
    };
  }, [events]);

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
          reflection={projection.reflection}
          plan={projection.plan}
          action={projection.action}
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
