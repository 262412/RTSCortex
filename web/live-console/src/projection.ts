import { findLatest, moduleName, moduleOutput, readNumber } from "./data";
import type { JsonObject, JsonValue, StoredEvent } from "./types";

const PLANNER_TERMINAL_EVENTS = new Set([
  "planner_cycle",
  "planner_timeout",
  "planner_error",
  "plan_accepted",
  "module_failed",
  "macro_plan_accepted",
  "macro_plan_rejected",
  "specialist_failed",
]);

function asObject(value: JsonValue | undefined): JsonObject | undefined {
  return typeof value === "object" && value !== null && !Array.isArray(value) ? value : undefined;
}

function latestModule(events: StoredEvent[], name: string): StoredEvent | undefined {
  return findLatest(
    events,
    (event) => event.event_type === "module_result" && moduleName(event)?.toLowerCase() === name,
  );
}

function legacyProposedActions(
  planning: StoredEvent | undefined,
  action: StoredEvent | undefined,
): JsonValue | undefined {
  const planningOutput = asObject(moduleOutput(planning));
  const plan = asObject(planningOutput?.plan);
  if (plan?.proposed_actions !== undefined) return plan.proposed_actions;
  return asObject(moduleOutput(action))?.commands;
}

export function plannerProjection(events: StoredEvent[]) {
  const started = findLatest(events, (event) => event.event_type === "planner_started");
  const terminal = findLatest(events, (event) => PLANNER_TERMINAL_EVENTS.has(event.event_type));
  return {
    started,
    running: Boolean(started && (!terminal || started.event_id > terminal.event_id)),
  };
}

export function decisionRailProjection(events: StoredEvent[]) {
  const currentPlanner = findLatest(events, (event) => event.event_type === "planner_started");
  const cycleEvents = currentPlanner
    ? events.filter((event) => event.event_id >= currentPlanner.event_id)
    : events;
  const planning = latestModule(cycleEvents, "planning");
  const action = latestModule(cycleEvents, "action");
  const cortexCandidates = findLatest(
    cycleEvents,
    (event) => event.event_type === "candidate_set_built",
  );
  return {
    situation: findLatest(events, (event) => event.event_type === "situation_assessed"),
    reflection: latestModule(cycleEvents, "reflection"),
    goalProgress: findLatest(events, (event) => event.event_type === "goal_progress"),
    plan:
      findLatest(cycleEvents, (event) => event.event_type === "macro_plan_accepted")
      ?? findLatest(cycleEvents, (event) => event.event_type === "plan_accepted")
      ?? planning,
    candidateActions:
      cortexCandidates?.payload.candidates ?? legacyProposedActions(planning, action),
    context: findLatest(cycleEvents, (event) => event.event_type === "context_prepared"),
    decision: findLatest(cycleEvents, (event) => event.event_type === "decision"),
    execution: findLatest(events, (event) => event.event_type === "execution"),
    observation: findLatest(events, (event) => event.event_type === "observation"),
  };
}

function generationUsage(event: StoredEvent): JsonObject | undefined {
  const direct = asObject(event.payload.generation_metadata);
  if (direct) return direct;
  const plan = asObject(event.payload.plan);
  const rawProposal = asObject(plan?.raw_proposal);
  const proposal = asObject(rawProposal?.proposal);
  return asObject(proposal?.generation_metadata);
}

export function modelTelemetryProjection(events: StoredEvent[]): JsonObject {
  const legacyCalls = events.filter(
    (event) => event.event_type === "module_result" && event.payload.model_call === true,
  );
  const specialistCalls = events.filter((event) =>
    event.event_type === "macro_plan_accepted" || event.event_type === "macro_plan_rejected"
  );
  let promptTokens = 0;
  let completionTokens = 0;
  for (const event of legacyCalls) {
    const usage = asObject(event.payload.usage);
    promptTokens += readNumber(usage ?? {}, "prompt_tokens") ?? 0;
    completionTokens += readNumber(usage ?? {}, "completion_tokens") ?? 0;
  }
  for (const event of specialistCalls) {
    const usage = generationUsage(event);
    promptTokens += readNumber(usage ?? {}, "prompt_token_count") ?? 0;
    completionTokens += readNumber(usage ?? {}, "completion_token_count") ?? 0;
  }
  const calls = [...legacyCalls, ...specialistCalls].sort(
    (left, right) => left.event_id - right.event_id,
  );
  const latest = calls.at(-1);
  return {
    retained_request_count: calls.length,
    specialist_request_count: specialistCalls.length,
    latest_latency_ms: latest ? Math.round(readNumber(latest.payload, "latency_ms") ?? 0) : 0,
    prompt_tokens: promptTokens,
    completion_tokens: completionTokens,
    total_tokens: promptTokens + completionTokens,
  };
}
