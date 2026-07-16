import type {
  ConsoleAction,
  ConsoleSession,
  ConsoleState,
  EventCategory,
  FrameMetadata,
  JsonObject,
  JsonValue,
  StoredEvent,
} from "./types";

export const DEFAULT_EVENT_LIMIT = 5_000;

export function createInitialState(eventLimit = DEFAULT_EVENT_LIMIT): ConsoleState {
  return {
    connection: "connecting",
    session: null,
    events: [],
    frames: {},
    runStatus: "unknown",
    lastHeartbeatAt: null,
    selectedEventId: null,
    eventLimit,
    error: null,
  };
}

function mergeEvents(current: StoredEvent[], incoming: StoredEvent[], limit: number): StoredEvent[] {
  const eventsById = new Map(current.map((event) => [event.event_id, event]));
  for (const event of incoming) {
    eventsById.set(event.event_id, event);
  }
  return [...eventsById.values()]
    .sort((left, right) => left.event_id - right.event_id)
    .slice(-limit);
}

export function consoleReducer(state: ConsoleState, action: ConsoleAction): ConsoleState {
  switch (action.type) {
    case "connection":
      return { ...state, connection: action.status };
    case "session": {
      const eventLimit = action.session.frontend_event_limit ?? state.eventLimit;
      return {
        ...state,
        session: action.session,
        eventLimit,
        runStatus: action.session.run_status ?? state.runStatus,
        frames: { ...state.frames, ...action.session.latest_frames },
        events: state.events.slice(-eventLimit),
      };
    }
    case "events":
      return { ...state, events: mergeEvents(state.events, action.events, state.eventLimit) };
    case "event":
      return { ...state, events: mergeEvents(state.events, [action.event], state.eventLimit) };
    case "frame":
      return { ...state, frames: { ...state.frames, [action.frame.kind]: action.frame } };
    case "run_status":
      return {
        ...state,
        runStatus: action.status,
        session:
          state.session === null
            ? null
            : { ...state.session, run_status: action.status, game_loop: action.gameLoop ?? state.session.game_loop },
      };
    case "heartbeat":
      return { ...state, lastHeartbeatAt: Date.now() };
    case "select_event":
      return { ...state, selectedEventId: action.eventId };
    case "error":
      return { ...state, error: action.error };
  }
}

function asObject(value: JsonValue | undefined): JsonObject | null {
  return typeof value === "object" && value !== null && !Array.isArray(value) ? value : null;
}

export function eventPayload(event: StoredEvent): JsonObject {
  return event.payload;
}

export function readString(payload: JsonObject, ...keys: string[]): string | undefined {
  for (const key of keys) {
    const value = payload[key];
    if (typeof value === "string") return value;
  }
  return undefined;
}

export function readNumber(payload: JsonObject, ...keys: string[]): number | undefined {
  for (const key of keys) {
    const value = payload[key];
    if (typeof value === "number") return value;
  }
  return undefined;
}

export function findLatest(events: StoredEvent[], predicate: (event: StoredEvent) => boolean): StoredEvent | undefined {
  for (let index = events.length - 1; index >= 0; index -= 1) {
    const event = events[index];
    if (event && predicate(event)) return event;
  }
  return undefined;
}

export function moduleName(event: StoredEvent): string | undefined {
  const payload = eventPayload(event);
  return readString(payload, "module", "module_name", "agent_module");
}

export function moduleOutput(event: StoredEvent | undefined): JsonValue | undefined {
  if (!event) return undefined;
  const payload = eventPayload(event);
  return payload.output ?? payload.result ?? payload.module_result ?? payload;
}

export function commandIds(event: StoredEvent): string[] {
  const discovered = new Set<string>();
  const visit = (value: JsonValue): void => {
    if (Array.isArray(value)) {
      value.forEach(visit);
      return;
    }
    if (typeof value !== "object" || value === null) return;
    for (const [key, nested] of Object.entries(value)) {
      if (key === "command_id" && typeof nested === "string") discovered.add(nested);
      visit(nested);
    }
  };
  visit(event.payload);
  return [...discovered];
}

export function commandId(event: StoredEvent): string | undefined {
  const ids = commandIds(event);
  if (ids.length > 0) return ids[0];
  const direct = readString(event.payload, "id");
  if (direct) return direct;
  for (const key of ["command", "execution", "report", "action"]) {
    const nested = asObject(event.payload[key]);
    const id = nested ? readString(nested, "id") : undefined;
    if (id) return id;
  }
  return undefined;
}

export function actionName(event: StoredEvent): string | undefined {
  const payload = eventPayload(event);
  const direct = readString(payload, "action_name", "action");
  if (direct) return direct;
  for (const key of ["command", "execution", "report"]) {
    const nested = asObject(payload[key]);
    const name = nested ? readString(nested, "action_name", "action", "name") : undefined;
    if (name) return name;
  }
  const batch = asObject(payload.batch);
  for (const value of [payload.commands, batch?.commands, payload.planner_candidates, payload.reflex_candidates]) {
    if (!Array.isArray(value)) continue;
    for (const command of value) {
      const nested = asObject(command);
      const name = nested ? readString(nested, "action_name", "action", "name") : undefined;
      if (name) return name;
    }
  }
  return undefined;
}

export function isFailure(event: StoredEvent): boolean {
  const payload = eventPayload(event);
  const status = readString(payload, "status", "result");
  return (
    event.event_type.includes("failed") ||
    event.event_type.includes("error") ||
    ["failed", "rejected", "expired", "unconfirmed"].includes(status ?? "") ||
    typeof payload.failure_code === "string" ||
    typeof payload.failure_reason === "string"
  );
}

export function eventCategory(event: StoredEvent): Exclude<EventCategory, "all"> | "system" {
  if (isFailure(event)) return "failure";
  const payloadText = JSON.stringify(event.payload).toLowerCase();
  const type = event.event_type.toLowerCase();
  const source = readString(event.payload, "source")?.toLowerCase();
  const action = actionName(event)?.toLowerCase();
  if (source === "reflex" || type.includes("reflex")) return "reflex";
  if (type === "goal_progress") return "planner";
  if (
    action?.startsWith("train_") ||
    payloadText.includes('"train_') ||
    payloadText.includes('"effect_kind":"production"') ||
    type.includes("production")
  ) return "production";
  if (action?.includes("build") || payloadText.includes("structure") || type.includes("build")) return "build";
  if (action?.includes("attack") || payloadText.includes("combatgroup") || type.includes("combat")) return "combat";
  if (["planner", "module", "context", "plan"].some((token) => type.includes(token))) return "planner";
  return "system";
}

export function eventMatches(event: StoredEvent, category: EventCategory, showObservations: boolean): boolean {
  if (!showObservations && event.event_type === "observation") return false;
  return category === "all" || eventCategory(event) === category;
}

export function eventSummary(event: StoredEvent): string {
  const payload = eventPayload(event);
  const action = actionName(event);
  const status = readString(payload, "status", "state");
  const code = readString(payload, "failure_code", "reason", "idle_reason");
  const module = moduleName(event);
  if (action) return [action, status, code].filter(Boolean).join(" · ");
  if (module) return [module, status].filter(Boolean).join(" · ");
  return code ?? status ?? commandId(event) ?? "Recorded runtime event";
}

export function normalizeEvent(value: unknown): StoredEvent | null {
  if (typeof value !== "object" || value === null) return null;
  const raw = value as Record<string, unknown>;
  const eventId = raw.event_id;
  if (typeof eventId !== "number") return null;
  const rawPayload = raw.payload ?? raw.data ?? {};
  const payload = typeof rawPayload === "object" && rawPayload !== null && !Array.isArray(rawPayload) ? rawPayload : {};
  return {
    event_id: eventId,
    event_type:
      typeof raw.event_type === "string"
        ? raw.event_type
        : typeof raw.type === "string"
          ? raw.type
          : "unknown",
    created_at: typeof raw.created_at === "string" ? raw.created_at : undefined,
    run_id: typeof raw.run_id === "string" ? raw.run_id : undefined,
    episode_id: typeof raw.episode_id === "string" ? raw.episode_id : undefined,
    step_id: typeof raw.step_id === "string" || typeof raw.step_id === "number" ? raw.step_id : undefined,
    game_loop:
      typeof raw.game_loop === "number"
        ? raw.game_loop
        : typeof (payload as Record<string, unknown>).game_loop === "number"
          ? ((payload as Record<string, unknown>).game_loop as number)
          : undefined,
    payload: payload as JsonObject,
  };
}

export function normalizeSession(value: unknown): ConsoleSession {
  if (typeof value !== "object" || value === null) return {};
  const snapshot = value as Record<string, unknown>;
  const rawSession =
    typeof snapshot.session === "object" && snapshot.session !== null
      ? (snapshot.session as Record<string, unknown>)
      : snapshot;
  const rawFrames =
    typeof snapshot.frames === "object" && snapshot.frames !== null
      ? (snapshot.frames as Record<string, unknown>)
      : undefined;
  const latestFrames: Partial<Record<"screen" | "minimap", FrameMetadata>> = {};
  for (const kind of ["screen", "minimap"] as const) {
    const frame = normalizeFrame(rawFrames?.[kind]);
    if (frame) latestFrames[kind] = frame;
  }
  return {
    ...(rawSession as ConsoleSession),
    run_status:
      typeof rawSession.status === "string"
        ? (rawSession.status as ConsoleSession["run_status"])
        : (rawSession.run_status as ConsoleSession["run_status"]),
    latest_frames: latestFrames,
  };
}

export function normalizeFrame(value: unknown): FrameMetadata | null {
  if (typeof value !== "object" || value === null) return null;
  const frame = value as Record<string, unknown>;
  if ((frame.kind !== "screen" && frame.kind !== "minimap") || typeof frame.frame_sequence !== "number") return null;
  return frame as unknown as FrameMetadata;
}
