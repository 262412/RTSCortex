export type JsonPrimitive = string | number | boolean | null;
export type JsonValue = JsonPrimitive | JsonValue[] | { [key: string]: JsonValue };
export type JsonObject = { [key: string]: JsonValue };

export type ConnectionStatus = "connecting" | "live" | "reconnecting" | "disconnected";
export type RunStatus = "starting" | "running" | "completed" | "failed" | "historical" | "unknown";
export type FrameKind = "screen" | "minimap";

export interface StoredEvent {
  event_id: number;
  event_type: string;
  created_at?: string;
  run_id?: string;
  episode_id?: string;
  step_id?: string | number;
  game_loop?: number;
  payload: JsonObject;
}

export interface FrameMetadata {
  kind: FrameKind;
  frame_sequence: number;
  game_loop?: number;
  captured_at?: string;
  width?: number;
  height?: number;
}

export interface ConsoleSession {
  run_id?: string;
  episode_id?: string;
  scenario?: string;
  map_name?: string;
  seed?: number;
  model?: string;
  provider?: string;
  run_status?: RunStatus;
  game_loop?: number;
  protocol_version?: string;
  frontend_event_limit?: number;
  stale_after_seconds?: number;
  latest_frames?: Partial<Record<FrameKind, FrameMetadata>>;
}

export type ConsoleMessage =
  | { type: "stored_event"; event: StoredEvent }
  | { type: "frame_available"; frame: FrameMetadata }
  | { type: "run_status"; session: ConsoleSession }
  | { type: "heartbeat"; sent_at?: string; latest_event_id?: number }
  | { type: "resync_required"; after_event_id?: number };

export interface ConsoleState {
  connection: ConnectionStatus;
  session: ConsoleSession | null;
  events: StoredEvent[];
  frames: Partial<Record<FrameKind, FrameMetadata>>;
  runStatus: RunStatus;
  lastHeartbeatAt: number | null;
  selectedEventId: number | null;
  eventLimit: number;
  error: string | null;
}

export type ConsoleAction =
  | { type: "connection"; status: ConnectionStatus }
  | { type: "session"; session: ConsoleSession }
  | { type: "events"; events: StoredEvent[] }
  | { type: "event"; event: StoredEvent }
  | { type: "frame"; frame: FrameMetadata }
  | { type: "run_status"; status: RunStatus; gameLoop?: number }
  | { type: "heartbeat" }
  | { type: "select_event"; eventId: number | null }
  | { type: "error"; error: string | null };

export type EventCategory = "all" | "planner" | "reflex" | "build" | "combat" | "failure";
