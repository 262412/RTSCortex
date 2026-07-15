import { normalizeEvent, normalizeFrame, normalizeSession } from "./data";
import type { ConsoleMessage, ConsoleSession, StoredEvent } from "./types";

const API_ROOT = "/console/api/v1";

async function readJson(response: Response): Promise<unknown> {
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return response.json() as Promise<unknown>;
}

export async function fetchSession(signal?: AbortSignal): Promise<ConsoleSession> {
  return normalizeSession(await readJson(await fetch(`${API_ROOT}/session`, { signal })));
}

export async function fetchEvents(afterEventId: number, limit = 5_000, signal?: AbortSignal): Promise<StoredEvent[]> {
  let cursor = afterEventId;
  let hasMore = true;
  let retained: StoredEvent[] = [];
  while (hasMore) {
    const params = new URLSearchParams({ after_event_id: String(cursor), limit: String(limit) });
    const response = await readJson(await fetch(`${API_ROOT}/events?${params}`, { signal }));
    const page = typeof response === "object" && response !== null ? (response as Record<string, unknown>) : {};
    const rawEvents = Array.isArray(response) ? response : Array.isArray(page.events) ? page.events : [];
    const events = rawEvents.map(normalizeEvent).filter((event): event is StoredEvent => event !== null);
    retained = [...retained, ...events].slice(-limit);
    const nextCursor = typeof page.next_after_event_id === "number" ? page.next_after_event_id : events.at(-1)?.event_id;
    hasMore = page.has_more === true && nextCursor !== undefined && nextCursor > cursor;
    if (nextCursor !== undefined) cursor = nextCursor;
  }
  return retained;
}

export function frameUrl(kind: "screen" | "minimap", sequence?: number): string {
  const suffix = sequence === undefined ? "" : `?frame_sequence=${encodeURIComponent(sequence)}`;
  return `${API_ROOT}/frames/${kind}${suffix}`;
}

export function streamUrl(afterEventId: number): string {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}${API_ROOT}/stream?after_event_id=${afterEventId}`;
}

export function normalizeMessage(value: unknown): ConsoleMessage | null {
  if (typeof value !== "object" || value === null || !("type" in value)) return null;
  const raw = value as Record<string, unknown>;
  if (raw.type === "stored_event") {
    const event = normalizeEvent(raw.event);
    return event ? { type: "stored_event", event } : null;
  }
  if (raw.type === "frame_available") {
    const frame = normalizeFrame(raw.frame ?? raw);
    return frame ? { type: "frame_available", frame } : null;
  }
  if (raw.type === "run_status") {
    return { type: "run_status", session: normalizeSession(raw.session) };
  }
  if (raw.type === "heartbeat") {
    return {
      type: "heartbeat",
      sent_at: typeof raw.sent_at === "string" ? raw.sent_at : undefined,
      latest_event_id: typeof raw.latest_event_id === "number" ? raw.latest_event_id : undefined,
    };
  }
  if (raw.type === "resync_required") {
    return {
      type: "resync_required",
      after_event_id: typeof raw.after_event_id === "number" ? raw.after_event_id : undefined,
    };
  }
  return null;
}

interface StreamCallbacks {
  getAfterEventId: () => number;
  onMessage: (message: ConsoleMessage) => void;
  onStatus: (status: "connecting" | "live" | "reconnecting" | "disconnected") => void;
  onBackfill: (events: StoredEvent[]) => void;
  onError: (message: string) => void;
}

export class ConsoleStream {
  private socket: WebSocket | null = null;
  private stopped = false;
  private reconnectTimer: number | null = null;
  private reconnectAttempts = 0;

  constructor(private readonly callbacks: StreamCallbacks) {}

  async start(): Promise<void> {
    this.stopped = false;
    await this.connect(false);
  }

  stop(): void {
    this.stopped = true;
    if (this.reconnectTimer !== null) window.clearTimeout(this.reconnectTimer);
    this.socket?.close();
    this.socket = null;
    this.callbacks.onStatus("disconnected");
  }

  private async connect(reconnecting: boolean): Promise<void> {
    if (this.stopped) return;
    this.callbacks.onStatus(reconnecting ? "reconnecting" : "connecting");

    try {
      if (reconnecting) {
        const events = await fetchEvents(this.callbacks.getAfterEventId());
        this.callbacks.onBackfill(events);
      }
    } catch (error) {
      this.callbacks.onError(error instanceof Error ? error.message : "Event backfill failed");
    }

    if (this.stopped) return;
    const socket = new WebSocket(streamUrl(this.callbacks.getAfterEventId()));
    this.socket = socket;
    socket.addEventListener("open", () => {
      this.reconnectAttempts = 0;
      this.callbacks.onStatus("live");
    });
    socket.addEventListener("message", (event) => {
      try {
        const message = normalizeMessage(JSON.parse(String(event.data)));
        if (message) this.callbacks.onMessage(message);
      } catch {
        this.callbacks.onError("The console received an invalid stream message.");
      }
    });
    socket.addEventListener("close", () => this.scheduleReconnect());
    socket.addEventListener("error", () => socket.close());
  }

  private scheduleReconnect(): void {
    if (this.stopped || this.reconnectTimer !== null) return;
    this.callbacks.onStatus("reconnecting");
    const delay = Math.min(10_000, 500 * 2 ** this.reconnectAttempts);
    this.reconnectAttempts += 1;
    this.reconnectTimer = window.setTimeout(() => {
      this.reconnectTimer = null;
      void this.connect(true);
    }, delay);
  }
}
