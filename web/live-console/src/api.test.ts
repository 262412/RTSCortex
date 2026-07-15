import { afterEach, describe, expect, it, vi } from "vitest";

import { fetchEvents, frameUrl, normalizeMessage } from "./api";

afterEach(() => vi.unstubAllGlobals());

describe("console API wire messages", () => {
  it("normalizes stored events", () => {
    const message = normalizeMessage({
      type: "stored_event",
      event: { event_id: 10, event_type: "planner_started", payload: { game_loop: 100 } },
    });
    expect(message?.type).toBe("stored_event");
    if (message?.type === "stored_event") expect(message.event.game_loop).toBe(100);
  });

  it("normalizes run status session snapshots", () => {
    const message = normalizeMessage({
      type: "run_status",
      session: { run_id: "run-1", status: "completed", protocol_version: "1.1" },
    });
    expect(message?.type).toBe("run_status");
    if (message?.type === "run_status") expect(message.session.run_status).toBe("completed");
  });

  it("normalizes frame messages and cache-busting URLs", () => {
    const message = normalizeMessage({
      type: "frame_available",
      frame: { kind: "minimap", frame_sequence: 11, game_loop: 256 },
    });
    expect(message?.type).toBe("frame_available");
    expect(frameUrl("minimap", 11)).toBe("/console/api/v1/frames/minimap?frame_sequence=11");
  });

  it("rejects malformed messages", () => {
    expect(normalizeMessage({ type: "frame_available", frame: { kind: "screen" } })).toBeNull();
    expect(normalizeMessage({ type: "unknown" })).toBeNull();
  });

  it("follows paginated backfill while retaining only the configured event cap", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            events: [
              { event_id: 1, event_type: "decision", payload: {} },
              { event_id: 2, event_type: "decision", payload: {} },
            ],
            next_after_event_id: 2,
            has_more: true,
          }),
          { status: 200, headers: { "content-type": "application/json" } },
        ),
      )
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            events: [
              { event_id: 3, event_type: "execution", payload: {} },
              { event_id: 4, event_type: "execution", payload: {} },
            ],
            next_after_event_id: 4,
            has_more: false,
          }),
          { status: 200, headers: { "content-type": "application/json" } },
        ),
      );
    vi.stubGlobal("fetch", fetchMock);

    const events = await fetchEvents(0, 2);
    expect(events.map((event) => event.event_id)).toEqual([3, 4]);
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(String(fetchMock.mock.calls[1]?.[0])).toContain("after_event_id=2");
  });
});
