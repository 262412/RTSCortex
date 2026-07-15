import { describe, expect, it } from "vitest";

import {
  commandId,
  commandIds,
  consoleReducer,
  createInitialState,
  eventCategory,
  eventMatches,
  normalizeEvent,
  normalizeSession,
} from "./data";
import type { StoredEvent } from "./types";

function storedEvent(eventId: number, eventType = "decision", payload = {}): StoredEvent {
  return { event_id: eventId, event_type: eventType, payload };
}

describe("consoleReducer", () => {
  it("sorts, deduplicates, and caps stored events", () => {
    const initial = createInitialState(2);
    const first = consoleReducer(initial, {
      type: "events",
      events: [storedEvent(2), storedEvent(1), storedEvent(2, "execution")],
    });
    expect(first.events.map((event) => [event.event_id, event.event_type])).toEqual([
      [1, "decision"],
      [2, "execution"],
    ]);

    const second = consoleReducer(first, { type: "event", event: storedEvent(3) });
    expect(second.events.map((event) => event.event_id)).toEqual([2, 3]);
  });

  it("updates only the matching latest frame", () => {
    const state = consoleReducer(createInitialState(), {
      type: "frame",
      frame: { kind: "screen", frame_sequence: 9, game_loop: 120 },
    });
    expect(state.frames.screen?.frame_sequence).toBe(9);
    expect(state.frames.minimap).toBeUndefined();
  });
});

describe("wire normalization", () => {
  it("normalizes the wrapped session snapshot and frame metadata", () => {
    const session = normalizeSession({
      session: {
        run_id: "run-1",
        status: "running",
        scenario: "Simple64",
        protocol_version: "1.1",
      },
      latest_event_id: 4,
      frames: {
        screen: { kind: "screen", frame_sequence: 3, game_loop: 99 },
        minimap: null,
      },
    });
    expect(session.run_id).toBe("run-1");
    expect(session.run_status).toBe("running");
    expect(session.latest_frames?.screen?.frame_sequence).toBe(3);
  });

  it("projects game_loop from an event payload", () => {
    const event = normalizeEvent({
      event_id: 7,
      event_type: "observation",
      step_id: 4,
      payload: { game_loop: 224 },
    });
    expect(event?.game_loop).toBe(224);
    expect(event?.step_id).toBe(4);
  });
});

describe("event projection", () => {
  it("finds command IDs nested in execution payloads", () => {
    const event = storedEvent(1, "execution", { report: { command_id: "cmd-42", action_name: "Attack_Unit" } });
    expect(commandId(event)).toBe("cmd-42");
    expect(eventCategory(event)).toBe("combat");
  });

  it("finds every command ID in nested decision batches", () => {
    const event = storedEvent(2, "decision", {
      batch: {
        commands: [{ command_id: "cmd-1" }, { command_id: "cmd-2" }],
      },
    });

    expect(commandIds(event)).toEqual(["cmd-1", "cmd-2"]);
  });

  it("makes failures dominant and hides observations by default", () => {
    const failed = storedEvent(2, "execution", { action_name: "Build_Pylon", status: "failed" });
    const observation = storedEvent(3, "observation", { game_loop: 10 });
    expect(eventCategory(failed)).toBe("failure");
    expect(eventMatches(observation, "all", false)).toBe(false);
    expect(eventMatches(observation, "all", true)).toBe(true);
  });
});
