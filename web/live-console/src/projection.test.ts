import { describe, expect, it } from "vitest";

import {
  decisionRailProjection,
  modelTelemetryProjection,
  plannerProjection,
} from "./projection";
import type { JsonObject, StoredEvent } from "./types";

function event(eventId: number, eventType: string, payload: JsonObject): StoredEvent {
  return { event_id: eventId, event_type: eventType, payload };
}

describe("Cortex decision-rail projection", () => {
  it("terminates planner timing and exposes the HIMA plan and candidates", () => {
    const events = [
      event(1, "planner_started", { runtime_kind: "cortex" }),
      event(2, "situation_assessed", { phase: "early" }),
      event(3, "macro_plan_accepted", {
        latency_ms: 125,
        plan: {
          plan_id: "plan-1",
          raw_proposal: {
            proposal: {
              generation_metadata: {
                prompt_token_count: 42,
                completion_token_count: 18,
              },
            },
          },
        },
      }),
      event(4, "candidate_set_built", {
        candidates: [{ candidate_id: "candidate-1", action_name: "Build_Pylon_Screen" }],
      }),
    ];

    expect(plannerProjection(events).running).toBe(false);
    const projection = decisionRailProjection(events);
    expect(projection.situation?.event_id).toBe(2);
    expect(projection.plan?.event_id).toBe(3);
    expect(projection.candidateActions).toEqual([
      { candidate_id: "candidate-1", action_name: "Build_Pylon_Screen" },
    ]);
    expect(modelTelemetryProjection(events)).toEqual({
      retained_request_count: 1,
      specialist_request_count: 1,
      latest_latency_ms: 125,
      prompt_tokens: 42,
      completion_tokens: 18,
      total_tokens: 60,
    });
  });

  it("treats a specialist failure as a terminal planner event", () => {
    const events = [
      event(1, "planner_started", {}),
      event(2, "specialist_failed", { message: "timeout" }),
      event(3, "macro_plan_rejected", { latency_ms: 12 }),
    ];

    expect(plannerProjection(events).running).toBe(false);
    expect(modelTelemetryProjection(events).retained_request_count).toBe(1);
  });
});
