import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import type { StoredEvent } from "../types";
import { DecisionRail } from "./DecisionRail";

function goalProgressEvent(): StoredEvent {
  return {
    event_id: 23,
    event_type: "goal_progress",
    game_loop: 512,
    payload: {
      run_id: "run-1",
      episode_id: "episode-1",
      step_id: 14,
      game_loop: 512,
      goal_id: "protoss-opening",
      strategic_goal: "建立基础科技",
      status: "blocked",
      achieved: [],
      missing: [
        {
          requirement_id: "gateway",
          kind: "structure",
          target: "Gateway",
          required_count: 1,
          current_count: 0,
          in_progress_count: 0,
        },
      ],
      blockers: [
        {
          requirement_id: "gateway",
          kind: "insufficient_minerals",
          detail: "Gateway requires 150 minerals",
          action_name: "Build_Gateway_Screen",
        },
      ],
      advancing_actions: [],
      unique_next_action: null,
      defensive_hold_required: false,
    },
  };
}

describe("DecisionRail", () => {
  it("shows the latest goal progress as readable reflection evidence", () => {
    const markup = renderToStaticMarkup(
      <DecisionRail
        goalProgress={goalProgressEvent()}
        plannerRunning={false}
        modelTelemetry={{ retained_request_count: 0 }}
      />,
    );

    expect(markup).toContain("目标进度检查（复盘依据）");
    expect(markup).toContain("暂时受阻（blocked）");
    expect(markup).toContain("尚缺少");
    expect(markup).toContain("晶体矿不足");
    expect(markup).not.toContain("等待确定性目标进度检查");
  });

  it("shows strategic cortex cards without requiring raw JSON", () => {
    const markup = renderToStaticMarkup(
      <DecisionRail
        raceProfile={{
          event_id: 1,
          event_type: "race_profile_activated",
          payload: { race: "protoss", live_worker_ready: true },
        }}
        roleIntent={{
          event_id: 2,
          event_type: "role_intent_emitted",
          payload: { intent: { role: "defense", action_names: ["Move_Minimap"] } },
        }}
        arbitration={{
          event_id: 3,
          event_type: "intent_arbitrated",
          payload: { mode: "shadow", arbitration: { decisions: [] } },
        }}
        playbookRule={{
          event_id: 4,
          event_type: "playbook_rule_applied",
          payload: { rule_id: "rule-1", reason: "matched" },
        }}
        plannerRunning={false}
        modelTelemetry={{ retained_request_count: 0 }}
      />,
    );

    expect(markup).toContain("当前种族与能力");
    expect(markup).toContain("职责 Agent");
    expect(markup).toContain("战略意图仲裁");
    expect(markup).toContain("CortexPlaybook 约束");
    expect(markup).toContain("rule-1");
  });
});
