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
});
