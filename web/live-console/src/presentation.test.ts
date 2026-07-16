import { describe, expect, it } from "vitest";

import { actionName } from "./data";
import {
  actionLabel,
  eventSemanticPayload,
  eventSummary,
  eventTitle,
  fieldLabel,
  moduleSemanticOutput,
  semanticScalar,
} from "./presentation";
import type { JsonObject, StoredEvent } from "./types";

function event(eventType: string, payload: JsonObject, eventId = 1): StoredEvent {
  return { event_id: eventId, event_type: eventType, payload };
}

describe("Chinese event presentation", () => {
  it("summarizes the planner lifecycle without exposing JSON syntax", () => {
    const started = event("planner_started", { started_game_loop: 24 });
    const context = event("context_prepared", {
      module: "planning",
      original_chars: 8_200,
      final_chars: 3_900,
      statistics: { retained_observations: 4 },
    });
    const result = event("module_result", {
      module: "planning",
      latency_ms: 1_260,
      model_call: true,
      output: { plan: { strategic_goal: "Establish Protoss production" } },
      usage: { total_tokens: 1_110 },
    });

    expect(eventTitle(started)).toBe("规划器开始思考");
    expect(eventSummary(started)).toBe("开始生成下一轮战略计划 · loop 24");
    expect(eventSummary(context)).toBe("战略规划上下文：8,200 → 3,900 字符 · 保留 4 条观察");
    expect(eventTitle(result)).toBe("战略规划完成");
    expect(eventSummary(result)).toContain("1.26 秒 · 1,110 tokens · Establish Protoss production");
  });

  it("explains decision, lifecycle, execution, and failure stages", () => {
    const decision = event("decision", {
      planner_candidates: [{ name: "Build_Pylon_Screen" }],
      batch: {
        commands: [
          {
            command_id: "cmd-pylon-1",
            name: "Build_Pylon_Screen",
            actor: "Builder/Builder-Probe-1",
          },
        ],
        rejected_commands: [],
      },
    });
    const lifecycle = event("command_lifecycle", {
      command: { command_id: "cmd-pylon-1", name: "Build_Pylon_Screen" },
      status: "dispatched",
    });
    const failed = event("execution", {
      command_id: "cmd-pylon-1",
      action_name: "Build_Pylon_Screen",
      status: "failed",
      execution_stage: "effect_verification",
      failure_code: "target_not_created",
    });

    expect(actionName(decision)).toBe("Build_Pylon_Screen");
    expect(actionName(lifecycle)).toBe("Build_Pylon_Screen");
    expect(eventSummary(decision)).toBe("派发 1 个动作：建造水晶塔");
    expect(eventSummary(lifecycle)).toBe("建造水晶塔：已派发");
    expect(eventSummary(failed)).toBe("建造水晶塔 · 失败 · 效果验证 · 目标建筑未出现");
    expect(semanticScalar("target_not_created", "failure_code")).toBe(
      "目标建筑未出现（target_not_created）",
    );
  });

  it("reduces observations to useful state and retains canonical protocol names", () => {
    const observation = event("observation", {
      game_loop: 120,
      state: {
        economy: { minerals: 400, vespene: 50, workers: 18 },
        own_units: [{ unit_type: "Probe" }],
        own_structures: [{ unit_type: "Nexus" }],
        visible_enemies: [],
        production_queue: [],
        upgrades: [],
      },
      available_actions: [{ name: "Build_Pylon_Screen" }],
      alerts: [],
      text_observation: "large upstream prompt that should not be in the semantic view",
    });
    const view = eventSemanticPayload(observation);

    expect(eventSummary(observation)).toBe("晶体矿 400 · 瓦斯 50 · 工人 18 · 可见敌军 0");
    expect(JSON.stringify(view)).not.toContain("large upstream prompt");
    expect(JSON.stringify(view)).toContain("Build_Pylon_Screen");
    expect(actionLabel("Build_Pylon_Screen")).toBe("建造水晶塔（Build_Pylon_Screen）");
  });

  it("handles first-step reflection and PySC2 terminal actions explicitly", () => {
    const reflection = event("module_result", {
      module: "reflection",
      model_call: false,
      output: { reflection: null, lessons: [] },
    });
    const execution = event("execution", {
      action_name: "Stop",
      status: "succeeded",
      execution_stage: "pysc2_acceptance",
      effect_evidence: null,
    });

    expect(moduleSemanticOutput(reflection)).toMatchObject({ reflection: "首轮没有上一条决策，已跳过复盘。" });
    expect(eventSemanticPayload(execution)).toMatchObject({
      effect_evidence: "该动作以 PySC2 接受为终态，不需要独立游戏效果校验。",
    });
  });

  it("distinguishes direct production confirmation from acceptance-only training", () => {
    const confirmed = event("execution", {
      action_name: "Train_Adept",
      actor: "Gateway/0x1",
      status: "succeeded",
      execution_stage: "effect_verification",
      effect_evidence: {
        effect_kind: "production",
        producer_tag: "0x1",
        producer_type: "Gateway",
        expected_unit_type: "Adept",
        expected_order_id: 54,
        production_order_seen: true,
        confirmation_kind: "producer_order",
      },
    });
    const acceptanceOnly = event("execution", {
      action_name: "Train_VoidRay",
      actor: "Stargate/0x2",
      status: "succeeded",
      execution_stage: "pysc2_acceptance",
      effect_evidence: null,
    });

    expect(actionLabel("Build_Stargate_Screen")).toBe("建造星门（Build_Stargate_Screen）");
    expect(actionLabel("Build_ShieldBattery_Screen")).toBe(
      "建造护盾充能站（Build_ShieldBattery_Screen）",
    );
    expect(actionLabel("Train_Adept")).toBe("训练使徒（Train_Adept）");
    expect(actionLabel("Train_VoidRay")).toBe("训练虚空辉光舰（Train_VoidRay）");
    expect(fieldLabel("producer_tag")).toBe("生产建筑 Tag");
    expect(fieldLabel("confirmation_kind")).toBe("生产确认方式");
    expect(semanticScalar("production", "effect_kind")).toBe("生产（production）");
    expect(semanticScalar("producer_order", "confirmation_kind")).toBe(
      "生产订单（producer_order）",
    );
    expect(semanticScalar("production_provenance_missing", "failure_code")).toBe(
      "生产效果证据缺少必要来源信息（production_provenance_missing）",
    );
    expect(semanticScalar("production_source_invalidated", "failure_code")).toBe(
      "生产建筑在执行前已失效（production_source_invalidated）",
    );
    expect(eventSummary(confirmed)).toBe("训练使徒 · 成功 · 效果验证");
    expect(eventSummary(acceptanceOnly)).toBe(
      "训练虚空辉光舰 · 成功 · PySC2 接受 · 尚未验证生产订单或新单位",
    );
    expect(eventSemanticPayload(confirmed)).toMatchObject({
      effect_evidence: {
        effect_kind: "production",
        producer_tag: "0x1",
        confirmation_kind: "producer_order",
      },
    });
    expect(eventSemanticPayload(acceptanceOnly)).toMatchObject({
      effect_evidence: "训练动作仅确认 PySC2 接受，未验证生产订单或新单位；不计入生产效果成功率。",
    });
  });

  it("presents deterministic goal progress as a readable review checkpoint", () => {
    const progress = event("goal_progress", {
      run_id: "run-1",
      episode_id: "episode-1",
      step_id: 12,
      game_loop: 448,
      goal_id: "protoss-opening",
      strategic_goal: "建立基础科技并开始生产追猎者",
      status: "actionable",
      achieved: [
        {
          requirement_id: "pylon",
          kind: "structure",
          target: "Pylon",
          required_count: 1,
          current_count: 1,
          in_progress_count: 0,
        },
      ],
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
      blockers: [],
      advancing_actions: ["Build_Gateway_Screen"],
      unique_next_action: "Build_Gateway_Screen",
      defensive_hold_required: false,
    });

    expect(eventTitle(progress)).toBe("目标进度检查");
    expect(eventSummary(progress)).toBe(
      "可以立即推进（actionable） · 已完成 1 项 · 待完成 1 项 · 下一步：建造传送门",
    );
    expect(eventSemanticPayload(progress)).toEqual({
      strategic_goal: "建立基础科技并开始生产追猎者",
      status: "actionable",
      achieved: progress.payload.achieved,
      missing: progress.payload.missing,
      blockers: [],
      advancing_actions: ["Build_Gateway_Screen"],
      unique_next_action: "Build_Gateway_Screen",
      defensive_hold_required: false,
      game_loop: 448,
    });
    expect(semanticScalar("Build_Gateway_Screen", "unique_next_action")).toBe(
      "建造传送门（Build_Gateway_Screen）",
    );
    expect(semanticScalar("missing_prerequisite", "kind")).toBe("缺少科技前置条件");
  });

  it("presents the Cortex specialist-to-executor pipeline without raw JSON", () => {
    const situation = event("situation_assessed", {
      source_kind: "deterministic",
      assessment: {
        game_phase: "early",
        threat_level: "low",
        army_readiness: "not_ready",
      },
    });
    const macro = event("macro_plan_accepted", {
      model_id: "hima-a",
      plan: { plan_id: "plan-1", steps: [{ action: "Pylon" }, { action: "Gateway" }] },
      runtime_frontier: "Build_Pylon_Screen",
    });
    const intent = event("intent_emitted", {
      role: "macro",
      intent_id: "intent-1",
      intent: { action_names: ["Build_Pylon_Screen"] },
    });
    const candidates = event("candidate_set_built", {
      intent_id: "intent-1",
      candidates: [
        {
          candidate_id: "candidate-1",
          action_name: "Build_Pylon_Screen",
          actor: "Builder/Probe-1",
        },
      ],
    });
    const selection = event("executor_selection", {
      executor_id: "deterministic",
      selected_candidate_id: "candidate-1",
      latency_ms: 1.4,
    });
    const lineage = event("command_lineage", {
      command_id: "command-1",
      macro_plan_id: "plan-1",
      intent_id: "intent-1",
      candidate_id: "candidate-1",
    });
    const macroStep = event("macro_step_updated", {
      plan_id: "plan-1",
      step: {
        semantic_action: "BUILD PYLON",
        runtime_actions: ["Build_Pylon_Screen"],
        status: "confirmed",
        completed_repeats: 1,
        repeat: 1,
      },
    });
    const failure = event("specialist_failed", {
      role: "macro",
      model_id: "hima-a",
      message: "request timed out",
    });

    expect(eventTitle(situation)).toBe("战况分析完成");
    expect(eventSummary(situation)).toBe(
      "来源：确定性规则（deterministic） · 阶段：开局阶段（early） · 威胁：低（low） · 军队：尚未准备（not_ready）",
    );
    expect(eventSummary(macro)).toBe("计划 plan-1 · hima-a · 2 步 · 当前：建造水晶塔");
    expect(eventSummary(intent)).toBe("宏观决策（macro） · 建造水晶塔 · intent-1");
    expect(eventSummary(candidates)).toBe("为意图 intent-1 生成 1 个合法候选：建造水晶塔");
    expect(eventSummary(selection)).toBe("deterministic · 选择 candidate-1 · 1 毫秒");
    expect(eventSummary(lineage)).toBe(
      "command-1 · 计划 plan-1 → 意图 intent-1 → 候选 candidate-1",
    );
    expect(eventSummary(macroStep)).toBe("建造水晶塔 · 已确认 · 1/1");
    expect(eventSummary(failure)).toBe("宏观决策（macro） · hima-a · request timed out");
    expect(eventSemanticPayload(selection)).toEqual({
      candidate_id: "candidate-1",
      executor_id: "deterministic",
      latency_ms: 1.4,
    });
  });

  it("explains Race Brain coordination and Playbook learning", () => {
    const coordinated = event("race_brain_coordinated", {
      selected_member_id: "hima-protoss-b",
      members: [{}, {}, {}],
      valid_member_count: 2,
      degraded_member_ids: ["hima-protoss-c"],
      playbook_lesson_ids: ["lesson-1"],
      rationale: "mapped_legal_now, promoted playbook support",
    });
    const retrieved = event("playbook_retrieved", {
      phase: "technology",
      hit_count: 2,
      lesson_ids: ["lesson-1", "lesson-2"],
    });
    const promoted = event("playbook_lesson_promoted", {
      statement: "BUILD STARGATE had a verified effect in winning episodes.",
      support_count: 2,
    });

    expect(eventTitle(coordinated)).toBe("种族大脑已汇总三位专家");
    expect(eventSummary(coordinated)).toContain(
      "3 位 HIMA 专家已提案 · 1 位输出异常 · 采用 hima-protoss-b · 引用 1 条战术经验",
    );
    expect(eventSummary(retrieved)).toBe("科技阶段（technology） · 找到 2 条可复用战术经验");
    expect(eventSummary(promoted)).toContain("已晋升 · 2 局支持");

    const revalidated = event("macro_proposal_revalidated", {
      source_game_loop: 112,
      current_game_loop: 184,
    });
    expect(eventSummary(revalidated)).toContain("loop 112 → 184");
  });

  it("falls back safely for future event types", () => {
    const unknown = event("world_model_projection", { horizon: 32 });
    expect(eventTitle(unknown)).toBe("world model projection");
    expect(eventSummary(unknown)).toBe("已记录运行事件");
  });
});
