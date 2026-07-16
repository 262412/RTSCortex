import {
  actionName,
  commandId,
  eventPayload,
  moduleName,
  readNumber,
  readString,
} from "./data";
import type { EventCategory, JsonObject, JsonValue, StoredEvent } from "./types";

const EVENT_TITLES: Record<string, string> = {
  observation: "环境观察",
  module_started: "模块开始运行",
  context_prepared: "上下文已准备",
  module_result: "模块输出",
  module_failed: "模块运行失败",
  planner_started: "规划器开始思考",
  planner_cycle: "规划周期完成",
  planner_timeout: "规划器超时",
  planner_error: "规划器错误",
  plan_accepted: "新计划已采用",
  decision: "动作决策",
  command_lifecycle: "动作状态变化",
  execution: "动作执行结果",
  validation_failed: "动作验证失败",
  goal_progress: "目标进度检查",
  situation_assessed: "战况分析完成",
  macro_plan_accepted: "专用宏观计划已采用",
  macro_plan_rejected: "专用宏观计划被拒绝",
  macro_step_updated: "宏观计划步骤已更新",
  intent_emitted: "决策意图已生成",
  candidate_set_built: "合法执行候选已生成",
  executor_selection: "快速执行器已选择",
  command_lineage: "动作决策链已关联",
  specialist_failed: "专用模型运行失败",
  specialist_ready: "专用模型已就绪",
  specialist_recovered: "专用模型已恢复",
  race_brain_coordinated: "种族大脑已汇总三位专家",
  macro_proposal_revalidated: "宏观提案已按最新战况复核",
  playbook_retrieved: "战术笔记已检索",
  playbook_case_recorded: "关键决策案例已记录",
  playbook_lesson_candidate: "候选战术经验已更新",
  playbook_lesson_promoted: "战术经验已晋升",
  postgame_review_completed: "赛后复盘已完成",
  episode_summary: "对局总结",
  episode_result: "对局结果",
};

const FIELD_LABELS: Record<string, string> = {
  module: "模块",
  model: "模型",
  provider: "模型服务",
  model_call: "调用了模型",
  latency_ms: "耗时",
  tick_latency_ms: "本次决策耗时",
  reflex_latency_ms: "快速反应耗时",
  reflex_latency_target_ms: "快速反应目标",
  command_count: "生成动作数",
  retained_request_count: "已保留模型请求",
  latest_latency_ms: "最近响应耗时",
  usage: "Token 用量",
  prompt_tokens: "输入 Token",
  completion_tokens: "输出 Token",
  total_tokens: "总 Token",
  output: "输出",
  result: "结果",
  context_compaction: "上下文压缩",
  compression_ratio: "保留比例",
  estimated_tokens: "估算 Token",
  original_chars: "压缩前字符",
  final_chars: "压缩后字符",
  budget_chars: "字符预算",
  prompt_version: "提示词版本",
  statistics: "压缩统计",
  compacted: "已执行压缩",
  retained_observations: "保留观察",
  retained_recent_events: "保留近期事件",
  retained_lessons: "保留经验",
  retained_episode_summaries: "保留对局总结",
  dropped_recent_events: "丢弃近期事件",
  dropped_lessons: "丢弃经验",
  dropped_episode_summaries: "丢弃对局总结",
  dropped_spatial_lines: "丢弃空间描述",
  aggregated_own_units: "合并己方单位",
  aggregated_own_structures: "合并己方建筑",
  aggregated_visible_enemies: "合并可见敌军",
  reflection: "反思",
  lessons: "得到的经验",
  lesson: "经验",
  plan: "计划",
  plan_summary: "计划摘要",
  strategic_goal: "战略目标",
  goal: "目标",
  goal_id: "目标 ID",
  requirement_id: "检查项 ID",
  kind: "检查类型",
  target: "目标对象",
  required_count: "目标数量",
  current_count: "已完成数量",
  in_progress_count: "进行中数量",
  description: "检查说明",
  achieved: "已完成",
  missing: "尚缺少",
  blockers: "推进阻塞",
  detail: "阻塞说明",
  advancing_actions: "可推进动作",
  unique_next_action: "唯一下一动作",
  defensive_hold_required: "当前必须防守",
  assessment: "战况分析",
  game_phase: "游戏阶段",
  threat_level: "威胁等级",
  army_readiness: "军队准备度",
  information_gaps: "信息缺口",
  source_kind: "分析来源",
  role: "模型职责",
  specialist: "专用模块",
  model_id: "模型 ID",
  selected_member_id: "选中的专家",
  coordinator_version: "协调器版本",
  valid_member_count: "有效专家数",
  degraded_member_ids: "降级专家",
  score_reasons: "评分依据",
  playbook_lesson_ids: "引用战术经验",
  lesson_ids: "经验 ID",
  hit_count: "匹配经验数",
  quality: "决策质量",
  failure_owner: "失败归属",
  consequence: "观察到的后果",
  statement: "战术经验",
  recommended_action: "建议动作",
  avoid_action: "应避免动作",
  support_count: "支持对局数",
  contradiction_count: "矛盾对局数",
  plan_id: "计划 ID",
  macro_plan_id: "宏观计划 ID",
  intent: "决策意图",
  intent_id: "意图 ID",
  intent_kind: "意图类型",
  candidate_id: "候选 ID",
  selected_candidate_id: "选中候选 ID",
  candidate_count: "合法候选数量",
  candidates: "合法执行候选",
  executor: "快速执行器",
  executor_id: "快速执行器 ID",
  confidence: "选择置信度",
  abstain: "放弃选择",
  fallback: "使用确定性回退",
  fallback_reason: "回退原因",
  lineage: "动作决策链",
  steps: "计划步骤",
  semantic_action: "SC2 宏动作",
  runtime_actions: "可执行 Runtime 动作",
  completed_repeats: "已完成次数",
  repeat: "目标次数",
  proposed_actions: "模型建议动作",
  planner_candidates: "规划器候选动作",
  reflex_candidates: "快速反应候选动作",
  validated_candidates: "验证通过的动作",
  rejected_commands: "被拒绝的动作",
  preemptions: "抢占记录",
  busy_actor_candidates: "因执行者忙碌而等待",
  batch: "最终动作批次",
  commands: "派发动作",
  command: "动作",
  command_id: "动作 ID",
  decision_id: "决策 ID",
  actor: "执行者",
  name: "动作",
  action_name: "动作",
  arguments: "参数",
  requested_arguments: "请求参数",
  resolved_arguments: "实际参数",
  priority: "优先级",
  source: "决策来源",
  ttl_game_loops: "有效期",
  created_game_loop: "创建时刻",
  accepted_game_loop: "接纳时刻",
  source_game_loop: "模型观察时刻",
  source_step_id: "来源决策步",
  plan_age_game_loops: "接纳时计划年龄",
  retained_command_ids: "继续沿用的动作 ID",
  is_revision: "是否为计划修订",
  lifecycle_protocol: "生命周期协议",
  fingerprint: "计划指纹",
  preconditions: "执行前置条件",
  max_supply_free: "人口空余上限",
  no_pending_structure: "不得已有在建建筑",
  planner_pending: "规划器仍在运行",
  idle_reason: "空闲原因",
  status: "状态",
  state: "状态",
  reason: "原因",
  success: "执行成功",
  execution_stage: "执行阶段",
  failure_code: "失败类型",
  failure_reason: "失败原因",
  game_result: "比赛结果",
  pysc2_function: "PySC2 调用",
  primitive_trace: "底层操作链",
  effect_evidence: "效果证据",
  effect_kind: "效果类型",
  accepted: "SC2 已接受",
  function: "底层函数",
  origin: "操作来源",
  ordinal: "链中序号",
  total: "链长度",
  game_loop: "游戏循环",
  emitted_function_id: "实际函数 ID",
  requested_function_id: "请求函数 ID",
  raw_reason: "底层原因",
  dispatch_loop: "派发时刻",
  accept_loop: "接受时刻",
  accepted_loop: "接受时刻",
  confirm_loop: "确认时刻",
  dispatch_game_loop: "派发时刻",
  dispatched_loop: "派发时刻",
  confirmed_game_loop: "效果确认时刻",
  confirmed_loop: "确认时刻",
  elapsed_game_loops: "验证耗时",
  target_type: "目标类型",
  new_structure_tag: "新建筑 Tag",
  builder_tag: "建造工 Tag",
  target_tag: "目标 Tag",
  target_position: "目标位置",
  expected_structure: "预期建筑",
  producer_tag: "生产建筑 Tag",
  producer_type: "生产建筑类型",
  expected_unit_type: "预期单位",
  expected_order_id: "预期生产订单 ID",
  baseline_structure_tags: "原有同类建筑",
  baseline_unit_tags: "原有同类单位",
  order_seen: "观察到建造订单",
  worker_orders: "工人订单",
  new_unit_tag: "新单位 Tag",
  baseline_producer_orders: "生产建筑原有订单",
  producer_orders: "生产建筑当前订单",
  production_order_seen: "观察到生产订单",
  confirmation_kind: "生产确认方式",
  resource_delta: "资源变化",
  mineral_delta: "晶体矿变化",
  builder_displacement: "建造工位移",
  move_order_seen: "观察到移动订单",
  active_order_extension: "因活动订单延长验证",
  alerts: "告警",
  available_actions: "当前可用动作",
  economy: "经济",
  minerals: "晶体矿",
  vespene: "高能瓦斯",
  workers: "工人",
  supply_used: "已用人口",
  supply_cap: "人口上限",
  army_supply: "军队人口",
  own_units: "己方单位",
  own_structures: "己方建筑",
  visible_enemies: "可见敌军",
  production_queue: "生产队列",
  upgrades: "科技升级",
  unit_type: "单位类型",
  unit_id: "单位 Tag",
  health_fraction: "生命比例",
  energy: "能量",
  position: "位置",
  protocol_version: "协议版本",
  run_id: "运行 ID",
  episode_id: "对局 ID",
  step_id: "决策步",
  summary: "摘要",
  score: "得分",
  outcome: "比赛结果",
  failure: "失败",
};

const ACTION_LABELS: Record<string, string> = {
  No_Operation: "暂不行动",
  Stop: "停止当前指令",
  Hold_Position: "原地防守",
  Move_Screen: "移动到屏幕位置",
  Move_Minimap: "移动到小地图位置",
  Attack_Unit: "攻击可见敌军",
  Attack_Screen: "攻击屏幕位置",
  Attack_Minimap: "攻击小地图位置",
  Build_Pylon_Screen: "建造水晶塔",
  Build_Gateway_Screen: "建造传送门",
  Build_CyberneticsCore_Screen: "建造控制芯核",
  Build_Cybernetics_Core_Screen: "建造控制芯核",
  Build_Assimilator_Near: "在气矿建造吸收舱",
  Build_Nexus_Near: "在扩张点建造星灵枢纽",
  Build_Stargate_Screen: "建造星门",
  Build_ShieldBattery_Screen: "建造护盾充能站",
  Train_Probe: "训练探机",
  Train_Zealot: "训练狂热者",
  Train_Stalker: "训练追猎者",
  Train_Adept: "训练使徒",
  Train_VoidRay: "训练虚空辉光舰",
  Train_Oracle: "训练先知",
  Train_Phoenix: "训练凤凰战机",
  Research_Warp_Gate: "研究折跃门",
  Research_WarpGate: "研究折跃门",
  Retreat: "撤退",
};

const VALUE_LABELS: Record<string, string> = {
  memory: "记忆检索",
  reflection: "复盘反思",
  planning: "战略规划",
  action: "动作生成",
  planner: "规划器",
  reflex: "快速反应",
  situation: "战况分析",
  macro: "宏观决策",
  tactical: "战术决策",
  motor: "快速执行",
  deterministic: "确定性规则",
  deterministic_reflex: "确定性战术与快速反应",
  fallback: "回退策略",
  translator: "动作翻译器",
  orchestration: "环境编排",
  openai_compatible: "OpenAI 兼容接口",
  pending: "等待处理",
  deferred: "暂缓执行",
  dispatched: "已派发",
  confirmed: "已确认",
  succeeded: "成功",
  success: "成功",
  failed: "失败",
  cancelled: "已取消",
  unconfirmed: "尚未确认",
  expired: "已过期",
  rejected: "已拒绝",
  obsolete: "已经失效",
  superseded: "已被新计划替代",
  actionable: "可以立即推进",
  in_progress: "正在推进",
  achieved: "目标已达成",
  structure: "建筑",
  unit: "单位",
  production: "生产",
  producer_order: "生产订单",
  new_unit: "新单位出现",
  upgrade: "科技升级",
  goal_dependency: "前序目标尚未完成",
  missing_prerequisite: "缺少科技前置条件",
  prerequisite_in_progress: "科技前置条件正在建设",
  effect_in_progress: "目标动作正在生效",
  insufficient_minerals: "晶体矿不足",
  insufficient_vespene: "高能瓦斯不足",
  insufficient_supply: "人口不足",
  action_unavailable: "推进动作当前不可用",
  no_progress_action: "没有可推进目标的动作",
  starting: "正在启动",
  running: "运行中",
  completed: "已完成",
  healthy: "正常",
  draw: "平局",
  victory: "胜利",
  defeat: "失败",
  pre_dispatch: "派发前检查",
  translation: "动作翻译",
  pysc2_acceptance: "PySC2 接受",
  effect_verification: "效果验证",
  episode_end: "对局结束",
  waiting_for_planner: "等待规划结果",
  plan_commands_deferred: "计划动作暂缓",
  plan_exhausted: "当前计划已执行完",
  no_legal_action: "当前没有合法动作",
  planner_timeout: "规划器超时",
  noop_baseline: "空动作基线",
  early: "开局阶段",
  technology: "科技阶段",
  combat: "战斗阶段",
  low: "低",
  medium: "中",
  high: "高",
  none: "无",
  critical: "紧急",
  empty: "尚无军队",
  forming: "正在集结",
  ready: "已准备",
  engaged: "正在交战",
  not_ready: "尚未准备",
  abstain: "主动放弃选择",
  friendly_target: "目标属于己方",
  target_not_visible: "目标当前不可见",
  no_legal_placement: "没有合法建造位置",
  need_power: "建造位置缺少水晶塔能量场",
  blocked: "建造位置被阻挡",
  not_pathable: "位置不可通行",
  invalid_geyser_tag: "气矿目标无效",
  invalid_expansion_anchor: "扩张点目标无效",
  translator_rejected: "动作翻译器拒绝",
  pysc2_rejected: "PySC2 拒绝动作",
  effect_timeout: "未在时限内观察到效果",
  bridge_integrity_error: "Bridge 归因异常",
  candidate_invalidated: "候选参数在派发前失效",
  production_source_unavailable: "生产建筑当前不可用",
  production_source_invalidated: "生产建筑在执行前已失效",
  production_provenance_missing: "生产效果证据缺少必要来源信息",
  actor_not_available: "执行者当前不可用",
  no_build_order_observed: "未观察到建造订单",
  worker_order_replaced: "工人订单被替换",
  target_not_created: "目标建筑未出现",
  builder_not_observable: "无法继续观察建造工",
  producer_not_observable: "无法继续观察生产建筑",
  no_production_order_observed: "未观察到生产订单",
  production_order_replaced: "生产订单被替换",
  episode_ended_unconfirmed: "对局结束时效果仍未确认",
  episode_ended_before_dispatch: "对局结束前动作尚未派发",
  worker_terminated_before_execution_report: "Worker 终止前未回报执行结果",
  bridge_execution_report_missing: "Bridge 未回报动作终态",
  episode_ended: "对局结束前动作未完成",
  "action is not available": "当前观察中该动作不可用",
};

const CATEGORY_LABELS: Record<Exclude<EventCategory, "all"> | "system", string> = {
  planner: "规划",
  reflex: "快速反应",
  build: "建造",
  production: "生产",
  combat: "战斗",
  failure: "失败",
  system: "系统",
};

const GOAL_PROGRESS_STATUS_LABELS: Record<string, string> = {
  actionable: "可以立即推进",
  in_progress: "正在推进",
  blocked: "暂时受阻",
  achieved: "目标已达成",
};

function asObject(value: JsonValue | undefined): JsonObject | undefined {
  return typeof value === "object" && value !== null && !Array.isArray(value) ? value : undefined;
}

function asArray(value: JsonValue | undefined): JsonValue[] {
  return Array.isArray(value) ? value : [];
}

function firstObject(value: JsonValue | undefined): JsonObject | undefined {
  return asArray(value).map(asObject).find((item) => item !== undefined);
}

function formatCount(value: number, unit: string): string {
  return `${value.toLocaleString()} ${unit}`;
}

function formatDuration(value: number): string {
  return value >= 1_000 ? `${(value / 1_000).toFixed(2)} 秒` : `${Math.round(value)} 毫秒`;
}

function truncate(value: string, limit = 120): string {
  const normalized = value.replace(/\s+/g, " ").trim();
  return normalized.length > limit ? `${normalized.slice(0, limit - 1)}…` : normalized;
}

function commandObjects(payload: JsonObject): JsonObject[] {
  const batch = asObject(payload.batch);
  const candidates = [
    payload.commands,
    payload.candidates,
    batch?.commands,
    payload.validated_candidates,
    payload.planner_candidates,
    payload.reflex_candidates,
  ];
  for (const candidate of candidates) {
    const commands = asArray(candidate).map(asObject).filter((value): value is JsonObject => value !== undefined);
    if (commands.length > 0) return commands;
  }
  const command = asObject(payload.command);
  return command ? [command] : [];
}

function commandAction(command: JsonObject | undefined): string | undefined {
  return command ? readString(command, "action_name", "action", "name") : undefined;
}

function actionFromEvent(event: StoredEvent): string | undefined {
  return actionName(event) ?? commandAction(firstObject(event.payload.commands)) ?? commandAction(commandObjects(event.payload)[0]);
}

export function fieldLabel(key: string): string {
  return FIELD_LABELS[key] ?? key.replaceAll("_", " ");
}

export function actionLabel(action: string, includeProtocolName = true): string {
  const translated = ACTION_LABELS[action] ?? action.replaceAll("_", " ");
  return includeProtocolName && translated !== action ? `${translated}（${action}）` : translated;
}

export function actorLabel(actor: string): string {
  if (actor.startsWith("Builder/")) return `建造工编队（${actor}）`;
  if (actor.startsWith("CombatGroup/")) return `战斗编队（${actor}）`;
  if (actor.startsWith("Developer/")) return `宏观指挥层（${actor}）`;
  return actor;
}

export function semanticScalar(value: string | number | boolean | null, key?: string): string {
  if (value === null) return "无";
  if (typeof value === "boolean") return value ? "是" : "否";
  if (typeof value === "number") {
    if (key?.endsWith("latency_ms")) return formatDuration(value);
    if (key === "compression_ratio") return `${(value * 100).toFixed(1)}%`;
    if (key === "health_fraction") return `${Math.round(value * 100)}%`;
    if (key?.endsWith("game_loops") || key?.endsWith("_loop") || key === "game_loop") return `${value} loops`;
    return value.toLocaleString();
  }
  if (
    key === "action_name" ||
    key === "action" ||
    key === "name" ||
    key === "unique_next_action" ||
    key === "advancing_actions"
  ) return actionLabel(value);
  if (key === "actor") return actorLabel(value);
  if (key === "status" && GOAL_PROGRESS_STATUS_LABELS[value.toLowerCase()]) {
    return `${GOAL_PROGRESS_STATUS_LABELS[value.toLowerCase()]}（${value}）`;
  }
  const translated = VALUE_LABELS[value.toLowerCase()];
  const protocolEnum = [
    "status",
    "state",
    "source",
    "origin",
    "execution_stage",
    "failure_code",
    "idle_reason",
    "reason",
    "effect_kind",
    "confirmation_kind",
    "role",
    "source_kind",
    "game_phase",
    "threat_level",
    "army_readiness",
    "selection",
  ].includes(key ?? "");
  return translated && protocolEnum ? `${translated}（${value}）` : translated ?? value;
}

export function categoryLabel(category: Exclude<EventCategory, "all"> | "system"): string {
  return CATEGORY_LABELS[category];
}

export function eventTitle(event: StoredEvent): string {
  const base = EVENT_TITLES[event.event_type] ?? event.event_type.replaceAll("_", " ");
  const module = moduleName(event);
  if ((event.event_type === "module_started" || event.event_type === "module_result") && module) {
    return `${semanticScalar(module)}${event.event_type === "module_started" ? "开始" : "完成"}`;
  }
  return base;
}

export function eventSummary(event: StoredEvent): string {
  const payload = eventPayload(event);
  const module = moduleName(event);
  const latency = readNumber(payload, "latency_ms");
  const status = readString(payload, "status", "state");
  const failure = readString(payload, "failure_code", "failure_reason", "reason", "message");
  const action = actionFromEvent(event);
  const commands = commandObjects(payload);

  if (event.event_type === "module_started") {
    return `开始运行${module ? semanticScalar(module) : "Agent"}模块${readString(payload, "model") ? ` · ${readString(payload, "model")}` : ""}`;
  }
  if (event.event_type === "context_prepared") {
    const original = readNumber(payload, "original_chars") ?? 0;
    const final = readNumber(payload, "final_chars") ?? 0;
    const statistics = asObject(payload.statistics);
    const retained = readNumber(statistics ?? payload, "retained_observations") ?? 0;
    return `${module ? semanticScalar(module) : "模型"}上下文：${original.toLocaleString()} → ${final.toLocaleString()} 字符 · 保留 ${retained} 条观察`;
  }
  if (event.event_type === "module_result") {
    const output = asObject(payload.output);
    const plan = asObject(output?.plan);
    const reflection = readString(output ?? {}, "reflection");
    const goal = readString(plan ?? output ?? {}, "strategic_goal", "goal");
    const usage = asObject(payload.usage);
    const tokens = readNumber(usage ?? {}, "total_tokens");
    const detail = reflection ?? goal;
    return [
      `${module ? semanticScalar(module) : "模块"}已完成`,
      latency === undefined ? undefined : formatDuration(latency),
      tokens === undefined ? undefined : formatCount(tokens, "tokens"),
      detail ? truncate(detail) : undefined,
    ].filter(Boolean).join(" · ");
  }
  if (event.event_type === "module_failed") {
    return `${module ? semanticScalar(module) : "模块"}运行失败${failure ? `：${semanticScalar(failure)}` : ""}`;
  }
  if (event.event_type === "planner_started") {
    const loop = readNumber(payload, "started_game_loop") ?? event.game_loop;
    return `开始生成下一轮战略计划${loop === undefined ? "" : ` · loop ${loop}`}`;
  }
  if (event.event_type === "planner_cycle") {
    return [status ? semanticScalar(status) : "规划完成", latency === undefined ? undefined : formatDuration(latency)].filter(Boolean).join(" · ");
  }
  if (event.event_type === "plan_accepted") {
    const plan = asObject(payload.plan);
    const goal = readString(plan ?? payload, "strategic_goal", "goal");
    return goal ? `采用目标：${truncate(goal)}` : "新的战略计划已进入执行队列";
  }
  if (event.event_type === "goal_progress") {
    const achieved = asArray(payload.achieved).length;
    const missing = asArray(payload.missing).length;
    const blockers = asArray(payload.blockers).length;
    const nextAction = readString(payload, "unique_next_action");
    return [
      status ? semanticScalar(status, "status") : "目标进度已更新",
      `已完成 ${achieved} 项`,
      `待完成 ${missing} 项`,
      nextAction ? `下一步：${actionLabel(nextAction, false)}` : undefined,
      blockers > 0 ? `阻塞 ${blockers} 项` : undefined,
    ].filter(Boolean).join(" · ");
  }
  if (event.event_type === "situation_assessed") {
    const assessment = asObject(payload.assessment) ?? payload;
    const source = readString(payload, "source_kind", "source_id", "source", "model") ?? "unknown";
    const phase = readString(assessment, "game_phase", "phase") ?? "unknown";
    const threat = readString(assessment, "threat_level", "threat") ?? "unknown";
    const readiness = readString(assessment, "army_readiness", "readiness") ?? "unknown";
    return `来源：${semanticScalar(source, "source_kind")} · 阶段：${semanticScalar(phase, "game_phase")} · 威胁：${semanticScalar(threat, "threat_level")} · 军队：${semanticScalar(readiness, "army_readiness")}`;
  }
  if (event.event_type === "macro_plan_accepted" || event.event_type === "macro_plan_rejected") {
    const plan = asObject(payload.plan) ?? payload;
    const planId = readString(payload, "plan_id") ?? readString(plan, "plan_id") ?? "unknown";
    const model = readString(payload, "model_id", "source_model_id", "model", "specialist") ?? readString(plan, "source_model_id", "model_id") ?? "unknown";
    if (event.event_type === "macro_plan_rejected") {
      return `计划 ${planId} · ${model} · ${failure ? semanticScalar(failure) : "原因未记录"}`;
    }
    const steps = asArray(plan.steps ?? payload.steps).length;
    const frontier = readString(payload, "runtime_frontier", "frontier_action");
    return `计划 ${planId} · ${model} · ${steps} 步${frontier ? ` · 当前：${actionLabel(frontier, false)}` : ""}`;
  }
  if (event.event_type === "race_brain_coordinated") {
    const selected = readString(payload, "selected_member_id") ?? "unknown";
    const rationale = readString(payload, "rationale");
    const members = asArray(payload.members).length;
    const degraded = asArray(payload.degraded_member_ids).length;
    const lessons = asArray(payload.playbook_lesson_ids).length;
    return `${members} 位 HIMA 专家已提案${degraded ? ` · ${degraded} 位输出异常` : ""} · 采用 ${selected}${lessons ? ` · 引用 ${lessons} 条战术经验` : ""}${rationale ? ` · ${truncate(rationale)}` : ""}`;
  }
  if (event.event_type === "macro_proposal_revalidated") {
    const sourceLoop = readNumber(payload, "source_game_loop") ?? 0;
    const currentLoop = readNumber(payload, "current_game_loop") ?? 0;
    return `提案生成期间动作结果发生变化 · loop ${sourceLoop} → ${currentLoop} · 已重新检查当前合法动作`;
  }
  if (event.event_type === "playbook_retrieved") {
    const count = readNumber(payload, "hit_count") ?? asArray(payload.lesson_ids).length;
    const phase = readString(payload, "phase");
    return `${phase ? `${semanticScalar(phase, "game_phase")} · ` : ""}找到 ${count} 条可复用战术经验`;
  }
  if (event.event_type === "playbook_case_recorded") {
    const semanticAction = readString(payload, "semantic_action") ?? "unknown";
    const quality = readString(payload, "quality") ?? "unknown";
    const consequence = readString(payload, "consequence");
    return `${actionLabel(semanticAction, false)} · ${semanticScalar(quality)}${consequence ? ` · ${truncate(consequence)}` : ""}`;
  }
  if (event.event_type === "playbook_lesson_candidate" || event.event_type === "playbook_lesson_promoted") {
    const statement = readString(payload, "statement") ?? "战术经验已更新";
    const support = readNumber(payload, "support_count") ?? 0;
    return `${event.event_type === "playbook_lesson_promoted" ? "已晋升" : "仍待验证"} · ${support} 局支持 · ${truncate(statement)}`;
  }
  if (event.event_type === "postgame_review_completed") {
    const cases = readNumber(payload, "case_count") ?? 0;
    const lessons = readNumber(payload, "lesson_update_count") ?? 0;
    return `复盘 ${cases} 个关键决策 · 更新 ${lessons} 条战术经验`;
  }
  if (event.event_type === "macro_step_updated") {
    const step = asObject(payload.step) ?? payload;
    const semanticAction = readString(step, "semantic_action", "action") ?? "unknown";
    const runtimeAction = asArray(step.runtime_actions).find((value): value is string => typeof value === "string");
    const stepStatus = readString(step, "status") ?? "unknown";
    const completed = readNumber(step, "completed_repeats") ?? 0;
    const repeat = readNumber(step, "repeat") ?? 1;
    const reason = readString(step, "reason");
    return `${actionLabel(runtimeAction ?? semanticAction, false)} · ${semanticScalar(stepStatus)} · ${completed}/${repeat}${reason ? ` · ${semanticScalar(reason)}` : ""}`;
  }
  if (event.event_type === "intent_emitted") {
    const intent = asObject(payload.intent) ?? payload;
    const role = readString(payload, "role", "source_role", "intent_kind", "source") ?? readString(intent, "role", "source_role", "intent_kind", "source");
    const intentId = readString(payload, "intent_id") ?? readString(intent, "intent_id");
    const intentAction = readString(payload, "action_name", "action")
      ?? readString(intent, "action_name", "action")
      ?? asArray(intent.action_names).find((value): value is string => typeof value === "string");
    return [
      role ? semanticScalar(role, "role") : "未知职责",
      intentAction ? actionLabel(intentAction, false) : "未指定动作",
      intentId,
    ].filter(Boolean).join(" · ");
  }
  if (event.event_type === "candidate_set_built") {
    const candidates = asArray(payload.candidates);
    const count = readNumber(payload, "candidate_count") ?? candidates.length;
    const names = candidates.slice(0, 3).map(asObject).map((candidate) => commandAction(candidate)).filter((name): name is string => Boolean(name));
    return `为意图 ${readString(payload, "intent_id") ?? "unknown"} 生成 ${count} 个合法候选${names.length ? `：${names.map((name) => actionLabel(name, false)).join("、")}` : ""}`;
  }
  if (event.event_type === "executor_selection") {
    const executor = readString(payload, "executor_id", "executor", "model") ?? "unknown";
    const selected = readString(payload, "selected_candidate_id", "candidate_id");
    const fallback = readString(payload, "fallback_reason");
    return [
      executor,
      selected ? `选择 ${selected}` : "主动放弃选择",
      latency === undefined ? undefined : formatDuration(latency),
      fallback ? `回退：${semanticScalar(fallback)}` : undefined,
    ].filter(Boolean).join(" · ");
  }
  if (event.event_type === "command_lineage") {
    const lineage = asObject(payload.lineage) ?? payload;
    const lineageCommand = readString(payload, "command_id") ?? readString(lineage, "command_id") ?? "unknown";
    const planId = readString(payload, "macro_plan_id", "plan_id") ?? readString(lineage, "macro_plan_id", "plan_id") ?? "none";
    const intentId = readString(payload, "intent_id") ?? readString(lineage, "intent_id") ?? "unknown";
    const candidateId = readString(payload, "candidate_id") ?? readString(lineage, "candidate_id") ?? "unknown";
    return `${lineageCommand} · 计划 ${planId} → 意图 ${intentId} → 候选 ${candidateId}`;
  }
  if (["specialist_failed", "specialist_ready", "specialist_recovered"].includes(event.event_type)) {
    const role = readString(payload, "role", "specialist", "module") ?? "unknown";
    const model = readString(payload, "model_id", "model") ?? "unknown";
    if (event.event_type === "specialist_failed") {
      return `${semanticScalar(role, "role")} · ${model} · ${failure ? semanticScalar(failure) : "原因未记录"}`;
    }
    return `${semanticScalar(role, "role")} · ${model} ${event.event_type === "specialist_ready" ? "已就绪" : "已恢复"}`;
  }
  if (event.event_type === "decision") {
    const batch = asObject(payload.batch);
    const dispatched = asArray(batch?.commands ?? payload.commands).length;
    const rejected = asArray(batch?.rejected_commands ?? payload.rejected_commands).length;
    const idleReason = readString(batch ?? payload, "idle_reason");
    if (dispatched === 0) return idleReason ? `本轮未派发动作：${semanticScalar(idleReason)}` : "本轮没有派发动作";
    const names = commands.slice(0, 2).map((command) => commandAction(command)).filter((name): name is string => Boolean(name));
    return `派发 ${dispatched} 个动作${names.length > 0 ? `：${names.map((name) => actionLabel(name, false)).join("、")}` : ""}${rejected > 0 ? ` · 拒绝 ${rejected} 个` : ""}`;
  }
  if (event.event_type === "command_lifecycle") {
    return `${action ? actionLabel(action, false) : "动作"}：${status ? semanticScalar(status) : "状态已更新"}${failure ? ` · ${semanticScalar(failure)}` : ""}`;
  }
  if (event.event_type === "execution" || event.event_type === "validation_failed") {
    const stage = readString(payload, "execution_stage");
    const acceptanceOnlyProduction =
      event.event_type === "execution" &&
      action?.startsWith("Train_") &&
      status === "succeeded" &&
      stage === "pysc2_acceptance" &&
      (payload.effect_evidence === null || payload.effect_evidence === undefined);
    return [
      action ? actionLabel(action, false) : "动作",
      status ? semanticScalar(status) : event.event_type === "validation_failed" ? "验证失败" : "已执行",
      stage ? semanticScalar(stage) : undefined,
      acceptanceOnlyProduction ? "尚未验证生产订单或新单位" : undefined,
      failure ? semanticScalar(failure) : undefined,
    ].filter(Boolean).join(" · ");
  }
  if (event.event_type === "observation") {
    const state = asObject(payload.state) ?? payload;
    const economy = asObject(state.economy);
    const minerals = readNumber(economy ?? {}, "minerals");
    const vespene = readNumber(economy ?? {}, "vespene");
    const workers = readNumber(economy ?? {}, "workers");
    const enemies = asArray(state.visible_enemies).length;
    return [
      minerals === undefined ? undefined : `晶体矿 ${minerals}`,
      vespene === undefined ? undefined : `瓦斯 ${vespene}`,
      workers === undefined ? undefined : `工人 ${workers}`,
      `可见敌军 ${enemies}`,
    ].filter(Boolean).join(" · ");
  }
  if (event.event_type === "episode_result" || event.event_type === "episode_summary") {
    const outcome = readString(payload, "outcome", "game_result", "status");
    const summary = readString(payload, "summary", "failure_reason");
    return [outcome ? semanticScalar(outcome) : "对局已结束", summary ? truncate(summary) : undefined].filter(Boolean).join(" · ");
  }
  if (action) return [actionLabel(action, false), status ? semanticScalar(status) : undefined, failure ? semanticScalar(failure) : undefined].filter(Boolean).join(" · ");
  if (module) return [semanticScalar(module), status ? semanticScalar(status) : undefined].filter(Boolean).join(" · ");
  if (failure) return semanticScalar(failure);
  if (status) return semanticScalar(status);
  return commandId(event) ? `记录动作 ${commandId(event)}` : "已记录运行事件";
}

function compactObject(entries: [string, JsonValue | undefined][]): JsonObject {
  return Object.fromEntries(entries.filter((entry): entry is [string, JsonValue] => entry[1] !== undefined));
}

export function eventSemanticPayload(event: StoredEvent): JsonValue {
  const payload = event.payload;
  if (event.event_type === "observation") {
    const state = asObject(payload.state) ?? {};
    return compactObject([
      ["game_loop", payload.game_loop],
      ["economy", state.economy],
      ["own_units", asArray(state.own_units).length],
      ["own_structures", asArray(state.own_structures).length],
      ["visible_enemies", asArray(state.visible_enemies).length],
      ["production_queue", state.production_queue],
      ["upgrades", state.upgrades],
      ["alerts", payload.alerts],
      ["available_actions", asArray(payload.available_actions).map((value) => readString(asObject(value) ?? {}, "name") ?? value)],
    ]);
  }
  if (event.event_type === "module_result") {
    return compactObject([
      ["module", payload.module],
      ["model", payload.model],
      ["latency_ms", payload.latency_ms],
      ["model_call", payload.model_call],
      ["command_count", payload.command_count],
      ["output", payload.output],
      ["usage", payload.usage],
    ]);
  }
  if (event.event_type === "goal_progress") {
    return compactObject([
      ["strategic_goal", payload.strategic_goal],
      ["status", payload.status],
      ["achieved", payload.achieved],
      ["missing", payload.missing],
      ["blockers", payload.blockers],
      ["advancing_actions", payload.advancing_actions],
      ["unique_next_action", payload.unique_next_action],
      ["defensive_hold_required", payload.defensive_hold_required],
      ["game_loop", payload.game_loop],
    ]);
  }
  if (event.event_type === "situation_assessed") {
    const assessment = asObject(payload.assessment) ?? payload;
    return compactObject([
      ["source_kind", payload.source_kind ?? payload.source ?? payload.model],
      ["game_phase", assessment.game_phase ?? assessment.phase],
      ["threat_level", assessment.threat_level ?? assessment.threat],
      ["army_readiness", assessment.army_readiness ?? assessment.readiness],
      ["information_gaps", assessment.information_gaps],
      ["assessment", payload.assessment],
    ]);
  }
  if (["macro_plan_accepted", "macro_plan_rejected", "macro_step_updated", "intent_emitted", "candidate_set_built", "executor_selection", "command_lineage", "specialist_failed", "specialist_ready", "specialist_recovered"].includes(event.event_type)) {
    return compactObject([
      ["role", payload.role ?? payload.source_role ?? payload.intent_kind ?? payload.specialist],
      ["model_id", payload.model_id ?? payload.source_model_id ?? payload.model],
      ["plan_id", payload.plan_id ?? payload.macro_plan_id],
      ["intent_id", payload.intent_id],
      ["candidate_id", payload.selected_candidate_id ?? payload.candidate_id],
      ["executor_id", payload.executor_id ?? payload.executor],
      ["action_name", payload.action_name ?? payload.action],
      ["status", payload.status],
      ["reason", payload.reason ?? payload.failure_code ?? payload.failure_reason ?? payload.message],
      ["latency_ms", payload.latency_ms],
      ["fallback_reason", payload.fallback_reason],
      ["plan", payload.plan],
      ["steps", payload.step],
      ["intent", payload.intent],
      ["candidates", payload.candidates],
      ["lineage", payload.lineage],
    ]);
  }
  if (event.event_type === "decision") {
    const batch = asObject(payload.batch) ?? {};
    return compactObject([
      ["strategic_goal", batch.strategic_goal],
      ["planner_candidates", payload.planner_candidates],
      ["reflex_candidates", payload.reflex_candidates],
      ["validated_candidates", payload.validated_candidates],
      ["rejected_commands", batch.rejected_commands ?? payload.rejected_commands],
      ["preemptions", payload.preemptions],
      ["commands", batch.commands ?? payload.commands],
      ["idle_reason", batch.idle_reason ?? payload.idle_reason],
      ["planner_pending", batch.planner_pending ?? payload.planner_pending],
      ["reflex_latency_ms", payload.reflex_latency_ms],
      ["tick_latency_ms", payload.tick_latency_ms],
    ]);
  }
  if (event.event_type === "plan_accepted") {
    return compactObject([
      ["strategic_goal", payload.strategic_goal],
      ["is_revision", payload.is_revision],
      ["commands", payload.commands],
      ["retained_command_ids", payload.retained_command_ids],
      ["accepted_game_loop", payload.accepted_game_loop],
      ["source_game_loop", payload.source_game_loop],
      ["plan_age_game_loops", payload.plan_age_game_loops],
    ]);
  }
  if (event.event_type === "execution") {
    const status = readString(payload, "status");
    const stage = readString(payload, "execution_stage");
    const action = readString(payload, "action_name");
    const lacksEffectEvidence = payload.effect_evidence === null || payload.effect_evidence === undefined;
    if (
      action?.startsWith("Train_") &&
      lacksEffectEvidence &&
      status === "succeeded" &&
      stage === "pysc2_acceptance"
    ) {
      return compactObject([
        ["action_name", payload.action_name],
        ["actor", payload.actor],
        ["status", payload.status],
        ["execution_stage", payload.execution_stage],
        ["pysc2_function", payload.pysc2_function],
        ["requested_arguments", payload.requested_arguments],
        ["resolved_arguments", payload.resolved_arguments],
        ["primitive_trace", payload.primitive_trace],
        ["effect_evidence", "训练动作仅确认 PySC2 接受，未验证生产订单或新单位；不计入生产效果成功率。"],
      ]);
    }
    if (lacksEffectEvidence && status === "succeeded" && stage === "pysc2_acceptance") {
      return compactObject([
        ["action_name", payload.action_name],
        ["actor", payload.actor],
        ["status", payload.status],
        ["execution_stage", payload.execution_stage],
        ["pysc2_function", payload.pysc2_function],
        ["requested_arguments", payload.requested_arguments],
        ["resolved_arguments", payload.resolved_arguments],
        ["primitive_trace", payload.primitive_trace],
        ["effect_evidence", "该动作以 PySC2 接受为终态，不需要独立游戏效果校验。"],
      ]);
    }
    return compactObject([
      ["action_name", payload.action_name],
      ["actor", payload.actor],
      ["source", payload.source],
      ["status", payload.status],
      ["execution_stage", payload.execution_stage],
      ["failure_code", payload.failure_code],
      ["failure_reason", payload.failure_reason],
      ["game_result", payload.game_result],
      ["pysc2_function", payload.pysc2_function],
      ["requested_arguments", payload.requested_arguments],
      ["resolved_arguments", payload.resolved_arguments],
      ["primitive_trace", payload.primitive_trace],
      ["effect_evidence", payload.effect_evidence],
    ]);
  }
  return payload;
}

export function moduleSemanticOutput(event: StoredEvent | undefined): JsonValue | undefined {
  if (!event) return undefined;
  const output = asObject(event.payload.output);
  if (
    moduleName(event)?.toLowerCase() === "reflection" &&
    output?.reflection === null &&
    event.payload.model_call === false
  ) {
    return {
      ...output,
      reflection: "首轮没有上一条决策，已跳过复盘。",
    };
  }
  if (moduleName(event)?.toLowerCase() === "reflection" && output) {
    return compactObject([
      ["reflection", output.reflection],
      ["lessons", output.lessons],
    ]);
  }
  return event.payload.output ?? event.payload.result ?? event.payload.module_result ?? event.payload;
}

export function isTechnicalField(key: string): boolean {
  return key.endsWith("_id") || key.endsWith("_tag") || key === "pysc2_function" || key === "function";
}
