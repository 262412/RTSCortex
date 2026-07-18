import { BookOpenCheck, BrainCircuit, Braces, CheckCircle2, Clock3, Crosshair, Database, Scale, ShieldCheck, Target, UsersRound, Workflow, XCircle } from "lucide-react";

import { moduleOutput } from "../data";
import { eventSemanticPayload, moduleSemanticOutput, semanticScalar } from "../presentation";
import type { JsonValue, StoredEvent } from "../types";
import { SemanticValue } from "./SemanticValue";

interface DecisionRailProps {
  raceProfile?: StoredEvent;
  situation?: StoredEvent;
  tacticalShadow?: StoredEvent;
  roleIntent?: StoredEvent;
  arbitration?: StoredEvent;
  playbookRule?: StoredEvent;
  reflection?: StoredEvent;
  goalProgress?: StoredEvent;
  plan?: StoredEvent;
  candidateActions?: JsonValue;
  context?: StoredEvent;
  decision?: StoredEvent;
  execution?: StoredEvent;
  plannerRunning: boolean;
  activeModule?: string;
  moduleElapsedSeconds?: number;
  modelTelemetry: JsonValue;
}

interface InsightCardProps {
  title: string;
  icon: React.ReactNode;
  value?: JsonValue;
  tone?: "default" | "success" | "danger" | "active";
  empty?: string;
}

function InsightCard({ title, icon, value, tone = "default", empty = "Waiting for runtime data" }: InsightCardProps) {
  const hasValue = value !== undefined;
  return (
    <section className={`insight-card tone-${tone}`}>
      <header>
        <span className="insight-icon" aria-hidden="true">{icon}</span>
        <h3>{title}</h3>
      </header>
      {hasValue ? <SemanticValue value={value} compact /> : <p className="empty-copy">{empty}</p>}
    </section>
  );
}

export function DecisionRail({
  raceProfile,
  situation,
  tacticalShadow,
  roleIntent,
  arbitration,
  playbookRule,
  reflection,
  goalProgress,
  plan,
  candidateActions,
  context,
  decision,
  execution,
  plannerRunning,
  activeModule,
  moduleElapsedSeconds,
  modelTelemetry,
}: DecisionRailProps) {
  const executionPayload = execution?.payload;
  const executionStatus = typeof executionPayload?.status === "string" ? executionPayload.status : undefined;
  const executionTone = executionStatus === "succeeded" ? "success" : executionStatus === "failed" ? "danger" : "default";
  const progressStatus = typeof goalProgress?.payload.status === "string" ? goalProgress.payload.status : undefined;
  const progressTone = progressStatus === "achieved" ? "success" : progressStatus === "blocked" ? "danger" : "active";
  const planValue = plan && ["plan_accepted", "macro_plan_accepted"].includes(plan.event_type)
    ? eventSemanticPayload(plan)
    : moduleOutput(plan);

  return (
    <aside className="decision-rail" aria-label="Agent decision pipeline">
      <div className="rail-heading">
        <div>
          <p className="eyebrow">当前决策周期</p>
          <h2>Agent 决策流水线</h2>
        </div>
        <span className={`planner-chip ${plannerRunning ? "active" : ""}`}>
          {plannerRunning ? <Clock3 size={13} aria-hidden="true" /> : <BrainCircuit size={13} aria-hidden="true" />}
          {plannerRunning ? `${activeModule ? semanticScalar(activeModule) : "战略规划"} · ${moduleElapsedSeconds ?? 0} 秒` : "空闲"}
        </span>
      </div>

      <div className="insight-stack">
        <InsightCard title="模型调用" icon={<Clock3 size={15} />} value={modelTelemetry} />
        <InsightCard
          title="当前种族与能力"
          icon={<ShieldCheck size={15} />}
          value={raceProfile ? eventSemanticPayload(raceProfile) : undefined}
          empty="等待 RaceProfile 激活"
        />
        <InsightCard
          title="当前战况分析"
          icon={<BrainCircuit size={15} />}
          value={situation ? eventSemanticPayload(situation) : undefined}
          empty="等待战况分析"
        />
        <InsightCard
          title="影子战术策略"
          icon={<Crosshair size={15} />}
          value={tacticalShadow ? eventSemanticPayload(tacticalShadow) : undefined}
          empty="未启用战术模型 shadow"
        />
        <InsightCard
          title="职责 Agent"
          icon={<UsersRound size={15} />}
          value={roleIntent ? eventSemanticPayload(roleIntent) : undefined}
          empty="等待职责 Agent 提交 Intent"
        />
        <InsightCard
          title="战略意图仲裁"
          icon={<Scale size={15} />}
          value={arbitration ? eventSemanticPayload(arbitration) : undefined}
          empty="等待 Intent Arbiter"
        />
        <InsightCard
          title="CortexPlaybook 约束"
          icon={<BookOpenCheck size={15} />}
          value={playbookRule ? eventSemanticPayload(playbookRule) : undefined}
          empty="当前没有适用的可执行规则"
        />
        <InsightCard title="上下文压缩" icon={<Database size={15} />} value={context?.payload} />
        <InsightCard title="复盘反思" icon={<BrainCircuit size={15} />} value={moduleSemanticOutput(reflection)} />
        <InsightCard
          title="目标进度检查（复盘依据）"
          icon={<Target size={15} />}
          value={goalProgress ? eventSemanticPayload(goalProgress) : undefined}
          tone={progressTone}
          empty="等待确定性目标进度检查"
        />
        <InsightCard title="当前采用的计划" icon={<Workflow size={15} />} value={planValue} />
        <InsightCard title="模型建议动作" icon={<Crosshair size={15} />} value={candidateActions} />
        <InsightCard title="验证与派发" icon={<ShieldCheck size={15} />} value={decision ? eventSemanticPayload(decision) : undefined} />
        <InsightCard
          title="最近执行与效果"
          icon={executionTone === "success" ? <CheckCircle2 size={15} /> : executionTone === "danger" ? <XCircle size={15} /> : <Braces size={15} />}
          value={execution ? eventSemanticPayload(execution) : undefined}
          tone={executionTone}
        />
      </div>
    </aside>
  );
}
