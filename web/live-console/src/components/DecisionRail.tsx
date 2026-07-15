import { BrainCircuit, Braces, CheckCircle2, Clock3, Crosshair, Database, ShieldCheck, Workflow, XCircle } from "lucide-react";

import { moduleOutput } from "../data";
import { eventSemanticPayload, moduleSemanticOutput, semanticScalar } from "../presentation";
import type { JsonValue, StoredEvent } from "../types";
import { SemanticValue } from "./SemanticValue";

interface DecisionRailProps {
  reflection?: StoredEvent;
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
  reflection,
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
  const planValue = plan?.event_type === "plan_accepted" ? eventSemanticPayload(plan) : moduleOutput(plan);

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
        <InsightCard title="上下文压缩" icon={<Database size={15} />} value={context?.payload} />
        <InsightCard title="复盘反思" icon={<BrainCircuit size={15} />} value={moduleSemanticOutput(reflection)} />
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
