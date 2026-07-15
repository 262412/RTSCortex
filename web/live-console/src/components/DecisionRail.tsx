import { BrainCircuit, Braces, CheckCircle2, Clock3, Crosshair, Database, ShieldCheck, Workflow, XCircle } from "lucide-react";

import { moduleOutput } from "../data";
import type { JsonValue, StoredEvent } from "../types";

interface DecisionRailProps {
  reflection?: StoredEvent;
  plan?: StoredEvent;
  action?: StoredEvent;
  context?: StoredEvent;
  decision?: StoredEvent;
  execution?: StoredEvent;
  plannerRunning: boolean;
  activeModule?: string;
  moduleElapsedSeconds?: number;
  modelTelemetry: JsonValue;
}

function pretty(value: JsonValue | undefined): string {
  if (value === undefined) return "No data yet";
  if (typeof value === "string") return value;
  return JSON.stringify(value, null, 2);
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
      {hasValue ? <pre>{pretty(value)}</pre> : <p className="empty-copy">{empty}</p>}
    </section>
  );
}

export function DecisionRail({
  reflection,
  plan,
  action,
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

  return (
    <aside className="decision-rail" aria-label="Agent decision pipeline">
      <div className="rail-heading">
        <div>
          <p className="eyebrow">Current cycle</p>
          <h2>Decision pipeline</h2>
        </div>
        <span className={`planner-chip ${plannerRunning ? "active" : ""}`}>
          {plannerRunning ? <Clock3 size={13} aria-hidden="true" /> : <BrainCircuit size={13} aria-hidden="true" />}
          {plannerRunning ? `${activeModule ?? "Planning"} · ${moduleElapsedSeconds ?? 0}s` : "Idle"}
        </span>
      </div>

      <div className="insight-stack">
        <InsightCard title="Model telemetry" icon={<Clock3 size={15} />} value={modelTelemetry} />
        <InsightCard title="Context prepared" icon={<Database size={15} />} value={context?.payload} />
        <InsightCard title="Reflection" icon={<BrainCircuit size={15} />} value={moduleOutput(reflection)} />
        <InsightCard title="Current plan" icon={<Workflow size={15} />} value={moduleOutput(plan)} />
        <InsightCard title="Candidate actions" icon={<Crosshair size={15} />} value={moduleOutput(action)} />
        <InsightCard title="Validation & dispatch" icon={<ShieldCheck size={15} />} value={decision?.payload} />
        <InsightCard
          title="Execution & effect"
          icon={executionTone === "success" ? <CheckCircle2 size={15} /> : executionTone === "danger" ? <XCircle size={15} /> : <Braces size={15} />}
          value={executionPayload}
          tone={executionTone}
        />
      </div>
    </aside>
  );
}
