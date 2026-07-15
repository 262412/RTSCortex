import { actionLabel, actorLabel, fieldLabel, isTechnicalField, semanticScalar } from "../presentation";
import type { JsonObject, JsonValue } from "../types";

interface SemanticValueProps {
  value: JsonValue | undefined;
  field?: string;
  compact?: boolean;
  depth?: number;
}

function asObject(value: JsonValue): JsonObject | undefined {
  return typeof value === "object" && value !== null && !Array.isArray(value) ? value : undefined;
}

function commandHeading(value: JsonObject): { action: string; actor?: string } | undefined {
  const action = [value.action_name, value.action, value.name].find((candidate) => typeof candidate === "string");
  if (typeof action !== "string") return undefined;
  return {
    action: actionLabel(action),
    actor: typeof value.actor === "string" ? actorLabel(value.actor) : undefined,
  };
}

function ObjectValue({ value, compact, depth }: { value: JsonObject; compact: boolean; depth: number }) {
  const command = commandHeading(value);
  const hiddenKeys = command ? new Set(["action_name", "action", "name", "actor"]) : new Set<string>();
  const entries = Object.entries(value).filter(([key, nested]) => !hiddenKeys.has(key) && nested !== "");
  return (
    <div className={`semantic-object ${command ? "semantic-command" : ""}`}>
      {command && (
        <div className="semantic-command-heading">
          <strong>{command.action}</strong>
          {command.actor && <span>{command.actor}</span>}
        </div>
      )}
      {entries.length === 0 ? (
        <span className="semantic-empty">无额外信息</span>
      ) : (
        <dl className={compact ? "semantic-fields compact" : "semantic-fields"}>
          {entries.map(([key, nested]) => (
            <div className="semantic-field" key={key}>
              <dt>{fieldLabel(key)}</dt>
              <dd className={isTechnicalField(key) ? "semantic-technical" : undefined}>
                <SemanticValue value={nested} field={key} compact={compact} depth={depth + 1} />
              </dd>
            </div>
          ))}
        </dl>
      )}
    </div>
  );
}

export function SemanticValue({ value, field, compact = false, depth = 0 }: SemanticValueProps) {
  if (value === undefined || value === null) return <span className="semantic-empty">无</span>;
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
    const content = semanticScalar(value, field);
    const longText = typeof value === "string" && (value.length > 100 || value.includes("\n"));
    return longText ? <p className="semantic-prose">{content}</p> : <span>{content}</span>;
  }
  if (Array.isArray(value)) {
    if (value.length === 0) return <span className="semantic-empty">无</span>;
    return (
      <ol className={`semantic-list ${depth > 1 ? "nested" : ""}`}>
        {value.map((item, index) => (
          <li key={`${field ?? "item"}-${index}`}>
            <SemanticValue value={item} compact={compact} depth={depth + 1} />
          </li>
        ))}
      </ol>
    );
  }
  const object = asObject(value);
  return object ? <ObjectValue value={object} compact={compact} depth={depth} /> : null;
}
