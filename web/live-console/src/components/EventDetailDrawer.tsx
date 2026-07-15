import { Braces, Copy, Languages, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { actionName, commandIds, eventCategory } from "../data";
import { categoryLabel, eventSemanticPayload, eventSummary, eventTitle } from "../presentation";
import type { StoredEvent } from "../types";
import { SemanticValue } from "./SemanticValue";

interface EventDetailDrawerProps {
  event: StoredEvent | null;
  events: StoredEvent[];
  onClose: () => void;
  onSelect: (eventId: number) => void;
}

export function EventDetailDrawer({ event, events, onClose, onSelect }: EventDetailDrawerProps) {
  const closeButtonRef = useRef<HTMLButtonElement>(null);
  const drawerRef = useRef<HTMLElement>(null);
  const [copiedEventId, setCopiedEventId] = useState<number | null>(null);

  useEffect(() => {
    if (!event) return;
    const previousFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    closeButtonRef.current?.focus();
    const onKeyDown = (keyEvent: KeyboardEvent) => {
      if (keyEvent.key === "Escape") onClose();
      if (keyEvent.key !== "Tab") return;
      const focusable = drawerRef.current?.querySelectorAll<HTMLElement>(
        "button, summary, [href], input, select, textarea, [tabindex]:not([tabindex='-1'])",
      );
      if (!focusable || focusable.length === 0) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (keyEvent.shiftKey && document.activeElement === first) {
        keyEvent.preventDefault();
        last?.focus();
      } else if (!keyEvent.shiftKey && document.activeElement === last) {
        keyEvent.preventDefault();
        first?.focus();
      }
    };
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("keydown", onKeyDown);
      previousFocus?.focus();
    };
  }, [event, onClose]);

  if (!event) return null;
  const selectedCommandIds = new Set(commandIds(event));
  const relatedEvents =
    selectedCommandIds.size === 0
      ? []
      : events.filter((candidate) =>
          commandIds(candidate).some((id) => selectedCommandIds.has(id)),
        );
  const serialized = JSON.stringify(event, null, 2);
  const copied = copiedEventId === event.event_id;
  const copy = async () => {
    await navigator.clipboard.writeText(serialized);
    setCopiedEventId(event.event_id);
  };

  return (
    <div className="drawer-layer" role="presentation" onMouseDown={(mouseEvent) => mouseEvent.target === mouseEvent.currentTarget && onClose()}>
      <aside ref={drawerRef} className="detail-drawer" role="dialog" aria-modal="true" aria-labelledby="event-detail-title">
        <header className="drawer-header">
          <div>
            <p className="eyebrow">事件 #{event.event_id} · {event.event_type}</p>
            <h2 id="event-detail-title">{eventTitle(event)}</h2>
          </div>
          <button ref={closeButtonRef} type="button" className="icon-button" onClick={onClose} aria-label="关闭事件详情">
            <X size={18} aria-hidden="true" />
          </button>
        </header>
        <div className="drawer-badges">
          <span>{categoryLabel(eventCategory(event))}</span>
          <span className="mono">loop {event.game_loop ?? "—"}</span>
          {actionName(event) && <span className="mono">{actionName(event)}</span>}
        </div>
        <section className="semantic-event" aria-labelledby="semantic-event-title">
          <div className="json-heading">
            <span id="semantic-event-title"><Languages size={14} aria-hidden="true" /> 可读解释</span>
          </div>
          <p className="semantic-event-summary">{eventSummary(event)}</p>
          <SemanticValue value={eventSemanticPayload(event)} />
        </section>
        {selectedCommandIds.size > 0 && (
          <div className="command-box">
            <span>动作 ID{selectedCommandIds.size > 1 ? "（多个）" : ""}</span>
            {[...selectedCommandIds].map((id) => <code key={id}>{id}</code>)}
          </div>
        )}
        {relatedEvents.length > 0 && (
          <section className="lifecycle-section" aria-labelledby="command-lifecycle-title">
            <div className="json-heading">
              <span id="command-lifecycle-title">动作完整生命周期</span>
              <span className="mono">{relatedEvents.length} 条事件</span>
            </div>
            <div className="lifecycle-list">
              {relatedEvents.map((related) => (
                <button
                  type="button"
                  key={related.event_id}
                  className={related.event_id === event.event_id ? "lifecycle-row selected" : "lifecycle-row"}
                  onClick={() => onSelect(related.event_id)}
                >
                  <span className="mono">#{related.event_id} · L{related.game_loop ?? "—"}</span>
                  <strong>{eventTitle(related)}</strong>
                  <small>{eventSummary(related)}</small>
                </button>
              ))}
            </div>
          </section>
        )}
        <details className="raw-json-disclosure">
          <summary><Braces size={14} aria-hidden="true" /> 技术详情：查看原始 JSON</summary>
          <div className="json-heading">
            <span>协议原始事件</span>
            <button type="button" className="text-button" onClick={() => void copy()}>
              <Copy size={14} aria-hidden="true" /> {copied ? "已复制" : "复制 JSON"}
            </button>
          </div>
          <pre className="json-detail" tabIndex={0} aria-label="原始事件 JSON">{serialized}</pre>
        </details>
        <span className="sr-only" aria-live="polite">{copied ? "原始 JSON 已复制" : ""}</span>
      </aside>
    </div>
  );
}
