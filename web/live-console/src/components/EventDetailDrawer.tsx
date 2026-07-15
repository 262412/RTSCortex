import { Braces, Copy, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { actionName, commandIds, eventCategory, eventSummary } from "../data";
import type { StoredEvent } from "../types";

interface EventDetailDrawerProps {
  event: StoredEvent | null;
  events: StoredEvent[];
  onClose: () => void;
  onSelect: (eventId: number) => void;
}

export function EventDetailDrawer({ event, events, onClose, onSelect }: EventDetailDrawerProps) {
  const closeButtonRef = useRef<HTMLButtonElement>(null);
  const [copiedEventId, setCopiedEventId] = useState<number | null>(null);

  useEffect(() => {
    if (!event) return;
    closeButtonRef.current?.focus();
    const onKeyDown = (keyEvent: KeyboardEvent) => {
      if (keyEvent.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
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
      <aside className="detail-drawer" role="dialog" aria-modal="true" aria-labelledby="event-detail-title">
        <header className="drawer-header">
          <div>
            <p className="eyebrow">Event #{event.event_id}</p>
            <h2 id="event-detail-title">{event.event_type.replaceAll("_", " ")}</h2>
          </div>
          <button ref={closeButtonRef} type="button" className="icon-button" onClick={onClose} aria-label="Close event details">
            <X size={18} aria-hidden="true" />
          </button>
        </header>
        <div className="drawer-badges">
          <span>{eventCategory(event)}</span>
          <span className="mono">loop {event.game_loop ?? "—"}</span>
          {actionName(event) && <span className="mono">{actionName(event)}</span>}
        </div>
        {selectedCommandIds.size > 0 && (
          <div className="command-box">
            <span>Command ID{selectedCommandIds.size > 1 ? "s" : ""}</span>
            {[...selectedCommandIds].map((id) => <code key={id}>{id}</code>)}
          </div>
        )}
        {relatedEvents.length > 0 && (
          <section className="lifecycle-section" aria-labelledby="command-lifecycle-title">
            <div className="json-heading">
              <span id="command-lifecycle-title">Command lifecycle</span>
              <span className="mono">{relatedEvents.length} events</span>
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
                  <strong>{related.event_type.replaceAll("_", " ")}</strong>
                  <small>{eventSummary(related)}</small>
                </button>
              ))}
            </div>
          </section>
        )}
        <div className="json-heading">
          <span><Braces size={14} aria-hidden="true" /> Structured event</span>
          <button type="button" className="text-button" onClick={() => void copy()}>
            <Copy size={14} aria-hidden="true" /> {copied ? "Copied" : "Copy JSON"}
          </button>
        </div>
        <pre className="json-detail">{serialized}</pre>
      </aside>
    </div>
  );
}
