import { Activity, ChevronRight, Crosshair, Eye, EyeOff, Filter, Hammer, Pause, Play, RotateCcw, TriangleAlert, Zap } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { actionName, commandId, eventCategory, eventMatches, isFailure } from "../data";
import { actionLabel, eventSummary, eventTitle } from "../presentation";
import type { EventCategory, StoredEvent } from "../types";

interface EventTimelineProps {
  events: StoredEvent[];
  onSelect: (eventId: number) => void;
}

const categories: { id: EventCategory; label: string; icon: React.ReactNode }[] = [
  { id: "all", label: "全部", icon: <Activity size={13} /> },
  { id: "planner", label: "规划", icon: <Filter size={13} /> },
  { id: "reflex", label: "快速反应", icon: <Zap size={13} /> },
  { id: "build", label: "建造", icon: <Hammer size={13} /> },
  { id: "combat", label: "战斗", icon: <Crosshair size={13} /> },
  { id: "failure", label: "失败", icon: <TriangleAlert size={13} /> },
];
const ROW_HEIGHT = 48;
const VIRTUAL_OVERSCAN = 12;

function formatEventTime(event: StoredEvent): string {
  if (!event.created_at) return `#${event.event_id}`;
  const date = new Date(event.created_at);
  return Number.isNaN(date.getTime()) ? `#${event.event_id}` : date.toLocaleTimeString([], { hour12: false });
}

export function EventTimeline({ events, onSelect }: EventTimelineProps) {
  const [category, setCategory] = useState<EventCategory>("all");
  const [showObservations, setShowObservations] = useState(false);
  const [followLive, setFollowLive] = useState(true);
  const [visibleRange, setVisibleRange] = useState({ start: 0, end: 40 });
  const scrollRef = useRef<HTMLDivElement>(null);

  const visibleEvents = useMemo(
    () => events.filter((event) => eventMatches(event, category, showObservations)),
    [category, events, showObservations],
  );
  const updateVisibleRange = useCallback(() => {
    const element = scrollRef.current;
    if (!element) return;
    const firstVisible = Math.floor(element.scrollTop / ROW_HEIGHT);
    const visibleCount = Math.ceil(element.clientHeight / ROW_HEIGHT);
    setVisibleRange({
      start: Math.max(0, firstVisible - VIRTUAL_OVERSCAN),
      end: Math.min(visibleEvents.length, firstVisible + visibleCount + VIRTUAL_OVERSCAN),
    });
  }, [visibleEvents.length]);

  useEffect(() => {
    if (followLive && visibleEvents.length > 0) {
      scrollRef.current?.scrollTo({ top: visibleEvents.length * ROW_HEIGHT });
    }
    const frame = window.requestAnimationFrame(updateVisibleRange);
    return () => window.cancelAnimationFrame(frame);
  }, [followLive, updateVisibleRange, visibleEvents.length]);

  return (
    <section className="timeline-panel" aria-label="决策事件时间线">
      <div className="timeline-toolbar">
        <div className="timeline-title">
          <p className="eyebrow">观察 → 计划 → 动作 → 效果</p>
          <h2>决策时间线</h2>
        </div>
        <div className="timeline-actions">
          <button
            type="button"
            className="text-button"
            onClick={() => setShowObservations((value) => !value)}
            aria-pressed={showObservations}
          >
            {showObservations ? <Eye size={14} aria-hidden="true" /> : <EyeOff size={14} aria-hidden="true" />}
            显示观察
          </button>
          <button
            type="button"
            className={`text-button ${followLive ? "is-active" : ""}`}
            onClick={() => setFollowLive((value) => !value)}
            aria-pressed={followLive}
          >
            {followLive ? <Pause size={14} aria-hidden="true" /> : <Play size={14} aria-hidden="true" />}
            {followLive ? "暂停跟随" : "继续跟随"}
          </button>
          {!followLive && (
            <button type="button" className="icon-button" onClick={() => setFollowLive(true)} aria-label="跳到最新事件">
              <RotateCcw size={15} aria-hidden="true" />
            </button>
          )}
        </div>
      </div>

      <div className="filter-row" role="toolbar" aria-label="筛选时间线事件">
        {categories.map((item) => (
          <button
            type="button"
            key={item.id}
            className={`filter-chip ${category === item.id ? "selected" : ""}`}
            onClick={() => setCategory(item.id)}
            aria-pressed={category === item.id}
          >
            <span aria-hidden="true">{item.icon}</span>
            {item.label}
          </button>
        ))}
        <span className="event-count mono" aria-live="polite">{visibleEvents.length.toLocaleString()} 条事件</span>
      </div>

      <div className="timeline-scroll" ref={scrollRef} tabIndex={0} aria-label="Runtime 事件" onScroll={updateVisibleRange}>
        {visibleEvents.length === 0 ? (
          <div className="timeline-empty">当前筛选条件下没有事件。</div>
        ) : (
          <div className="virtual-event-list" style={{ height: `${visibleEvents.length * ROW_HEIGHT}px` }}>
            {visibleEvents.slice(visibleRange.start, visibleRange.end).map((event, offset) => {
              const index = visibleRange.start + offset;
              if (!event) return null;
              const categoryName = eventCategory(event);
              const failed = isFailure(event);
              const action = actionName(event);
              return (
                <button
                  type="button"
                  className={`event-row category-${categoryName} ${failed ? "event-failed" : ""}`}
                  key={event.event_id}
                  style={{ transform: `translateY(${index * ROW_HEIGHT}px)` }}
                  onClick={() => onSelect(event.event_id)}
                  aria-label={`${eventTitle(event)}，${eventSummary(event)}，loop ${event.game_loop ?? "未知"}`}
                  aria-posinset={index + 1}
                  aria-setsize={visibleEvents.length}
                >
                  <span className="event-rail" aria-hidden="true" />
                  <span className="event-time mono">{formatEventTime(event)}</span>
                  <span className="event-loop mono">L{event.game_loop ?? "—"}</span>
                  <span className="event-content">
                    <span className="event-type">{eventTitle(event)}</span>
                    <span className="event-summary">{eventSummary(event)}</span>
                  </span>
                  {(action || commandId(event)) && (
                    <span className="event-tag mono">{action ? actionLabel(action) : commandId(event)?.slice(-18)}</span>
                  )}
                  <ChevronRight className="event-chevron" size={15} aria-hidden="true" />
                </button>
              );
            })}
          </div>
        )}
      </div>
    </section>
  );
}
