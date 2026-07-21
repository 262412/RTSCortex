"""Deterministic, structure-aware prompt context compaction."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from rtscortex.contracts import ActionBatch, ExecutionReport, ObservationEnvelope
from rtscortex.contracts.interfaces import ActivePlanSnapshot
from rtscortex.progress import GoalProgressReport


class ContextBudgetExceeded(ValueError):
    """Raised when mandatory current-state context cannot fit the configured budget."""


@dataclass(frozen=True)
class ContextBudget:
    """Character budget and bounded history sizes for one complete model prompt."""

    max_prompt_chars: int = 9_000
    max_recent_events: int = 8
    max_lessons: int = 6
    max_episode_summaries: int = 1

    def __post_init__(self) -> None:
        if self.max_prompt_chars < 2_000:
            raise ValueError("max_prompt_chars must be at least 2000")
        for name in ("max_recent_events", "max_lessons"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1")
        if self.max_episode_summaries < 0:
            raise ValueError("max_episode_summaries must be non-negative")


@dataclass(frozen=True)
class PromptContext:
    user_prompt: str
    statistics: dict[str, int | bool]


def compact_spatial_context(text_observation: str) -> list[str]:
    """Retain actionable screen/minimap coordinates without upstream prompt bulk."""

    lines: list[str] = []
    for raw_line in text_observation.splitlines():
        line = raw_line.strip()
        if (
            (line.startswith("[") and line.endswith("]"))
            or (line.startswith("Team ") and line.endswith(" Info:"))
            or "Team minimap position:" in line
            or "ScreenPos:" in line
            or (line.startswith("Build_") and " candidates:" in line)
        ):
            lines.append(line)
            if len(lines) == 48:
                break
    return lines


def model_observation(observation: ObservationEnvelope) -> tuple[dict[str, Any], dict[str, int]]:
    """Project current state while aggregating unit lists that grow throughout a match."""

    payload = observation.model_dump(mode="json")
    payload.pop("observed_at")
    payload.pop("text_observation")
    payload.pop("image_uri")
    spatial_context = compact_spatial_context(observation.text_observation)
    if spatial_context:
        payload["spatial_context"] = spatial_context

    state = payload["state"]
    own_units = state["own_units"]
    own_structures = state["own_structures"]
    visible_enemies = state["visible_enemies"]
    state["own_units"], own_groups = _aggregate_units(own_units, limit=12)
    state["own_structures"], structure_groups = _aggregate_units(own_structures, limit=16)
    state["visible_enemies"], enemy_groups = _aggregate_units(visible_enemies, limit=16)
    if own_groups:
        state["own_unit_groups"] = own_groups
    if structure_groups:
        state["own_structure_groups"] = structure_groups
    if enemy_groups:
        state["visible_enemy_groups"] = enemy_groups
    return payload, {
        "aggregated_own_units": len(own_units) - len(state["own_units"]),
        "aggregated_own_structures": len(own_structures) - len(state["own_structures"]),
        "aggregated_visible_enemies": len(visible_enemies) - len(state["visible_enemies"]),
    }


def compact_memory_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse semantically identical history entries and keep their latest position."""

    collapsed: dict[str, tuple[int, dict[str, Any]]] = {}
    for index, raw_event in enumerate(events):
        event = _bounded_event(raw_event)
        repeats = max(1, int(raw_event.get("repeat_count", 1)))
        first_step = int(raw_event.get("first_step_id", event.get("step_id", 0)))
        ignored_signature_keys = {"step_id", "first_step_id", "repeat_count"}
        if event.get("event_type") == "execution":
            ignored_signature_keys.add("command_id")
        signature_payload = {
            key: value for key, value in event.items() if key not in ignored_signature_keys
        }
        signature = json.dumps(signature_payload, ensure_ascii=False, sort_keys=True)
        previous = collapsed.get(signature)
        if previous is None:
            event["first_step_id"] = first_step
            event["repeat_count"] = repeats
        else:
            previous_event = previous[1]
            event["first_step_id"] = int(previous_event["first_step_id"])
            event["repeat_count"] = int(previous_event["repeat_count"]) + repeats
        collapsed[signature] = (index, event)

    result: list[dict[str, Any]] = []
    for _, event in sorted(collapsed.values(), key=lambda item: item[0]):
        if event["repeat_count"] == 1:
            event.pop("first_step_id")
            event.pop("repeat_count")
        result.append(event)
    return result


def compact_execution_payload(
    execution: ExecutionReport | dict[str, Any],
) -> dict[str, Any]:
    """Retain actionable execution provenance without primitive-level prompt bulk."""

    raw = execution.model_dump(mode="json") if isinstance(execution, ExecutionReport) else execution
    payload: dict[str, Any] = {}
    for key in (
        "protocol_version",
        "step_id",
        "command_id",
        "success",
        "action_name",
        "actor",
        "source",
        "status",
        "execution_stage",
        "failure_code",
        "pysc2_function",
    ):
        if key in raw:
            payload[key] = _bounded_json_value(raw[key])

    if "failure_reason" in raw:
        reason = raw["failure_reason"]
        payload["failure_reason"] = None if reason is None else _bounded_text(str(reason), 240)
    for key in ("requested_arguments", "resolved_arguments"):
        if key in raw:
            payload[key] = _bounded_json_value(raw[key])

    raw_evidence = raw.get("effect_evidence")
    if raw_evidence is not None:
        if hasattr(raw_evidence, "model_dump"):
            raw_evidence = raw_evidence.model_dump(mode="json")
        if isinstance(raw_evidence, dict):
            evidence: dict[str, Any] = {}
            for key in (
                "target_type",
                "target_position",
                "target_tag",
                "builder_tag",
                "actor_tag",
                "new_structure_tag",
                "dispatch_game_loop",
                "accepted_game_loop",
                "confirmed_game_loop",
                "worker_orders",
                "order_seen",
                "order_last_seen_game_loop",
                "post_order_grace_game_loops",
                "mineral_delta",
                "resource_delta",
                "elapsed_game_loops",
                "base_timeout_game_loops",
                "effective_timeout_game_loops",
                "active_order_extension",
            ):
                if key in raw_evidence and (key != "actor_tag" or raw_evidence[key] is not None):
                    evidence[key] = _bounded_json_value(raw_evidence[key])
            payload["effect_evidence"] = evidence
    return payload


def build_planning_context(
    *,
    observation: ObservationEnvelope,
    memory: dict[str, Any],
    active_plan: ActivePlanSnapshot | None,
    last_execution: ExecutionReport | None,
    goal_progress: GoalProgressReport | None,
    budget: ContextBudget,
    system_prompt: str,
) -> PromptContext:
    raw_payload: dict[str, Any] = {
        "observation": _raw_model_observation(observation),
        "memory": memory,
    }
    if active_plan is not None:
        raw_payload["active_plan"] = _compact_active_plan(active_plan)
    if last_execution is not None:
        raw_payload["last_execution"] = last_execution.model_dump(mode="json")
    if goal_progress is not None:
        raw_payload["goal_progress"] = goal_progress.model_dump(mode="json")

    projected_observation, unit_stats = model_observation(observation)
    raw_events = list(memory.get("recent_events", []))
    event_count = sum(max(1, int(event.get("repeat_count", 1))) for event in raw_events)
    recent_events = compact_memory_events(raw_events)[-budget.max_recent_events :]
    lessons = (
        _latest_unique_lessons(list(memory.get("lessons", [])))[-budget.max_lessons :]
        if budget.max_lessons
        else []
    )
    summaries = (
        list(memory.get("episode_summaries", []))[-budget.max_episode_summaries :]
        if budget.max_episode_summaries
        else []
    )
    compact_memory: dict[str, Any] = {
        "recent_events": recent_events,
        "lessons": lessons,
        "episode_summaries": summaries,
    }
    reflection = memory.get("reflection")
    if reflection:
        compact_memory["reflection"] = _bounded_text(str(reflection), 500)

    payload: dict[str, Any] = {
        "observation": projected_observation,
        "memory": compact_memory,
    }
    if active_plan is not None:
        payload["active_plan"] = _compact_active_plan(active_plan)
    if last_execution is not None:
        payload["last_execution"] = _compact_execution(last_execution)
    if goal_progress is not None:
        payload["goal_progress"] = goal_progress.model_dump(mode="json")
    statistics: dict[str, int | bool] = {
        "budget_chars": budget.max_prompt_chars,
        "original_chars": _prompt_chars(system_prompt, raw_payload),
        "final_chars": 0,
        "retained_observations": 1,
        "retained_recent_events": len(recent_events),
        "retained_lessons": len(lessons),
        "retained_episode_summaries": len(summaries),
        "dropped_recent_events": event_count - len(recent_events),
        "dropped_lessons": len(list(memory.get("lessons", []))) - len(lessons),
        "dropped_episode_summaries": len(list(memory.get("episode_summaries", [])))
        - len(summaries),
        **unit_stats,
        "compacted": False,
    }
    return _fit_prompt(payload, statistics, budget, system_prompt)


def build_reflection_context(
    *,
    observation: ObservationEnvelope,
    last_decision: ActionBatch,
    last_execution: ExecutionReport | None,
    goal_progress: GoalProgressReport | None,
    budget: ContextBudget,
    system_prompt: str,
) -> PromptContext:
    raw_payload = {
        "observation": _raw_model_observation(observation),
        "last_decision": last_decision.model_dump(mode="json"),
        "last_execution": (
            None if last_execution is None else last_execution.model_dump(mode="json")
        ),
    }
    if goal_progress is not None:
        raw_payload["goal_progress"] = goal_progress.model_dump(mode="json")
    projected_observation, unit_stats = model_observation(observation)
    payload = {
        "observation": projected_observation,
        "last_decision": _compact_decision(last_decision, include_rejections=True),
        "last_execution": (None if last_execution is None else _compact_execution(last_execution)),
    }
    if goal_progress is not None:
        payload["goal_progress"] = goal_progress.model_dump(mode="json")
    statistics: dict[str, int | bool] = {
        "budget_chars": budget.max_prompt_chars,
        "original_chars": _prompt_chars(system_prompt, raw_payload),
        "final_chars": 0,
        "retained_observations": 1,
        "retained_recent_events": 0,
        "retained_lessons": 0,
        "retained_episode_summaries": 0,
        "dropped_recent_events": 0,
        "dropped_lessons": 0,
        "dropped_episode_summaries": 0,
        **unit_stats,
        "compacted": False,
    }
    return _fit_prompt(payload, statistics, budget, system_prompt)


def _fit_prompt(
    payload: dict[str, Any],
    statistics: dict[str, int | bool],
    budget: ContextBudget,
    system_prompt: str,
) -> PromptContext:
    memory = payload.get("memory", {})
    droppable_lists = [
        (memory.get("episode_summaries", []), "dropped_episode_summaries", False),
        (memory.get("recent_events", []), "dropped_recent_events", False),
        (memory.get("lessons", []), "dropped_lessons", False),
    ]
    state = payload["observation"]["state"]
    droppable_lists.extend(
        [
            (state["own_units"], "aggregated_own_units", True),
            (state["visible_enemies"], "aggregated_visible_enemies", True),
            (state["own_structures"], "aggregated_own_structures", True),
            (
                payload["observation"].get("spatial_context", []),
                "dropped_spatial_lines",
                False,
            ),
        ]
    )
    statistics.setdefault("dropped_spatial_lines", 0)
    while _rendered_prompt_chars(payload, statistics, system_prompt) > budget.max_prompt_chars:
        dropped = False
        for items, statistic_name, drop_from_end in droppable_lists:
            minimum = (
                1
                if statistic_name
                in {
                    "aggregated_own_units",
                    "aggregated_own_structures",
                    "aggregated_visible_enemies",
                }
                and items
                else 0
            )
            if len(items) > minimum:
                items.pop() if drop_from_end else items.pop(0)
                statistics[statistic_name] = int(statistics[statistic_name]) + 1
                dropped = True
                break
        if not dropped:
            mandatory_chars = _rendered_prompt_chars(payload, statistics, system_prompt)
            raise ContextBudgetExceeded(
                "mandatory observation and action schema require "
                f"{mandatory_chars} characters, exceeding max_prompt_chars="
                f"{budget.max_prompt_chars}"
            )

    statistics["compacted"] = int(statistics["original_chars"]) > _prompt_chars(
        system_prompt, payload
    ) or any(
        int(statistics[name]) > 0
        for name in (
            "dropped_recent_events",
            "dropped_lessons",
            "dropped_episode_summaries",
            "aggregated_own_units",
            "aggregated_own_structures",
            "aggregated_visible_enemies",
            "dropped_spatial_lines",
        )
    )
    user_prompt = _render_with_stable_final_size(payload, statistics, system_prompt)
    if len(system_prompt) + len(user_prompt) > budget.max_prompt_chars:
        raise ContextBudgetExceeded("context statistics exceeded the configured prompt budget")
    return PromptContext(user_prompt=user_prompt, statistics=dict(statistics))


def _rendered_prompt_chars(
    payload: dict[str, Any], statistics: dict[str, int | bool], system_prompt: str
) -> int:
    probe = dict(statistics)
    probe["final_chars"] = statistics["budget_chars"]
    rendered_payload = {**payload, "context_compaction": probe}
    return _prompt_chars(system_prompt, rendered_payload)


def _render_with_stable_final_size(
    payload: dict[str, Any], statistics: dict[str, int | bool], system_prompt: str
) -> str:
    user_prompt = ""
    for _ in range(4):
        rendered_payload = {**payload, "context_compaction": statistics}
        user_prompt = json.dumps(rendered_payload, ensure_ascii=False, sort_keys=True)
        final_chars = len(system_prompt) + len(user_prompt)
        if statistics["final_chars"] == final_chars:
            return user_prompt
        statistics["final_chars"] = final_chars
    return json.dumps(
        {**payload, "context_compaction": statistics}, ensure_ascii=False, sort_keys=True
    )


def _raw_model_observation(observation: ObservationEnvelope) -> dict[str, Any]:
    payload = observation.model_dump(mode="json")
    payload.pop("observed_at")
    payload.pop("text_observation")
    payload.pop("image_uri")
    spatial_context = compact_spatial_context(observation.text_observation)
    if spatial_context:
        payload["spatial_context"] = spatial_context
    return payload


def _aggregate_units(
    units: list[dict[str, Any]], *, limit: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for unit in units:
        groups[str(unit["unit_type"])].append(unit)
    summaries = []
    for unit_type in sorted(groups):
        members = groups[unit_type]
        health = [float(member["health_fraction"]) for member in members]
        positions = [member["position"] for member in members if member["position"] is not None]
        summaries.append(
            {
                "unit_type": unit_type,
                "count": len(members),
                "min_health_fraction": min(health),
                "average_health_fraction": round(sum(health) / len(health), 4),
                "sample_positions": positions[:2],
            }
        )
    if len(units) <= limit:
        return units, []

    urgent = sorted(
        (unit for unit in units if float(unit["health_fraction"]) < 0.35),
        key=lambda unit: (float(unit["health_fraction"]), str(unit["unit_id"])),
    )
    representatives = [groups[unit_type][0] for unit_type in sorted(groups)]
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for unit in [*urgent, *representatives, *units]:
        unit_id = str(unit["unit_id"])
        if unit_id in selected_ids:
            continue
        selected.append(unit)
        selected_ids.add(unit_id)
        if len(selected) == limit:
            break
    return selected, summaries


def _latest_unique_lessons(lessons: list[Any]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_lesson in reversed(lessons):
        if isinstance(raw_lesson, dict):
            content = str(raw_lesson.get("content", "")).strip()
            source_step_id = raw_lesson.get("source_step_id")
        else:
            content = str(raw_lesson).strip()
            source_step_id = None
        if not content or content in seen:
            continue
        lesson: dict[str, Any] = {"content": _bounded_text(content, 300)}
        if source_step_id is not None:
            lesson["source_step_id"] = source_step_id
        selected.append(lesson)
        seen.add(content)
    selected.reverse()
    return selected


def _compact_decision(decision: ActionBatch, *, include_rejections: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "step_id": decision.step_id,
        "strategic_goal": _bounded_text(decision.strategic_goal, 240),
        "summary": _bounded_text(decision.summary, 500),
        "commands": [
            {
                "command_id": command.command_id,
                "actor": command.actor,
                "name": command.name,
                "arguments": command.arguments,
                "source": command.source.value,
                "priority": command.priority,
                "ttl_game_loops": command.ttl_game_loops,
                "created_game_loop": command.created_game_loop,
            }
            for command in decision.commands
        ],
    }
    if include_rejections:
        payload["rejected_commands"] = [
            _bounded_text(reason, 180) for reason in decision.rejected_commands[-3:]
        ]
    return payload


def _compact_active_plan(active_plan: ActivePlanSnapshot) -> dict[str, Any]:
    return {
        "strategic_goal": _bounded_text(active_plan.strategic_goal, 240),
        "summary": _bounded_text(active_plan.summary, 500),
        "commands": [
            {
                "command_id": command.command_id,
                "actor": command.actor,
                "name": command.name,
                "arguments": list(command.arguments),
                "source": command.source,
                "status": command.status,
                "reason": (None if command.reason is None else _bounded_text(command.reason, 180)),
                "created_game_loop": command.created_game_loop,
                "ttl_game_loops": command.ttl_game_loops,
                "expires_at_game_loop": command.expires_at_game_loop,
            }
            for command in active_plan.commands
        ],
    }


def _compact_execution(execution: ExecutionReport) -> dict[str, Any]:
    return compact_execution_payload(execution)


def _bounded_event(event: dict[str, Any]) -> dict[str, Any]:
    if event.get("event_type") == "execution":
        return {"event_type": "execution", **compact_execution_payload(event)}

    bounded: dict[str, Any] = {}
    for key in (
        "event_type",
        "step_id",
        "strategic_goal",
        "summary",
        "commands",
        "rejected_commands",
        "command_id",
        "success",
        "failure_reason",
        "pysc2_function",
        "error_type",
        "message",
        "module",
    ):
        if key not in event:
            continue
        value = event[key]
        bounded[key] = _bounded_text(value, 300) if isinstance(value, str) else value
    return bounded


def _bounded_json_value(value: Any, *, depth: int = 0) -> Any:
    if isinstance(value, str):
        return _bounded_text(value, 160)
    if depth >= 3:
        return _bounded_text(str(value), 160)
    if isinstance(value, (list, tuple)):
        return [_bounded_json_value(item, depth=depth + 1) for item in value[:8]]
    if isinstance(value, dict):
        return {
            _bounded_text(str(key), 80): _bounded_json_value(item, depth=depth + 1)
            for key, item in list(value.items())[:8]
        }
    return value


def _bounded_text(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 1] + "…"


def _prompt_chars(system_prompt: str, payload: dict[str, Any]) -> int:
    return len(system_prompt) + len(json.dumps(payload, ensure_ascii=False, sort_keys=True))
