"""Human-readable timelines derived from append-only runtime event journals."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from rtscortex.agents.models import PlanningOutput
from rtscortex.contracts import (
    ActionBatch,
    ActionCommand,
    EpisodeResult,
    EpisodeSummary,
    ExecutionReport,
    ObservationEnvelope,
    UnitState,
)
from rtscortex.evaluation.metrics import compute_episode_metrics
from rtscortex.memory import StoredEvent, read_event_log

REPORT_FILENAME = "timeline.md"
ModelT = TypeVar("ModelT", bound=BaseModel)


class ReportError(ValueError):
    """Raised when a run directory cannot produce a trustworthy timeline."""


def write_timeline_report(run_dir: Path) -> Path:
    """Render ``events.jsonl`` in a run directory to ``timeline.md``."""

    resolved_run_dir = run_dir.expanduser().resolve()
    if not resolved_run_dir.is_dir():
        raise ReportError(f"Run directory does not exist or is not a directory: {resolved_run_dir}")

    journal_path = resolved_run_dir / "events.jsonl"
    if not journal_path.is_file():
        raise ReportError(f"Run directory does not contain events.jsonl: {resolved_run_dir}")
    try:
        events = list(read_event_log(journal_path))
    except (json.JSONDecodeError, TypeError, UnicodeDecodeError) as error:
        raise ReportError(f"Invalid event journal {journal_path}: {error}") from error
    except OSError as error:
        raise ReportError(f"Could not read event journal {journal_path}: {error}") from error
    if not events:
        raise ReportError(f"Event journal is empty: {journal_path}")

    report = render_timeline(events)
    output_path = resolved_run_dir / REPORT_FILENAME
    try:
        output_path.write_text(report, encoding="utf-8")
    except OSError as error:
        raise ReportError(f"Could not write timeline {output_path}: {error}") from error
    return output_path


def render_timeline(events: Sequence[StoredEvent]) -> str:
    """Render events in append order, grouped by run and episode."""

    if not events:
        raise ReportError("Event journal is empty")
    ordered = sorted(events, key=lambda event: event.event_id)
    episodes: dict[tuple[str, str], list[StoredEvent]] = {}
    for event in ordered:
        episodes.setdefault((event.run_id, event.episode_id), []).append(event)

    lines = [
        "# RTSCortex Run Timeline",
        "",
        "Raw event journal: [events.jsonl](events.jsonl)",
        "",
    ]
    for (run_id, episode_id), episode_events in episodes.items():
        lines.extend(_render_episode(run_id, episode_id, episode_events))
    return "\n".join(lines).rstrip() + "\n"


def _render_episode(
    run_id: str,
    episode_id: str,
    events: list[StoredEvent],
) -> list[str]:
    result = _last_model(events, "episode_result", EpisodeResult)
    summary = _last_model(events, "episode_summary", EpisodeSummary)
    observations = [event for event in events if event.event_type == "observation"]
    decisions = [event for event in events if event.event_type == "decision"]
    plans = [event for event in events if event.event_type == "plan_accepted"]
    executions = [event for event in events if event.event_type == "execution"]
    model_events = [
        event
        for event in events
        if event.event_type == "module_result" and event.payload.get("model_call") is True
    ]
    rejected = sum(len(_payload_list(event, "batch", "rejected_commands")) for event in decisions)
    successful_executions = sum(event.payload.get("success") is True for event in executions)
    total_tokens = sum(_total_tokens(event.payload.get("usage")) for event in model_events)

    scenario = result.scenario if result is not None else summary.scenario if summary else "unknown"
    seed = result.seed if result is not None else summary.seed if summary else None
    outcome = result.outcome.value if result is not None else "incomplete"
    score = _number(result.score) if result is not None else "n/a"
    steps = str(result.steps) if result is not None else "n/a"
    execution_rate = (
        f"{successful_executions}/{len(executions)} ({successful_executions / len(executions):.1%})"
        if executions
        else "0/0"
    )
    provider_models = _provider_models(model_events)

    lines = [
        f"## Episode {_code(episode_id)}",
        "",
        f"- Run: {_code(run_id)}",
        f"- Provider/model: {provider_models}",
        "",
        "### Result",
        "",
        "| Scenario | Outcome | Seed | Score | SC2 steps |",
        "|---|---|---:|---:|---:|",
        (
            f"| {_code(scenario)} | {_code(outcome)} | "
            f"{seed if seed is not None else 'n/a'} | {score} | {steps} |"
        ),
        "",
        "| Agent ticks | Decisions | Plans | Executions | Rejected | Model calls | Tokens |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| {len(observations)} | {len(decisions)} | {len(plans)} | {execution_rate} | "
            f"{rejected} | {len(model_events)} | {total_tokens} |"
        ),
    ]
    if result is not None:
        metrics = compute_episode_metrics(events, result)
        lines.extend(
            [
                "",
                (
                    "- Latency p50/p95: planner "
                    f"`{metrics.planner_latency_ms_p50:.2f}/{metrics.planner_latency_ms_p95:.2f} "
                    "ms`, reflex "
                    f"`{metrics.reflex_latency_ms_p50:.2f}/{metrics.reflex_latency_ms_p95:.2f} "
                    "ms`, tick "
                    f"`{metrics.tick_latency_ms_p50:.2f}/{metrics.tick_latency_ms_p95:.2f} ms`."
                ),
                (
                    f"- Plan revisions: `{metrics.plan_revisions}`; reflex preemptions: "
                    f"`{metrics.reflex_preemptions}`; duration: "
                    f"`{metrics.episode_duration_seconds:.2f} s`."
                ),
            ]
        )
    else:
        lines.extend(["", "- No terminal episode result was recorded; this run is incomplete."])

    lines.extend(["", "### Timeline", ""])
    timeline_started = False
    for event in events:
        if event.event_type == "observation":
            observation = _validate(event, ObservationEnvelope)
            lines.extend(_render_observation(event, observation))
            timeline_started = True
            continue
        if event.event_type == "episode_result":
            terminal = _validate(event, EpisodeResult)
            lines.extend(
                [
                    "",
                    f"#### Episode end · event {event.event_id}",
                    "",
                    (
                        f"- Outcome: {_code(terminal.outcome.value)}; score: "
                        f"`{_number(terminal.score)}`; SC2 steps: `{terminal.steps}`."
                    ),
                ]
            )
            if terminal.failure_reason:
                lines.append(f"- Failure reason: {_inline(terminal.failure_reason)}")
            timeline_started = True
            continue
        if not timeline_started:
            lines.extend(["#### Before first observation", ""])
            timeline_started = True
        lines.extend(_render_event(event))
    lines.append("")
    return lines


def _render_observation(event: StoredEvent, observation: ObservationEnvelope) -> list[str]:
    state = observation.state
    economy = state.economy
    lines = [
        "",
        (
            f"#### Tick {observation.step_id} · game loop {observation.game_loop} · "
            f"event {event.event_id}"
        ),
        "",
        (
            f"- State: own {_describe_units(state.own_units)}; structures "
            f"{_describe_units(state.own_structures)}; visible enemies "
            f"{_describe_units(state.visible_enemies)}."
        ),
        (
            f"- Economy: `{economy.minerals}` minerals, `{economy.vespene}` gas, "
            f"supply `{economy.supply_used}/{economy.supply_cap}`, "
            f"workers `{economy.workers}`, army supply `{economy.army_supply}`."
        ),
    ]
    if observation.alerts:
        lines.append(f"- Alerts: {', '.join(_code(alert) for alert in observation.alerts)}")
    if state.production_queue:
        production = ", ".join(
            f"{_code(item.name)} {item.progress:.0%}" for item in state.production_queue
        )
        lines.append(f"- Production: {production}")
    return lines


def _render_event(event: StoredEvent) -> list[str]:
    if event.event_type == "plan_accepted":
        return _render_plan(event)
    if event.event_type == "decision":
        return _render_decision(event)
    if event.event_type == "execution":
        return _render_execution(event)
    if event.event_type == "module_result":
        return _render_module_result(event)
    if event.event_type == "planner_cycle":
        status = _inline(event.payload.get("status", "unknown"))
        latency = _milliseconds(event.payload.get("latency_ms"))
        return [f"- Event {event.event_id} · Planner cycle: {status} · {latency}."]
    if event.event_type == "episode_summary":
        summary = _validate(event, EpisodeSummary)
        lines = [f"- Event {event.event_id} · Episode summary: {_inline(summary.summary)}"]
        lines.extend(f"  - Lesson: {_inline(lesson)}" for lesson in summary.lessons)
        return lines
    if event.event_type in {"planner_timeout", "planner_error", "module_error"}:
        details = _error_details(event.payload)
        return [
            f"- Event {event.event_id} · {_inline(event.event_type)}"
            f"{f': {details}' if details else '.'}"
        ]
    return [f"- Event {event.event_id} · {_code(event.event_type)} (payload not rendered)."]


def _render_plan(event: StoredEvent) -> list[str]:
    payload = event.payload
    goal = _inline(payload.get("strategic_goal", "")) or "none"
    summary = _inline(payload.get("summary", "")) or "none"
    source_step = payload.get("source_step_id", "unknown")
    revision = "revision" if payload.get("is_revision") is True else "accepted"
    lines = [
        (
            f"- Event {event.event_id} · Plan {revision} from tick `{source_step}`: "
            f"**{goal}** — {summary}."
        )
    ]
    if "plan_age_game_loops" in payload:
        lines.append(
            "  - Freshness: source loop "
            f"`{payload.get('source_game_loop', 'unknown')}`, accepted loop "
            f"`{payload.get('accepted_game_loop', 'unknown')}`, age "
            f"`{payload['plan_age_game_loops']}` loops."
        )
    for command in _commands(event, payload.get("commands", []), "commands"):
        lines.append(f"  - {_describe_command(command)}")
    return lines


def _render_decision(event: StoredEvent) -> list[str]:
    batch = _validate_nested(event, "batch", ActionBatch)
    payload = event.payload
    lines = [
        (
            f"- Event {event.event_id} · Decision: **"
            f"{_inline(batch.strategic_goal) or 'no strategic goal'}**"
            f"{f' — {_inline(batch.summary)}' if batch.summary else ''}."
        )
    ]
    for field, label in (
        ("planner_candidates", "Planner candidate"),
        ("reflex_candidates", "Reflex candidate"),
    ):
        if field in payload:
            for command in _commands(event, payload[field], field):
                lines.append(f"  - {label}: {_describe_command(command)}")
    if batch.commands:
        lines.extend(f"  - Selected: {_describe_command(command)}" for command in batch.commands)
    else:
        lines.append("  - Selected: none")
    lines.extend(f"  - Rejected: {_inline(reason)}" for reason in batch.rejected_commands)
    for preemption in payload.get("preemptions", []):
        if not isinstance(preemption, dict):
            raise ReportError(f"Invalid preemption payload at event {event.event_id}")
        lines.append(
            "  - Reflex preemption: "
            f"actor {_code(preemption.get('actor', 'unknown'))}; winner "
            f"{_code(preemption.get('winner_command_id', 'unknown'))}; loser "
            f"{_code(preemption.get('loser_command_id', 'unknown'))}."
        )
    lines.append(
        f"  - Latency: reflex {_milliseconds(payload.get('reflex_latency_ms'))}; "
        f"tick {_milliseconds(payload.get('tick_latency_ms'))}."
    )
    return lines


def _render_execution(event: StoredEvent) -> list[str]:
    report = _validate(event, ExecutionReport)
    status = "SUCCESS" if report.success else "FAILED"
    function = report.pysc2_function or "not reported"
    lines = [
        (
            f"- Event {event.event_id} · Execution **{status}** for "
            f"{_code(report.command_id)}: {_code(function)} · "
            f"{report.latency_ms:.2f} ms."
        )
    ]
    if report.failure_reason:
        lines.append(f"  - Failure reason: {_inline(report.failure_reason)}")
    if report.game_result:
        lines.append(f"  - Game result: {_code(report.game_result)}")
    return lines


def _render_module_result(event: StoredEvent) -> list[str]:
    payload = event.payload
    module = str(payload.get("module", "unknown"))
    model_call = payload.get("model_call") is True
    kind = "model call" if model_call else "module"
    provider = payload.get("provider")
    model = payload.get("model")
    identity = (
        f" · {_code(provider)}/{_code(model)}" if provider is not None or model is not None else ""
    )
    lines = [
        f"- Event {event.event_id} · {_inline(module).title()} {kind}{identity} · "
        f"{_milliseconds(payload.get('latency_ms'))}."
    ]
    if model_call:
        lines.append(f"  - Usage: `{_total_tokens(payload.get('usage'))}` total tokens.")
    output = payload.get("output")
    if output is None and module in {"reflection", "planning"}:
        lines.append("  - Structured output not recorded (legacy event).")
    elif output is not None:
        lines.extend(_render_module_output(event, module, output))
    return lines


def _render_module_output(event: StoredEvent, module: str, output: object) -> list[str]:
    if not isinstance(output, dict):
        raise ReportError(f"Invalid module output at event {event.event_id}")
    if module == "reflection":
        reflection = output.get("reflection")
        lines = [f"  - Reflection: {_inline(reflection) if reflection else 'none'}"]
        lessons = output.get("lessons", [])
        if not isinstance(lessons, list):
            raise ReportError(f"Invalid reflection lessons at event {event.event_id}")
        lines.extend(f"  - Lesson: {_inline(lesson)}" for lesson in lessons)
        return lines
    if module == "planning":
        raw_plan = output.get("plan")
        if raw_plan is None:
            return ["  - Plan output: none"]
        try:
            plan = PlanningOutput.model_validate(raw_plan)
        except ValidationError as error:
            raise ReportError(
                f"Invalid planning output at event {event.event_id}: {error}"
            ) from error
        lines = [f"  - Proposed goal: **{_inline(plan.strategic_goal)}**"]
        lines.extend(f"  - Step: {_inline(step)}" for step in plan.steps)
        for proposal in plan.proposed_actions:
            arguments = json.dumps(proposal.arguments, ensure_ascii=False)
            lines.append(
                f"  - Proposed: [{_code(proposal.actor)}] {_code(proposal.name)}"
                f"({arguments}) · priority `{proposal.priority}`, TTL "
                f"`{proposal.ttl_game_loops}` loops."
            )
        return lines
    return []


def _describe_units(units: list[UnitState]) -> str:
    if not units:
        return "none"
    grouped: dict[str, list[float]] = {}
    for unit in units:
        grouped.setdefault(unit.unit_type, []).append(unit.health_fraction)
    descriptions: list[str] = []
    for unit_type, health_values in grouped.items():
        if len(health_values) == 1:
            health = f"HP {health_values[0]:.0%}"
        else:
            health = (
                f"min/avg HP {min(health_values):.0%}/{sum(health_values) / len(health_values):.0%}"
            )
        descriptions.append(f"{_code(unit_type)} x{len(health_values)} ({health})")
    return ", ".join(descriptions)


def _describe_command(command: ActionCommand) -> str:
    arguments = json.dumps(command.arguments, ensure_ascii=False)
    return (
        f"[{_code(command.source.value)}] {_code(command.actor)} -> {_code(command.name)}"
        f"({arguments}) · priority `{command.priority}`, TTL `{command.ttl_game_loops}` loops"
    )


def _commands(event: StoredEvent, value: object, field: str) -> list[ActionCommand]:
    if not isinstance(value, list):
        raise ReportError(f"Invalid {field} at event {event.event_id}")
    commands: list[ActionCommand] = []
    for raw_command in value:
        try:
            commands.append(ActionCommand.model_validate(raw_command))
        except ValidationError as error:
            raise ReportError(
                f"Invalid command in {field} at event {event.event_id}: {error}"
            ) from error
    return commands


def _last_model(
    events: list[StoredEvent],
    event_type: str,
    model_type: type[ModelT],
) -> ModelT | None:
    selected = [event for event in events if event.event_type == event_type]
    return None if not selected else _validate(selected[-1], model_type)


def _validate(event: StoredEvent, model_type: type[ModelT]) -> ModelT:
    try:
        return model_type.model_validate(event.payload)
    except ValidationError as error:
        raise ReportError(
            f"Invalid {event.event_type} payload at event {event.event_id}: {error}"
        ) from error


def _validate_nested(event: StoredEvent, field: str, model_type: type[ModelT]) -> ModelT:
    if field not in event.payload:
        raise ReportError(f"Missing {field} in {event.event_type} event {event.event_id}")
    try:
        return model_type.model_validate(event.payload[field])
    except ValidationError as error:
        raise ReportError(
            f"Invalid {field} in {event.event_type} event {event.event_id}: {error}"
        ) from error


def _payload_list(event: StoredEvent, container: str, field: str) -> list[object]:
    nested = event.payload.get(container)
    if not isinstance(nested, dict):
        raise ReportError(f"Invalid {container} in event {event.event_id}")
    value = nested.get(field, [])
    if not isinstance(value, list):
        raise ReportError(f"Invalid {container}.{field} in event {event.event_id}")
    return value


def _provider_models(events: list[StoredEvent]) -> str:
    identities: list[str] = []
    for event in events:
        provider = event.payload.get("provider", "unknown")
        model = event.payload.get("model", "unknown")
        identity = f"{_code(provider)}/{_code(model)}"
        if identity not in identities:
            identities.append(identity)
    return ", ".join(identities) if identities else "none"


def _total_tokens(usage: object) -> int:
    if not isinstance(usage, dict):
        return 0
    total = usage.get("total_tokens")
    if isinstance(total, int | float):
        return int(total)
    prompt = usage.get("prompt_tokens", 0)
    completion = usage.get("completion_tokens", 0)
    prompt_value = int(prompt) if isinstance(prompt, int | float) else 0
    completion_value = int(completion) if isinstance(completion, int | float) else 0
    return prompt_value + completion_value


def _error_details(payload: dict[str, Any]) -> str:
    fields = ["module", "status", "error_type", "message", "latency_ms"]
    return ", ".join(f"{field}={_inline(payload[field])}" for field in fields if field in payload)


def _milliseconds(value: object) -> str:
    return f"{float(value):.2f} ms" if isinstance(value, int | float) else "not reported"


def _number(value: float) -> str:
    return f"{value:.2f}"


def _inline(value: object) -> str:
    return " ".join(str(value).split()).replace("|", "\\|")


def _code(value: object) -> str:
    escaped = str(value).replace("`", "'")
    return f"`{escaped}`"
