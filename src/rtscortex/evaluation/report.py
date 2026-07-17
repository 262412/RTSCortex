"""Human-readable timelines derived from append-only runtime event journals."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, TypeVar

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
from rtscortex.evaluation.cortex import (
    CORTEX_EVENT_TYPES,
    CortexObservabilityMetrics,
    compute_cortex_observability,
)
from rtscortex.evaluation.metrics import (
    EpisodeMetrics,
    ExecutionMetrics,
    compute_episode_metrics,
    compute_execution_metrics,
)
from rtscortex.memory import StoredEvent, read_event_log

REPORT_FILENAME = "timeline.md"
SUMMARY_FILENAME = "summary.json"
ModelT = TypeVar("ModelT", bound=BaseModel)
GateScalar = bool | int | float


@dataclass(frozen=True)
class RunReportArtifacts:
    """Paths produced from one immutable runtime journal."""

    timeline_path: Path
    summary_path: Path


@dataclass(frozen=True)
class AcceptanceGate:
    """One machine-readable live acceptance check."""

    name: str
    value: GateScalar
    comparison: Literal["==", ">=", "<="]
    threshold: GateScalar
    unit: Literal["boolean", "count", "game_loops", "ratio"]
    passed: bool | None

    def as_dict(self) -> dict[str, object]:
        return {
            "value": self.value,
            "comparison": self.comparison,
            "threshold": self.threshold,
            "unit": self.unit,
            "passed": self.passed,
        }


class ReportError(ValueError):
    """Raised when a run directory cannot produce a trustworthy timeline."""


def write_timeline_report(run_dir: Path) -> Path:
    """Render ``events.jsonl`` in a run directory to ``timeline.md``."""

    resolved_run_dir, events = _read_run_events(run_dir)
    report = render_timeline(events)
    output_path = resolved_run_dir / REPORT_FILENAME
    try:
        output_path.write_text(report, encoding="utf-8")
    except OSError as error:
        raise ReportError(f"Could not write timeline {output_path}: {error}") from error
    return output_path


def write_run_reports(run_dir: Path) -> RunReportArtifacts:
    """Idempotently derive the Markdown timeline and JSON summary from a journal."""

    resolved_run_dir, events = _read_run_events(run_dir)
    timeline_path = resolved_run_dir / REPORT_FILENAME
    summary_path = resolved_run_dir / SUMMARY_FILENAME
    timeline = render_timeline(events)
    summary = _build_run_summary(events)
    try:
        timeline_path.write_text(timeline, encoding="utf-8")
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except OSError as error:
        raise ReportError(
            f"Could not write run reports below {resolved_run_dir}: {error}"
        ) from error
    return RunReportArtifacts(timeline_path=timeline_path, summary_path=summary_path)


def _read_run_events(run_dir: Path) -> tuple[Path, list[StoredEvent]]:
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
    return resolved_run_dir, events


def _build_run_summary(events: Sequence[StoredEvent]) -> dict[str, object]:
    episodes = _group_episodes(events)
    runs: dict[str, dict[str, object]] = {}
    for (run_id, episode_id), episode_events in episodes.items():
        result = _last_model(episode_events, "episode_result", EpisodeResult)
        execution = compute_execution_metrics(episode_events)
        episode_metrics = (
            compute_episode_metrics(episode_events, result) if result is not None else None
        )
        cortex = compute_cortex_observability(episode_events)
        gates = _acceptance_gates(execution, episode_metrics, cortex)
        run = runs.setdefault(run_id, {"episodes": {}})
        run_episodes = run["episodes"]
        assert isinstance(run_episodes, dict)
        run_episodes[episode_id] = {
            "run_id": run_id,
            "episode_id": episode_id,
            "complete": result is not None,
            "result": result.model_dump(mode="json") if result is not None else None,
            "metrics": (
                episode_metrics.as_dict()
                if episode_metrics is not None
                else {"execution": asdict(execution)}
            ),
            "classification_conservation": _classification_conservation(execution),
            "terminal_reports": _terminal_report_summary(execution),
            "cortex": cortex.as_dict(),
            "hard_acceptance": _hard_acceptance_summary(
                gates,
                complete=result is not None,
            ),
        }
    return {
        "format_version": "1.0",
        "source_journal": "events.jsonl",
        "runs": runs,
    }


def _group_episodes(events: Sequence[StoredEvent]) -> dict[tuple[str, str], list[StoredEvent]]:
    ordered = sorted(events, key=lambda event: event.event_id)
    episodes: dict[tuple[str, str], list[StoredEvent]] = {}
    for event in ordered:
        episodes.setdefault((event.run_id, event.episode_id), []).append(event)
    return episodes


def _classification_conservation(metrics: ExecutionMetrics) -> dict[str, int | bool]:
    succeeded = metrics.status_counts.get("succeeded", 0)
    failed = metrics.status_counts.get("failed", 0)
    cancelled = metrics.status_counts.get("cancelled", 0)
    unconfirmed = metrics.status_counts.get("unconfirmed", 0)
    classified = succeeded + failed + cancelled + unconfirmed
    return {
        "reported": metrics.execution_reports,
        "succeeded": succeeded,
        "failed": failed,
        "cancelled": cancelled,
        "unconfirmed": unconfirmed,
        "classified": classified,
        "conserved": classified == metrics.execution_reports,
    }


def _terminal_report_summary(metrics: ExecutionMetrics) -> dict[str, int | float | bool]:
    exactly_once = (
        metrics.missing_terminal_reports == 0
        and metrics.duplicate_terminal_reports == 0
        and metrics.unexpected_terminal_reports == 0
    )
    return {
        "dispatched_commands": metrics.dispatched_commands,
        "known_lifecycle_commands": metrics.known_lifecycle_commands,
        "reported_commands": metrics.terminal_commands_reported,
        "missing_reports": metrics.missing_terminal_reports,
        "duplicate_reports": metrics.duplicate_terminal_reports,
        "unexpected_reports": metrics.unexpected_terminal_reports,
        "duplicate_dispatches": metrics.duplicate_dispatches,
        "coverage": metrics.terminal_report_coverage,
        "exactly_once": exactly_once,
    }


def _acceptance_gates(
    execution: ExecutionMetrics,
    episode: EpisodeMetrics | None,
    cortex: CortexObservabilityMetrics | None = None,
) -> tuple[AcceptanceGate, ...]:
    classified = sum(
        execution.status_counts.get(status, 0)
        for status in ("succeeded", "failed", "cancelled", "unconfirmed")
    )
    terminal_exactly_once = (
        execution.missing_terminal_reports == 0
        and execution.duplicate_terminal_reports == 0
        and execution.unexpected_terminal_reports == 0
    )
    plan_samples = episode.plan_accept_gap_samples if episode is not None else 0
    accepted_builds = execution.build_funnel.get("pysc2_accepted", 0)
    proposed_builds = execution.build_funnel.get("proposed", 0)
    accepted_production = execution.production_funnel.get("pysc2_accepted", 0)
    production_applicable = execution.production_metrics_applicable
    cortex_metrics = cortex or CortexObservabilityMetrics()
    return (
        _gate("semantic_control_noops", execution.control_noops, "==", 0, "count"),
        _gate("planner_noop_proposals", execution.planner_noop_proposals, "==", 0, "count"),
        _gate("duplicate_command_dispatches", execution.duplicate_dispatches, "==", 0, "count"),
        _gate(
            "unexpected_terminal_reports",
            execution.unexpected_terminal_reports,
            "==",
            0,
            "count",
        ),
        _gate(
            "planner_proposal_audit_complete",
            execution.planner_proposal_audit_complete,
            "==",
            True,
            "boolean",
            applicable=execution.planner_module_results > 0,
        ),
        _gate(
            "planner_builder_attack_proposals",
            execution.planner_builder_attack_proposals,
            "==",
            0,
            "count",
        ),
        _gate(
            "planner_friendly_target_attack_proposals",
            execution.planner_friendly_target_attack_proposals,
            "==",
            0,
            "count",
        ),
        _gate(
            "planner_unsafe_attack_dispatched",
            execution.planner_unsafe_attack_dispatched,
            "==",
            0,
            "count",
        ),
        _gate("friendly_target_attacks", execution.friendly_target_attacks, "==", 0, "count"),
        _gate("builder_attack_commands", execution.builder_attack_commands, "==", 0, "count"),
        _gate(
            "generic_translation_failures",
            execution.generic_translation_failures,
            "==",
            0,
            "count",
        ),
        _gate("unattributed_primitives", execution.unattributed_primitives, "==", 0, "count"),
        _gate(
            "upstream_placement_rejections",
            execution.upstream_placement_rejections,
            "==",
            0,
            "count",
        ),
        _gate(
            "candidate_outside_pysc2_dispatches",
            execution.candidate_outside_pysc2_dispatches,
            "==",
            0,
            "count",
        ),
        _gate(
            "orchestration_573_terminal_reports",
            execution.orchestration_573_terminal_reports,
            "==",
            0,
            "count",
        ),
        _gate(
            "classification_conservation",
            classified,
            "==",
            execution.execution_reports,
            "count",
        ),
        _gate("terminal_report_exactly_once", terminal_exactly_once, "==", True, "boolean"),
        _gate(
            "failure_classification_coverage",
            execution.failure_classification_coverage,
            ">=",
            1.0,
            "ratio",
        ),
        _gate(
            "plan_accept_gap_p50_game_loops",
            episode.plan_accept_gap_game_loops_p50 if episode is not None else 0.0,
            "<=",
            140,
            "game_loops",
            applicable=plan_samples > 0,
        ),
        _gate(
            "plan_accept_gap_p95_game_loops",
            episode.plan_accept_gap_game_loops_p95 if episode is not None else 0.0,
            "<=",
            170,
            "game_loops",
            applicable=plan_samples > 0,
        ),
        _gate(
            "meaningful_command_success_rate",
            execution.meaningful_action_success_rate,
            ">=",
            0.70,
            "ratio",
            applicable=execution.meaningful_commands > 0,
        ),
        _gate(
            "completed_execution_success_rate",
            execution.completed_execution_success_rate,
            ">=",
            0.75,
            "ratio",
            applicable=execution.completed_meaningful_commands > 0,
        ),
        _gate(
            "terminal_backlog_rate",
            execution.terminal_backlog_rate,
            "<=",
            0.05,
            "ratio",
            applicable=execution.meaningful_commands > 0,
        ),
        _gate(
            "build_effect_confirmed_rate",
            execution.build_effect_confirmed_rate,
            ">=",
            0.90,
            "ratio",
            applicable=accepted_builds > 0,
        ),
        _gate(
            "build_effect_timeout_rate",
            execution.build_effect_timeout_rate,
            "<=",
            0.10,
            "ratio",
            applicable=accepted_builds > 0,
        ),
        _gate(
            "build_pre_dispatch_rejection_rate",
            execution.build_pre_dispatch_rejection_rate,
            "<=",
            0.05,
            "ratio",
            applicable=proposed_builds > 0,
        ),
        _gate(
            "production_acceptance_only",
            execution.production_funnel.get("acceptance_only", 0),
            "==",
            0,
            "count",
            applicable=production_applicable,
        ),
        _gate(
            "production_provenance_coverage",
            execution.production_provenance_coverage,
            ">=",
            1.0,
            "ratio",
            applicable=production_applicable and accepted_production > 0,
        ),
        _gate(
            "production_effect_confirmed_rate",
            execution.production_effect_confirmed_rate,
            ">=",
            0.90,
            "ratio",
            applicable=production_applicable and accepted_production > 0,
        ),
        _gate(
            "production_timeout_rate",
            execution.production_timeout_rate,
            "<=",
            0.10,
            "ratio",
            applicable=production_applicable and accepted_production > 0,
        ),
        _gate(
            "cortex_executor_candidate_violations",
            cortex_metrics.executor_candidate_violations,
            "==",
            0,
            "count",
            applicable=cortex_metrics.observed,
        ),
        _gate(
            "cortex_lineage_integrity_violations",
            cortex_metrics.lineage_integrity_violations,
            "==",
            0,
            "count",
            applicable=cortex_metrics.observed,
        ),
        _gate(
            "cortex_duplicate_command_lineage",
            cortex_metrics.duplicate_lineage_commands,
            "==",
            0,
            "count",
            applicable=cortex_metrics.observed,
        ),
        _gate(
            "cortex_missing_command_lineage",
            cortex_metrics.missing_lineage_commands,
            "==",
            0,
            "count",
            applicable=cortex_metrics.observed,
        ),
        _gate(
            "cortex_orphan_command_lineage",
            cortex_metrics.orphan_lineage_commands,
            "==",
            0,
            "count",
            applicable=cortex_metrics.observed,
        ),
    )


def _gate(
    name: str,
    value: GateScalar,
    comparison: Literal["==", ">=", "<="],
    threshold: GateScalar,
    unit: Literal["boolean", "count", "game_loops", "ratio"],
    *,
    applicable: bool = True,
) -> AcceptanceGate:
    passed: bool | None = None
    if applicable:
        if comparison == "==":
            passed = value == threshold
        elif comparison == ">=":
            passed = value >= threshold
        else:
            passed = value <= threshold
    return AcceptanceGate(
        name=name,
        value=value,
        comparison=comparison,
        threshold=threshold,
        unit=unit,
        passed=passed,
    )


def _hard_acceptance_summary(
    gates: Sequence[AcceptanceGate],
    *,
    complete: bool,
) -> dict[str, object]:
    passed_gates = sum(gate.passed is True for gate in gates)
    failed_gates = sum(gate.passed is False for gate in gates)
    not_applicable_gates = sum(gate.passed is None for gate in gates)
    return {
        "passed": complete and failed_gates == 0,
        "passed_gates": passed_gates,
        "failed_gates": failed_gates,
        "not_applicable_gates": not_applicable_gates,
        "gates": {gate.name: gate.as_dict() for gate in gates},
    }


def render_timeline(events: Sequence[StoredEvent]) -> str:
    """Render events in append order, grouped by run and episode."""

    if not events:
        raise ReportError("Event journal is empty")
    episodes = _group_episodes(events)

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
    legacy_plans = [event for event in events if event.event_type == "plan_accepted"]
    macro_plan_events = [
        event
        for event in events
        if event.event_type in {"macro_plan_accepted", "macro_plan_rejected"}
    ]
    plans = [
        *legacy_plans,
        *(event for event in macro_plan_events if event.event_type == "macro_plan_accepted"),
    ]
    executions = [event for event in events if event.event_type == "execution"]
    model_events = [
        event
        for event in events
        if event.event_type == "module_result" and event.payload.get("model_call") is True
    ]
    rejected = sum(len(_payload_list(event, "batch", "rejected_commands")) for event in decisions)
    successful_executions = sum(event.payload.get("success") is True for event in executions)
    execution_metrics = compute_execution_metrics(events)
    cortex_metrics = compute_cortex_observability(events)
    episode_metrics = compute_episode_metrics(events, result) if result is not None else None
    command_index = _decision_command_index(decisions)
    total_tokens = sum(_total_tokens(event.payload.get("usage")) for event in model_events)
    total_tokens += sum(_macro_generation_tokens(event) for event in macro_plan_events)
    model_call_count = len(model_events) + len(macro_plan_events)

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
    provider_models = _provider_models([*model_events, *macro_plan_events])

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
        (
            "| Agent ticks | Decisions | Plans | Legacy executions (deprecated) | "
            "Rejected | Model calls | Tokens |"
        ),
        "|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| {len(observations)} | {len(decisions)} | {len(plans)} | {execution_rate} | "
            f"{rejected} | {model_call_count} | {total_tokens} |"
        ),
    ]
    if episode_metrics is not None:
        accept_gap_summary = (
            f"`{episode_metrics.plan_accept_gap_game_loops_p50:.0f}/"
            f"{episode_metrics.plan_accept_gap_game_loops_p95:.0f}` loops "
            f"({episode_metrics.plan_accept_gap_samples} samples)"
            if episode_metrics.plan_accept_gap_samples
            else "insufficient samples (requires at least two accepted plans)"
        )
        lines.extend(
            [
                "",
                (
                    "- Latency p50/p95: planner "
                    f"`{episode_metrics.planner_latency_ms_p50:.2f}/"
                    f"{episode_metrics.planner_latency_ms_p95:.2f} "
                    "ms`, reflex "
                    f"`{episode_metrics.reflex_latency_ms_p50:.2f}/"
                    f"{episode_metrics.reflex_latency_ms_p95:.2f} "
                    "ms`, tick "
                    f"`{episode_metrics.tick_latency_ms_p50:.2f}/"
                    f"{episode_metrics.tick_latency_ms_p95:.2f} ms`."
                ),
                (
                    f"- Plan revisions: `{episode_metrics.plan_revisions}`; "
                    f"reflex preemptions: `{episode_metrics.reflex_preemptions}`; duration: "
                    f"`{episode_metrics.episode_duration_seconds:.2f} s`."
                ),
                (
                    "- Plan age p50/p95: "
                    f"`{episode_metrics.plan_age_game_loops_p50:.0f}/"
                    f"{episode_metrics.plan_age_game_loops_p95:.0f}` loops; "
                    "accept gap p50/p95: "
                    f"{accept_gap_summary}."
                ),
            ]
        )
    else:
        lines.extend(["", "- No terminal episode result was recorded; this run is incomplete."])

    lines.extend(_render_execution_metrics(execution_metrics))
    lines.extend(_render_cortex_metrics(cortex_metrics))
    lines.extend(
        _render_hard_acceptance(
            _acceptance_gates(execution_metrics, episode_metrics, cortex_metrics),
            complete=result is not None,
        )
    )
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
        lines.extend(_render_event(event, command_index))
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


def _render_event(
    event: StoredEvent,
    command_index: dict[str, ActionCommand],
) -> list[str]:
    if event.event_type == "plan_accepted":
        return _render_plan(event)
    if event.event_type == "decision":
        return _render_decision(event)
    if event.event_type == "execution":
        return _render_execution(event, command_index)
    if event.event_type == "module_result":
        return _render_module_result(event)
    if event.event_type == "planner_cycle":
        status = _inline(event.payload.get("status", "unknown"))
        latency = _milliseconds(event.payload.get("latency_ms"))
        return [f"- Event {event.event_id} · Planner cycle: {status} · {latency}."]
    if event.event_type in CORTEX_EVENT_TYPES:
        return _render_cortex_event(event)
    if event.event_type == "episode_summary":
        summary = _validate(event, EpisodeSummary)
        lines = [f"- Event {event.event_id} · Episode summary: {_inline(summary.summary)}"]
        lines.extend(f"  - Lesson: {_inline(lesson)}" for lesson in summary.lessons)
        return lines
    if event.event_type in {
        "planner_timeout",
        "planner_error",
        "module_error",
        "module_failed",
    }:
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


def _render_execution(
    event: StoredEvent,
    command_index: dict[str, ActionCommand],
) -> list[str]:
    report = _validate(event, ExecutionReport)
    command = command_index.get(report.command_id)
    if report.protocol_version == "1.0" and command is not None:
        report = report.model_copy(
            update={
                "action_name": report.action_name or command.name,
                "actor": report.actor or command.actor,
                "source": report.source or command.source,
                "requested_arguments": report.requested_arguments or command.arguments,
                "resolved_arguments": report.resolved_arguments or command.arguments,
            }
        )
    status = report.status.value.upper()
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
    if report.action_name or report.actor or report.source:
        lines.append(
            "  - Command: "
            f"{_code(report.action_name or 'unknown')} by "
            f"{_code(report.actor or 'unknown')} from "
            f"{_code(report.source.value if report.source else 'unknown')}."
        )
    if report.execution_stage or report.failure_code:
        lines.append(
            "  - Classification: stage "
            f"{_code(report.execution_stage.value if report.execution_stage else 'unknown')}; "
            f"code {_code(report.failure_code or 'none')}."
        )
    if report.primitive_trace:
        accepted = sum(primitive.accepted for primitive in report.primitive_trace)
        lines.append(f"  - Primitive trace: `{accepted}/{len(report.primitive_trace)}` accepted.")
    if report.effect_evidence and report.effect_evidence.new_structure_tag:
        lines.append(
            "  - Effect confirmed by new structure "
            f"{_code(report.effect_evidence.new_structure_tag)}."
        )
    if report.effect_evidence and report.effect_evidence.effect_kind == "production":
        evidence = report.effect_evidence
        producer = evidence.producer_type or "unknown producer"
        producer_tag = evidence.producer_tag or "unknown"
        unit_type = evidence.expected_unit_type or "unknown unit"
        if evidence.confirmation_kind == "producer_order":
            lines.append(
                "  - Production confirmed by order on "
                f"{_code(producer)} {_code(producer_tag)} for {_code(unit_type)}."
            )
        elif evidence.confirmation_kind == "new_unit":
            lines.append(
                "  - Production confirmed by new unit "
                f"{_code(evidence.new_unit_tag or 'unknown')} ({_code(unit_type)})."
            )
    elif (
        report.action_name is not None
        and report.action_name.startswith("Train_")
        and report.status.value == "succeeded"
        and report.execution_stage is not None
        and report.execution_stage.value == "pysc2_acceptance"
    ):
        lines.append(
            "  - Production acceptance only (deprecated): PySC2 accepted the Train action, "
            "but no producer order or new unit was verified."
        )
    if report.game_result:
        lines.append(f"  - Game result: {_code(report.game_result)}")
    return lines


def _render_cortex_event(event: StoredEvent) -> list[str]:
    payload = event.payload
    event_id = event.event_id
    if event.event_type == "situation_assessed":
        assessment = _nested_payload(payload, "assessment")
        phase = _payload_text(assessment, "game_phase", "phase") or "unknown"
        threat = _payload_text(assessment, "threat_level", "threat") or "unknown"
        readiness = _payload_text(assessment, "army_readiness", "readiness") or "unknown"
        source = _payload_text(payload, "source_kind", "source", "model") or "unknown"
        return [
            f"- Event {event_id} · Situation assessed by {_code(source)}: phase "
            f"{_code(phase)}; threat {_code(threat)}; readiness {_code(readiness)}."
        ]
    if event.event_type in {"macro_plan_accepted", "macro_plan_rejected"}:
        plan = _nested_payload(payload, "plan")
        plan_id = _payload_text(payload, "plan_id") or _payload_text(plan, "plan_id") or "unknown"
        model = (
            _payload_text(payload, "model_id", "source_model_id", "model", "specialist")
            or _payload_text(plan, "source_model_id", "model_id")
            or "unknown"
        )
        if event.event_type == "macro_plan_rejected":
            reason = _payload_text(
                payload,
                "reason",
                "failure_code",
                "failure_reason",
                "message",
            )
            return [
                f"- Event {event_id} · Macro plan {_code(plan_id)} from {_code(model)} "
                f"rejected: {_inline(reason or 'unspecified')}."
            ]
        steps = plan.get("steps", payload.get("steps", []))
        step_count = len(steps) if isinstance(steps, list) else 0
        frontier = _payload_text(payload, "runtime_frontier", "frontier_action")
        detail = f"; frontier {_code(frontier)}" if frontier else ""
        return [
            f"- Event {event_id} · Macro plan {_code(plan_id)} from {_code(model)} accepted "
            f"with `{step_count}` steps{detail}."
        ]
    if event.event_type == "macro_step_updated":
        step = _nested_payload(payload, "step")
        action = _payload_text(step, "semantic_action", "action") or "unknown"
        status = _payload_text(step, "status") or "unknown"
        completed = step.get("completed_repeats", 0)
        repeat = step.get("repeat", 1)
        reason = _payload_text(step, "reason")
        suffix = f"; reason {_inline(reason)}" if reason else ""
        return [
            f"- Event {event_id} · Macro step {_code(action)} is {_code(status)} "
            f"(`{completed}/{repeat}`){suffix}."
        ]
    if event.event_type == "intent_emitted":
        intent = _nested_payload(payload, "intent")
        intent_id = _payload_text(payload, "intent_id") or _payload_text(intent, "intent_id")
        role = _payload_text(
            payload, "role", "source_role", "intent_kind", "source"
        ) or _payload_text(intent, "role", "source_role", "intent_kind", "source")
        intent_action = _payload_text(payload, "action_name", "action") or _payload_text(
            intent, "action_name", "action"
        )
        if intent_action is None:
            action_names = intent.get("action_names")
            if isinstance(action_names, list) and action_names:
                first_action = action_names[0]
                if isinstance(first_action, str):
                    intent_action = first_action
        return [
            f"- Event {event_id} · {_inline(role or 'unknown')} intent "
            f"{_code(intent_id or 'unknown')}: {_code(intent_action or 'no action')}."
        ]
    if event.event_type == "candidate_set_built":
        candidates = payload.get("candidates", [])
        candidate_count = len(candidates) if isinstance(candidates, list) else 0
        declared_count = payload.get("candidate_count")
        if isinstance(declared_count, int) and not isinstance(declared_count, bool):
            candidate_count = declared_count
        intent_id = _payload_text(payload, "intent_id") or "unknown"
        actions = _candidate_actions(candidates)
        suffix = f": {', '.join(_code(action) for action in actions)}" if actions else "."
        return [
            f"- Event {event_id} · Built `{candidate_count}` executable candidates for "
            f"intent {_code(intent_id)}{suffix}"
        ]
    if event.event_type == "executor_selection":
        executor = _payload_text(payload, "executor_id", "executor", "model") or "unknown"
        selected = _payload_text(payload, "selected_candidate_id", "candidate_id")
        latency = _milliseconds(payload.get("latency_ms"))
        choice = f"selected {_code(selected)}" if selected else "abstained"
        fallback = _payload_text(payload, "fallback_reason")
        suffix = f"; fallback {_inline(fallback)}" if fallback else ""
        return [f"- Event {event_id} · Executor {_code(executor)} {choice} in {latency}{suffix}."]
    if event.event_type == "command_lineage":
        lineage = _nested_payload(payload, "lineage")
        command_id = _payload_text(payload, "command_id") or _payload_text(lineage, "command_id")
        intent_id = _payload_text(payload, "intent_id") or _payload_text(lineage, "intent_id")
        candidate_id = _payload_text(payload, "candidate_id") or _payload_text(
            lineage, "candidate_id"
        )
        lineage_plan_id = _payload_text(payload, "macro_plan_id", "plan_id") or _payload_text(
            lineage, "macro_plan_id", "plan_id"
        )
        return [
            f"- Event {event_id} · Command lineage {_code(command_id or 'unknown')}: plan "
            f"{_code(lineage_plan_id or 'none')} → intent {_code(intent_id or 'unknown')} → "
            "candidate "
            f"{_code(candidate_id or 'unknown')}."
        ]
    role = _payload_text(payload, "role", "specialist", "module") or "unknown"
    model = _payload_text(payload, "model_id", "model") or "unknown"
    if event.event_type == "specialist_failed":
        reason = _payload_text(
            payload,
            "reason",
            "failure_code",
            "failure_reason",
            "message",
        )
        return [
            f"- Event {event_id} · Specialist {_code(role)}/{_code(model)} failed: "
            f"{_inline(reason or 'unspecified')}."
        ]
    if event.event_type == "specialist_ready":
        return [f"- Event {event_id} · Specialist {_code(role)}/{_code(model)} is ready."]
    return [f"- Event {event_id} · Specialist {_code(role)}/{_code(model)} recovered."]


def _nested_payload(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    return value if isinstance(value, dict) else payload


def _payload_text(payload: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _candidate_actions(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    actions: list[str] = []
    for candidate in value:
        if not isinstance(candidate, dict):
            continue
        action = _payload_text(candidate, "action_name", "action", "name")
        if action and action not in actions:
            actions.append(action)
    return actions[:4]


def _decision_command_index(decisions: Sequence[StoredEvent]) -> dict[str, ActionCommand]:
    commands: dict[str, ActionCommand] = {}
    for event in decisions:
        batch = _validate_nested(event, "batch", ActionBatch)
        for command in batch.commands:
            commands.setdefault(command.command_id, command)
    return commands


def _render_execution_metrics(metrics: ExecutionMetrics) -> list[str]:
    terminal_cancelled = metrics.status_counts.get("cancelled", 0)
    terminal_unconfirmed = metrics.status_counts.get("unconfirmed", 0)
    meaningful_rate = _rate(
        metrics.meaningful_successes,
        metrics.meaningful_commands,
    )
    completed_rate = _rate(
        metrics.meaningful_successes,
        metrics.completed_meaningful_commands,
    )
    legacy_rate = _rate(metrics.legacy_successes, metrics.execution_reports)
    classified = sum(
        metrics.status_counts.get(status, 0)
        for status in ("succeeded", "failed", "cancelled", "unconfirmed")
    )
    lines = [
        "",
        "### Decision activity",
        "",
        "| Decisions | Fallback | Planner pending | Unique validation rejections |",
        "|---:|---:|---:|---:|",
        (
            f"| {metrics.decision_count} | {metrics.fallback_decisions} | "
            f"{metrics.planner_pending_decisions} | "
            f"{metrics.unique_validation_rejected_command_ids} |"
        ),
    ]
    if metrics.idle_reason_counts:
        lines.extend(_render_count_table("Idle reasons", metrics.idle_reason_counts))
    lines.extend(
        [
            "",
            "### Meaningful outcomes",
            "",
            (
                f"- Tracked control NoOps: `{metrics.control_noops}` "
                f"(`{metrics.control_noop_successes}` succeeded)."
            ),
            (f"- Untracked transport NoOp primitives: `{metrics.transport_noop_primitives}`."),
            (
                f"- Meaningful commands: `{metrics.meaningful_commands}` — "
                f"`{metrics.meaningful_successes}` succeeded, "
                f"`{metrics.meaningful_failures}` failed, "
                f"`{metrics.meaningful_cancelled}` cancelled, "
                f"`{metrics.meaningful_unconfirmed}` unconfirmed."
            ),
            f"- Meaningful success: {meaningful_rate}.",
            f"- Completed execution success: {completed_rate}.",
            (
                f"- Terminal cancelled: `{terminal_cancelled}`; "
                f"unconfirmed: `{terminal_unconfirmed}`."
            ),
            (
                "- Terminal backlog rate: "
                f"`{metrics.terminal_backlog_rate:.1%}` "
                f"(`{metrics.meaningful_cancelled + metrics.meaningful_unconfirmed}/"
                f"{metrics.meaningful_commands}`)."
            ),
            f"- Classification conservation: `{classified}/{metrics.execution_reports}`.",
            (
                "- Terminal report coverage: "
                f"`{metrics.terminal_commands_reported}/{metrics.dispatched_commands}` "
                f"({metrics.terminal_report_coverage:.1%}); "
                f"known lifecycle commands `{metrics.known_lifecycle_commands}`, "
                f"missing `{metrics.missing_terminal_reports}`, unexpected reports "
                f"`{metrics.unexpected_terminal_reports}`, duplicate reports "
                f"`{metrics.duplicate_terminal_reports}`, duplicate dispatches "
                f"`{metrics.duplicate_dispatches}`."
            ),
            (
                "- Explicit failure stage/code coverage: "
                f"`{metrics.explicitly_classified_failures}/{metrics.failure_reports}` "
                f"({metrics.failure_classification_coverage:.1%})."
            ),
            f"- Legacy execution-report rate: {legacy_rate} (deprecated).",
            "",
            "### Build funnel",
            "",
            "| Raw Planner proposed | Candidate validated | Translator accepted | PySC2 accepted | "
            "Effect confirmed |",
            "|---:|---:|---:|---:|---:|",
            (
                f"| {metrics.build_funnel.get('proposed', 0)} | "
                f"{metrics.build_funnel.get('candidate_validated', 0)} | "
                f"{metrics.build_funnel.get('translator_accepted', 0)} | "
                f"{metrics.build_funnel.get('pysc2_accepted', 0)} | "
                f"{metrics.build_funnel.get('effect_confirmed', 0)} |"
            ),
            (
                "- Build effect confirmed rate: "
                f"`{metrics.build_effect_confirmed_rate:.1%}`; effect timeout rate: "
                f"`{metrics.build_effect_timeout_rate:.1%}` "
                f"(`{metrics.build_effect_timeouts}` commands)."
            ),
            (
                "- Build pre-dispatch race rejection rate: "
                f"`{metrics.build_pre_dispatch_rejection_rate:.1%}` "
                f"(`{metrics.build_pre_dispatch_rejections}` commands)."
            ),
            "",
            "### Production funnel",
            "",
            (
                "| Raw Planner proposed | Candidate validated | Translator accepted | "
                "PySC2 accepted | Order confirmed | Unit fallback confirmed | "
                "Effect confirmed | Acceptance only (deprecated) |"
            ),
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
            (
                f"| {metrics.production_funnel.get('proposed', 0)} | "
                f"{metrics.production_funnel.get('candidate_validated', 0)} | "
                f"{metrics.production_funnel.get('translator_accepted', 0)} | "
                f"{metrics.production_funnel.get('pysc2_accepted', 0)} | "
                f"{metrics.production_funnel.get('order_confirmed', 0)} | "
                f"{metrics.production_funnel.get('unit_fallback_confirmed', 0)} | "
                f"{metrics.production_funnel.get('effect_confirmed', 0)} | "
                f"{metrics.production_funnel.get('acceptance_only', 0)} |"
            ),
            (
                "- Production effect confirmed rate: "
                f"`{metrics.production_effect_confirmed_rate:.1%}`; provenance coverage: "
                f"`{metrics.production_provenance_coverage:.1%}`; timeout rate: "
                f"`{metrics.production_timeout_rate:.1%}` "
                f"(`{metrics.production_effect_timeouts}` commands)."
            ),
            (
                "- Production confirmation latency p50/p95: "
                f"`{metrics.confirmation_latency_game_loops_p50:.0f}/"
                f"{metrics.confirmation_latency_game_loops_p95:.0f}` game loops "
                f"(`{metrics.confirmation_latency_game_loops_samples}` samples)."
            ),
            (
                "- Production metric gates: "
                + (
                    "protocol v1.1 applicable."
                    if metrics.production_metrics_applicable
                    else "not applicable to legacy/no-production reports; acceptance-only is "
                    "shown for compatibility only."
                )
            ),
            *_render_count_table("Production by action", metrics.production_by_action),
            *_render_count_table("Production by producer", metrics.production_by_producer),
            "",
            "### Safety and attribution invariants",
            "",
            (
                "- Planner proposal audit: "
                f"`{metrics.planner_proposal_audited_results}/"
                f"{metrics.planner_module_results}` module results complete."
            ),
            (
                f"- Planner unsafe Attack proposals: `{metrics.planner_unsafe_attack_proposals}` "
                f"(Builder actor `{metrics.planner_builder_attack_proposals}`, friendly target "
                f"`{metrics.planner_friendly_target_attack_proposals}`); rejected before dispatch "
                f"`{metrics.planner_unsafe_attack_rejected_before_dispatch}`; dispatched "
                f"`{metrics.planner_unsafe_attack_dispatched}`."
            ),
            (
                f"- Dispatched unsafe Attack commands: Builder actor "
                f"`{metrics.builder_attack_commands}`; friendly target "
                f"`{metrics.friendly_target_attacks}`."
            ),
            "",
            "| Planner NoOp proposals | Generic translation failures | "
            "Upstream placement rejections | Unattributed primitives | "
            "Candidate-external PySC2 dispatches | Orchestration 573 terminal reports |",
            "|---:|---:|---:|---:|---:|---:|",
            (
                f"| {metrics.planner_noop_proposals} | "
                f"{metrics.generic_translation_failures} | "
                f"{metrics.upstream_placement_rejections} | "
                f"{metrics.unattributed_primitives} | "
                f"{metrics.candidate_outside_pysc2_dispatches} | "
                f"{metrics.orchestration_573_terminal_reports} |"
            ),
            "",
            "### Failure taxonomy",
        ]
    )
    lines.extend(_render_count_table("By stage", metrics.failure_by_stage))
    lines.extend(_render_count_table("By code", metrics.failure_by_code))
    lines.extend(_render_count_table("By action", metrics.failure_by_action))
    lines.extend(_render_count_table("By actor", metrics.failure_by_actor))
    lines.extend(
        _render_count_table("Commands by action and actor", metrics.command_by_action_actor)
    )
    lines.extend(
        _render_count_table(
            "Failures by action, stage, and code",
            metrics.failure_by_action_stage_code,
        )
    )
    return lines


def _render_cortex_metrics(metrics: CortexObservabilityMetrics) -> list[str]:
    if not metrics.observed:
        return []
    event_counts = metrics.event_counts
    coverage = (
        f"`{metrics.lineage_commands}/{metrics.dispatched_commands}` "
        f"({metrics.command_lineage_coverage:.1%})"
        if metrics.dispatched_commands
        else "not applicable (no dispatched Cortex commands)"
    )
    lines = [
        "",
        "### Cortex observability",
        "",
        "| Situation | Macro accepted | Macro rejected | Intents | Candidate sets | "
        "Executor selections |",
        "|---:|---:|---:|---:|---:|---:|",
        (
            f"| {event_counts.get('situation_assessed', 0)} | "
            f"{event_counts.get('macro_plan_accepted', 0)} | "
            f"{event_counts.get('macro_plan_rejected', 0)} | "
            f"{event_counts.get('intent_emitted', 0)} | "
            f"{event_counts.get('candidate_set_built', 0)} | "
            f"{event_counts.get('executor_selection', 0)} |"
        ),
        (
            "- Executor outcomes: "
            f"`{metrics.executor_selections}` selected, "
            f"`{metrics.executor_abstentions}` abstained, "
            f"`{metrics.executor_fallbacks}` fallback; candidate-domain violations "
            f"`{metrics.executor_candidate_violations}`."
        ),
        (
            "- Executor latency p50/p95: "
            f"`{metrics.executor_latency_ms_p50:.2f}/"
            f"{metrics.executor_latency_ms_p95:.2f}` ms."
        ),
        (
            "- Macro specialist requests and latency p50/p95: "
            f"`{metrics.macro_requests}`; `{metrics.macro_latency_ms_p50:.2f}/"
            f"{metrics.macro_latency_ms_p95:.2f}` ms."
        ),
        (
            "- Race Brain / Playbook: "
            f"`{event_counts.get('race_brain_coordinated', 0)}` coordinated cycles, "
            f"`{event_counts.get('playbook_retrieved', 0)}` retrievals, "
            f"`{event_counts.get('playbook_case_recorded', 0)}` cases, "
            f"`{event_counts.get('playbook_lesson_promoted', 0)}` promoted lessons."
        ),
        (
            f"- Command lineage coverage: {coverage}; missing "
            f"`{metrics.missing_lineage_commands}`, orphan `{metrics.orphan_lineage_commands}`, "
            f"duplicates `{metrics.duplicate_lineage_commands}`, integrity violations "
            f"`{metrics.lineage_integrity_violations}`."
        ),
        *_render_count_table("Cortex intents by role", metrics.intent_counts),
        *_render_count_table("Cortex selections by executor", metrics.executor_counts),
        *_render_count_table("Specialist failures", metrics.specialist_failure_counts),
        *_render_count_table("Specialists ready", metrics.specialist_ready_counts),
        *_render_count_table("Specialist recoveries", metrics.specialist_recovery_counts),
    ]
    return lines


def _render_hard_acceptance(
    gates: Sequence[AcceptanceGate],
    *,
    complete: bool,
) -> list[str]:
    summary = _hard_acceptance_summary(gates, complete=complete)
    overall = "PASS" if summary["passed"] is True else "FAIL"
    lines = [
        "",
        "### Hard acceptance gates",
        "",
        (
            f"- Overall: **{overall}** — `{summary['passed_gates']}` passed, "
            f"`{summary['failed_gates']}` failed, "
            f"`{summary['not_applicable_gates']}` not applicable."
        ),
        "",
        "| Gate | Value | Requirement | Result |",
        "|---|---:|---:|---|",
    ]
    for gate in gates:
        result = "PASS" if gate.passed is True else "FAIL" if gate.passed is False else "N/A"
        lines.append(
            f"| `{gate.name}` | {_format_gate_scalar(gate.value, gate.unit)} | "
            f"`{gate.comparison}` {_format_gate_scalar(gate.threshold, gate.unit)} | "
            f"**{result}** |"
        )
    return lines


def _format_gate_scalar(
    value: GateScalar,
    unit: Literal["boolean", "count", "game_loops", "ratio"],
) -> str:
    if unit == "ratio":
        return f"`{float(value):.1%}`"
    if unit == "boolean":
        return _code(str(bool(value)).lower())
    if unit == "game_loops":
        return f"`{float(value):.0f}` loops"
    return _code(value)


def _render_count_table(title: str, counts: dict[str, int]) -> list[str]:
    lines = ["", f"#### {title}", ""]
    if not counts:
        return [*lines, "None."]
    lines.extend(["| Value | Count |", "|---|---:|"])
    lines.extend(
        f"| {_code(key)} | {value} |"
        for key, value in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    )
    return lines


def _rate(numerator: int, denominator: int) -> str:
    if not denominator:
        return "`0/0`"
    return f"`{numerator}/{denominator}` ({numerator / denominator:.1%})"


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
        lines.extend(_render_context_compaction(event, output))
    return lines


def _render_module_output(event: StoredEvent, module: str, output: object) -> list[str]:
    if not isinstance(output, dict):
        raise ReportError(f"Invalid module output at event {event.event_id}")
    if module == "reflection":
        reflection = output.get("summary") or output.get("reflection")
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
            plan = PlanningOutput.model_validate(_without_legacy_planner_ttl(raw_plan))
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
                f"({arguments}) · priority `{proposal.priority}`."
            )
        return lines
    return []


def _without_legacy_planner_ttl(raw_plan: object) -> object:
    if not isinstance(raw_plan, dict):
        return raw_plan
    proposed_actions = raw_plan.get("proposed_actions")
    if not isinstance(proposed_actions, list):
        return raw_plan
    normalized = dict(raw_plan)
    normalized["proposed_actions"] = [
        {key: value for key, value in proposal.items() if key != "ttl_game_loops"}
        if isinstance(proposal, dict)
        else proposal
        for proposal in proposed_actions
    ]
    return normalized


def _render_context_compaction(event: StoredEvent, output: object) -> list[str]:
    if not isinstance(output, dict):
        return []
    statistics = output.get("context_compaction")
    if statistics is None:
        return []
    if not isinstance(statistics, dict):
        raise ReportError(f"Invalid context compaction output at event {event.event_id}")

    budget = statistics.get("budget_chars", "unknown")
    original = statistics.get("original_chars", "unknown")
    final = statistics.get("final_chars", "unknown")
    compacted = "yes" if statistics.get("compacted") is True else "no"
    lines = [
        f"  - Context budget: {_code(budget)} chars; {_code(original)} → "
        f"{_code(final)} chars; compacted: {_code(compacted)}."
    ]
    reductions = []
    for field, label in (
        ("aggregated_own_units", "own units"),
        ("aggregated_own_structures", "own structures"),
        ("aggregated_visible_enemies", "visible enemies"),
        ("dropped_recent_events", "recent events"),
        ("dropped_lessons", "lessons"),
        ("dropped_episode_summaries", "episode summaries"),
        ("dropped_spatial_lines", "spatial lines"),
    ):
        if field in statistics:
            reductions.append(f"{label} {_code(statistics[field])}")
    if reductions:
        lines.append(f"  - Context reductions: {'; '.join(reductions)}.")
    return lines


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
        if event.event_type in {"macro_plan_accepted", "macro_plan_rejected"}:
            plan = _nested_payload(event.payload, "plan")
            provider = "hima"
            model = (
                _payload_text(
                    event.payload,
                    "model_id",
                    "source_model_id",
                    "model",
                )
                or _payload_text(plan, "source_model_id", "model_id")
                or "unknown"
            )
        else:
            provider = event.payload.get("provider", "unknown")
            model = event.payload.get("model", "unknown")
        identity = f"{_code(provider)}/{_code(model)}"
        if identity not in identities:
            identities.append(identity)
    return ", ".join(identities) if identities else "none"


def _macro_generation_tokens(event: StoredEvent) -> int:
    metadata = event.payload.get("generation_metadata")
    if not isinstance(metadata, dict):
        plan = _nested_payload(event.payload, "plan")
        raw_response = plan.get("raw_proposal")
        if isinstance(raw_response, dict):
            proposal = raw_response.get("proposal")
            if isinstance(proposal, dict):
                nested = proposal.get("generation_metadata")
                metadata = nested if isinstance(nested, dict) else None
    if not isinstance(metadata, dict):
        return 0
    prompt = metadata.get("prompt_token_count", 0)
    completion = metadata.get("completion_token_count", 0)
    prompt_value = int(prompt) if isinstance(prompt, int | float) else 0
    completion_value = int(completion) if isinstance(completion, int | float) else 0
    return prompt_value + completion_value


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
