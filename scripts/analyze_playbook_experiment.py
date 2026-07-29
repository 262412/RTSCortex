"""Build conserved acceptance metrics for Playbook experiment run sets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import resource
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from rtscortex.evaluation.engineering import (
    ENGINEERING_GATES_FILENAME,
    REQUIRED_ENGINEERING_GATES,
    EngineeringAccumulator,
    build_engineering_gate_report,
)
from rtscortex.memory import read_event_log

_ERROR_CONSEQUENCES = frozenset(
    {
        "threat_unanswered",
        "expansion_delayed",
        "production_imbalance",
        "timing_attack_failed",
        "unnecessary_retreat",
        "advantage_not_converted",
    }
)
MAX_RULE_EVALUATIONS = 100_000
MAX_METRIC_STATE_KEYS = 100_000


@dataclass
class _MetricStateBudget:
    """Bound all command-, operation-, and signature-keyed analyzer state."""

    limit: int = MAX_METRIC_STATE_KEYS
    retained_key_count: int = 0
    overflow_count: int = 0

    def increment(self, values: Counter[str], key: str) -> None:
        if key in values:
            values[key] += 1
            return
        if self.retained_key_count >= self.limit:
            self.overflow_count += 1
            return
        values[key] = 1
        self.retained_key_count += 1

    def add(self, values: set[str], key: str) -> None:
        if key in values:
            return
        if self.retained_key_count >= self.limit:
            self.overflow_count += 1
            return
        values.add(key)
        self.retained_key_count += 1


@dataclass(frozen=True)
class RunMetrics:
    mode: str
    seed: int
    arm: str
    exit_code: int
    run_dir: str
    natural_terminal: bool
    outcome: str | None
    score: float
    terminal_report_count: int
    missing_terminal_reports: int
    duplicate_terminal_reports: int
    duplicate_dispatches: int
    candidate_outside_dispatch: int
    playbook_applications: int
    playbook_nonzero_score_applications: int
    playbook_blocks: int
    repeated_eligible_errors: int
    repeated_errors_per_10k_game_loops: float
    repeated_errors_per_lineaged_operation: float
    lineaged_operation_count: int
    strategic_consequences: dict[str, int]
    hard_rule_false_block_count: int
    hard_rule_shadow_state_count: int
    hard_rule_unresolved_block_count: int
    hard_rule_false_block_rate: float
    journal_bytes: int
    artifact_bytes: int
    event_count: int
    max_game_loop: int
    events_per_game_loop: float
    bytes_per_game_loop: float
    effective_game_loops_per_second: float
    writer_queue_peak: int
    writer_lag_ms_p95: float
    writer_lag_ms_max: float
    blocked_append_count: int
    sampled_drop_supported: bool
    playbook_before_sha256: str
    playbook_after_sha256: str
    experiment_kind: str = "behavior"
    subject_arm: str | None = None
    active_hard_block_keys: tuple[str, ...] = ()
    shadow_would_block_keys: tuple[str, ...] = ()
    resolved_counterfactual_keys: tuple[str, ...] = ()
    strategic_regret_count: int = 0
    strategic_resolved_count: int = 0
    invalid_counterfactual_evidence_count: int = 0
    engineering_gates: dict[str, bool] = field(
        default_factory=lambda: {name: True for name in REQUIRED_ENGINEERING_GATES}
    )
    engineering_missing_metrics: tuple[str, ...] = ()
    engineering_accepted: bool = True
    source_tree_clean: bool = True
    source_commit_matches_expected_sha: bool = True
    source_attestation_consistent: bool = True
    source_attestation_fingerprint: str | None = "source-baseline"
    retained_event_count: int = 0
    rule_evaluation_count: int = 0
    analysis_peak_rss_kib: int = 0
    analysis_rss_per_10k_game_loops: float = 0.0
    analysis_evidence_overflow_count: int = 0
    metric_state_retained_key_count: int = 0
    metric_state_retention_limit: int = MAX_METRIC_STATE_KEYS
    active_hard_block_records: tuple[tuple[str, int, str], ...] = ()
    counterfactual_state_records: tuple[tuple[int, str, str], ...] = ()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_set_dir", type=Path)
    parser.add_argument("--baseline-sha256", required=True)
    parser.add_argument("--expected-git-sha", required=True)
    parser.add_argument("--engineering-baseline", type=Path, required=True)
    parser.add_argument("--recovery-evidence", type=Path, required=True)
    parser.add_argument("--counterfactual-canary", type=Path, required=True)
    parser.add_argument(
        "--expected-seed",
        type=int,
        action="append",
        dest="expected_seeds",
    )
    arguments = parser.parse_args()
    run_set = arguments.run_set_dir.resolve()
    engineering_baseline = json.loads(arguments.engineering_baseline.read_text(encoding="utf-8"))
    baseline_bytes_per_loop = float(engineering_baseline["natural_run_bytes_per_game_loop"])
    recovery_evidence = json.loads(arguments.recovery_evidence.read_text(encoding="utf-8"))
    counterfactual_canary = json.loads(arguments.counterfactual_canary.read_text(encoding="utf-8"))
    rows = list(csv.DictReader((run_set / "experiment-status.tsv").open(), delimiter="\t"))
    metrics = [
        _run_metrics(
            row,
            natural_run_baseline_bytes_per_loop=baseline_bytes_per_loop,
            expected_git_sha=arguments.expected_git_sha,
            recovery_evidence=recovery_evidence,
        )
        for row in rows
    ]
    report = _comparison(
        metrics,
        baseline_sha256=arguments.baseline_sha256,
        counterfactual_canary=counterfactual_canary,
        expected_git_sha=arguments.expected_git_sha,
        expected_seeds=tuple(arguments.expected_seeds or (0, 1, 2)),
    )
    (run_set / "comparison.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_set / "report.md").write_text(_markdown(report), encoding="utf-8")
    raise SystemExit(0 if report["accepted"] else 1)


def _run_metrics(
    row: dict[str, str],
    *,
    natural_run_baseline_bytes_per_loop: float | None = None,
    expected_git_sha: str | None = None,
    recovery_evidence: dict[str, Any] | None = None,
    metric_state_limit: int = MAX_METRIC_STATE_KEYS,
) -> RunMetrics:
    run_dir_value = row.get("run_dir", "").strip()
    run_dir = (
        Path(run_dir_value).expanduser()
        if run_dir_value
        else Path("/__rtscortex_missing_run_dir__")
    )
    journal = run_dir / "events.jsonl"
    episode_result: dict[str, Any] = {}
    terminal_counts: Counter[str] = Counter()
    dispatch_counts: Counter[str] = Counter()
    consequences: Counter[str] = Counter()
    error_signatures: Counter[str] = Counter()
    playbook_applications = 0
    playbook_nonzero_score_applications = 0
    playbook_blocks = 0
    candidate_outside_dispatch = 0
    max_game_loop = 0
    lineaged_operation_ids: set[str] = set()
    rule_evaluations: dict[str, dict[str, Any]] = {}
    rule_evaluation_overflow_count = 0
    first_timestamp: datetime | None = None
    last_timestamp: datetime | None = None
    event_count = 0
    performance: dict[str, Any] = {}
    engineering_accumulator = EngineeringAccumulator()
    metric_state_budget = _MetricStateBudget(limit=metric_state_limit)
    if journal.is_file():
        for event in read_event_log(journal):
            engineering_accumulator.ingest(event)
            event_count += 1
            try:
                timestamp = datetime.fromisoformat(event.created_at)
            except ValueError:
                timestamp = None
            if timestamp is not None:
                first_timestamp = timestamp if first_timestamp is None else first_timestamp
                last_timestamp = timestamp
            value = event.payload.get("game_loop")
            if isinstance(value, int | float):
                max_game_loop = max(max_game_loop, int(value))
            if event.event_type == "episode_result":
                episode_result = event.payload
            elif event.event_type == "execution":
                command_id = str(event.payload.get("command_id"))
                metric_state_budget.increment(terminal_counts, command_id)
                candidate_outside_dispatch += (
                    event.payload.get("failure_code") == "candidate_outside_dispatch"
                )
            elif (
                event.event_type == "command_lifecycle"
                and event.payload.get("status") == "dispatched"
            ):
                command = event.payload.get("command")
                if isinstance(command, dict):
                    metric_state_budget.increment(
                        dispatch_counts,
                        str(command.get("command_id")),
                    )
            elif event.event_type == "command_lineage":
                lineage = event.payload.get("lineage", event.payload)
                operation_id = lineage.get("operation_id") if isinstance(lineage, dict) else None
                if isinstance(operation_id, str):
                    metric_state_budget.add(lineaged_operation_ids, operation_id)
            elif event.event_type == "strategic_consequence_attributed":
                consequence_type = str(event.payload.get("consequence_type", "unknown"))
                metric_state_budget.increment(consequences, consequence_type)
                if consequence_type in _ERROR_CONSEQUENCES:
                    metric_state_budget.increment(
                        error_signatures,
                        _consequence_signature(event.payload),
                    )
            elif event.event_type == "playbook_rule_applied":
                playbook_applications += 1
                playbook_nonzero_score_applications += (
                    float(event.payload.get("score_delta", 0.0)) != 0.0
                )
                playbook_blocks += event.payload.get("blocked") is True
            elif event.event_type == "event_store_performance":
                performance = event.payload
            elif event.event_type == "playbook_rule_evaluated":
                evaluation_id = event.payload.get("evaluation_id")
                if isinstance(evaluation_id, str):
                    if (
                        evaluation_id in rule_evaluations
                        or len(rule_evaluations) < MAX_RULE_EVALUATIONS
                    ):
                        rule_evaluations[evaluation_id] = event.payload
                    else:
                        rule_evaluation_overflow_count += 1

    elapsed_seconds = (
        0.0
        if first_timestamp is None or last_timestamp is None
        else max(0.0, (last_timestamp - first_timestamp).total_seconds())
    )
    repeated_errors = sum(max(0, count - 1) for count in error_signatures.values())
    lineaged_operation_count = len(lineaged_operation_ids) or len(dispatch_counts)
    hard_evaluations = [
        evaluation
        for evaluation in rule_evaluations.values()
        if evaluation.get("strength_at_evaluation") == "hard"
        and evaluation.get("status_at_evaluation") == "active"
    ]
    would_block_evaluations = [
        evaluation
        for evaluation in hard_evaluations
        if evaluation.get("shadow_decision") == "would_block"
    ]
    invalid_counterfactual_evidence = [
        evaluation
        for evaluation in would_block_evaluations
        if not (
            isinstance(evaluation.get("counterfactual_key"), str)
            and isinstance(evaluation.get("counterfactual_signature"), str)
            and isinstance(evaluation.get("behavior_before_hash"), str)
            and isinstance(evaluation.get("decision_epoch"), int)
            and evaluation.get("counterfactual_observable") in {True, False}
            and evaluation.get("rule_kind") in {"execution_guard", "strategy"}
        )
    ]
    blocking_counterfactuals = [
        evaluation
        for evaluation in hard_evaluations
        if evaluation.get("shadow_decision") == "would_block"
        and evaluation.get("counterfactual_observable") is True
        and evaluation.get("rule_kind") == "execution_guard"
        and isinstance(evaluation.get("counterfactual_signature"), str)
        and isinstance(evaluation.get("behavior_before_hash"), str)
        and isinstance(evaluation.get("decision_epoch"), int)
    ]
    false_blocks = sum(
        evaluation.get("execution_false_block", evaluation.get("false_block")) is True
        for evaluation in blocking_counterfactuals
    )
    resolved_blocks = sum(
        isinstance(
            evaluation.get("execution_false_block", evaluation.get("false_block")),
            bool,
        )
        for evaluation in blocking_counterfactuals
    )
    unresolved_blocks = sum(
        evaluation.get("execution_false_block", evaluation.get("false_block")) is None
        for evaluation in blocking_counterfactuals
    )
    active_hard_block_keys = tuple(
        sorted(
            {
                str(evaluation["counterfactual_key"])
                for evaluation in hard_evaluations
                if evaluation.get("actual_outcome") == "blocked"
                and isinstance(evaluation.get("counterfactual_signature"), str)
                and isinstance(evaluation.get("behavior_before_hash"), str)
                and isinstance(evaluation.get("decision_epoch"), int)
                and isinstance(evaluation.get("counterfactual_key"), str)
            }
        )
    )
    shadow_would_block_keys = tuple(
        sorted(
            {
                str(evaluation["counterfactual_key"])
                for evaluation in hard_evaluations
                if evaluation.get("shadow_decision") == "would_block"
                and evaluation.get("actual_outcome") != "blocked"
                and isinstance(evaluation.get("counterfactual_key"), str)
            }
        )
    )
    active_hard_block_records = tuple(
        sorted(
            (
                str(evaluation["counterfactual_key"]),
                int(evaluation["decision_epoch"]),
                str(evaluation["behavior_before_hash"]),
            )
            for evaluation in hard_evaluations
            if evaluation.get("actual_outcome") == "blocked"
            and isinstance(evaluation.get("counterfactual_key"), str)
            and isinstance(evaluation.get("decision_epoch"), int)
            and isinstance(evaluation.get("behavior_before_hash"), str)
        )
    )
    counterfactual_state_records = tuple(
        sorted(
            (
                int(evaluation["decision_epoch"]),
                str(evaluation["counterfactual_signature"]),
                str(evaluation["behavior_before_hash"]),
            )
            for evaluation in hard_evaluations
            if evaluation.get("shadow_decision") == "would_block"
            and isinstance(evaluation.get("decision_epoch"), int)
            and isinstance(evaluation.get("counterfactual_signature"), str)
            and isinstance(evaluation.get("behavior_before_hash"), str)
        )
    )
    resolved_counterfactual_keys = tuple(
        sorted(
            {
                str(evaluation["counterfactual_key"])
                for evaluation in hard_evaluations
                if evaluation.get("counterfactual_observable") is True
                and isinstance(evaluation.get("counterfactual_signature"), str)
                and isinstance(evaluation.get("behavior_before_hash"), str)
                and isinstance(evaluation.get("decision_epoch"), int)
                and (
                    isinstance(evaluation.get("execution_false_block"), bool)
                    or isinstance(evaluation.get("strategic_regret"), bool)
                )
                and isinstance(evaluation.get("counterfactual_key"), str)
            }
        )
    )
    strategic_evaluations = [
        evaluation
        for evaluation in hard_evaluations
        if evaluation.get("shadow_decision") == "would_block"
        and evaluation.get("counterfactual_observable") is True
        and evaluation.get("rule_kind") == "strategy"
        and isinstance(evaluation.get("counterfactual_signature"), str)
        and isinstance(evaluation.get("behavior_before_hash"), str)
        and isinstance(evaluation.get("decision_epoch"), int)
    ]
    terminal_report_count = sum(terminal_counts.values())
    dispatched_ids = set(dispatch_counts)
    terminal_ids = set(terminal_counts)
    journal_bytes = journal.stat().st_size if journal.is_file() else 0
    artifact_bytes = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
    engineering = build_engineering_gate_report(
        engineering_accumulator,
        run_dir=run_dir,
        natural_run_baseline_bytes_per_loop=natural_run_baseline_bytes_per_loop,
        recovery_evidence=recovery_evidence,
        expected_git_sha=expected_git_sha,
    )
    if run_dir.is_dir():
        (run_dir / ENGINEERING_GATES_FILENAME).write_text(
            json.dumps(engineering, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    engineering_gates = {
        name: engineering.get("gates", {}).get(name, {}).get("passed") is True
        for name in REQUIRED_ENGINEERING_GATES
    }
    git_head_before = row.get("git_head_before") or row.get("git_head", "")
    git_head_after = row.get("git_head_after") or row.get("git_head", "")
    dirty_before = row.get("superproject_dirty_before") or row.get("git_dirty", "")
    dirty_after = row.get("superproject_dirty_after") or row.get("git_dirty", "")
    submodule_commit_before = row.get("submodule_commit_before", "")
    submodule_commit_after = row.get("submodule_commit_after", "")
    submodule_dirty_before = row.get("submodule_dirty_before", "")
    submodule_dirty_after = row.get("submodule_dirty_after", "")
    submodule_diff_before = row.get("submodule_diff_sha256_before", "")
    submodule_diff_after = row.get("submodule_diff_sha256_after", "")
    has_source_attestation = all(
        (
            git_head_before,
            git_head_after,
            dirty_before,
            dirty_after,
            submodule_commit_before,
            submodule_commit_after,
            submodule_dirty_before,
            submodule_dirty_after,
            submodule_diff_before,
            submodule_diff_after,
        )
    )
    source_attestation_consistent = (
        has_source_attestation
        and git_head_before == git_head_after
        and dirty_before == dirty_after
        and submodule_commit_before == submodule_commit_after
        and submodule_dirty_before == submodule_dirty_after
        and submodule_diff_before == submodule_diff_after
    )
    source_attestation_fingerprint = (
        _source_attestation_fingerprint(
            git_head=git_head_before,
            superproject_dirty=dirty_before,
            submodule_commit=submodule_commit_before,
            submodule_dirty=submodule_dirty_before,
            submodule_diff_sha256=submodule_diff_before,
        )
        if source_attestation_consistent
        else None
    )
    peak_rss_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    analysis_evidence_overflow_count = (
        engineering_accumulator.retention_overflow_count
        + rule_evaluation_overflow_count
        + metric_state_budget.overflow_count
    )
    return RunMetrics(
        mode=row["mode"],
        seed=int(row["seed"]),
        arm=row["arm"],
        exit_code=int(row["exit_code"]),
        run_dir=str(run_dir),
        natural_terminal=(
            episode_result.get("outcome") in {"victory", "defeat", "draw"}
            and not bool(episode_result.get("failure_reason"))
        ),
        outcome=(
            str(episode_result["outcome"]) if episode_result.get("outcome") is not None else None
        ),
        score=float(episode_result.get("score", 0.0)),
        terminal_report_count=terminal_report_count,
        missing_terminal_reports=len(dispatched_ids - terminal_ids),
        duplicate_terminal_reports=sum(
            count - 1 for count in terminal_counts.values() if count > 1
        ),
        duplicate_dispatches=sum(count - 1 for count in dispatch_counts.values() if count > 1),
        candidate_outside_dispatch=candidate_outside_dispatch,
        playbook_applications=playbook_applications,
        playbook_nonzero_score_applications=playbook_nonzero_score_applications,
        playbook_blocks=playbook_blocks,
        repeated_eligible_errors=repeated_errors,
        repeated_errors_per_10k_game_loops=(
            repeated_errors * 10_000 / max_game_loop if max_game_loop else 0.0
        ),
        repeated_errors_per_lineaged_operation=(
            repeated_errors / lineaged_operation_count if lineaged_operation_count else 0.0
        ),
        lineaged_operation_count=lineaged_operation_count,
        strategic_consequences=dict(sorted(consequences.items())),
        hard_rule_false_block_count=false_blocks,
        hard_rule_shadow_state_count=resolved_blocks,
        hard_rule_unresolved_block_count=unresolved_blocks,
        hard_rule_false_block_rate=(false_blocks / resolved_blocks if resolved_blocks else 0.0),
        journal_bytes=journal_bytes,
        artifact_bytes=artifact_bytes,
        event_count=event_count,
        max_game_loop=max_game_loop,
        events_per_game_loop=(event_count / max_game_loop if max_game_loop else 0.0),
        bytes_per_game_loop=(journal_bytes / max_game_loop if max_game_loop else 0.0),
        effective_game_loops_per_second=(
            max_game_loop / elapsed_seconds if elapsed_seconds else 0.0
        ),
        writer_queue_peak=int(performance.get("max_queue_depth", 0)),
        writer_lag_ms_p95=float(performance.get("writer_lag_ms_p95", 0.0)),
        writer_lag_ms_max=float(performance.get("writer_lag_ms_max", 0.0)),
        blocked_append_count=int(performance.get("blocked_append_count", 0)),
        sampled_drop_supported=bool(performance.get("sampled_drop_supported", False)),
        playbook_before_sha256=row["playbook_before_sha256"],
        playbook_after_sha256=row["playbook_after_sha256"],
        experiment_kind=row.get("experiment_kind") or "behavior",
        subject_arm=row.get("subject_arm") or None,
        active_hard_block_keys=active_hard_block_keys,
        shadow_would_block_keys=shadow_would_block_keys,
        resolved_counterfactual_keys=resolved_counterfactual_keys,
        strategic_regret_count=sum(
            evaluation.get("strategic_regret") is True for evaluation in strategic_evaluations
        ),
        strategic_resolved_count=sum(
            isinstance(evaluation.get("strategic_regret"), bool)
            for evaluation in strategic_evaluations
        ),
        invalid_counterfactual_evidence_count=len(invalid_counterfactual_evidence),
        engineering_gates=engineering_gates,
        engineering_missing_metrics=tuple(engineering["missing_required_metrics"]),
        engineering_accepted=engineering["accepted"] is True,
        source_tree_clean=dirty_before == "false" and dirty_after == "false",
        source_commit_matches_expected_sha=(
            expected_git_sha is not None
            and git_head_before == expected_git_sha
            and git_head_after == expected_git_sha
        ),
        source_attestation_consistent=source_attestation_consistent,
        source_attestation_fingerprint=source_attestation_fingerprint,
        retained_event_count=len(engineering_accumulator.events),
        rule_evaluation_count=len(rule_evaluations),
        analysis_peak_rss_kib=peak_rss_kib,
        analysis_rss_per_10k_game_loops=(
            peak_rss_kib * 10_000 / max_game_loop if max_game_loop else 0.0
        ),
        analysis_evidence_overflow_count=analysis_evidence_overflow_count,
        metric_state_retained_key_count=metric_state_budget.retained_key_count,
        metric_state_retention_limit=metric_state_budget.limit,
        active_hard_block_records=active_hard_block_records,
        counterfactual_state_records=counterfactual_state_records,
    )


def _consequence_signature(payload: dict[str, Any]) -> str:
    condition = payload.get("condition")
    normalized_condition = condition if isinstance(condition, dict) else {}
    return json.dumps(
        {
            "consequence_type": payload.get("consequence_type"),
            "role": payload.get("role"),
            "semantic_action": payload.get("semantic_action"),
            "condition": normalized_condition,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _source_attestation_fingerprint(
    *,
    git_head: str,
    superproject_dirty: str,
    submodule_commit: str,
    submodule_dirty: str,
    submodule_diff_sha256: str,
) -> str:
    payload = json.dumps(
        {
            "git_head": git_head,
            "superproject_dirty": superproject_dirty,
            "submodule_commit": submodule_commit,
            "submodule_dirty": submodule_dirty,
            "submodule_diff_sha256": submodule_diff_sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def counterfactual_canary_is_valid(
    artifact: dict[str, Any],
    *,
    baseline_sha256: str,
    expected_git_sha: str | None,
) -> bool:
    gates = artifact.get("gates")
    behavior = artifact.get("behavior")
    shadow = artifact.get("shadow")
    readiness = artifact.get("hard_readiness")
    return (
        artifact.get("schema_version") == "1.0"
        and artifact.get("canary_kind") == "production"
        and artifact.get("canary_fixture") is False
        and artifact.get("accepted") is True
        and artifact.get("baseline_sha256") == baseline_sha256
        and (expected_git_sha is None or artifact.get("expected_git_sha") == expected_git_sha)
        and isinstance(gates, dict)
        and bool(gates)
        and all(value is True for value in gates.values())
        and isinstance(readiness, dict)
        and readiness.get("canary_runnable") is True
        and readiness.get("baseline_sha256") == baseline_sha256
        and readiness.get("expected_git_sha") == expected_git_sha
        and int(readiness.get("context_applicable_blocking_hard_count", 0)) >= 1
        and not bool(readiness.get("canary_fixture_rule_ids"))
        and int(artifact.get("active_hard_block_count", 0)) > 0
        and int(artifact.get("matched_counterfactual_count", 0)) > 0
        and int(artifact.get("unmatched_active_hard_block_count", -1)) == 0
        and isinstance(behavior, dict)
        and isinstance(shadow, dict)
        and behavior.get("source_attestation_fingerprint")
        == shadow.get("source_attestation_fingerprint")
        and behavior.get("analysis_evidence_overflow_count") == 0
        and shadow.get("analysis_evidence_overflow_count") == 0
    )


def _comparison(
    metrics: list[RunMetrics],
    *,
    baseline_sha256: str,
    counterfactual_canary: dict[str, Any] | None = None,
    expected_git_sha: str | None = None,
    expected_seeds: tuple[int, ...] = (0, 1, 2),
) -> dict[str, Any]:
    behavior = [metric for metric in metrics if metric.experiment_kind == "behavior"]
    calibration = [metric for metric in metrics if metric.experiment_kind == "calibration"]
    split_matrix = any(metric.subject_arm is not None for metric in metrics)
    unique_expected_seeds = tuple(dict.fromkeys(expected_seeds))
    if len(unique_expected_seeds) != 3:
        raise ValueError("formal comparison requires exactly three distinct held-out seeds")
    expected_behavior_matrix = {
        (mode, seed, arm)
        for mode in ("independent_paired", "sequential_learning")
        for seed in unique_expected_seeds
        for arm in ("frozen", "evolving")
    }
    observed_behavior_matrix = {(metric.mode, metric.seed, metric.arm) for metric in behavior}
    expected_calibration_matrix = {
        (mode, seed, subject_arm)
        for mode in ("independent_paired", "sequential_learning")
        for seed in unique_expected_seeds
        for subject_arm in ("frozen", "evolving")
    }
    observed_calibration_matrix = {
        (metric.mode, metric.seed, metric.subject_arm) for metric in calibration
    }
    paired: list[dict[str, Any]] = []
    for mode in ("independent_paired", "sequential_learning"):
        for seed in sorted({metric.seed for metric in behavior if metric.mode == mode}):
            pair = {
                metric.arm: metric
                for metric in behavior
                if metric.mode == mode and metric.seed == seed
            }
            if set(pair) != {"frozen", "evolving"}:
                continue
            frozen = pair["frozen"]
            evolving = pair["evolving"]
            paired.append(
                {
                    "mode": mode,
                    "seed": seed,
                    "score_delta": evolving.score - frozen.score,
                    "repeated_error_delta": (
                        evolving.repeated_eligible_errors - frozen.repeated_eligible_errors
                    ),
                    "repeated_error_rate_per_10k_delta": (
                        evolving.repeated_errors_per_10k_game_loops
                        - frozen.repeated_errors_per_10k_game_loops
                    ),
                    "repeated_error_per_operation_delta": (
                        evolving.repeated_errors_per_lineaged_operation
                        - frozen.repeated_errors_per_lineaged_operation
                    ),
                    "win_delta": _win(evolving.outcome) - _win(frozen.outcome),
                }
            )
    independent_rows = [metric for metric in behavior if metric.mode == "independent_paired"]
    sequential_rows = [metric for metric in behavior if metric.mode == "sequential_learning"]
    independent_pairs = [item for item in paired if item["mode"] == "independent_paired"]
    baseline_identity = all(
        metric.playbook_before_sha256 == baseline_sha256 for metric in independent_rows
    )
    frozen_immutable = all(
        metric.playbook_before_sha256 == metric.playbook_after_sha256
        for metric in behavior
        if metric.arm == "frozen"
    )
    evolving_sequence = sorted(
        (metric for metric in sequential_rows if metric.arm == "evolving"),
        key=lambda metric: metric.seed,
    )
    sequential_carry = all(
        current.playbook_before_sha256 == previous.playbook_after_sha256
        for previous, current in zip(evolving_sequence, evolving_sequence[1:], strict=False)
    )
    behavior_by_subject = {(metric.mode, metric.seed, metric.arm): metric for metric in behavior}
    shadow_baseline_identity = all(
        (reference := behavior_by_subject.get((metric.mode, metric.seed, metric.subject_arm or "")))
        is not None
        and metric.playbook_before_sha256 == reference.playbook_before_sha256
        for metric in calibration
    )
    shadow_immutable = all(
        metric.playbook_before_sha256 == metric.playbook_after_sha256 for metric in calibration
    )
    frozen_errors = sum(
        metric.repeated_eligible_errors for metric in independent_rows if metric.arm == "frozen"
    )
    evolving_errors = sum(
        metric.repeated_eligible_errors for metric in independent_rows if metric.arm == "evolving"
    )
    frozen_game_loops = sum(
        metric.max_game_loop for metric in independent_rows if metric.arm == "frozen"
    )
    evolving_game_loops = sum(
        metric.max_game_loop for metric in independent_rows if metric.arm == "evolving"
    )
    frozen_error_rate = frozen_errors * 10_000 / frozen_game_loops if frozen_game_loops else 0.0
    evolving_error_rate = (
        evolving_errors * 10_000 / evolving_game_loops if evolving_game_loops else 0.0
    )
    reduction = (
        0.0
        if frozen_error_rate == 0.0
        else (frozen_error_rate - evolving_error_rate) / frozen_error_rate
    )
    counterfactual_rows = calibration if split_matrix else behavior
    false_blocks = sum(metric.hard_rule_false_block_count for metric in counterfactual_rows)
    resolved_blocks = sum(metric.hard_rule_shadow_state_count for metric in counterfactual_rows)
    unresolved_hard_blocks = sum(
        metric.hard_rule_unresolved_block_count for metric in counterfactual_rows
    )
    active_hard_block_keys = {key for metric in behavior for key in metric.active_hard_block_keys}
    resolved_counterfactual_keys = {
        key for metric in counterfactual_rows for key in metric.resolved_counterfactual_keys
    }
    calibration_by_subject = {
        (metric.mode, metric.seed, metric.subject_arm): metric for metric in calibration
    }
    unmatched_active_hard_blocks: list[str] = []
    for metric in behavior:
        matched_keys = resolved_counterfactual_keys
        if split_matrix:
            matched = calibration_by_subject.get((metric.mode, metric.seed, metric.arm))
            matched_keys = set() if matched is None else set(matched.resolved_counterfactual_keys)
        unmatched_active_hard_blocks.extend(
            f"{metric.mode}:{metric.seed}:{metric.arm}:{key}"
            for key in metric.active_hard_block_keys
            if key not in matched_keys
        )
    unmatched_active_hard_blocks.sort()
    engineering_gate_results = {
        name: all(metric.engineering_gates.get(name) is True for metric in metrics)
        for name in REQUIRED_ENGINEERING_GATES
    }
    missing_engineering_metrics = sorted(
        {name for metric in metrics for name in metric.engineering_missing_metrics}
    )
    invalid_counterfactual_evidence_count = sum(
        metric.invalid_counterfactual_evidence_count for metric in metrics
    )
    source_fingerprints = {
        metric.source_attestation_fingerprint
        for metric in metrics
        if metric.source_attestation_fingerprint is not None
    }
    gates = {
        "complete_unique_run_matrix": (
            len(behavior) == len(expected_behavior_matrix)
            and observed_behavior_matrix == expected_behavior_matrix
            and (
                not split_matrix
                or (
                    len(calibration) == len(expected_calibration_matrix)
                    and observed_calibration_matrix == expected_calibration_matrix
                )
            )
        ),
        "all_runs_exit_zero": all(metric.exit_code == 0 for metric in metrics),
        "all_runs_natural_terminal": all(metric.natural_terminal for metric in metrics),
        "source_tree_clean": all(metric.source_tree_clean for metric in metrics),
        "source_commit_matches_expected_sha": all(
            metric.source_commit_matches_expected_sha for metric in metrics
        ),
        "source_attestation_consistent": all(
            metric.source_attestation_consistent for metric in metrics
        )
        and all(metric.source_attestation_fingerprint is not None for metric in metrics)
        and len(source_fingerprints) == 1,
        "independent_baseline_identity": baseline_identity,
        "frozen_hash_unchanged": frozen_immutable,
        "sequential_evolving_carry": sequential_carry,
        "shadow_calibration_baseline_identity": (
            shadow_baseline_identity if split_matrix else True
        ),
        "shadow_calibration_hash_unchanged": (shadow_immutable if split_matrix else True),
        "duplicate_dispatch_zero": all(metric.duplicate_dispatches == 0 for metric in metrics),
        "terminal_report_exactly_once": all(
            metric.duplicate_terminal_reports == 0 and metric.missing_terminal_reports == 0
            for metric in metrics
        ),
        "candidate_outside_dispatch_zero": all(
            metric.candidate_outside_dispatch == 0 for metric in metrics
        ),
        "hard_false_block_rate_at_most_1_percent": (
            resolved_blocks > 0
            and unresolved_hard_blocks == 0
            and false_blocks / resolved_blocks <= 0.01
        ),
        "active_hard_blocks_have_matched_shadow_evidence": (not unmatched_active_hard_blocks),
        "counterfactual_schema_complete": invalid_counterfactual_evidence_count == 0,
        "counterfactual_canary_accepted": (
            True
            if counterfactual_canary is None
            else counterfactual_canary_is_valid(
                counterfactual_canary,
                baseline_sha256=baseline_sha256,
                expected_git_sha=expected_git_sha,
            )
        ),
        "analysis_memory_budget_respected": all(
            metric.analysis_evidence_overflow_count == 0 for metric in metrics
        ),
        "required_engineering_metrics_complete": not missing_engineering_metrics,
        "all_engineering_gates_pass": (
            bool(metrics)
            and not missing_engineering_metrics
            and all(metric.engineering_accepted for metric in metrics)
            and all(engineering_gate_results.values())
        ),
        "repeated_error_reduction_at_least_50_percent": reduction >= 0.5,
        "matched_win_rate_not_reduced": (sum(item["win_delta"] for item in independent_pairs) >= 0),
    }
    return {
        "schema_version": "1.2",
        "baseline_sha256": baseline_sha256,
        "expected_seed_ids": list(unique_expected_seeds),
        "runs": [asdict(metric) for metric in metrics],
        "paired_differences": paired,
        "aggregate": {
            "independent_frozen_repeated_errors": frozen_errors,
            "independent_evolving_repeated_errors": evolving_errors,
            "independent_frozen_repeated_errors_per_10k_game_loops": frozen_error_rate,
            "independent_evolving_repeated_errors_per_10k_game_loops": (evolving_error_rate),
            "independent_repeated_error_reduction": reduction,
            "hard_rule_false_block_count": false_blocks,
            "hard_rule_resolved_block_count": resolved_blocks,
            "hard_rule_shadow_state_count": resolved_blocks,
            "hard_rule_unresolved_block_count": unresolved_hard_blocks,
            "hard_rule_false_block_rate": (
                false_blocks / resolved_blocks if resolved_blocks else 0.0
            ),
            "active_hard_block_count": len(active_hard_block_keys),
            "unmatched_active_hard_block_count": len(unmatched_active_hard_blocks),
            "unmatched_active_hard_block_keys": unmatched_active_hard_blocks,
            "strategic_regret_count": sum(
                metric.strategic_regret_count for metric in counterfactual_rows
            ),
            "strategic_resolved_count": sum(
                metric.strategic_resolved_count for metric in counterfactual_rows
            ),
            "invalid_counterfactual_evidence_count": invalid_counterfactual_evidence_count,
            "analysis_evidence_overflow_count": sum(
                metric.analysis_evidence_overflow_count for metric in metrics
            ),
            "engineering_gates": engineering_gate_results,
            "missing_engineering_metrics": missing_engineering_metrics,
        },
        "gates": gates,
        "accepted": all(gates.values()),
    }


def _win(outcome: str | None) -> int:
    return int(outcome in {"victory", "win"})


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Playbook Experiment Report",
        "",
        f"Accepted: **{report['accepted']}**",
        "",
        "## Engineering and causal gates",
        "",
        "| Gate | Result |",
        "|---|---:|",
    ]
    lines.extend(
        f"| `{name}` | {'PASS' if passed else 'FAIL'} |" for name, passed in report["gates"].items()
    )
    lines.extend(
        [
            "",
            "## Aggregate engineering gates",
            "",
            "| Engineering gate | Result |",
            "|---|---:|",
        ]
    )
    lines.extend(
        f"| `{name}` | {'PASS' if passed else 'FAIL'} |"
        for name, passed in report["aggregate"]["engineering_gates"].items()
    )
    aggregate = report["aggregate"]
    lines.extend(
        [
            "",
            "## Counterfactual calibration",
            "",
            f"- Resolved execution-guard blocks: `{aggregate['hard_rule_resolved_block_count']}`",
            f"- Unresolved observable execution-guard blocks: "
            f"`{aggregate['hard_rule_unresolved_block_count']}`",
            f"- Execution false blocks: `{aggregate['hard_rule_false_block_count']}`",
            f"- Strategic outcomes resolved: `{aggregate['strategic_resolved_count']}`; "
            f"regret: `{aggregate['strategic_regret_count']}`",
            f"- Active hard blocks without matched shadow evidence: "
            f"`{aggregate['unmatched_active_hard_block_count']}`",
        ]
    )
    lines.extend(
        [
            "",
            "## Paired differences",
            "",
            "| Mode | Seed | Score delta | Error delta | Error/10k delta "
            "| Error/op delta | Win delta |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    lines.extend(
        (
            f"| {item['mode']} | {item['seed']} | {item['score_delta']:.3f} | "
            f"{item['repeated_error_delta']} | "
            f"{item['repeated_error_rate_per_10k_delta']:.3f} | "
            f"{item['repeated_error_per_operation_delta']:.3f} | "
            f"{item['win_delta']} |"
        )
        for item in report["paired_differences"]
    )
    lines.extend(
        [
            "",
            "## Runtime persistence",
            "",
            "| Mode | Seed | Arm | loops/s | events/loop | bytes/loop "
            "| artifact bytes | queue peak | writer p95 ms | blocked | sampled drop |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    lines.extend(
        (
            f"| {item['mode']} | {item['seed']} | {item['arm']} | "
            f"{item['effective_game_loops_per_second']:.3f} | "
            f"{item['events_per_game_loop']:.3f} | "
            f"{item['bytes_per_game_loop']:.3f} | "
            f"{item['artifact_bytes']} | {item['writer_queue_peak']} | "
            f"{item['writer_lag_ms_p95']:.3f} | {item['blocked_append_count']} | "
            f"{'supported' if item['sampled_drop_supported'] else 'unsupported'} |"
        )
        for item in report["runs"]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
