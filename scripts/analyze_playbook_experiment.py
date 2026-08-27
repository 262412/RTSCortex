"""Build conserved acceptance metrics for Playbook experiment run sets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
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
from rtscortex.evaluation.metrics import compute_execution_metrics
from rtscortex.evaluation.report import _build_run_summary
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
GAMEPLAY_GATE_NAMES = (
    "upstream_placement_rejections",
    "meaningful_command_success_rate",
    "completed_execution_success_rate",
    "build_pre_dispatch_rejection_rate",
)
CANARY_GATE_KEYS_BASE = frozenset(
    {
        "hard_readiness_accepted",
        "runs_exit_zero",
        "execution_seed_contract",
        "runs_complete_for_kind",
        "source_attestation_matches",
        "active_hard_block_observed",
        "all_active_hard_blocks_matched",
        "matched_prestate_identity",
        "rule_evaluation_kind_consistent",
        "analysis_memory_budget_respected",
        "active_role_contract",
        "shadow_role_contract",
        "run_identity_contract",
        "matching_provenance",
        "active_block_prestate_identity",
    }
)
CANARY_GATE_KEYS_PRODUCTION = CANARY_GATE_KEYS_BASE | {"terminal_counterfactual_resolved"}
CANARY_GATE_KEYS_FIXTURE = CANARY_GATE_KEYS_BASE | {"matched_shadow_guard_allow_observed"}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


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
    hard_rule_kind_records: tuple[tuple[str, str], ...] = ()
    gameplay_metrics: dict[str, int | float] = field(default_factory=dict)
    gameplay_gate_results: dict[str, bool] = field(
        default_factory=lambda: {name: False for name in GAMEPLAY_GATE_NAMES}
    )
    gameplay_acceptance_accepted: bool = False
    events_sha256: str | None = None
    summary_sha256: str | None = None
    artifact_integrity_valid: bool = False
    legacy_acceptance_valid: bool = False
    event_identity_valid: bool = True
    event_run_id: str | None = None
    event_episode_ids: tuple[str, ...] = ()
    source_attestation: dict[str, str] = field(default_factory=dict)


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
        run_set_dir=run_set,
        counterfactual_canary_run_set_dir=(
            arguments.counterfactual_canary.expanduser().resolve().parent
        ),
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
    persist_engineering_report: bool = True,
) -> RunMetrics:
    run_dir_value = row.get("run_dir", "").strip()
    run_dir = (
        Path(run_dir_value).expanduser().resolve()
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
    hard_rule_kind_records = tuple(
        sorted(
            {
                (str(evaluation["rule_id"]), str(evaluation["rule_kind"]))
                for evaluation in hard_evaluations
                if isinstance(evaluation.get("rule_id"), str)
                and isinstance(evaluation.get("rule_kind"), str)
            }
        )
    )
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
    if persist_engineering_report and run_dir.is_dir():
        (run_dir / ENGINEERING_GATES_FILENAME).write_text(
            json.dumps(engineering, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    engineering_gates = {
        name: engineering.get("gates", {}).get(name, {}).get("passed") is True
        for name in REQUIRED_ENGINEERING_GATES
    }
    gameplay_metrics, gameplay_gate_results = _recompute_gameplay_acceptance(journal)
    (
        events_sha256,
        summary_sha256,
        artifact_integrity_valid,
        legacy_acceptance_valid,
    ) = _validate_run_artifacts(
        run_dir,
        row,
    )
    event_identity_valid, event_run_id, event_episode_ids = _validate_event_identity(
        run_dir,
        row,
    )
    git_head_before = row.get("git_head_before") or row.get("git_head", "")
    git_head_after = row.get("git_head_after") or row.get("git_head", "")
    dirty_before = row.get("superproject_dirty_before") or row.get("git_dirty", "")
    dirty_after = row.get("superproject_dirty_after") or row.get("git_dirty", "")
    submodule_commit_before = row.get("submodule_commit_before", "")
    submodule_commit_after = row.get("submodule_commit_after", "")
    submodule_dirty_before = row.get("submodule_dirty_before", "")
    submodule_dirty_after = row.get("submodule_dirty_after", "")
    submodule_gitlink_before = row.get("submodule_gitlink_before", "")
    submodule_gitlink_after = row.get("submodule_gitlink_after", "")
    submodule_diff_before = row.get("submodule_diff_sha256_before", "")
    submodule_diff_after = row.get("submodule_diff_sha256_after", "")
    reviewed_source_commit_before = row.get("reviewed_source_commit_before", "")
    reviewed_source_commit_after = row.get("reviewed_source_commit_after", "")
    reviewed_source_diff_before = row.get("reviewed_source_diff_sha256_before", "")
    reviewed_source_diff_after = row.get("reviewed_source_diff_sha256_after", "")
    reviewed_source_tree_before = row.get("reviewed_source_tree_sha256_before", "")
    reviewed_source_tree_after = row.get("reviewed_source_tree_sha256_after", "")
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
            submodule_gitlink_before,
            submodule_gitlink_after,
            submodule_diff_before,
            submodule_diff_after,
            reviewed_source_commit_before,
            reviewed_source_commit_after,
            reviewed_source_diff_before,
            reviewed_source_diff_after,
            reviewed_source_tree_before,
            reviewed_source_tree_after,
        )
    )
    source_attestation_consistent = (
        has_source_attestation
        and git_head_before == git_head_after
        and dirty_before == dirty_after
        and submodule_commit_before == submodule_commit_after
        and submodule_dirty_before == submodule_dirty_after == "false"
        and submodule_gitlink_before == submodule_gitlink_after
        and submodule_commit_before == submodule_gitlink_before
        and submodule_diff_before == submodule_diff_after
        and reviewed_source_commit_before
        == reviewed_source_commit_after
        == submodule_gitlink_before
        and reviewed_source_diff_before == reviewed_source_diff_after
        and reviewed_source_tree_before == reviewed_source_tree_after
    )
    source_attestation_fingerprint = (
        _source_attestation_fingerprint(
            git_head=git_head_before,
            superproject_dirty=dirty_before,
            submodule_commit=submodule_commit_before,
            submodule_dirty=submodule_dirty_before,
            submodule_gitlink=submodule_gitlink_before,
            submodule_diff_sha256=submodule_diff_before,
            reviewed_source_commit=reviewed_source_commit_before,
            reviewed_source_diff_sha256=reviewed_source_diff_before,
            reviewed_source_tree_sha256=reviewed_source_tree_before,
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
        hard_rule_kind_records=hard_rule_kind_records,
        gameplay_metrics=gameplay_metrics,
        gameplay_gate_results=gameplay_gate_results,
        gameplay_acceptance_accepted=(
            artifact_integrity_valid
            and legacy_acceptance_valid
            and all(gameplay_gate_results.values())
        ),
        events_sha256=events_sha256,
        summary_sha256=summary_sha256,
        artifact_integrity_valid=artifact_integrity_valid,
        legacy_acceptance_valid=legacy_acceptance_valid,
        event_identity_valid=event_identity_valid,
        event_run_id=event_run_id,
        event_episode_ids=event_episode_ids,
        source_attestation={
            "git_head_before": git_head_before,
            "git_head_after": git_head_after,
            "superproject_dirty_before": dirty_before,
            "superproject_dirty_after": dirty_after,
            "submodule_commit_before": submodule_commit_before,
            "submodule_commit_after": submodule_commit_after,
            "submodule_dirty_before": submodule_dirty_before,
            "submodule_dirty_after": submodule_dirty_after,
            "submodule_gitlink_before": submodule_gitlink_before,
            "submodule_gitlink_after": submodule_gitlink_after,
            "submodule_diff_sha256_before": submodule_diff_before,
            "submodule_diff_sha256_after": submodule_diff_after,
            "reviewed_source_commit_before": reviewed_source_commit_before,
            "reviewed_source_commit_after": reviewed_source_commit_after,
            "reviewed_source_diff_sha256_before": reviewed_source_diff_before,
            "reviewed_source_diff_sha256_after": reviewed_source_diff_after,
            "reviewed_source_tree_sha256_before": reviewed_source_tree_before,
            "reviewed_source_tree_sha256_after": reviewed_source_tree_after,
        },
    )


def _recompute_gameplay_acceptance(
    journal: Path,
) -> tuple[dict[str, int | float], dict[str, bool]]:
    """Recompute the gameplay contract directly from the immutable event journal."""

    if not journal.is_file():
        return {}, {name: False for name in GAMEPLAY_GATE_NAMES}
    try:
        execution = compute_execution_metrics(list(read_event_log(journal)))
    except (OSError, TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return {}, {name: False for name in GAMEPLAY_GATE_NAMES}
    values: dict[str, int | float] = {
        "upstream_placement_rejections": execution.upstream_placement_rejections,
        "meaningful_command_success_rate": execution.meaningful_action_success_rate,
        "completed_execution_success_rate": execution.completed_execution_success_rate,
        "build_pre_dispatch_rejection_rate": execution.build_pre_dispatch_rejection_rate,
    }
    gates = {
        "upstream_placement_rejections": (
            type(values["upstream_placement_rejections"]) is int
            and values["upstream_placement_rejections"] == 0
        ),
        "meaningful_command_success_rate": _finite_ratio_at_least(
            values["meaningful_command_success_rate"], 0.70
        ),
        "completed_execution_success_rate": _finite_ratio_at_least(
            values["completed_execution_success_rate"], 0.75
        ),
        "build_pre_dispatch_rejection_rate": _finite_ratio_at_most(
            values["build_pre_dispatch_rejection_rate"], 0.05
        ),
    }
    return values, gates


def _finite_ratio_at_least(value: int | float, threshold: float) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= threshold
    )


def _finite_ratio_at_most(value: int | float, threshold: float) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value <= threshold
    )


def _validate_run_artifacts(
    run_dir: Path,
    row: dict[str, str],
) -> tuple[str | None, str | None, bool, bool]:
    """Validate status-file hashes and the summary schema before accepting a run."""

    events_path = run_dir / "events.jsonl"
    summary_path = run_dir / "summary.json"
    events_sha256 = row.get("events_sha256")
    summary_sha256 = row.get("summary_sha256")
    if not (
        isinstance(events_sha256, str)
        and isinstance(summary_sha256, str)
        and _SHA256_RE.fullmatch(events_sha256)
        and _SHA256_RE.fullmatch(summary_sha256)
        and events_path.is_file()
        and summary_path.is_file()
    ):
        return events_sha256 or None, summary_sha256 or None, False, False
    try:
        actual_events_sha256 = _sha256_file(events_path)
        actual_summary_sha256 = _sha256_file(summary_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        events = list(read_event_log(events_path))
        canonical_summary = _build_run_summary(events)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        return events_sha256, summary_sha256, False, False
    schema_valid = _summary_schema_is_valid(summary)
    canonical_valid = summary == canonical_summary
    return (
        events_sha256,
        summary_sha256,
        actual_events_sha256 == events_sha256
        and actual_summary_sha256 == summary_sha256
        and schema_valid
        and canonical_valid,
        schema_valid and canonical_valid and _summary_legacy_acceptance_passed(summary),
    )


def _validate_event_identity(
    run_dir: Path,
    row: dict[str, str],
) -> tuple[bool, str | None, tuple[str, ...]]:
    """Bind one runtime-emitted experiment identity to one status row."""

    required = bool(
        row.get("experiment_kind") or row.get("run_id") or row.get("event_identity_required")
    )
    journal = run_dir / "events.jsonl"
    if not journal.is_file():
        return (not required), None, ()
    try:
        events = list(read_event_log(journal))
    except (OSError, UnicodeDecodeError, TypeError, ValueError, json.JSONDecodeError):
        return False, None, ()
    if not events:
        return (not required), None, ()

    run_ids = {event.run_id for event in events if isinstance(event.run_id, str) and event.run_id}
    episode_ids = tuple(
        sorted(
            {
                event.episode_id
                for event in events
                if isinstance(event.episode_id, str) and event.episode_id
            }
        )
    )
    identity_events = [event for event in events if event.event_type == "experiment_run_identity"]
    if not required and not identity_events:
        return len(run_ids) == 1 and bool(episode_ids), next(iter(run_ids), None), episode_ids
    if len(identity_events) != 1 or len(run_ids) != 1 or len(episode_ids) != 1:
        return False, next(iter(run_ids), None), episode_ids
    identity_event = identity_events[0]
    identity = identity_event.payload if isinstance(identity_event.payload, dict) else {}
    if set(identity) != {
        "schema_version",
        "source",
        "run_id",
        "episode_id",
        "seed",
        "mode",
        "experiment_kind",
        "arm",
        "subject_arm",
    }:
        return False, next(iter(run_ids), None), episode_ids
    expected_seed: int | None
    try:
        expected_seed = int(row["seed"])
    except (KeyError, TypeError, ValueError):
        expected_seed = None
    expected_arm = row.get("arm") or None
    expected_kind = row.get("experiment_kind") or None
    expected_subject_arm = row.get("subject_arm") or None
    expected_mode = row.get("mode") or None
    event_run_id = next(iter(run_ids))
    event_episode_id = episode_ids[0]
    valid = (
        identity.get("schema_version") == "1.0"
        and identity.get("source") == "runner_environment"
        and identity_event.run_id == event_run_id == identity.get("run_id")
        and identity_event.episode_id == event_episode_id == identity.get("episode_id")
        and (not row.get("run_id") or row.get("run_id") == event_run_id)
        and (not row.get("episode_id") or row.get("episode_id") == event_episode_id)
        and expected_seed is not None
        and type(identity.get("seed")) is int
        and identity.get("seed") == expected_seed
        and expected_mode is not None
        and identity.get("mode") == expected_mode
        and expected_arm is not None
        and identity.get("arm") == expected_arm
        and expected_kind is not None
        and identity.get("experiment_kind") == expected_kind
        and identity.get("subject_arm") == expected_subject_arm
    )
    return valid, event_run_id, episode_ids


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summary_schema_is_valid(summary: Any) -> bool:
    if not isinstance(summary, dict):
        return False
    if set(summary) != {"format_version", "source_journal", "runs"}:
        return False
    if summary.get("format_version") != "1.0" or summary.get("source_journal") != "events.jsonl":
        return False
    runs = summary.get("runs")
    if not isinstance(runs, dict) or not runs:
        return False
    for run in runs.values():
        if (
            not isinstance(run, dict)
            or set(run) != {"episodes"}
            or not isinstance(run.get("episodes"), dict)
            or not run["episodes"]
        ):
            return False
        for episode in run["episodes"].values():
            if not isinstance(episode, dict):
                return False
            hard_acceptance = episode.get("hard_acceptance")
            if (
                not isinstance(hard_acceptance, dict)
                or type(hard_acceptance.get("passed")) is not bool
            ):
                return False
    return True


def _summary_legacy_acceptance_passed(summary: dict[str, Any]) -> bool:
    runs = summary.get("runs")
    if not isinstance(runs, dict) or not runs:
        return False
    for run in runs.values():
        if not isinstance(run, dict):
            return False
        episodes = run.get("episodes")
        if not isinstance(episodes, dict) or not episodes:
            return False
        for episode in episodes.values():
            if not isinstance(episode, dict):
                return False
            hard_acceptance = episode.get("hard_acceptance")
            if not isinstance(hard_acceptance, dict) or hard_acceptance.get("passed") is not True:
                return False
    return True


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
    submodule_gitlink: str,
    submodule_diff_sha256: str,
    reviewed_source_commit: str,
    reviewed_source_diff_sha256: str,
    reviewed_source_tree_sha256: str,
) -> str:
    payload = json.dumps(
        {
            "git_head": git_head,
            "superproject_dirty": superproject_dirty,
            "submodule_commit": submodule_commit,
            "submodule_dirty": submodule_dirty,
            "submodule_gitlink": submodule_gitlink,
            "submodule_diff_sha256": submodule_diff_sha256,
            "reviewed_source_commit": reviewed_source_commit,
            "reviewed_source_diff_sha256": reviewed_source_diff_sha256,
            "reviewed_source_tree_sha256": reviewed_source_tree_sha256,
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
    expected_evaluation_seeds: tuple[int, ...],
    approved_rule_set_sha256: str | None = None,
    run_set_dir: Path | None = None,
    expected_run_set_dir: Path | None = None,
    natural_run_baseline_bytes_per_loop: float | None = None,
    recovery_evidence: dict[str, Any] | None = None,
) -> bool:
    gates = artifact.get("gates")
    gameplay_gates = artifact.get("gameplay_gates")
    behavior = artifact.get("behavior")
    shadow = artifact.get("shadow")
    readiness = artifact.get("hard_readiness")
    context_blocking_count = (
        readiness.get("context_applicable_blocking_hard_count")
        if isinstance(readiness, dict)
        else None
    )
    active_hard_block_count = artifact.get("active_hard_block_count")
    terminal_resolved_count = artifact.get("terminal_counterfactual_resolved_count")
    behavior_run_dir = (
        _canonical_run_directory(str(behavior.get("run_dir", "")))
        if isinstance(behavior, dict)
        else Path("/__rtscortex_missing_run_dir__")
    )
    shadow_run_dir = (
        _canonical_run_directory(str(shadow.get("run_dir", "")))
        if isinstance(shadow, dict)
        else Path("/__rtscortex_missing_run_dir__")
    )
    if not isinstance(gates, dict) or set(gates) != CANARY_GATE_KEYS_PRODUCTION:
        return False
    if not all(type(value) is bool for value in gates.values()) or not all(gates.values()):
        return False
    expected_gameplay_gate_keys = {
        "runs_exit_zero",
        "runs_complete_for_kind",
        "behavior_artifact_integrity_valid",
        "shadow_artifact_integrity_valid",
        "behavior_legacy_acceptance_valid",
        "shadow_legacy_acceptance_valid",
        "behavior_per_run",
        "shadow_per_run",
    }
    if not isinstance(gameplay_gates, dict) or set(gameplay_gates) != expected_gameplay_gate_keys:
        return False
    if not all(
        type(gameplay_gates.get(name)) is bool
        for name in (
            "runs_exit_zero",
            "runs_complete_for_kind",
            "behavior_artifact_integrity_valid",
            "shadow_artifact_integrity_valid",
            "behavior_legacy_acceptance_valid",
            "shadow_legacy_acceptance_valid",
        )
    ):
        return False
    for name in ("behavior_per_run", "shadow_per_run"):
        per_run = gameplay_gates.get(name)
        if not isinstance(per_run, dict) or set(per_run) != set(GAMEPLAY_GATE_NAMES):
            return False
        if not all(type(value) is bool for value in per_run.values()):
            return False
    for name in (
        "counterfactual_qualification_accepted",
        "gameplay_acceptance_accepted",
        "production_authorizing_accepted",
        "canonical_report_valid",
    ):
        if type(artifact.get(name)) is not bool:
            return False
    if not isinstance(artifact.get("canonical_report_sha256"), str) or not _SHA256_RE.fullmatch(
        artifact["canonical_report_sha256"]
    ):
        return False
    if artifact["canonical_report_sha256"] != _canonical_report_digest(artifact):
        return False
    try:
        derived_fields_valid = _canary_derived_fields_are_consistent(artifact)
    except (TypeError, ValueError, KeyError):
        return False
    if not derived_fields_valid:
        return False
    basic_valid = (
        artifact.get("schema_version") == "1.1"
        and artifact.get("canary_kind") == "production"
        and artifact.get("canary_fixture") is False
        and artifact.get("accepted") is True
        and artifact.get("counterfactual_qualification_accepted") is True
        and artifact.get("gameplay_acceptance_accepted") is True
        and artifact.get("production_authorizing_accepted") is True
        and artifact.get("canonical_report_valid") is True
        and artifact.get("baseline_sha256") == baseline_sha256
        and tuple(artifact.get("evaluation_seed_ids", ())) == expected_evaluation_seeds
        and artifact.get("execution_seed") in expected_evaluation_seeds
        and (expected_git_sha is None or artifact.get("expected_git_sha") == expected_git_sha)
        and isinstance(readiness, dict)
        and readiness.get("canary_runnable") is True
        and readiness.get("baseline_sha256") == baseline_sha256
        and readiness.get("expected_git_sha") == expected_git_sha
        and tuple(readiness.get("evaluation_seed_ids", ())) == expected_evaluation_seeds
        and readiness.get("schema_version") == "1.1"
        and bool(readiness.get("approved_hard_rule_ids"))
        and readiness.get("approved_hard_rule_ids")
        == readiness.get("runtime_selected_hard_rule_ids")
        and not bool(readiness.get("rejected_runtime_hard_rule_ids"))
        and readiness.get("hard_rule_limit_exceeded") is False
        and artifact.get("approved_rule_set_sha256") == readiness.get("approved_rule_set_sha256")
        and (
            approved_rule_set_sha256 is None
            or artifact.get("approved_rule_set_sha256") == approved_rule_set_sha256
        )
        and type(context_blocking_count) is int
        and context_blocking_count >= 1
        and not bool(readiness.get("canary_fixture_rule_ids"))
        and type(active_hard_block_count) is int
        and active_hard_block_count > 0
        and artifact.get("terminal_counterfactual_required") is True
        and type(terminal_resolved_count) is int
        and terminal_resolved_count > 0
        and type(artifact.get("unmatched_active_hard_block_count")) is int
        and artifact.get("unmatched_active_hard_block_count") == 0
        and isinstance(behavior, dict)
        and isinstance(shadow, dict)
        and behavior.get("artifact_integrity_valid") is True
        and shadow.get("artifact_integrity_valid") is True
        and behavior.get("legacy_acceptance_valid") is True
        and shadow.get("legacy_acceptance_valid") is True
        and behavior.get("gameplay_acceptance_accepted") is True
        and shadow.get("gameplay_acceptance_accepted") is True
        and isinstance(behavior.get("events_sha256"), str)
        and isinstance(behavior.get("summary_sha256"), str)
        and isinstance(shadow.get("events_sha256"), str)
        and isinstance(shadow.get("summary_sha256"), str)
        and _SHA256_RE.fullmatch(behavior["events_sha256"]) is not None
        and _SHA256_RE.fullmatch(behavior["summary_sha256"]) is not None
        and _SHA256_RE.fullmatch(shadow["events_sha256"]) is not None
        and _SHA256_RE.fullmatch(shadow["summary_sha256"]) is not None
        and isinstance(behavior.get("gameplay_gate_results"), dict)
        and isinstance(shadow.get("gameplay_gate_results"), dict)
        and set(behavior["gameplay_gate_results"]) == set(GAMEPLAY_GATE_NAMES)
        and set(shadow["gameplay_gate_results"]) == set(GAMEPLAY_GATE_NAMES)
        and all(type(value) is bool for value in behavior["gameplay_gate_results"].values())
        and all(type(value) is bool for value in shadow["gameplay_gate_results"].values())
        and all(behavior["gameplay_gate_results"].values())
        and all(shadow["gameplay_gate_results"].values())
        and behavior.get("source_attestation_fingerprint")
        == shadow.get("source_attestation_fingerprint")
        and behavior.get("analysis_evidence_overflow_count") == 0
        and shadow.get("analysis_evidence_overflow_count") == 0
        and behavior.get("event_identity_valid") is True
        and shadow.get("event_identity_valid") is True
        and behavior.get("mode") == "causal_canary"
        and shadow.get("mode") == "causal_canary"
        and behavior.get("arm") == "active"
        and shadow.get("arm") == "shadow"
        and behavior.get("experiment_kind") == "behavior"
        and shadow.get("experiment_kind") == "calibration"
        and behavior.get("seed") == shadow.get("seed") == artifact.get("execution_seed")
        and behavior_run_dir != Path("/__rtscortex_missing_run_dir__")
        and shadow_run_dir != Path("/__rtscortex_missing_run_dir__")
        and behavior_run_dir != shadow_run_dir
        and behavior.get("playbook_before_sha256") == shadow.get("playbook_before_sha256")
    )
    if not basic_valid:
        return False
    return _strict_canary_report_is_canonical(
        artifact,
        baseline_sha256=baseline_sha256,
        expected_git_sha=expected_git_sha,
        expected_evaluation_seeds=expected_evaluation_seeds,
        run_set_dir=(expected_run_set_dir or run_set_dir),
        natural_run_baseline_bytes_per_loop=natural_run_baseline_bytes_per_loop,
        recovery_evidence=recovery_evidence,
    )


def _canary_derived_fields_are_consistent(artifact: dict[str, Any]) -> bool:
    behavior = artifact.get("behavior")
    shadow = artifact.get("shadow")
    gates = artifact.get("gates")
    gameplay_gates = artifact.get("gameplay_gates")
    if not isinstance(behavior, dict) or not isinstance(shadow, dict):
        return False
    if not isinstance(gates, dict) or not isinstance(gameplay_gates, dict):
        return False

    def records(value: Any, width: int) -> list[tuple[Any, ...]] | None:
        if not isinstance(value, (list, tuple)):
            return None
        parsed: list[tuple[Any, ...]] = []
        for item in value:
            if not isinstance(item, (list, tuple)) or len(item) != width:
                return None
            parsed.append(tuple(item))
        return parsed

    active_records = records(behavior.get("active_hard_block_records"), 3)
    if active_records is None:
        return False
    if len(active_records) != len(set(active_records)):
        return False
    active_keys = {str(item[0]) for item in active_records}
    reported_active_keys = behavior.get("active_hard_block_keys")
    if not isinstance(reported_active_keys, (list, tuple)) or list(reported_active_keys) != sorted(
        active_keys
    ):
        return False
    resolved = shadow.get("resolved_counterfactual_keys")
    if not isinstance(resolved, (list, tuple)) or list(resolved) != sorted(
        set(str(key) for key in resolved)
    ):
        return False
    resolved_keys = set(str(key) for key in resolved)
    unmatched = [item for item in active_records if str(item[0]) not in resolved_keys]
    if artifact.get("active_hard_block_count") != len(active_keys):
        return False
    if artifact.get("terminal_counterfactual_resolved_count") != len(active_keys & resolved_keys):
        return False
    if artifact.get("unmatched_active_hard_block_count") != len(unmatched):
        return False
    reported_unmatched = artifact.get("unmatched_active_hard_blocks")
    if not isinstance(reported_unmatched, (list, tuple)):
        return False
    expected_unmatched = [
        {"counterfactual_key": str(key), "game_loop": loop, "state_hash": str(state_hash)}
        for key, loop, state_hash in unmatched
    ]
    if reported_unmatched != expected_unmatched:
        return False
    first_unmatched = min((int(item[1]) for item in unmatched), default=None)
    if artifact.get("first_unmatched_game_loop") != first_unmatched:
        return False

    behavior_states = records(behavior.get("counterfactual_state_records"), 3)
    shadow_states = records(shadow.get("counterfactual_state_records"), 3)
    if behavior_states is None or shadow_states is None:
        return False
    behavior_state_keys = [(int(item[0]), str(item[1])) for item in behavior_states]
    shadow_state_keys = [(int(item[0]), str(item[1])) for item in shadow_states]
    if len(behavior_state_keys) != len(set(behavior_state_keys)) or len(shadow_state_keys) != len(
        set(shadow_state_keys)
    ):
        return False
    behavior_state_map = {(int(item[0]), str(item[1])): str(item[2]) for item in behavior_states}
    shadow_state_map = {(int(item[0]), str(item[1])): str(item[2]) for item in shadow_states}
    divergent = [
        key[0]
        for key, state_hash in behavior_state_map.items()
        if key in shadow_state_map and shadow_state_map[key] != state_hash
    ]
    first_divergence = min(divergent, default=first_unmatched)
    return artifact.get("first_state_hash_divergence_game_loop") == first_divergence


def _canonical_report_digest(report: dict[str, Any]) -> str:
    payload = _report_canonical_projection(report)
    payload.pop("canonical_report_sha256", None)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=list,
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


_VOLATILE_CANARY_METRIC_FIELDS = frozenset(
    {
        "analysis_peak_rss_kib",
        "analysis_rss_per_10k_game_loops",
    }
)


def _metric_canonical_projection(metric: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in metric.items() if key not in _VOLATILE_CANARY_METRIC_FIELDS
    }


def _report_canonical_projection(report: dict[str, Any]) -> dict[str, Any]:
    projected = dict(report)
    projected.pop("canonical_report_sha256", None)
    projected["behavior"] = _metric_canonical_projection(report["behavior"])
    projected["shadow"] = _metric_canonical_projection(report["shadow"])
    memory = report.get("analysis_memory")
    if isinstance(memory, dict):
        projected["analysis_memory"] = {
            key: value
            for key, value in memory.items()
            if key
            not in {
                "behavior_peak_rss_kib",
                "shadow_peak_rss_kib",
                "behavior_rss_per_10k_game_loops",
                "shadow_rss_per_10k_game_loops",
            }
        }
    normalized = json.loads(json.dumps(projected, sort_keys=True, default=list))
    if not isinstance(normalized, dict):
        raise TypeError("canonical canary report projection must be an object")
    return normalized


def _strict_canary_report_is_canonical(
    artifact: dict[str, Any],
    *,
    baseline_sha256: str,
    expected_git_sha: str | None,
    expected_evaluation_seeds: tuple[int, ...],
    run_set_dir: Path | None,
    natural_run_baseline_bytes_per_loop: float | None,
    recovery_evidence: dict[str, Any] | None,
) -> bool:
    from scripts.analyze_playbook_counterfactual_canary import build_canary_report

    behavior_data = artifact.get("behavior")
    shadow_data = artifact.get("shadow")
    if not isinstance(behavior_data, dict) or not isinstance(shadow_data, dict):
        return False
    expected_metric_keys = set(RunMetrics.__dataclass_fields__)
    if set(behavior_data) != expected_metric_keys or set(shadow_data) != expected_metric_keys:
        return False

    if run_set_dir is None:
        return False

    expected_root = run_set_dir.expanduser().resolve()
    status_path = expected_root / "experiment-status.tsv"
    baseline_snapshot = expected_root / "playbook.baseline.sqlite3"
    readiness_path = expected_root / "playbook-hard-readiness.json"
    recovery_path = expected_root / "recovery-canary.json"
    engineering_baseline_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "acceptance"
        / "protoss_natural_terminal_v1.json"
    )
    if not all(
        path.is_file()
        for path in (
            status_path,
            baseline_snapshot,
            readiness_path,
            recovery_path,
            engineering_baseline_path,
        )
    ):
        return False
    try:
        canonical_readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
        canonical_recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
        engineering_baseline = json.loads(engineering_baseline_path.read_text(encoding="utf-8"))
        canonical_bytes_per_loop = float(engineering_baseline["natural_run_bytes_per_game_loop"])
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return False
    if (
        _sha256_file(baseline_snapshot) != baseline_sha256
        or canonical_readiness != artifact.get("hard_readiness")
        or recovery_evidence is not None
        and recovery_evidence != canonical_recovery
        or natural_run_baseline_bytes_per_loop is not None
        and natural_run_baseline_bytes_per_loop != canonical_bytes_per_loop
    ):
        return False
    behavior_dir = _canonical_run_directory(str(behavior_data.get("run_dir", "")))
    shadow_dir = _canonical_run_directory(str(shadow_data.get("run_dir", "")))
    if behavior_dir == shadow_dir or not behavior_dir.is_dir() or not shadow_dir.is_dir():
        return False
    try:
        status_rows = list(csv.DictReader(status_path.open(), delimiter="\t"))
    except (OSError, UnicodeDecodeError, csv.Error):
        return False
    if len(status_rows) != 2:
        return False
    rows_by_directory: dict[Path, dict[str, str]] = {}
    for row in status_rows:
        directory = _canonical_run_directory(row.get("run_dir", ""))
        if directory in rows_by_directory:
            return False
        rows_by_directory[directory] = row
    if set(rows_by_directory) != {behavior_dir, shadow_dir}:
        return False
    behavior_row = rows_by_directory[behavior_dir]
    shadow_row = rows_by_directory[shadow_dir]
    if not (
        behavior_row.get("experiment_kind") == "behavior"
        and behavior_row.get("mode") == "causal_canary"
        and behavior_row.get("arm") == "active"
        and not behavior_row.get("subject_arm")
        and shadow_row.get("experiment_kind") == "calibration"
        and shadow_row.get("mode") == "causal_canary"
        and shadow_row.get("arm") == "shadow"
        and shadow_row.get("subject_arm") == "active"
        and behavior_row.get("seed") == shadow_row.get("seed")
    ):
        return False
    kwargs = {
        "natural_run_baseline_bytes_per_loop": canonical_bytes_per_loop,
        "expected_git_sha": expected_git_sha,
        "recovery_evidence": canonical_recovery,
        "persist_engineering_report": False,
    }
    try:
        behavior_metrics = _run_metrics(behavior_row, **kwargs)
        shadow_metrics = _run_metrics(shadow_row, **kwargs)
        expected = build_canary_report(
            behavior_metrics,
            shadow_metrics,
            baseline_sha256=baseline_sha256,
            expected_git_sha=str(artifact.get("expected_git_sha", expected_git_sha)),
            readiness_evidence=canonical_readiness,
            canary_kind="production",
        )
    except (OSError, KeyError, TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    return _report_canonical_projection(artifact) == _report_canonical_projection(expected)


def _comparison(
    metrics: list[RunMetrics],
    *,
    baseline_sha256: str,
    counterfactual_canary: dict[str, Any] | None = None,
    expected_git_sha: str | None = None,
    expected_seeds: tuple[int, ...] = (0, 1, 2),
    run_set_dir: Path | None = None,
    counterfactual_canary_run_set_dir: Path | None = None,
) -> dict[str, Any]:
    behavior = [metric for metric in metrics if metric.experiment_kind == "behavior"]
    calibration = [metric for metric in metrics if metric.experiment_kind == "calibration"]
    split_matrix = run_set_dir is not None or any(
        metric.subject_arm is not None or metric.experiment_kind != "behavior" for metric in metrics
    )
    unique_expected_seeds = tuple(dict.fromkeys(expected_seeds))
    if len(unique_expected_seeds) != 3:
        raise ValueError("formal comparison requires exactly three distinct held-out seeds")
    if unique_expected_seeds != tuple(sorted(unique_expected_seeds)):
        raise ValueError("held-out seeds must be in strictly increasing execution order")
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
    canonical_run_dirs = [_canonical_run_directory(metric.run_dir) for metric in metrics]
    canonical_run_dirs_unique = len(canonical_run_dirs) == len(set(canonical_run_dirs)) and all(
        path != Path("/__rtscortex_missing_run_dir__") for path in canonical_run_dirs
    )
    run_dirs_match_status = False
    if run_set_dir is not None:
        expected_root = run_set_dir.expanduser().resolve()
        status_path = expected_root / "experiment-status.tsv"
        try:
            status_rows = list(csv.DictReader(status_path.open(), delimiter="\t"))
        except (OSError, UnicodeDecodeError, csv.Error):
            status_rows = []
        status_run_dirs = [_canonical_run_directory(row.get("run_dir", "")) for row in status_rows]
        rows_by_run_dir = {
            directory: row for directory, row in zip(status_run_dirs, status_rows, strict=True)
        }
        metric_identity_matches_status = all(
            (row := rows_by_run_dir.get(directory)) is not None
            and row.get("experiment_kind") == metric.experiment_kind
            and row.get("mode") == metric.mode
            and row.get("seed") == str(metric.seed)
            and row.get("arm") == metric.arm
            and (row.get("subject_arm") or None) == metric.subject_arm
            and row.get("exit_code") == str(metric.exit_code)
            for directory, metric in zip(canonical_run_dirs, metrics, strict=True)
        )
        run_dirs_match_status = (
            len(status_rows) == 24
            and len(status_run_dirs) == len(set(status_run_dirs))
            and set(status_run_dirs) == set(canonical_run_dirs)
            and metric_identity_matches_status
            and all(path.is_dir() for path in canonical_run_dirs)
        )
    event_run_ids = [metric.event_run_id for metric in metrics if metric.event_run_id is not None]
    event_identity_unique = len(event_run_ids) == len(set(event_run_ids))
    event_identity_contract = (
        all(metric.event_identity_valid for metric in metrics)
        and len(event_run_ids) == len(metrics)
        and all(metric.event_episode_ids for metric in metrics)
        and event_identity_unique
    )
    formal_split_contract = True
    if split_matrix:
        formal_split_contract = (
            len(metrics) == 24
            and len(behavior) == 12
            and len(calibration) == 12
            and all(metric.experiment_kind in {"behavior", "calibration"} for metric in metrics)
            and all(
                metric.arm in {"frozen", "evolving"} and metric.subject_arm == metric.arm
                for metric in behavior
            )
            and all(
                metric.arm == "shadow" and metric.subject_arm in {"frozen", "evolving"}
                for metric in calibration
            )
            and run_set_dir is not None
            and run_dirs_match_status
        )
    paired: list[dict[str, Any]] = []
    for mode in ("independent_paired", "sequential_learning"):
        for seed in unique_expected_seeds:
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
    evolving_by_seed = {
        metric.seed: metric for metric in sequential_rows if metric.arm == "evolving"
    }
    evolving_sequence = [
        evolving_by_seed[seed] for seed in unique_expected_seeds if seed in evolving_by_seed
    ]
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
    gameplay_acceptance = bool(metrics) and all(
        _gameplay_contract_is_valid(metric) for metric in metrics
    )
    run_artifact_integrity = bool(metrics) and all(
        _artifact_attestation_is_valid(metric) for metric in metrics
    )
    canary_is_valid = counterfactual_canary is not None and counterfactual_canary_is_valid(
        counterfactual_canary,
        baseline_sha256=baseline_sha256,
        expected_git_sha=expected_git_sha,
        expected_evaluation_seeds=unique_expected_seeds,
        run_set_dir=counterfactual_canary_run_set_dir,
    )
    canary_counterfactual_qualification = (
        counterfactual_canary is not None
        and counterfactual_canary.get("counterfactual_qualification_accepted") is True
    )
    canary_gameplay_acceptance = (
        counterfactual_canary is not None
        and counterfactual_canary.get("gameplay_acceptance_accepted") is True
    )
    production_authorizing_acceptance = (
        canary_is_valid
        and canary_counterfactual_qualification
        and canary_gameplay_acceptance
        and gameplay_acceptance
    )
    gates = {
        "formal_split_contract": formal_split_contract,
        "canonical_run_directory_identity": (
            canonical_run_dirs_unique and run_dirs_match_status
            if split_matrix
            else canonical_run_dirs_unique
        ),
        "event_identity_contract": event_identity_contract,
        "complete_unique_run_matrix": (
            len(behavior) == len(expected_behavior_matrix)
            and observed_behavior_matrix == expected_behavior_matrix
            and (
                not split_matrix
                or (
                    len(calibration) == len(expected_calibration_matrix)
                    and observed_calibration_matrix == expected_calibration_matrix
                    and formal_split_contract
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
        "counterfactual_canary_accepted": (canary_is_valid),
        "counterfactual_qualification_accepted": canary_counterfactual_qualification,
        "all_runs_gameplay_acceptance_pass": gameplay_acceptance,
        "gameplay_acceptance_accepted": canary_gameplay_acceptance and gameplay_acceptance,
        "production_authorizing_acceptance": production_authorizing_acceptance,
        "run_artifact_integrity_valid": run_artifact_integrity,
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
        "counterfactual_qualification_accepted": canary_counterfactual_qualification,
        "gameplay_acceptance_accepted": canary_gameplay_acceptance and gameplay_acceptance,
        "production_authorizing_accepted": production_authorizing_acceptance,
        "accepted": all(gates.values()),
    }


def _gameplay_contract_is_valid(metric: RunMetrics) -> bool:
    results = metric.gameplay_gate_results
    return (
        isinstance(results, dict)
        and set(results) == set(GAMEPLAY_GATE_NAMES)
        and all(type(value) is bool for value in results.values())
        and metric.gameplay_acceptance_accepted is True
        and metric.artifact_integrity_valid is True
        and metric.legacy_acceptance_valid is True
        and all(results.values())
    )


def _artifact_attestation_is_valid(metric: RunMetrics) -> bool:
    return (
        metric.artifact_integrity_valid is True
        and isinstance(metric.events_sha256, str)
        and isinstance(metric.summary_sha256, str)
        and _SHA256_RE.fullmatch(metric.events_sha256) is not None
        and _SHA256_RE.fullmatch(metric.summary_sha256) is not None
    )


def _canonical_run_directory(value: str | Path) -> Path:
    raw = (
        Path(value).expanduser()
        if value is not None and str(value).strip()
        else Path("/__rtscortex_missing_run_dir__")
    )
    return raw.resolve()


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
