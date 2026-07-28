"""Build conserved acceptance metrics for Playbook experiment run sets."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_set_dir", type=Path)
    parser.add_argument("--baseline-sha256", required=True)
    arguments = parser.parse_args()
    run_set = arguments.run_set_dir.resolve()
    rows = list(csv.DictReader((run_set / "experiment-status.tsv").open(), delimiter="\t"))
    metrics = [_run_metrics(row) for row in rows]
    report = _comparison(metrics, baseline_sha256=arguments.baseline_sha256)
    (run_set / "comparison.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_set / "report.md").write_text(_markdown(report), encoding="utf-8")


def _run_metrics(row: dict[str, str]) -> RunMetrics:
    run_dir = Path(row["run_dir"]).expanduser()
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
    first_timestamp: datetime | None = None
    last_timestamp: datetime | None = None
    event_count = 0
    performance: dict[str, Any] = {}
    if journal.is_file():
        for event in read_event_log(journal):
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
                terminal_counts[command_id] += 1
                candidate_outside_dispatch += (
                    event.payload.get("failure_code") == "candidate_outside_dispatch"
                )
            elif (
                event.event_type == "command_lifecycle"
                and event.payload.get("status") == "dispatched"
            ):
                command = event.payload.get("command")
                if isinstance(command, dict):
                    dispatch_counts[str(command.get("command_id"))] += 1
            elif event.event_type == "command_lineage":
                lineage = event.payload.get("lineage", event.payload)
                operation_id = lineage.get("operation_id") if isinstance(lineage, dict) else None
                if isinstance(operation_id, str):
                    lineaged_operation_ids.add(operation_id)
            elif event.event_type == "strategic_consequence_attributed":
                consequence_type = str(event.payload.get("consequence_type", "unknown"))
                consequences[consequence_type] += 1
                if consequence_type in _ERROR_CONSEQUENCES:
                    error_signatures[_consequence_signature(event.payload)] += 1
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
                    rule_evaluations[evaluation_id] = event.payload

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
    false_blocks = sum(evaluation.get("false_block") is True for evaluation in hard_evaluations)
    shadow_states = sum(
        isinstance(evaluation.get("false_block"), bool) for evaluation in hard_evaluations
    )
    unresolved_blocks = sum(
        evaluation.get("shadow_decision") == "would_block"
        and evaluation.get("actual_outcome") == "blocked"
        and evaluation.get("false_block") is None
        for evaluation in hard_evaluations
    )
    terminal_report_count = sum(terminal_counts.values())
    dispatched_ids = set(dispatch_counts)
    terminal_ids = set(terminal_counts)
    journal_bytes = journal.stat().st_size if journal.is_file() else 0
    artifact_bytes = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
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
        hard_rule_shadow_state_count=shadow_states,
        hard_rule_unresolved_block_count=unresolved_blocks,
        hard_rule_false_block_rate=(false_blocks / shadow_states if shadow_states else 0.0),
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


def _comparison(metrics: list[RunMetrics], *, baseline_sha256: str) -> dict[str, Any]:
    expected_matrix = {
        (mode, seed, arm)
        for mode in ("independent_paired", "sequential_learning")
        for seed in (0, 1, 2)
        for arm in ("frozen", "evolving")
    }
    observed_matrix = {(metric.mode, metric.seed, metric.arm) for metric in metrics}
    paired: list[dict[str, Any]] = []
    for mode in ("independent_paired", "sequential_learning"):
        for seed in sorted({metric.seed for metric in metrics if metric.mode == mode}):
            pair = {
                metric.arm: metric
                for metric in metrics
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
    independent_rows = [metric for metric in metrics if metric.mode == "independent_paired"]
    sequential_rows = [metric for metric in metrics if metric.mode == "sequential_learning"]
    independent_pairs = [item for item in paired if item["mode"] == "independent_paired"]
    baseline_identity = all(
        metric.playbook_before_sha256 == baseline_sha256 for metric in independent_rows
    )
    frozen_immutable = all(
        metric.playbook_before_sha256 == metric.playbook_after_sha256
        for metric in metrics
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
    false_blocks = sum(metric.hard_rule_false_block_count for metric in metrics)
    shadow_states = sum(metric.hard_rule_shadow_state_count for metric in metrics)
    unresolved_hard_blocks = sum(metric.hard_rule_unresolved_block_count for metric in metrics)
    gates = {
        "complete_unique_run_matrix": (
            len(metrics) == len(expected_matrix) and observed_matrix == expected_matrix
        ),
        "all_runs_exit_zero": all(metric.exit_code == 0 for metric in metrics),
        "all_runs_natural_terminal": all(metric.natural_terminal for metric in metrics),
        "independent_baseline_identity": baseline_identity,
        "frozen_hash_unchanged": frozen_immutable,
        "sequential_evolving_carry": sequential_carry,
        "duplicate_dispatch_zero": all(metric.duplicate_dispatches == 0 for metric in metrics),
        "terminal_report_exactly_once": all(
            metric.duplicate_terminal_reports == 0 and metric.missing_terminal_reports == 0
            for metric in metrics
        ),
        "candidate_outside_dispatch_zero": all(
            metric.candidate_outside_dispatch == 0 for metric in metrics
        ),
        "hard_false_block_rate_at_most_1_percent": (
            shadow_states > 0
            and unresolved_hard_blocks == 0
            and false_blocks / shadow_states <= 0.01
        ),
        "repeated_error_reduction_at_least_50_percent": reduction >= 0.5,
        "matched_win_rate_not_reduced": (sum(item["win_delta"] for item in independent_pairs) >= 0),
    }
    return {
        "schema_version": "1.1",
        "baseline_sha256": baseline_sha256,
        "runs": [asdict(metric) for metric in metrics],
        "paired_differences": paired,
        "aggregate": {
            "independent_frozen_repeated_errors": frozen_errors,
            "independent_evolving_repeated_errors": evolving_errors,
            "independent_frozen_repeated_errors_per_10k_game_loops": frozen_error_rate,
            "independent_evolving_repeated_errors_per_10k_game_loops": (evolving_error_rate),
            "independent_repeated_error_reduction": reduction,
            "hard_rule_false_block_count": false_blocks,
            "hard_rule_shadow_state_count": shadow_states,
            "hard_rule_unresolved_block_count": unresolved_hard_blocks,
            "hard_rule_false_block_rate": (false_blocks / shadow_states if shadow_states else 0.0),
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
