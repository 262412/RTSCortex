"""Build conserved acceptance metrics for Playbook experiment run sets."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
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
    strategic_consequences: dict[str, int]
    hard_rule_false_block_count: int
    hard_rule_shadow_state_count: int
    hard_rule_false_block_rate: float
    journal_bytes: int
    max_game_loop: int
    bytes_per_game_loop: float
    effective_game_loops_per_second: float
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
    events = list(read_event_log(journal)) if journal.is_file() else []
    episode_result = next(
        (event.payload for event in reversed(events) if event.event_type == "episode_result"),
        {},
    )
    executions = [event.payload for event in events if event.event_type == "execution"]
    terminal_counts = Counter(str(payload.get("command_id")) for payload in executions)
    dispatch_counts: Counter[str] = Counter()
    for event in events:
        if event.event_type != "command_lifecycle":
            continue
        if event.payload.get("status") == "dispatched":
            command = event.payload.get("command")
            if isinstance(command, dict):
                dispatch_counts[str(command.get("command_id"))] += 1
    consequence_events = [
        event.payload for event in events if event.event_type == "strategic_consequence_attributed"
    ]
    consequences = Counter(
        str(payload.get("consequence_type", "unknown")) for payload in consequence_events
    )
    error_signatures = Counter(
        _consequence_signature(payload)
        for payload in consequence_events
        if str(payload.get("consequence_type")) in _ERROR_CONSEQUENCES
    )
    applications = [
        event.payload for event in events if event.event_type == "playbook_rule_applied"
    ]
    observed_game_loops = [
        value
        for event in events
        if isinstance(value := event.payload.get("game_loop"), int | float)
    ]
    max_game_loop = max((*observed_game_loops, *(event.step_id for event in events)), default=0)
    elapsed_seconds = _elapsed_seconds(events)
    before_false_blocks, before_shadow_states = _hard_false_blocks(
        Path(row["playbook_before_snapshot"])
    )
    after_false_blocks, after_shadow_states = _hard_false_blocks(
        Path(row["playbook_after_snapshot"])
    )
    false_blocks = max(0, after_false_blocks - before_false_blocks)
    shadow_states = max(0, after_shadow_states - before_shadow_states)
    dispatched_ids = set(dispatch_counts)
    terminal_ids = set(terminal_counts)
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
        terminal_report_count=len(executions),
        missing_terminal_reports=len(dispatched_ids - terminal_ids),
        duplicate_terminal_reports=sum(
            count - 1 for count in terminal_counts.values() if count > 1
        ),
        duplicate_dispatches=sum(count - 1 for count in dispatch_counts.values() if count > 1),
        candidate_outside_dispatch=sum(
            payload.get("failure_code") == "candidate_outside_dispatch" for payload in executions
        ),
        playbook_applications=len(applications),
        playbook_nonzero_score_applications=sum(
            float(payload.get("score_delta", 0.0)) != 0.0 for payload in applications
        ),
        playbook_blocks=sum(payload.get("blocked") is True for payload in applications),
        repeated_eligible_errors=sum(max(0, count - 1) for count in error_signatures.values()),
        strategic_consequences=dict(sorted(consequences.items())),
        hard_rule_false_block_count=false_blocks,
        hard_rule_shadow_state_count=shadow_states,
        hard_rule_false_block_rate=(false_blocks / shadow_states if shadow_states else 0.0),
        journal_bytes=journal.stat().st_size if journal.is_file() else 0,
        max_game_loop=int(max_game_loop) if isinstance(max_game_loop, int | float) else 0,
        bytes_per_game_loop=(
            journal.stat().st_size / max_game_loop
            if journal.is_file() and isinstance(max_game_loop, int | float) and max_game_loop > 0
            else 0.0
        ),
        effective_game_loops_per_second=(
            max_game_loop / elapsed_seconds if elapsed_seconds else 0.0
        ),
        playbook_before_sha256=row["playbook_before_sha256"],
        playbook_after_sha256=row["playbook_after_sha256"],
    )


def _hard_false_blocks(path: Path) -> tuple[int, int]:
    if not path.is_file():
        return 0, 0
    connection = sqlite3.connect(path)
    try:
        rows = connection.execute("SELECT payload_json FROM playbook_rules_v2").fetchall()
    finally:
        connection.close()
    false_blocks = 0
    shadow_states = 0
    for (encoded,) in rows:
        payload = json.loads(str(encoded))
        if payload.get("strength") != "hard" or payload.get("status") != "active":
            continue
        false_blocks += int(payload.get("false_block_count", 0))
        shadow_states += int(payload.get("shadow_state_count", 0))
    return false_blocks, shadow_states


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


def _elapsed_seconds(events: list[Any]) -> float:
    timestamps: list[datetime] = []
    for event in events:
        try:
            timestamps.append(datetime.fromisoformat(event.created_at))
        except ValueError:
            continue
    if len(timestamps) < 2:
        return 0.0
    return max(0.0, (timestamps[-1] - timestamps[0]).total_seconds())


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
    reduction = 0.0 if frozen_errors == 0 else (frozen_errors - evolving_errors) / frozen_errors
    false_blocks = sum(metric.hard_rule_false_block_count for metric in metrics)
    shadow_states = sum(metric.hard_rule_shadow_state_count for metric in metrics)
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
            shadow_states > 0 and false_blocks / shadow_states <= 0.01
        ),
        "repeated_error_reduction_at_least_50_percent": reduction >= 0.5,
        "matched_win_rate_not_reduced": (sum(item["win_delta"] for item in independent_pairs) >= 0),
    }
    return {
        "schema_version": "1.0",
        "baseline_sha256": baseline_sha256,
        "runs": [asdict(metric) for metric in metrics],
        "paired_differences": paired,
        "aggregate": {
            "independent_frozen_repeated_errors": frozen_errors,
            "independent_evolving_repeated_errors": evolving_errors,
            "independent_repeated_error_reduction": reduction,
            "hard_rule_false_block_count": false_blocks,
            "hard_rule_shadow_state_count": shadow_states,
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
            "| Mode | Seed | Score delta | Repeated-error delta | Win delta |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    lines.extend(
        (
            f"| {item['mode']} | {item['seed']} | {item['score_delta']:.3f} | "
            f"{item['repeated_error_delta']} | {item['win_delta']} |"
        )
        for item in report["paired_differences"]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
