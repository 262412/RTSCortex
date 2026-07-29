"""Fail closed on a divergent Active/Shadow counterfactual canary."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

from scripts.analyze_playbook_experiment import RunMetrics, _run_metrics


def build_canary_report(
    behavior: RunMetrics,
    shadow: RunMetrics,
    *,
    baseline_sha256: str,
    expected_git_sha: str,
    readiness_evidence: dict[str, Any],
    canary_kind: Literal["production", "fixture"] = "production",
) -> dict[str, Any]:
    resolved = set(
        shadow.shadow_would_block_keys
        if canary_kind == "fixture"
        else shadow.resolved_counterfactual_keys
    )
    unmatched = [
        record for record in behavior.active_hard_block_records if record[0] not in resolved
    ]
    active_keys = {record[0] for record in behavior.active_hard_block_records}
    matched_count = len(active_keys & resolved)
    behavior_states = {
        (epoch, signature): state_hash
        for epoch, signature, state_hash in behavior.counterfactual_state_records
    }
    shadow_states = {
        (epoch, signature): state_hash
        for epoch, signature, state_hash in shadow.counterfactual_state_records
    }
    divergent_epochs = [
        epoch
        for (epoch, signature), state_hash in behavior_states.items()
        if (epoch, signature) in shadow_states and shadow_states[(epoch, signature)] != state_hash
    ]
    first_unmatched_loop = min((record[1] for record in unmatched), default=None)
    first_state_hash_divergence = min(divergent_epochs, default=first_unmatched_loop)
    readiness_valid = (
        readiness_evidence.get("canary_runnable") is True
        and readiness_evidence.get("baseline_sha256") == baseline_sha256
        and readiness_evidence.get("expected_git_sha") == expected_git_sha
        and int(readiness_evidence.get("context_applicable_blocking_hard_count", 0)) >= 1
        and (
            bool(readiness_evidence.get("canary_fixture_rule_ids"))
            if canary_kind == "fixture"
            else not bool(readiness_evidence.get("canary_fixture_rule_ids"))
        )
    )
    gates = {
        "hard_readiness_accepted": readiness_valid,
        "runs_exit_zero": behavior.exit_code == 0 and shadow.exit_code == 0,
        "runs_complete_for_kind": (
            behavior.natural_terminal and shadow.natural_terminal
            if canary_kind == "production"
            else behavior.exit_code == 0 and shadow.exit_code == 0
        ),
        "source_attestation_matches": (
            behavior.source_attestation_consistent
            and shadow.source_attestation_consistent
            and behavior.source_commit_matches_expected_sha
            and shadow.source_commit_matches_expected_sha
            and behavior.source_attestation_fingerprint == shadow.source_attestation_fingerprint
        ),
        "active_hard_block_observed": bool(active_keys),
        "matched_counterfactual_observed": matched_count > 0,
        "all_active_hard_blocks_matched": not unmatched,
        "matched_prestate_identity": not divergent_epochs,
        "analysis_memory_budget_respected": (
            behavior.analysis_evidence_overflow_count == 0
            and shadow.analysis_evidence_overflow_count == 0
        ),
    }
    return {
        "schema_version": "1.0",
        "canary_kind": canary_kind,
        "canary_fixture": canary_kind == "fixture",
        "baseline_sha256": baseline_sha256,
        "expected_git_sha": expected_git_sha,
        "hard_readiness": readiness_evidence,
        "behavior": asdict(behavior),
        "shadow": asdict(shadow),
        "active_hard_block_count": len(active_keys),
        "matched_counterfactual_count": matched_count,
        "unmatched_active_hard_block_count": len(unmatched),
        "unmatched_active_hard_blocks": [
            {"counterfactual_key": key, "game_loop": loop, "state_hash": state_hash}
            for key, loop, state_hash in unmatched
        ],
        "first_unmatched_game_loop": first_unmatched_loop,
        "first_state_hash_divergence_game_loop": first_state_hash_divergence,
        "analysis_memory": {
            "behavior_peak_rss_kib": behavior.analysis_peak_rss_kib,
            "shadow_peak_rss_kib": shadow.analysis_peak_rss_kib,
            "behavior_retained_event_count": behavior.retained_event_count,
            "shadow_retained_event_count": shadow.retained_event_count,
            "behavior_scanned_event_count": behavior.event_count,
            "shadow_scanned_event_count": shadow.event_count,
            "behavior_rule_evaluation_count": behavior.rule_evaluation_count,
            "shadow_rule_evaluation_count": shadow.rule_evaluation_count,
            "behavior_rss_per_10k_game_loops": behavior.analysis_rss_per_10k_game_loops,
            "shadow_rss_per_10k_game_loops": shadow.analysis_rss_per_10k_game_loops,
        },
        "gates": gates,
        "accepted": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_set_dir", type=Path)
    parser.add_argument("--baseline-sha256", required=True)
    parser.add_argument("--expected-git-sha", required=True)
    parser.add_argument("--engineering-baseline", type=Path, required=True)
    parser.add_argument("--recovery-evidence", type=Path, required=True)
    parser.add_argument("--readiness-evidence", type=Path, required=True)
    parser.add_argument(
        "--canary-kind",
        choices=("production", "fixture"),
        default="production",
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    engineering_baseline = json.loads(arguments.engineering_baseline.read_text(encoding="utf-8"))
    recovery_evidence = json.loads(arguments.recovery_evidence.read_text(encoding="utf-8"))
    readiness_evidence = json.loads(arguments.readiness_evidence.read_text(encoding="utf-8"))
    rows = list(
        csv.DictReader(
            (arguments.run_set_dir / "experiment-status.tsv").open(),
            delimiter="\t",
        )
    )
    behavior_rows = [row for row in rows if row.get("experiment_kind") == "behavior"]
    shadow_rows = [row for row in rows if row.get("experiment_kind") == "calibration"]
    if len(behavior_rows) != 1 or len(shadow_rows) != 1:
        raise SystemExit("counterfactual canary requires exactly one behavior and one shadow run")
    kwargs = {
        "natural_run_baseline_bytes_per_loop": float(
            engineering_baseline["natural_run_bytes_per_game_loop"]
        ),
        "expected_git_sha": arguments.expected_git_sha,
        "recovery_evidence": recovery_evidence,
    }
    report = build_canary_report(
        _run_metrics(behavior_rows[0], **kwargs),
        _run_metrics(shadow_rows[0], **kwargs),
        baseline_sha256=arguments.baseline_sha256,
        expected_git_sha=arguments.expected_git_sha,
        readiness_evidence=readiness_evidence,
        canary_kind=arguments.canary_kind,
    )
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    raise SystemExit(0 if report["accepted"] else 1)


if __name__ == "__main__":
    main()
