"""Fail closed on a divergent Active/Shadow counterfactual canary."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

from rtscortex.playbook import PlaybookRuleCategory, evaluation_kind
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
    terminal_resolved = set(shadow.resolved_counterfactual_keys)
    shadow_guard_allows = set(shadow.shadow_would_block_keys)
    resolved = shadow_guard_allows if canary_kind == "fixture" else terminal_resolved
    unmatched = [
        record for record in behavior.active_hard_block_records if record[0] not in resolved
    ]
    active_keys = {record[0] for record in behavior.active_hard_block_records}
    matched_count = len(active_keys & resolved)
    terminal_resolved_count = len(active_keys & terminal_resolved)
    matched_shadow_guard_allow_count = len(active_keys & shadow_guard_allows)
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
    approved_kinds, readiness_kind_contract_valid = _approved_evaluation_kinds(readiness_evidence)
    observed_kinds = (("behavior", *record) for record in behavior.hard_rule_kind_records)
    observed_kind_records = [
        *observed_kinds,
        *(("shadow", *record) for record in shadow.hard_rule_kind_records),
    ]
    rule_kind_mismatches = [
        {
            "arm": arm,
            "rule_id": rule_id,
            "observed": observed_kind,
            "expected": approved_kinds.get(rule_id),
        }
        for arm, rule_id, observed_kind in observed_kind_records
        if approved_kinds.get(rule_id) != observed_kind
    ]
    rule_kind_contract_valid = (
        readiness_kind_contract_valid and bool(observed_kind_records) and not rule_kind_mismatches
    )
    readiness_valid = (
        readiness_evidence.get("schema_version") == "1.1"
        and readiness_evidence.get("canary_runnable") is True
        and readiness_evidence.get("baseline_sha256") == baseline_sha256
        and readiness_evidence.get("expected_git_sha") == expected_git_sha
        and int(readiness_evidence.get("context_applicable_blocking_hard_count", 0)) >= 1
        and bool(readiness_evidence.get("approved_hard_rule_ids"))
        and readiness_evidence.get("approved_hard_rule_ids")
        == readiness_evidence.get("runtime_selected_hard_rule_ids")
        and isinstance(readiness_evidence.get("approved_rule_set_sha256"), str)
        and not bool(readiness_evidence.get("rejected_runtime_hard_rule_ids"))
        and readiness_evidence.get("hard_rule_limit_exceeded") is False
        and (
            bool(readiness_evidence.get("canary_fixture_rule_ids"))
            if canary_kind == "fixture"
            else not bool(readiness_evidence.get("canary_fixture_rule_ids"))
        )
    )
    evaluation_seed_ids = tuple(
        int(seed) for seed in readiness_evidence.get("evaluation_seed_ids", ())
    )
    gates = {
        "hard_readiness_accepted": readiness_valid,
        "runs_exit_zero": behavior.exit_code == 0 and shadow.exit_code == 0,
        "execution_seed_contract": (
            behavior.seed == shadow.seed and behavior.seed in evaluation_seed_ids
        ),
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
        (
            "matched_shadow_guard_allow_observed"
            if canary_kind == "fixture"
            else "terminal_counterfactual_resolved"
        ): matched_count > 0,
        "all_active_hard_blocks_matched": not unmatched,
        "matched_prestate_identity": not divergent_epochs,
        "rule_evaluation_kind_consistent": rule_kind_contract_valid,
        "analysis_memory_budget_respected": (
            behavior.analysis_evidence_overflow_count == 0
            and shadow.analysis_evidence_overflow_count == 0
        ),
    }
    return {
        "schema_version": "1.1",
        "canary_kind": canary_kind,
        "canary_fixture": canary_kind == "fixture",
        "baseline_sha256": baseline_sha256,
        "expected_git_sha": expected_git_sha,
        "execution_seed": behavior.seed,
        "evaluation_seed_ids": list(evaluation_seed_ids),
        "approved_rule_set_sha256": readiness_evidence.get("approved_rule_set_sha256"),
        "hard_readiness": readiness_evidence,
        "behavior": asdict(behavior),
        "shadow": asdict(shadow),
        "active_hard_block_count": len(active_keys),
        "terminal_counterfactual_required": canary_kind == "production",
        "terminal_counterfactual_resolved_count": terminal_resolved_count,
        "matched_shadow_guard_allow_count": matched_shadow_guard_allow_count,
        "unmatched_active_hard_block_count": len(unmatched),
        "unmatched_active_hard_blocks": [
            {"counterfactual_key": key, "game_loop": loop, "state_hash": state_hash}
            for key, loop, state_hash in unmatched
        ],
        "first_unmatched_game_loop": first_unmatched_loop,
        "first_state_hash_divergence_game_loop": first_state_hash_divergence,
        "rule_kind_mismatches": rule_kind_mismatches,
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


def _approved_evaluation_kinds(
    readiness_evidence: dict[str, Any],
) -> tuple[dict[str, str], bool]:
    approved_ids = {
        str(rule_id) for rule_id in readiness_evidence.get("approved_hard_rule_ids", ())
    }
    audits = readiness_evidence.get("rules")
    if not approved_ids or not isinstance(audits, list):
        return {}, False
    approved_kinds: dict[str, str] = {}
    valid = True
    for audit in audits:
        if not isinstance(audit, dict) or str(audit.get("rule_id")) not in approved_ids:
            continue
        rule_id = str(audit["rule_id"])
        try:
            expected = evaluation_kind(PlaybookRuleCategory(str(audit.get("category"))))
        except ValueError:
            valid = False
            continue
        expected_qualification = "execution" if expected.value == "execution_guard" else "strategic"
        if (
            audit.get("evaluation_kind") != expected.value
            or audit.get("qualification_kind") != expected_qualification
        ):
            valid = False
        approved_kinds[rule_id] = expected.value
    return approved_kinds, valid and set(approved_kinds) == approved_ids


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
