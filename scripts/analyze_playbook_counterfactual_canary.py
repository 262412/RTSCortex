"""Fail closed on a divergent Active/Shadow counterfactual canary."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

from rtscortex.playbook import PlaybookRuleCategory, evaluation_kind
from scripts.analyze_playbook_experiment import (
    GAMEPLAY_GATE_NAMES,
    RunMetrics,
    _canonical_report_digest,
    _run_metrics,
    counterfactual_canary_is_valid,
)


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
    unique_active_records = len(behavior.active_hard_block_records) == len(
        set(behavior.active_hard_block_records)
    )
    unique_behavior_states = len(behavior_states) == len(behavior.counterfactual_state_records)
    unique_shadow_states = len(shadow_states) == len(shadow.counterfactual_state_records)
    prestate_identity_valid = (
        unique_active_records
        and unique_behavior_states
        and unique_shadow_states
        and set(behavior_states) == set(shadow_states)
        and not divergent_epochs
    )
    active_block_prestate_identity = not behavior.counterfactual_state_records or all(
        any(
            epoch == loop and state_hash == prestate_hash
            for epoch, _signature, prestate_hash in behavior.counterfactual_state_records
        )
        for _key, loop, state_hash in behavior.active_hard_block_records
    )
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
    observed_rule_ids = {rule_id for _, rule_id, _ in observed_kind_records}
    rule_kind_contract_valid = (
        readiness_kind_contract_valid
        and observed_rule_ids == set(approved_kinds)
        and bool(observed_kind_records)
        and not rule_kind_mismatches
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
        "active_role_contract": (
            behavior.mode == ("fixture" if canary_kind == "fixture" else "causal_canary")
            and behavior.arm == "active"
            and behavior.experiment_kind == "behavior"
        ),
        "shadow_role_contract": (
            shadow.mode == ("fixture" if canary_kind == "fixture" else "causal_canary")
            and shadow.arm == "shadow"
            and shadow.experiment_kind == "calibration"
        ),
        "run_identity_contract": (
            behavior.seed == shadow.seed
            and behavior.run_dir != shadow.run_dir
            and behavior.event_identity_valid
            and shadow.event_identity_valid
            and (
                behavior.event_run_id is None
                or shadow.event_run_id is None
                or behavior.event_run_id != shadow.event_run_id
            )
        ),
        "matching_provenance": (
            behavior.playbook_before_sha256 == shadow.playbook_before_sha256
            and behavior.source_attestation_fingerprint is not None
            and behavior.source_attestation_fingerprint == shadow.source_attestation_fingerprint
        ),
        "active_hard_block_observed": bool(active_keys),
        (
            "matched_shadow_guard_allow_observed"
            if canary_kind == "fixture"
            else "terminal_counterfactual_resolved"
        ): matched_count > 0,
        "all_active_hard_blocks_matched": not unmatched,
        "matched_prestate_identity": prestate_identity_valid,
        "active_block_prestate_identity": active_block_prestate_identity,
        "rule_evaluation_kind_consistent": rule_kind_contract_valid,
        "analysis_memory_budget_respected": (
            behavior.analysis_evidence_overflow_count == 0
            and shadow.analysis_evidence_overflow_count == 0
        ),
    }
    counterfactual_gate_names = {
        "hard_readiness_accepted",
        "runs_exit_zero",
        "execution_seed_contract",
        "runs_complete_for_kind",
        "source_attestation_matches",
        "active_role_contract",
        "shadow_role_contract",
        "run_identity_contract",
        "matching_provenance",
        "active_block_prestate_identity",
        "active_hard_block_observed",
        "terminal_counterfactual_resolved"
        if canary_kind == "production"
        else "matched_shadow_guard_allow_observed",
        "all_active_hard_blocks_matched",
        "matched_prestate_identity",
        "rule_evaluation_kind_consistent",
        "analysis_memory_budget_respected",
    }
    counterfactual_qualification_accepted = all(gates[name] for name in counterfactual_gate_names)
    behavior_gameplay_gates = {
        name: behavior.gameplay_gate_results.get(name) is True for name in GAMEPLAY_GATE_NAMES
    }
    shadow_gameplay_gates = {
        name: shadow.gameplay_gate_results.get(name) is True for name in GAMEPLAY_GATE_NAMES
    }
    gameplay_gates: dict[str, Any] = {
        "runs_exit_zero": gates["runs_exit_zero"],
        "runs_complete_for_kind": gates["runs_complete_for_kind"],
        "behavior_artifact_integrity_valid": behavior.artifact_integrity_valid,
        "shadow_artifact_integrity_valid": shadow.artifact_integrity_valid,
        "behavior_legacy_acceptance_valid": behavior.legacy_acceptance_valid,
        "shadow_legacy_acceptance_valid": shadow.legacy_acceptance_valid,
        "behavior_per_run": behavior_gameplay_gates,
        "shadow_per_run": shadow_gameplay_gates,
    }
    gameplay_acceptance_accepted = (
        all(
            gameplay_gates[name] is True
            for name in (
                "runs_exit_zero",
                "runs_complete_for_kind",
                "behavior_artifact_integrity_valid",
                "shadow_artifact_integrity_valid",
                "behavior_legacy_acceptance_valid",
                "shadow_legacy_acceptance_valid",
            )
        )
        and all(behavior_gameplay_gates.values())
        and all(shadow_gameplay_gates.values())
    )
    production_authorizing_accepted = (
        canary_kind == "production"
        and counterfactual_qualification_accepted
        and gameplay_acceptance_accepted
    )
    report = {
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
        "gameplay_gates": gameplay_gates,
        "counterfactual_qualification_accepted": counterfactual_qualification_accepted,
        "gameplay_acceptance_accepted": gameplay_acceptance_accepted,
        "production_authorizing_accepted": production_authorizing_accepted,
        "canonical_report_valid": True,
        "accepted": (
            production_authorizing_accepted if canary_kind == "production" else all(gates.values())
        ),
    }
    report["canonical_report_sha256"] = _canonical_report_digest(report)
    return report


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
        if rule_id in approved_kinds:
            valid = False
            continue
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
    if (
        len(rows) != 2
        or len(behavior_rows) != 1
        or len(shadow_rows) != 1
        or any(row.get("experiment_kind") not in {"behavior", "calibration"} for row in rows)
    ):
        raise SystemExit("counterfactual canary requires exactly one behavior and one shadow run")
    kwargs = {
        "natural_run_baseline_bytes_per_loop": float(
            engineering_baseline["natural_run_bytes_per_game_loop"]
        ),
        "expected_git_sha": arguments.expected_git_sha,
        "recovery_evidence": recovery_evidence,
    }
    behavior_metrics = _run_metrics(behavior_rows[0], **kwargs)
    shadow_metrics = _run_metrics(shadow_rows[0], **kwargs)
    report = build_canary_report(
        behavior_metrics,
        shadow_metrics,
        baseline_sha256=arguments.baseline_sha256,
        expected_git_sha=arguments.expected_git_sha,
        readiness_evidence=readiness_evidence,
        canary_kind=arguments.canary_kind,
    )
    if arguments.canary_kind == "production":
        canonical_valid = counterfactual_canary_is_valid(
            report,
            baseline_sha256=arguments.baseline_sha256,
            expected_git_sha=arguments.expected_git_sha,
            expected_evaluation_seeds=tuple(
                int(seed) for seed in readiness_evidence.get("evaluation_seed_ids", ())
            ),
            run_set_dir=arguments.run_set_dir,
            natural_run_baseline_bytes_per_loop=float(
                engineering_baseline["natural_run_bytes_per_game_loop"]
            ),
            recovery_evidence=recovery_evidence,
        )
    else:
        canonical_valid = report["canonical_report_sha256"] == _canonical_report_digest(report)
    report["canonical_report_valid"] = canonical_valid
    if not canonical_valid:
        report["accepted"] = False
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    raise SystemExit(0 if report["accepted"] and canonical_valid else 1)


if __name__ == "__main__":
    main()
