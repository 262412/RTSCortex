from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from rtscortex.memory import StoredEvent
from rtscortex.playbook import (
    PlaybookCondition,
    PlaybookHardQualificationRunEvidence,
    PlaybookRetryGuardBinding,
    PlaybookRule,
    PlaybookRuleCategory,
    PlaybookRuleEffect,
    PlaybookRuleStatus,
    PlaybookRuleStrength,
    analyze_hard_qualification_evaluations,
    build_hard_qualification_manifest,
    playbook_predicate_fingerprint,
    playbook_rule_fingerprint,
)
from scripts.analyze_playbook_hard_qualification import write_hard_qualification_result
from scripts.prepare_playbook_hard_qualification import _eligible_parent

GIT_SHA = "a" * 40
BASELINE_SHA = "b" * 64
PROBE_SHA = "c" * 64


def _parent() -> PlaybookRule:
    return PlaybookRule(
        rule_id="playbook-rule:parent",
        canonical_key="parent",
        category=PlaybookRuleCategory.EXECUTION_GUARD,
        conditions=(
            PlaybookCondition(field="agent_race", value="protoss"),
            PlaybookCondition(field="opponent_race", value="zerg"),
            PlaybookCondition(field="map_name", value="Simple64"),
            PlaybookCondition(field="threat_level", value="high"),
        ),
        effect=PlaybookRuleEffect.AVOID,
        strength=PlaybookRuleStrength.SOFT,
        status=PlaybookRuleStatus.ACTIVE,
        action_names=("Attack_Unit",),
        confidence=0.95,
        source_run_ids=("source-0", "source-1", "source-2"),
        source_seeds=(0, 1, 2),
        shadow_state_count=96,
        code_revision=GIT_SHA,
        sc2_patch="4.10",
        retry_guard=PlaybookRetryGuardBinding(
            failure_code="attack_target_lost",
            max_age_game_loops=224,
        ),
        evidence={
            "rule_fingerprint": "d" * 64,
            "predicate_fingerprint": "e" * 64,
            "typed_retry_binding": {
                "schema_version": "1.0",
                "kind": "execution_guard_retry",
            },
        },
    )


def _evaluation_event(
    *,
    event_id: int,
    seed: int,
    outcome: str,
    false_block: bool | None,
    observable: bool = True,
) -> StoredEvent:
    evaluation_id = f"rule-evaluation:{seed:064x}"
    return StoredEvent(
        event_id=event_id,
        run_id=f"qualification-{seed}",
        episode_id="episode-0",
        step_id=event_id,
        event_type="playbook_rule_evaluated",
        created_at=f"2026-08-08T00:00:{event_id:02d}+00:00",
        payload={
            "evaluation_id": evaluation_id,
            "application_id": f"application-{seed}",
            "rule_id": "playbook-rule:parent",
            "run_id": f"qualification-{seed}",
            "episode_id": "episode-0",
            "step_id": event_id,
            "game_loop": 100 + event_id,
            "target_kind": "candidate",
            "target_id": f"candidate-{seed}",
            "rule_kind": "execution_guard",
            "action_name": "Attack_Unit",
            "counterfactual_key": "counterfactual:" + f"{seed + 1:064x}",
            "counterfactual_signature": f"{seed + 2:064x}",
            "behavior_before_hash": f"{seed + 3:064x}",
            "decision_epoch": 100,
            "rule_fingerprint": "d" * 64,
            "predicate_fingerprint": "e" * 64,
            "counterfactual_observable": observable,
            "strength_at_evaluation": "hard",
            "status_at_evaluation": "active",
            "shadow_decision": "would_block",
            "actual_outcome": outcome,
            "execution_false_block": false_block,
            "false_block": false_block,
        },
    )


def _application_event(*, event_id: int, seed: int) -> StoredEvent:
    return StoredEvent(
        event_id=event_id,
        run_id=f"qualification-{seed}",
        episode_id="episode-0",
        step_id=event_id,
        event_type="playbook_rule_applied",
        created_at=f"2026-08-08T00:01:{event_id:02d}+00:00",
        payload={
            "application_id": f"application-{seed}",
            "rule_id": "playbook-rule:parent",
            "run_id": f"qualification-{seed}",
            "episode_id": "episode-0",
            "step_id": event_id,
            "game_loop": 100,
            "target_kind": "candidate",
            "target_id": f"candidate-{seed}",
            "rule_kind": "execution_guard",
            "rule_fingerprint": "d" * 64,
            "predicate_fingerprint": "e" * 64,
            "matched": True,
            "blocked": False,
            "score_delta": 0.0,
            "reason": "shadow_would_block",
        },
    )


def _run_evidence(
    seed: int,
    *,
    resolved: int = 1,
    unresolved: int = 0,
    false_blocks: int = 0,
) -> PlaybookHardQualificationRunEvidence:
    probe = _parent().model_copy(
        update={
            "effect": PlaybookRuleEffect.FORBID,
            "strength": PlaybookRuleStrength.HARD,
        }
    )
    return PlaybookHardQualificationRunEvidence(
        seed_id=seed,
        run_id=f"qualification-{seed}",
        run_directory=f"/runs/qualification-{seed}",
        events_sha256=f"{seed + 1:064x}",
        engineering_gates_sha256=f"{seed + 2:064x}",
        worker_stderr_sha256=f"{seed + 3:064x}",
        playbook_before_sha256=PROBE_SHA,
        playbook_after_sha256=PROBE_SHA,
        git_sha=GIT_SHA,
        source_attestation_fingerprint=f"{9:064x}",
        sc2_build="B75689",
        sc2_patch="4.10",
        natural_terminal=True,
        engineering_accepted=True,
        analysis_evidence_overflow_count=0,
        invalid_counterfactual_evidence_count=0,
        shadow_would_block_application_count=max(1, resolved + unresolved),
        resolved_counterfactual_count=resolved,
        unresolved_counterfactual_count=unresolved,
        execution_false_block_count=false_blocks,
        structurally_unobservable_counterfactual_count=0,
        invalid_application_count=0,
        application_without_evaluation_count=0,
        evaluation_without_application_count=0,
        rule_fingerprint=playbook_rule_fingerprint(probe),
        predicate_fingerprint=playbook_predicate_fingerprint(probe),
    )


def test_hard_qualification_uses_latest_terminal_evaluation() -> None:
    pending = _evaluation_event(
        event_id=1,
        seed=0,
        outcome="pending",
        false_block=None,
        observable=False,
    )
    resolved = _evaluation_event(
        event_id=3,
        seed=0,
        outcome="failed",
        false_block=False,
    )

    metrics = analyze_hard_qualification_evaluations(
        (_application_event(event_id=2, seed=0), pending, resolved),
        rule_id="playbook-rule:parent",
    )

    assert metrics.shadow_would_block_application_count == 1
    assert metrics.resolved_counterfactual_count == 1
    assert metrics.unresolved_counterfactual_count == 0
    assert metrics.execution_false_block_count == 0
    assert metrics.invalid_counterfactual_evidence_count == 0
    assert metrics.invalid_application_count == 0
    assert metrics.application_conservation_valid is True


def test_successful_shadow_action_is_a_strict_false_block() -> None:
    metrics = analyze_hard_qualification_evaluations(
        (
            _application_event(event_id=1, seed=0),
            _evaluation_event(
                event_id=2,
                seed=0,
                outcome="succeeded",
                false_block=True,
            ),
        ),
        rule_id="playbook-rule:parent",
    )

    assert metrics.resolved_counterfactual_count == 1
    assert metrics.execution_false_block_count == 1
    assert metrics.execution_false_block_rate == 1.0


def test_not_selected_is_structurally_unobservable_not_unresolved() -> None:
    metrics = analyze_hard_qualification_evaluations(
        (
            _application_event(event_id=1, seed=0),
            _evaluation_event(
                event_id=2,
                seed=0,
                outcome="not_selected",
                false_block=None,
                observable=False,
            ),
        ),
        rule_id="playbook-rule:parent",
    )

    assert metrics.structurally_unobservable_counterfactual_count == 1
    assert metrics.resolved_counterfactual_count == 0
    assert metrics.unresolved_counterfactual_count == 0
    assert metrics.execution_false_block_count == 0


def test_observable_terminal_without_boolean_is_unresolved() -> None:
    metrics = analyze_hard_qualification_evaluations(
        (
            _application_event(event_id=1, seed=0),
            _evaluation_event(
                event_id=2,
                seed=0,
                outcome="failed",
                false_block=None,
                observable=True,
            ),
        ),
        rule_id="playbook-rule:parent",
    )

    assert metrics.structurally_unobservable_counterfactual_count == 0
    assert metrics.resolved_counterfactual_count == 0
    assert metrics.unresolved_counterfactual_count == 1


def test_application_without_evaluation_and_schema_mismatch_are_invalid() -> None:
    missing = analyze_hard_qualification_evaluations(
        (_application_event(event_id=1, seed=0),),
        rule_id="playbook-rule:parent",
    )
    mismatched = analyze_hard_qualification_evaluations(
        (
            _application_event(event_id=1, seed=0),
            _evaluation_event(event_id=2, seed=0, outcome="failed", false_block=False),
        ),
        rule_id="playbook-rule:parent",
    )
    mismatched_source = _evaluation_event(
        event_id=2,
        seed=0,
        outcome="failed",
        false_block=False,
    )
    mismatched_event = replace(
        mismatched_source,
        payload={**mismatched_source.payload, "predicate_fingerprint": "f" * 64},
    )
    mismatched_fingerprint = analyze_hard_qualification_evaluations(
        (_application_event(event_id=1, seed=0), mismatched_event),
        rule_id="playbook-rule:parent",
    )
    malformed_application = replace(
        _application_event(event_id=1, seed=0),
        payload={
            key: value
            for key, value in _application_event(event_id=1, seed=0).payload.items()
            if key != "application_id"
        },
    )
    malformed = analyze_hard_qualification_evaluations(
        (malformed_application,),
        rule_id="playbook-rule:parent",
    )

    assert missing.invalid_counterfactual_evidence_count == 1
    assert missing.invalid_application_count == 1
    assert missing.application_without_evaluation_count == 1
    assert missing.application_conservation_count == 1
    assert mismatched.invalid_counterfactual_evidence_count == 0
    assert mismatched_fingerprint.invalid_counterfactual_evidence_count == 1
    assert mismatched_fingerprint.invalid_application_count == 1
    assert malformed.invalid_counterfactual_evidence_count == 1
    assert malformed.shadow_would_block_application_count == 1
    assert malformed.invalid_application_count == 1
    assert malformed.application_conservation_count == 1


def test_multiple_evaluations_for_one_application_fail_closed_and_conserve() -> None:
    first = _evaluation_event(
        event_id=2,
        seed=0,
        outcome="failed",
        false_block=False,
    )
    second = replace(
        _evaluation_event(
            event_id=3,
            seed=0,
            outcome="failed",
            false_block=False,
        ),
        payload={
            **_evaluation_event(
                event_id=3,
                seed=0,
                outcome="failed",
                false_block=False,
            ).payload,
            "evaluation_id": f"rule-evaluation:{99:064x}",
        },
    )

    metrics = analyze_hard_qualification_evaluations(
        (_application_event(event_id=1, seed=0), first, second),
        rule_id="playbook-rule:parent",
    )

    assert metrics.invalid_application_count == 1
    assert metrics.evaluation_without_application_count == 1
    assert metrics.application_conservation_count == 1
    assert metrics.shadow_would_block_application_count == 1
    assert metrics.application_conservation_valid is True


def test_manifest_schema_is_bumped_and_legacy_manifest_is_rejected() -> None:
    manifest = build_hard_qualification_manifest(
        parent=_parent(),
        baseline_sha256=BASELINE_SHA,
        probe_baseline_sha256=PROBE_SHA,
        git_sha=GIT_SHA,
        sc2_patch="4.10",
        runs=(_run_evidence(0), _run_evidence(1), _run_evidence(2)),
    )

    assert manifest.schema_version == "1.2"


def test_prepare_excludes_legacy_execution_parent_without_retry_binding() -> None:
    legacy = _parent().model_copy(update={"retry_guard": None})

    assert (
        _eligible_parent(
            legacy,
            source_run_ids=set(legacy.source_run_ids),
            source_seeds=set(legacy.source_seeds),
        )
        is False
    )


def test_manifest_requires_resolved_evidence_from_every_parent_seed() -> None:
    with pytest.raises(ValueError, match="qualification_seed_coverage"):
        build_hard_qualification_manifest(
            parent=_parent(),
            baseline_sha256=BASELINE_SHA,
            probe_baseline_sha256=PROBE_SHA,
            git_sha=GIT_SHA,
            sc2_patch="4.10",
            runs=(_run_evidence(0), _run_evidence(1)),
        )


@pytest.mark.parametrize(
    ("runs", "reason"),
    [
        (
            (_run_evidence(0), _run_evidence(1), _run_evidence(2, unresolved=1)),
            "unresolved_counterfactuals",
        ),
        (
            (_run_evidence(0), _run_evidence(1), _run_evidence(2, false_blocks=1)),
            "execution_false_block_rate",
        ),
        (
            (
                _run_evidence(0),
                _run_evidence(1),
                _run_evidence(2).model_copy(update={"sc2_patch": "5.0"}),
            ),
            "sc2_patch",
        ),
        (
            (
                _run_evidence(0).model_copy(update={"run_id": "source-0"}),
                _run_evidence(1),
                _run_evidence(2),
            ),
            "qualification_run_source_overlap",
        ),
    ],
)
def test_manifest_fails_closed_on_counterfactual_or_patch_mismatch(
    runs: tuple[PlaybookHardQualificationRunEvidence, ...],
    reason: str,
) -> None:
    with pytest.raises(ValueError, match=reason):
        build_hard_qualification_manifest(
            parent=_parent(),
            baseline_sha256=BASELINE_SHA,
            probe_baseline_sha256=PROBE_SHA,
            git_sha=GIT_SHA,
            sc2_patch="4.10",
            runs=runs,
        )


def test_manifest_binds_real_run_evidence_and_parent_statistics() -> None:
    manifest = build_hard_qualification_manifest(
        parent=_parent(),
        baseline_sha256=BASELINE_SHA,
        probe_baseline_sha256=PROBE_SHA,
        git_sha=GIT_SHA,
        sc2_patch="4.10",
        runs=(_run_evidence(0), _run_evidence(1), _run_evidence(2)),
    )

    assert manifest.schema_version == "1.2"
    assert manifest.counterfactual_evidence_accepted is True
    assert manifest.probe_baseline_sha256 == PROBE_SHA
    assert manifest.qualification_seed_ids == (0, 1, 2)
    assert manifest.source_run_ids == ("source-0", "source-1", "source-2")
    assert tuple(run.run_id for run in manifest.qualification_runs) == (
        "qualification-0",
        "qualification-1",
        "qualification-2",
    )
    assert manifest.counterfactual_resolved_count == 3
    assert manifest.counterfactual_unresolved_count == 0
    assert manifest.counterfactual_false_block_count == 0
    assert manifest.counterfactual_invalid_application_count == 0
    assert manifest.shadow_state_count == 96
    assert manifest.execution_false_block_count == 0


def test_run_evidence_hashes_and_source_fingerprint_are_required() -> None:
    with pytest.raises(ValueError):
        PlaybookHardQualificationRunEvidence(
            seed_id=0,
            run_id="qualification-0",
            run_directory=str(Path("/runs/qualification-0")),
            events_sha256="not-a-hash",
            engineering_gates_sha256="a" * 64,
            worker_stderr_sha256="b" * 64,
            playbook_before_sha256=PROBE_SHA,
            playbook_after_sha256=PROBE_SHA,
            git_sha=GIT_SHA,
            source_attestation_fingerprint="c" * 64,
            sc2_build="B75689",
            sc2_patch="4.10",
            natural_terminal=True,
            engineering_accepted=True,
            analysis_evidence_overflow_count=0,
            invalid_counterfactual_evidence_count=0,
            shadow_would_block_application_count=1,
            resolved_counterfactual_count=1,
            unresolved_counterfactual_count=0,
            execution_false_block_count=0,
            structurally_unobservable_counterfactual_count=0,
            invalid_application_count=0,
            application_without_evaluation_count=0,
            evaluation_without_application_count=0,
            rule_fingerprint="d" * 64,
            predicate_fingerprint="e" * 64,
        )


def test_hard_qualification_runner_keeps_probes_shadow_only_and_fail_closed() -> None:
    runner = (
        Path(__file__).parents[2] / "scripts" / "run_protoss_playbook_hard_qualification.sh"
    ).read_text(encoding="utf-8")
    analyzer_call = "scripts.analyze_playbook_hard_qualification"
    qualify_call = "playbook qualify-hard"

    assert "seeds=(0 1 2)" in runner
    assert "prepare_playbook_hard_qualification" in runner
    assert analyzer_call in runner
    assert qualify_call in runner
    assert runner.index(analyzer_call) < runner.index(qualify_call)
    assert "--evaluation-seed 3" in runner
    assert "--evaluation-seed 4" in runner
    assert "--evaluation-seed 5" in runner
    assert "playbook hard-readiness" in runner
    assert "--allow-canary-fixture" not in runner
    assert "run_protoss_playbook_counterfactual_canary.sh" not in runner
    assert "run_protoss_playbook_formal_24.sh" not in runner

    config = (
        Path(__file__).parents[2]
        / "configs"
        / "experiments"
        / "live_simple64_hima_protoss_ensemble_cortex_v0_5_hard_qualification_shadow.yaml"
    ).read_text(encoding="utf-8")
    assert "learning_mode: frozen" in config
    assert "rule_mode: shadow" in config
    assert "hard_readiness_required: false" in config
    assert "allow_canary_fixture: false" in config


def test_hard_rejection_records_nonzero_exit_and_report_without_manifest(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "hard-qualification-report.json"
    manifest_path = tmp_path / "hard-qualification-manifest.json"

    exit_code, message = write_hard_qualification_result(
        {"accepted": False, "parents": []},
        None,
        report_output=report_path,
        manifest_output=manifest_path,
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert message == (f"hard_qualification rejected exit_code=1 report={report_path.resolve()}")
    assert payload["accepted"] is False
    assert payload["status"] == "rejected"
    assert payload["exit_code"] == 1
    assert payload["report_path"] == str(report_path.resolve())
    assert manifest_path.exists() is False
