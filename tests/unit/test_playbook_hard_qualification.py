from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

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
    PlaybookStore,
    TypedRetryCoverage,
    TypedRetryCoverageBySeed,
    analyze_hard_qualification_evaluations,
    build_hard_qualification_manifest,
    playbook_predicate_fingerprint,
    playbook_rule_fingerprint,
)
from scripts.analyze_playbook_hard_qualification import (
    _index_batch_status_rows,
    _qualification_config_fingerprint,
    _require_complete_batch_run_audit,
    _select_global_manifest,
    _values_for_parent_batch,
    build_incomplete_hard_qualification_report,
    write_hard_qualification_result,
)
from scripts.prepare_playbook_hard_qualification import (
    ProbeBatchCapacityError,
    _create_probe_batch,
    _eligible_parent,
    _load_source_rows,
    _partition_probe_batches,
)

GIT_SHA = "a" * 40
BASELINE_SHA = "b" * 64
PROBE_SHA = "c" * 64


def _write_source_status_fixture(
    tmp_path: Path,
    *,
    reviewed_tree_after: str = "5" * 64,
) -> tuple[Path, tuple[Path, ...]]:
    columns = (
        "seed",
        "exit_code",
        "run_dir",
        "accepted",
        "git_head_before",
        "git_head_after",
        "submodule_commit_before",
        "submodule_commit_after",
        "reviewed_commit_before",
        "reviewed_commit_after",
        "reviewed_tree_sha256_before",
        "reviewed_tree_sha256_after",
    )
    run_directories: list[Path] = []
    rows: list[dict[str, str]] = []
    for seed in (0, 1, 2):
        run_directory = tmp_path / f"source-{seed}"
        run_directory.mkdir()
        (run_directory / "engineering-gates.json").write_text(
            json.dumps(
                {
                    "accepted": True,
                    "evidence": {
                        "diagnostic_only": False,
                        "expected_git_sha": GIT_SHA,
                    },
                }
            ),
            encoding="utf-8",
        )
        (run_directory / "events.jsonl").write_text("", encoding="utf-8")
        (run_directory / "worker.stderr.log").write_text(
            "Version: B75689 (SC2.4.10)\n",
            encoding="utf-8",
        )
        (run_directory / "config.yaml").write_text(
            "environment:\n  agent_race: protoss\n  opponent_race: zerg\n  scenario: Simple64\n",
            encoding="utf-8",
        )
        run_directories.append(run_directory)
        rows.append(
            {
                "seed": str(seed),
                "exit_code": "0",
                "run_dir": str(run_directory),
                "accepted": "true",
                "git_head_before": GIT_SHA,
                "git_head_after": GIT_SHA,
                "submodule_commit_before": "8" * 40,
                "submodule_commit_after": "8" * 40,
                "reviewed_commit_before": "8" * 40,
                "reviewed_commit_after": "8" * 40,
                "reviewed_tree_sha256_before": "5" * 64,
                "reviewed_tree_sha256_after": reviewed_tree_after,
            }
        )
    status_path = tmp_path / "qualification-status.tsv"
    with status_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return status_path, tuple(run_directories)


def test_load_source_rows_accepts_source_qualification_reviewed_tree_schema(
    tmp_path: Path,
) -> None:
    status_path, run_directories = _write_source_status_fixture(tmp_path)

    source_rows, sc2_build, _, source_git_sha, attestation = _load_source_rows(
        status_path,
        run_directories,
        sc2_patch="4.10",
    )

    assert [row["seed_id"] for row in source_rows] == [0, 1, 2]
    assert sc2_build == "B75689"
    assert source_git_sha == GIT_SHA
    assert attestation["reviewed_tree"] == "5" * 64


def test_load_source_rows_rejects_changed_reviewed_tree_sha256(tmp_path: Path) -> None:
    status_path, run_directories = _write_source_status_fixture(
        tmp_path,
        reviewed_tree_after="6" * 64,
    )

    with pytest.raises(ValueError, match="failed attestation: reviewed_tree"):
        _load_source_rows(status_path, run_directories, sc2_patch="4.10")


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


def _parents(count: int, *, same_shadow_count: bool = False) -> tuple[PlaybookRule, ...]:
    return tuple(
        _parent().model_copy(
            update={
                "rule_id": f"playbook-rule:parent-{index:02d}",
                "canonical_key": f"parent-{index:02d}",
                "shadow_state_count": 96 if same_shadow_count else 200 - index,
            }
        )
        for index in range(count)
    )


def _batch_plan(tmp_path: Path) -> dict[str, object]:
    first_probe = tmp_path / "probe.batch-000.sqlite3"
    second_probe = tmp_path / "probe.batch-001.sqlite3"
    return {
        "schema_version": "2.0",
        "artifact_kind": "playbook-hard-qualification-plan",
        "expected_git_sha": GIT_SHA,
        "source_git_sha": "9" * 40,
        "source_attestation": {
            "submodule_commit": "8" * 40,
            "submodule_gitlink": "8" * 40,
            "submodule_diff_sha256": "7" * 64,
            "reviewed_commit": "8" * 40,
            "reviewed_diff_sha256": "6" * 64,
            "reviewed_tree_sha256": "5" * 64,
        },
        "source_runs": [{"seed_id": seed} for seed in (0, 1, 2)],
        "eligible_parent_ids": [
            "playbook-rule:parent-00",
            "playbook-rule:parent-01",
        ],
        "batches": [
            {
                "batch_id": "batch-000",
                "batch_index": 0,
                "parent_rule_ids": ["playbook-rule:parent-00"],
                "probe_baseline_path": str(first_probe.resolve()),
                "probe_baseline_sha256": "1" * 64,
            },
            {
                "batch_id": "batch-001",
                "batch_index": 1,
                "parent_rule_ids": ["playbook-rule:parent-01"],
                "probe_baseline_path": str(second_probe.resolve()),
                "probe_baseline_sha256": "2" * 64,
            },
        ],
    }


def _status_row(plan: dict[str, object], batch_index: int, seed: int) -> dict[str, str]:
    batch = plan["batches"][batch_index]  # type: ignore[index]
    source = plan["source_attestation"]
    assert isinstance(batch, dict)
    assert isinstance(source, dict)
    return {
        "batch_id": str(batch["batch_id"]),
        "batch_index": str(batch["batch_index"]),
        "seed": str(seed),
        "probe_path": str(batch["probe_baseline_path"]),
        "probe_sha256": str(batch["probe_baseline_sha256"]),
        "exit_code": "0",
        "git_head_before": GIT_SHA,
        "git_head_after": GIT_SHA,
        "superproject_dirty_before": "false",
        "superproject_dirty_after": "false",
        "submodule_dirty_before": "false",
        "submodule_dirty_after": "false",
        "submodule_commit_before": str(source["submodule_commit"]),
        "submodule_commit_after": str(source["submodule_commit"]),
        "submodule_gitlink_before": str(source["submodule_gitlink"]),
        "submodule_gitlink_after": str(source["submodule_gitlink"]),
        "submodule_diff_sha256_before": str(source["submodule_diff_sha256"]),
        "submodule_diff_sha256_after": str(source["submodule_diff_sha256"]),
        "reviewed_source_commit_before": str(source["reviewed_commit"]),
        "reviewed_source_commit_after": str(source["reviewed_commit"]),
        "reviewed_source_diff_sha256_before": str(source["reviewed_diff_sha256"]),
        "reviewed_source_diff_sha256_after": str(source["reviewed_diff_sha256"]),
        "reviewed_source_tree_sha256_before": str(source["reviewed_tree_sha256"]),
        "reviewed_source_tree_sha256_after": str(source["reviewed_tree_sha256"]),
    }


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
        typed_retry_opportunity_count=1,
        typed_retry_application_count=1,
        typed_retry_coverage_unavailable=False,
        typed_retry_coverage_reasons=(),
        config_sha256=f"{seed + 10:064x}",
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
            typed_retry_coverage=TypedRetryCoverageBySeed(),
        )
        is False
    )


def test_prepare_requires_typed_retry_opportunity_in_every_source_seed() -> None:
    parent = _parent()
    sparse = TypedRetryCoverageBySeed(
        by_seed={
            0: TypedRetryCoverage(typed_retry_opportunity_count=1),
            1: TypedRetryCoverage(typed_retry_opportunity_count=0),
            2: TypedRetryCoverage(typed_retry_opportunity_count=1),
        }
    )
    assert (
        _eligible_parent(
            parent,
            source_run_ids=set(parent.source_run_ids),
            source_seeds=set(parent.source_seeds),
            typed_retry_coverage=sparse,
        )
        is False
    )
    complete = TypedRetryCoverageBySeed(
        by_seed={
            seed: TypedRetryCoverage(typed_retry_opportunity_count=1)
            for seed in parent.source_seeds
        }
    )
    assert _eligible_parent(
        parent,
        source_run_ids=set(parent.source_run_ids),
        source_seeds=set(parent.source_seeds),
        typed_retry_coverage=complete,
    )


def test_manifest_rejects_missing_typed_retry_coverage() -> None:
    runs = tuple(
        _run_evidence(seed).model_copy(
            update={
                "typed_retry_opportunity_count": None,
                "typed_retry_application_count": None,
                "typed_retry_coverage_unavailable": None,
                "typed_retry_coverage_reasons": None,
                "config_sha256": None,
            }
        )
        for seed in (0, 1, 2)
    )
    with pytest.raises(ValueError, match="typed_retry_coverage_unavailable"):
        build_hard_qualification_manifest(
            parent=_parent(),
            baseline_sha256=BASELINE_SHA,
            probe_baseline_sha256=PROBE_SHA,
            git_sha=GIT_SHA,
            sc2_patch="4.10",
            runs=runs,
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


@pytest.mark.parametrize(
    ("candidate_count", "expected_sizes"),
    [(8, [8]), (9, [8, 1]), (10, [8, 2])],
)
def test_probe_candidates_are_partitioned_into_bounded_batches(
    candidate_count: int,
    expected_sizes: list[int],
) -> None:
    batches = _partition_probe_batches(
        _parents(candidate_count),
        max_probes_per_batch=8,
        max_probe_batches=2,
    )

    assert [len(batch) for batch in batches] == expected_sizes
    flattened = [rule.rule_id for batch in batches for rule in batch]
    assert flattened == [rule.rule_id for rule in _parents(candidate_count)]
    assert len(flattened) == len(set(flattened)) == candidate_count


def test_probe_batch_sort_is_stable_for_equal_shadow_counts() -> None:
    parents = tuple(reversed(_parents(10, same_shadow_count=True)))

    batches = _partition_probe_batches(
        parents,
        max_probes_per_batch=8,
        max_probe_batches=2,
    )

    assert [rule.rule_id for batch in batches for rule in batch] == sorted(
        rule.rule_id for rule in parents
    )


def test_probe_capacity_failure_reports_every_candidate() -> None:
    parents = _parents(17)

    with pytest.raises(ProbeBatchCapacityError) as raised:
        _partition_probe_batches(
            parents,
            max_probes_per_batch=8,
            max_probe_batches=2,
        )

    payload = raised.value.payload
    assert payload["eligible_parent_count"] == 17
    assert payload["required_batch_count"] == 3
    assert payload["max_probe_batches"] == 2
    assert payload["eligible_parent_ids"] == [rule.rule_id for rule in parents]


def test_probe_batch_activates_only_members_and_suspends_other_eligible_rules(
    tmp_path: Path,
) -> None:
    parents = _parents(10)
    baseline = tmp_path / "baseline.sqlite3"
    probe = tmp_path / "probe.batch-001.sqlite3"
    store = PlaybookStore(baseline)
    try:
        for parent in parents:
            store.upsert_rule(parent)
    finally:
        store.close()
    baseline_sha256 = hashlib.sha256(baseline.read_bytes()).hexdigest()

    validation = _create_probe_batch(
        baseline_path=baseline,
        probe_path=probe,
        eligible_parents=parents,
        batch_members=parents[8:],
        batch_id="batch-001",
        batch_index=1,
        baseline_sha256=baseline_sha256,
        expected_git_sha=GIT_SHA,
        sc2_build="B75689",
        sc2_patch="4.10",
        max_probes_per_batch=8,
    )

    assert validation["verified"] is True
    assert validation["active_hard_parent_ids"] == [
        "playbook-rule:parent-08",
        "playbook-rule:parent-09",
    ]
    probe_store = PlaybookStore(probe, read_only=True)
    try:
        rules = {rule.rule_id: rule for rule in probe_store.rules()}
        runtime_rules = probe_store.rules_for_guard(max_hard=8, max_soft=8)
    finally:
        probe_store.close()
    for parent in parents[:8]:
        isolated = rules[parent.rule_id]
        assert isolated.status is PlaybookRuleStatus.SUSPENDED
        assert isolated.strength is PlaybookRuleStrength.SOFT
        assert isolated.effect is PlaybookRuleEffect.AVOID
    for parent in parents[8:]:
        active = rules[parent.rule_id]
        assert active.status is PlaybookRuleStatus.ACTIVE
        assert active.strength is PlaybookRuleStrength.HARD
        assert active.effect is PlaybookRuleEffect.FORBID
    assert {rule.rule_id for rule in runtime_rules} == {
        "playbook-rule:parent-08",
        "playbook-rule:parent-09",
    }
    assert hashlib.sha256(baseline.read_bytes()).hexdigest() == baseline_sha256


def test_batch_status_requires_exact_batch_seed_cross_product(tmp_path: Path) -> None:
    plan = _batch_plan(tmp_path)
    rows = [_status_row(plan, batch, seed) for batch in (0, 1) for seed in (0, 1, 2)]

    indexed = _index_batch_status_rows(plan, rows)

    assert set(indexed) == {
        ("batch-000", 0),
        ("batch-000", 1),
        ("batch-000", 2),
        ("batch-001", 0),
        ("batch-001", 1),
        ("batch-001", 2),
    }


def test_batch_status_rejects_duplicate_source_seed(tmp_path: Path) -> None:
    plan = _batch_plan(tmp_path)
    plan["source_runs"] = [{"seed_id": seed} for seed in (0, 0, 2)]

    with pytest.raises(ValueError, match="seeds 0/1/2 exactly once"):
        _index_batch_status_rows(plan, [])


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("missing", "missing batch-seed status"),
        ("duplicate", "duplicate batch-seed status"),
        ("extra", "unknown batch ID"),
        ("probe_hash", "probe SHA"),
        ("git", "Git attestation"),
        ("submodule_dirty", "Git attestation"),
    ],
)
def test_batch_status_fails_closed_on_partition_or_attestation_mismatch(
    tmp_path: Path,
    mutation: str,
    reason: str,
) -> None:
    plan = _batch_plan(tmp_path)
    rows = [_status_row(plan, batch, seed) for batch in (0, 1) for seed in (0, 1, 2)]
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(dict(rows[0]))
    elif mutation == "extra":
        rows[-1]["batch_id"] = "batch-999"
    elif mutation == "probe_hash":
        rows[-1]["probe_sha256"] = "f" * 64
    elif mutation == "git":
        rows[-1]["git_head_after"] = "f" * 40
    else:
        rows[-1]["submodule_dirty_after"] = "true"

    with pytest.raises(ValueError, match=reason):
        _index_batch_status_rows(plan, rows)


def test_parent_evidence_is_scoped_to_its_own_batch() -> None:
    values = {
        ("batch-000", 0): "first-0",
        ("batch-000", 1): "first-1",
        ("batch-000", 2): "first-2",
        ("batch-001", 0): "second-0",
        ("batch-001", 1): "second-1",
        ("batch-001", 2): "second-2",
    }

    selected = _values_for_parent_batch(
        "playbook-rule:parent-01",
        batch_id_by_parent={
            "playbook-rule:parent-00": "batch-000",
            "playbook-rule:parent-01": "batch-001",
        },
        values_by_batch_seed=values,
        expected_seeds=(0, 1, 2),
    )

    assert selected == {0: "second-0", 1: "second-1", 2: "second-2"}


def test_common_batch_run_failure_rejects_the_global_audit() -> None:
    reports = [
        {"batch_id": "batch-000", "seed_id": 0, "accepted": True},
        {"batch_id": "batch-000", "seed_id": 1, "accepted": True},
        {"batch_id": "batch-001", "seed_id": 0, "accepted": False},
    ]

    with pytest.raises(ValueError, match="batch-001:0"):
        _require_complete_batch_run_audit(reports)


def test_complete_batch_run_audit_accepts_all_common_runs() -> None:
    _require_complete_batch_run_audit(
        [
            {"batch_id": batch_id, "seed_id": seed, "accepted": True}
            for batch_id in ("batch-000", "batch-001")
            for seed in (0, 1, 2)
        ]
    )


def test_global_manifest_selection_uses_frozen_order() -> None:
    candidates = [
        SimpleNamespace(
            parent_rule_id="parent-c",
            counterfactual_false_block_rate=0.1,
            counterfactual_resolved_count=9,
        ),
        SimpleNamespace(
            parent_rule_id="parent-b",
            counterfactual_false_block_rate=0.0,
            counterfactual_resolved_count=3,
        ),
        SimpleNamespace(
            parent_rule_id="parent-a",
            counterfactual_false_block_rate=0.0,
            counterfactual_resolved_count=3,
        ),
        SimpleNamespace(
            parent_rule_id="parent-z",
            counterfactual_false_block_rate=0.0,
            counterfactual_resolved_count=2,
        ),
    ]

    selected, ordered = _select_global_manifest(candidates)

    assert selected is not None
    assert selected.parent_rule_id == "parent-a"
    assert [item.parent_rule_id for item in ordered] == [
        "parent-a",
        "parent-b",
        "parent-z",
        "parent-c",
    ]


def test_incomplete_runner_report_lists_completed_failed_and_unrun(
    tmp_path: Path,
) -> None:
    plan = _batch_plan(tmp_path)
    completed = _status_row(plan, 0, 0)
    failed = _status_row(plan, 0, 1)
    failed["exit_code"] = "42"

    report = build_incomplete_hard_qualification_report(
        plan,
        [completed, failed],
        reason="runner_failed:batch-000:seed-1",
        runner_exit_code=42,
    )

    assert report["accepted"] is False
    assert report["batch_audit_complete"] is False
    assert report["completed_batch_seed_runs"] == ["batch-000:0"]
    assert report["failed_batch_seed_runs"] == ["batch-000:1"]
    assert report["unrun_batch_seed_runs"] == [
        "batch-000:2",
        "batch-001:0",
        "batch-001:1",
        "batch-001:2",
    ]
    eligible_parent_ids = plan["eligible_parent_ids"]
    assert isinstance(eligible_parent_ids, list)
    assert {parent["parent_rule_id"] for parent in report["parents"]} == set(eligible_parent_ids)
    assert all(parent["accepted"] is False for parent in report["parents"])


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
    assert "--max-probes-per-batch 8" in runner
    assert "--max-probe-batches 2" in runner
    assert "read -r batch_id batch_index probe_path probe_sha256" in runner
    assert "${batch_id}.seed-${seed}.log" in runner
    assert "write_terminal_rejection" in runner
    assert analyzer_call in runner
    assert qualify_call in runner
    assert runner.count(qualify_call) == 1
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


def test_qualification_config_binding_ignores_seed_but_detects_probe_drift(
    tmp_path: Path,
) -> None:
    seed_zero = tmp_path / "seed-0.yaml"
    seed_one = tmp_path / "seed-1.yaml"
    drifted = tmp_path / "drifted.yaml"
    seed_zero.write_text(
        "run:\n"
        "  output_root: /outputs\n"
        "  seed: 0\n"
        "cortex:\n"
        "  playbook:\n"
        "    enabled: true\n"
        "    learning_mode: frozen\n"
        "    rule_mode: shadow\n",
        encoding="utf-8",
    )
    seed_one.write_text(
        seed_zero.read_text(encoding="utf-8").replace("seed: 0", "seed: 1"),
        encoding="utf-8",
    )
    drifted.write_text(
        seed_zero.read_text(encoding="utf-8").replace("rule_mode: shadow", "rule_mode: active"),
        encoding="utf-8",
    )

    assert _qualification_config_fingerprint(seed_zero) == _qualification_config_fingerprint(
        seed_one
    )
    assert _qualification_config_fingerprint(seed_zero) != _qualification_config_fingerprint(
        drifted
    )


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
