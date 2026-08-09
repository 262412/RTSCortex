import hashlib
import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from rtscortex.cli.app import app
from rtscortex.playbook import (
    PlaybookCondition,
    PlaybookConditionOperator,
    PlaybookHardReadinessReport,
    PlaybookRetryGuardBinding,
    PlaybookRule,
    PlaybookRuleCategory,
    PlaybookRuleEffect,
    PlaybookRuleKind,
    PlaybookRuleStatus,
    PlaybookRuleStrength,
    PlaybookStore,
    analyze_hard_readiness_database,
    create_canary_fixture,
    evaluation_kind,
    playbook_predicate_fingerprint,
    playbook_rule_fingerprint,
    qualify_hard_rule,
)

GIT_SHA = "a" * 40


def _soft_rule() -> PlaybookRule:
    return PlaybookRule(
        rule_id="soft-rule",
        canonical_key="soft-rule",
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
        action_names=("Build_Pylon_Screen",),
        role_ids=("economy",),
        confidence=0.95,
        source_run_ids=("run-0", "run-1", "run-2"),
        source_seeds=(0, 1, 2),
        shadow_state_count=48,
        code_revision=GIT_SHA,
        sc2_patch="4.10",
        retry_guard=PlaybookRetryGuardBinding(
            failure_code="build_target_lost",
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


def _write_store(path: Path, rule: PlaybookRule) -> None:
    store = PlaybookStore(path)
    try:
        store.upsert_rule(rule)
    finally:
        store.close()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _qualification_manifest(
    path: Path,
    database: Path,
    parent: PlaybookRule,
    **updates: object,
) -> None:
    probe_sha256 = "c" * 64
    probe = parent.model_copy(
        update={
            "effect": PlaybookRuleEffect.FORBID,
            "strength": PlaybookRuleStrength.HARD,
        }
    )
    probe_rule_fingerprint = playbook_rule_fingerprint(probe)
    probe_predicate_fingerprint = playbook_predicate_fingerprint(probe)
    qualification_runs = [
        {
            "seed_id": seed,
            "run_id": f"qualification-{seed}",
            "run_directory": f"/qualification/{seed}",
            "events_sha256": f"{seed + 1:064x}",
            "engineering_gates_sha256": f"{seed + 2:064x}",
            "worker_stderr_sha256": f"{seed + 3:064x}",
            "playbook_before_sha256": probe_sha256,
            "playbook_after_sha256": probe_sha256,
            "git_sha": GIT_SHA,
            "source_attestation_fingerprint": "d" * 64,
            "sc2_build": "B75689",
            "sc2_patch": "4.10",
            "natural_terminal": True,
            "engineering_accepted": True,
            "analysis_evidence_overflow_count": 0,
            "invalid_counterfactual_evidence_count": 0,
            "shadow_would_block_application_count": 1,
            "resolved_counterfactual_count": 1,
            "unresolved_counterfactual_count": 0,
            "execution_false_block_count": 0,
            "structurally_unobservable_counterfactual_count": 0,
            "invalid_application_count": 0,
            "application_without_evaluation_count": 0,
            "evaluation_without_application_count": 0,
            "rule_fingerprint": probe_rule_fingerprint,
            "predicate_fingerprint": probe_predicate_fingerprint,
            "typed_retry_opportunity_count": 1,
            "typed_retry_application_count": 1,
            "typed_retry_coverage_unavailable": False,
            "typed_retry_coverage_reasons": [],
            "config_sha256": f"{seed + 10:064x}",
        }
        for seed in (0, 1, 2)
    ]
    payload: dict[str, object] = {
        "schema_version": "1.2",
        "artifact_kind": "playbook-hard-qualification",
        "parent_rule_id": parent.rule_id,
        "parent_canonical_key": parent.canonical_key,
        "baseline_sha256": _sha256(database),
        "probe_baseline_sha256": probe_sha256,
        "git_sha": GIT_SHA,
        "sc2_patch": "4.10",
        "qualification_seed_ids": [0, 1, 2],
        "source_run_ids": ["run-0", "run-1", "run-2"],
        "qualification_runs": qualification_runs,
        "engineering_accepted": True,
        "counterfactual_evidence_accepted": True,
        "analysis_evidence_overflow_count": 0,
        "counterfactual_resolved_count": 3,
        "counterfactual_unresolved_count": 0,
        "counterfactual_false_block_count": 0,
        "counterfactual_false_block_rate": 0.0,
        "shadow_state_count": 48,
        "execution_false_block_count": 0,
        "execution_false_block_rate": 0.0,
        "counterfactual_structurally_unobservable_count": 0,
        "counterfactual_invalid_application_count": 0,
        "application_without_evaluation_count": 0,
        "evaluation_without_application_count": 0,
        "rule_fingerprint": probe_rule_fingerprint,
        "predicate_fingerprint": probe_predicate_fingerprint,
        "typed_retry_opportunity_count_by_seed": {"0": 1, "1": 1, "2": 1},
        "typed_retry_application_count_by_seed": {"0": 1, "1": 1, "2": 1},
        "typed_retry_coverage_unavailable_by_seed": {},
    }
    payload.update(updates)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _strategic_ab_manifest(
    path: Path,
    qualification_manifest: Path,
    parent: PlaybookRule,
    **updates: object,
) -> None:
    qualification = json.loads(qualification_manifest.read_text(encoding="utf-8"))
    payload: dict[str, object] = {
        "schema_version": "1.0",
        "artifact_kind": "playbook-strategic-ab-qualification",
        "parent_rule_id": parent.rule_id,
        "parent_canonical_key": parent.canonical_key,
        "baseline_sha256": qualification["baseline_sha256"],
        "git_sha": GIT_SHA,
        "sc2_patch": "4.10",
        "qualification_seed_ids": [0, 1, 2],
        "source_run_ids": ["run-0", "run-1", "run-2"],
        "paired_seed_ids": [0, 1, 2],
        "paired_run_ids": ["run-0", "run-1", "run-2"],
        "accepted": True,
        "analysis_evidence_overflow_count": 0,
        "repeat_error_reduction": 0.5,
        "task_score_improvement": 0.1,
        "win_rate_delta": 0.0,
    }
    payload.update(updates)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _config(path: Path, *, allow_fixture: bool = False) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "environment": {
                    "agent_race": "protoss",
                    "opponent_race": "zerg",
                    "scenario": "Simple64",
                },
                "cortex": {
                    "playbook": {
                        "allow_canary_fixture": allow_fixture,
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def test_soft_rule_readiness_fails_with_audit_reasons(tmp_path: Path) -> None:
    database = tmp_path / "playbook.sqlite3"
    _write_store(database, _soft_rule())

    report = analyze_hard_readiness_database(
        database,
        expected_git_sha=GIT_SHA,
        sc2_patch="4.10",
        agent_race="protoss",
        opponent_race="zerg",
        map_name="Simple64",
        evaluation_seed_ids=(3, 4, 5),
    )

    assert report.active_soft_count == 1
    assert report.active_hard_count == 0
    assert report.context_applicable_blocking_hard_count == 0
    assert report.canary_runnable is False
    assert "strength_is_soft" in report.rejection_reasons["soft-rule"]
    assert "effect_is_avoid_not_forbid" in report.rejection_reasons["soft-rule"]
    audit = report.rules[0]
    assert audit.action_names == ("Build_Pylon_Screen",)
    assert audit.role_ids == ("economy",)
    assert audit.source_run_ids == ("run-0", "run-1", "run-2")
    assert audit.source_seed_ids == (0, 1, 2)
    assert audit.execution_false_block_rate == 0.0
    assert audit.code_revision == GIT_SHA
    assert audit.rule_fingerprint == playbook_rule_fingerprint(_soft_rule())
    assert audit.predicate_fingerprint == playbook_predicate_fingerprint(_soft_rule())


def test_qualification_creates_disjoint_hard_child_without_mutating_parent(
    tmp_path: Path,
) -> None:
    database = tmp_path / "playbook.sqlite3"
    manifest = tmp_path / "qualification.json"
    parent = _soft_rule()
    _write_store(database, parent)
    _qualification_manifest(manifest, database, parent)
    store = PlaybookStore(database)
    try:
        child = qualify_hard_rule(
            store,
            parent_rule_id="soft-rule",
            expected_git_sha=GIT_SHA,
            sc2_patch="4.10",
            qualification_manifest_path=manifest,
            evaluation_seed_ids=(3, 4, 5),
        )
        rules = {rule.rule_id: rule for rule in store.rules()}
    finally:
        store.close()

    assert child.parent_rule_id == "soft-rule"
    assert child.effect is PlaybookRuleEffect.FORBID
    assert child.strength is PlaybookRuleStrength.HARD
    assert child.qualification_seed_ids == (0, 1, 2)
    assert child.evaluation_seed_ids == (3, 4, 5)
    assert child.evidence["qualification_manifest_sha256"] in child.evidence_hashes
    assert rules["soft-rule"].strength is PlaybookRuleStrength.SOFT
    report = analyze_hard_readiness_database(
        database,
        expected_git_sha=GIT_SHA,
        sc2_patch="4.10",
        agent_race="protoss",
        opponent_race="zerg",
        map_name="Simple64",
        evaluation_seed_ids=(3, 4, 5),
    )
    assert report.context_applicable_blocking_hard_count == 1
    assert report.reachable_blocking_hard_count == 1
    assert report.approved_hard_rule_ids == (child.rule_id,)
    assert report.runtime_selected_hard_rule_ids == (child.rule_id,)
    assert report.canary_runnable is True


def test_readiness_rejects_qualification_manifest_without_typed_retry_coverage(
    tmp_path: Path,
) -> None:
    database = tmp_path / "playbook.sqlite3"
    manifest = tmp_path / "qualification.json"
    parent = _soft_rule()
    _write_store(database, parent)
    _qualification_manifest(manifest, database, parent)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    for run in payload["qualification_runs"]:
        run["typed_retry_opportunity_count"] = None
        run["typed_retry_application_count"] = None
        run["typed_retry_coverage_unavailable"] = None
        run["typed_retry_coverage_reasons"] = None
        run["config_sha256"] = None
    payload["typed_retry_opportunity_count_by_seed"] = {}
    payload["typed_retry_application_count_by_seed"] = {}
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    store = PlaybookStore(database)
    try:
        with pytest.raises(
            ValueError,
            match="qualification_manifest_typed_retry_coverage_unavailable",
        ):
            qualify_hard_rule(
                store,
                parent_rule_id=parent.rule_id,
                expected_git_sha=GIT_SHA,
                sc2_patch="4.10",
                qualification_manifest_path=manifest,
                evaluation_seed_ids=(3, 4, 5),
            )
    finally:
        store.close()


def test_tactical_response_qualification_kind_matches_shared_evaluation_kind(
    tmp_path: Path,
) -> None:
    database = tmp_path / "playbook.sqlite3"
    manifest = tmp_path / "qualification.json"
    parent = _soft_rule().model_copy(
        update={
            "rule_id": "tactical-soft-rule",
            "canonical_key": "tactical-soft-rule",
            "category": PlaybookRuleCategory.TACTICAL_RESPONSE,
            "retry_guard": None,
        }
    )
    _write_store(database, parent)
    _qualification_manifest(manifest, database, parent)
    store = PlaybookStore(database)
    try:
        child = qualify_hard_rule(
            store,
            parent_rule_id=parent.rule_id,
            expected_git_sha=GIT_SHA,
            sc2_patch="4.10",
            qualification_manifest_path=manifest,
            evaluation_seed_ids=(3, 4, 5),
        )
    finally:
        store.close()

    assert child.qualification_kind == "execution"
    assert evaluation_kind(child.category) is PlaybookRuleKind.EXECUTION_GUARD


def test_qualification_rejects_reused_evaluation_seed(tmp_path: Path) -> None:
    database = tmp_path / "playbook.sqlite3"
    manifest = tmp_path / "qualification.json"
    parent = _soft_rule()
    _write_store(database, parent)
    _qualification_manifest(manifest, database, parent)
    store = PlaybookStore(database)
    try:
        try:
            qualify_hard_rule(
                store,
                parent_rule_id="soft-rule",
                expected_git_sha=GIT_SHA,
                sc2_patch="4.10",
                qualification_manifest_path=manifest,
                evaluation_seed_ids=(2, 3, 4),
            )
        except ValueError as error:
            assert "overlap" in str(error)
        else:
            raise AssertionError("overlapping qualification/evaluation seeds were accepted")
    finally:
        store.close()


def test_strategic_hard_qualification_requires_paired_outcome_evidence(
    tmp_path: Path,
) -> None:
    database = tmp_path / "playbook.sqlite3"
    manifest = tmp_path / "qualification.json"
    strategic = _soft_rule().model_copy(
        update={"category": PlaybookRuleCategory.MATCHUP_STRATEGY, "retry_guard": None}
    )
    _write_store(database, strategic)
    _qualification_manifest(manifest, database, strategic)
    store = PlaybookStore(database)
    try:
        with pytest.raises(ValueError, match="paired outcome evidence"):
            qualify_hard_rule(
                store,
                parent_rule_id="soft-rule",
                expected_git_sha=GIT_SHA,
                sc2_patch="4.10",
                qualification_manifest_path=manifest,
                evaluation_seed_ids=(3, 4, 5),
            )
    finally:
        store.close()


def test_canary_fixture_is_only_runnable_when_explicitly_allowed(tmp_path: Path) -> None:
    database = tmp_path / "fixture.sqlite3"
    create_canary_fixture(database, expected_git_sha=GIT_SHA, sc2_patch="4.10")

    rejected = analyze_hard_readiness_database(
        database,
        expected_git_sha=GIT_SHA,
        sc2_patch="4.10",
        agent_race="protoss",
        opponent_race="zerg",
        map_name="Simple64",
    )
    allowed = analyze_hard_readiness_database(
        database,
        expected_git_sha=GIT_SHA,
        sc2_patch="4.10",
        agent_race="protoss",
        opponent_race="zerg",
        map_name="Simple64",
        allow_canary_fixture=True,
    )

    assert rejected.canary_runnable is False
    assert "canary_fixture_not_allowed" in next(iter(rejected.rejection_reasons.values()))
    assert allowed.canary_runnable is True
    assert allowed.canary_fixture_rule_ids


def test_readiness_cli_writes_artifact_and_exits_two_when_not_runnable(
    tmp_path: Path,
) -> None:
    database = tmp_path / "playbook.sqlite3"
    config = tmp_path / "config.yaml"
    output = tmp_path / "playbook-hard-readiness.json"
    _write_store(database, _soft_rule())
    _config(config)

    result = CliRunner().invoke(
        app,
        [
            "playbook",
            "hard-readiness",
            "--database",
            str(database),
            "--config",
            str(config),
            "--expected-git-sha",
            GIT_SHA,
            "--sc2-patch",
            "4.10",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 2
    report = PlaybookHardReadinessReport.model_validate(
        json.loads(output.read_text(encoding="utf-8"))
    )
    assert report.canary_runnable is False


def test_arbitrary_json_cannot_qualify_hard_rule(tmp_path: Path) -> None:
    database = tmp_path / "playbook.sqlite3"
    evidence = tmp_path / "unrelated.json"
    parent = _soft_rule()
    _write_store(database, parent)
    evidence.write_text('{"accepted": true}\n', encoding="utf-8")
    store = PlaybookStore(database)
    try:
        with pytest.raises(ValueError, match="invalid hard qualification manifest"):
            qualify_hard_rule(
                store,
                parent_rule_id=parent.rule_id,
                expected_git_sha=GIT_SHA,
                sc2_patch="4.10",
                qualification_manifest_path=evidence,
                evaluation_seed_ids=(3, 4, 5),
            )
    finally:
        store.close()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("parent_rule_id", "other-rule"),
        ("git_sha", "b" * 40),
        ("sc2_patch", "5.0"),
        ("qualification_seed_ids", [0, 1, 9]),
        ("source_run_ids", ["run-0", "run-1", "other"]),
        ("engineering_accepted", False),
        ("counterfactual_evidence_accepted", False),
        ("analysis_evidence_overflow_count", 1),
    ],
)
def test_qualification_manifest_must_match_parent_and_evidence(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    database = tmp_path / "playbook.sqlite3"
    manifest = tmp_path / "qualification.json"
    parent = _soft_rule()
    _write_store(database, parent)
    _qualification_manifest(manifest, database, parent, **{field: value})
    store = PlaybookStore(database)
    try:
        with pytest.raises(ValueError, match="does not match"):
            qualify_hard_rule(
                store,
                parent_rule_id=parent.rule_id,
                expected_git_sha=GIT_SHA,
                sc2_patch="4.10",
                qualification_manifest_path=manifest,
                evaluation_seed_ids=(3, 4, 5),
            )
    finally:
        store.close()


def test_parent_without_revision_bound_evidence_cannot_be_restamped(
    tmp_path: Path,
) -> None:
    database = tmp_path / "playbook.sqlite3"
    manifest = tmp_path / "qualification.json"
    parent = _soft_rule().model_copy(update={"code_revision": None, "sc2_patch": None})
    _write_store(database, parent)
    _qualification_manifest(manifest, database, parent)
    store = PlaybookStore(database)
    try:
        with pytest.raises(ValueError, match="soft parent must already be bound"):
            qualify_hard_rule(
                store,
                parent_rule_id=parent.rule_id,
                expected_git_sha=GIT_SHA,
                sc2_patch="4.10",
                qualification_manifest_path=manifest,
                evaluation_seed_ids=(3, 4, 5),
            )
    finally:
        store.close()


def test_legacy_execution_guard_without_typed_retry_binding_is_ineligible(
    tmp_path: Path,
) -> None:
    database = tmp_path / "playbook.sqlite3"
    manifest = tmp_path / "qualification.json"
    parent = _soft_rule().model_copy(update={"retry_guard": None})
    _write_store(database, parent)
    _qualification_manifest(manifest, database, parent)
    store = PlaybookStore(database)
    try:
        with pytest.raises(ValueError, match="typed retry binding"):
            qualify_hard_rule(
                store,
                parent_rule_id=parent.rule_id,
                expected_git_sha=GIT_SHA,
                sc2_patch="4.10",
                qualification_manifest_path=manifest,
                evaluation_seed_ids=(3, 4, 5),
            )
    finally:
        store.close()


def test_legacy_qualification_manifest_schema_is_rejected_fail_closed(
    tmp_path: Path,
) -> None:
    database = tmp_path / "playbook.sqlite3"
    manifest = tmp_path / "qualification.json"
    parent = _soft_rule()
    _write_store(database, parent)
    _qualification_manifest(manifest, database, parent, schema_version="1.1")
    store = PlaybookStore(database)
    try:
        with pytest.raises(ValueError, match="invalid hard qualification manifest"):
            qualify_hard_rule(
                store,
                parent_rule_id=parent.rule_id,
                expected_git_sha=GIT_SHA,
                sc2_patch="4.10",
                qualification_manifest_path=manifest,
                evaluation_seed_ids=(3, 4, 5),
            )
    finally:
        store.close()


def test_current_manifest_missing_conservation_field_is_rejected_fail_closed(
    tmp_path: Path,
) -> None:
    database = tmp_path / "playbook.sqlite3"
    manifest = tmp_path / "qualification.json"
    parent = _soft_rule()
    _write_store(database, parent)
    _qualification_manifest(manifest, database, parent)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    del payload["counterfactual_invalid_application_count"]
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    store = PlaybookStore(database)
    try:
        with pytest.raises(ValueError, match="invalid hard qualification manifest"):
            qualify_hard_rule(
                store,
                parent_rule_id=parent.rule_id,
                expected_git_sha=GIT_SHA,
                sc2_patch="4.10",
                qualification_manifest_path=manifest,
                evaluation_seed_ids=(3, 4, 5),
            )
    finally:
        store.close()


def test_strategic_ab_artifact_is_hashed_and_identity_bound(tmp_path: Path) -> None:
    children: list[PlaybookRule] = []
    for index, improvement in enumerate((0.1, 0.2)):
        database = tmp_path / f"playbook-{index}.sqlite3"
        manifest = tmp_path / f"qualification-{index}.json"
        strategic_path = tmp_path / f"strategic-{index}.json"
        parent = _soft_rule().model_copy(
            update={"category": PlaybookRuleCategory.MATCHUP_STRATEGY, "retry_guard": None}
        )
        _write_store(database, parent)
        _qualification_manifest(manifest, database, parent)
        _strategic_ab_manifest(
            strategic_path,
            manifest,
            parent,
            task_score_improvement=improvement,
        )
        store = PlaybookStore(database)
        try:
            children.append(
                qualify_hard_rule(
                    store,
                    parent_rule_id=parent.rule_id,
                    expected_git_sha=GIT_SHA,
                    sc2_patch="4.10",
                    qualification_manifest_path=manifest,
                    evaluation_seed_ids=(3, 4, 5),
                    strategic_ab_path=strategic_path,
                )
            )
        finally:
            store.close()

    assert children[0].canonical_key != children[1].canonical_key
    for child in children:
        assert child.evidence["strategic_ab_evidence_sha256"] in child.evidence_hashes
        assert child.evidence["strategic_ab_manifest"]


def test_one_valid_rule_does_not_hide_stale_active_hard_rule(tmp_path: Path) -> None:
    database = tmp_path / "playbook.sqlite3"
    manifest = tmp_path / "qualification.json"
    parent = _soft_rule()
    _write_store(database, parent)
    _qualification_manifest(manifest, database, parent)
    store = PlaybookStore(database)
    try:
        child = qualify_hard_rule(
            store,
            parent_rule_id=parent.rule_id,
            expected_git_sha=GIT_SHA,
            sc2_patch="4.10",
            qualification_manifest_path=manifest,
            evaluation_seed_ids=(3, 4, 5),
        )
        stale = child.model_copy(
            update={
                "rule_id": "stale-hard",
                "canonical_key": "stale-hard",
                "code_revision": "b" * 40,
                "effect": PlaybookRuleEffect.REQUIRE,
                "retry_guard": None,
            }
        )
        store.upsert_rule(stale)
    finally:
        store.close()

    report = analyze_hard_readiness_database(
        database,
        expected_git_sha=GIT_SHA,
        sc2_patch="4.10",
        agent_race="protoss",
        opponent_race="zerg",
        map_name="Simple64",
        evaluation_seed_ids=(3, 4, 5),
    )

    assert report.reachable_blocking_hard_count == 0
    assert report.approved_hard_rule_ids == ()
    assert report.approved_blocking_rule_ids == ()
    assert report.rejected_runtime_hard_rule_ids == ("stale-hard",)
    assert report.rejected_context_applicable_blocking_hard_rule_ids == ("stale-hard",)
    assert "code_revision_mismatch" in report.rejection_reasons["stale-hard"]
    assert "effect_is_require_not_forbid" in report.rejection_reasons["stale-hard"]
    assert report.canary_runnable is False


@pytest.mark.parametrize(
    "effect",
    (
        PlaybookRuleEffect.REQUIRE,
        PlaybookRuleEffect.PREFER,
        PlaybookRuleEffect.AVOID,
    ),
)
def test_non_forbid_active_hard_rule_cannot_bypass_readiness(
    tmp_path: Path,
    effect: PlaybookRuleEffect,
) -> None:
    database = tmp_path / "playbook.sqlite3"
    fixture = create_canary_fixture(
        database,
        expected_git_sha=GIT_SHA,
        sc2_patch="4.10",
    )
    extra = fixture.model_copy(
        update={
            "rule_id": f"hard-{effect.value}",
            "canonical_key": f"hard-{effect.value}",
            "effect": effect,
            "code_revision": "b" * 40,
        }
    )
    store = PlaybookStore(database)
    try:
        store.upsert_rule(extra)
    finally:
        store.close()

    report = analyze_hard_readiness_database(
        database,
        expected_git_sha=GIT_SHA,
        sc2_patch="4.10",
        agent_race="protoss",
        opponent_race="zerg",
        map_name="Simple64",
        evaluation_seed_ids=(0,),
        allow_canary_fixture=True,
    )

    assert report.canary_runnable is False
    assert report.approved_hard_rule_ids == ()
    assert f"hard-{effect.value}" in report.rejected_runtime_hard_rule_ids
    assert (
        f"effect_is_{effect.value}_not_forbid" in report.rejection_reasons[f"hard-{effect.value}"]
    )


def test_max_hard_selection_matches_readiness_and_overflow_fails_closed(
    tmp_path: Path,
) -> None:
    database = tmp_path / "playbook.sqlite3"
    first = create_canary_fixture(
        database,
        expected_git_sha=GIT_SHA,
        sc2_patch="4.10",
    )
    second = first.model_copy(
        update={
            "rule_id": "second-valid-hard",
            "canonical_key": "second-valid-hard",
            "confidence": 0.99,
        }
    )
    store = PlaybookStore(database)
    try:
        store.upsert_rule(second)
    finally:
        store.close()

    report = analyze_hard_readiness_database(
        database,
        expected_git_sha=GIT_SHA,
        sc2_patch="4.10",
        agent_race="protoss",
        opponent_race="zerg",
        map_name="Simple64",
        evaluation_seed_ids=(0,),
        allow_canary_fixture=True,
        max_hard_rules=1,
    )

    assert report.hard_rule_limit_exceeded is True
    assert report.runtime_candidate_hard_rule_ids == (first.rule_id, second.rule_id)
    assert report.runtime_selected_hard_rule_ids == (first.rule_id,)
    assert report.approved_hard_rule_ids == ()
    assert set(report.rejected_runtime_hard_rule_ids) == {first.rule_id, second.rule_id}
    assert report.canary_runnable is False


def test_invalid_static_operator_rejects_readiness(tmp_path: Path) -> None:
    database = tmp_path / "fixture.sqlite3"
    fixture = create_canary_fixture(
        database,
        expected_git_sha=GIT_SHA,
        sc2_patch="4.10",
    )
    store = PlaybookStore(database)
    try:
        invalid = fixture.model_copy(
            update={
                "conditions": (
                    PlaybookCondition(
                        field="map_name",
                        operator=PlaybookConditionOperator.GTE,
                        value="Simple64",
                    ),
                )
            }
        )
        store.upsert_rule(invalid)
    finally:
        store.close()

    report = analyze_hard_readiness_database(
        database,
        expected_git_sha=GIT_SHA,
        sc2_patch="4.10",
        agent_race="protoss",
        opponent_race="zerg",
        map_name="Simple64",
        allow_canary_fixture=True,
    )

    assert report.canary_runnable is False
    assert any(
        reason.startswith("unsupported_static_operator:map_name:gte")
        for reason in report.rejection_reasons[fixture.rule_id]
    )
