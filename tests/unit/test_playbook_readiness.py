import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from rtscortex.cli.app import app
from rtscortex.playbook import (
    PlaybookCondition,
    PlaybookHardReadinessReport,
    PlaybookRule,
    PlaybookRuleCategory,
    PlaybookRuleEffect,
    PlaybookRuleStatus,
    PlaybookRuleStrength,
    PlaybookStore,
    analyze_hard_readiness_database,
    create_canary_fixture,
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
    )


def _write_store(path: Path, rule: PlaybookRule) -> None:
    store = PlaybookStore(path)
    try:
        store.upsert_rule(rule)
    finally:
        store.close()


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
    assert audit.code_revision is None


def test_qualification_creates_disjoint_hard_child_without_mutating_parent(
    tmp_path: Path,
) -> None:
    database = tmp_path / "playbook.sqlite3"
    evidence = tmp_path / "evidence.json"
    evidence.write_text('{"accepted": true}\n', encoding="utf-8")
    _write_store(database, _soft_rule())
    store = PlaybookStore(database)
    try:
        child = qualify_hard_rule(
            store,
            parent_rule_id="soft-rule",
            expected_git_sha=GIT_SHA,
            sc2_patch="4.10",
            evidence_paths=(evidence,),
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
    assert report.canary_runnable is True


def test_qualification_rejects_reused_evaluation_seed(tmp_path: Path) -> None:
    database = tmp_path / "playbook.sqlite3"
    evidence = tmp_path / "evidence.json"
    evidence.write_text("{}\n", encoding="utf-8")
    _write_store(database, _soft_rule())
    store = PlaybookStore(database)
    try:
        try:
            qualify_hard_rule(
                store,
                parent_rule_id="soft-rule",
                expected_git_sha=GIT_SHA,
                sc2_patch="4.10",
                evidence_paths=(evidence,),
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
    evidence = tmp_path / "evidence.json"
    evidence.write_text("{}\n", encoding="utf-8")
    strategic = _soft_rule().model_copy(update={"category": PlaybookRuleCategory.MATCHUP_STRATEGY})
    _write_store(database, strategic)
    store = PlaybookStore(database)
    try:
        with pytest.raises(ValueError, match="paired outcome evidence"):
            qualify_hard_rule(
                store,
                parent_rule_id="soft-rule",
                expected_git_sha=GIT_SHA,
                sc2_patch="4.10",
                evidence_paths=(evidence,),
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
