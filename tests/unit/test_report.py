from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest
from typer.testing import CliRunner

import rtscortex.cli.app as cli_module
from rtscortex.contracts import (
    ActionBatch,
    ActionCommand,
    ActionSource,
    EffectEvidence,
    EpisodeOutcome,
    EpisodeResult,
    EpisodeSummary,
    ExecutionReport,
    ExecutionStage,
    ExecutionStatus,
)
from rtscortex.evaluation.report import (
    AcceptanceGate,
    ReportError,
    _hard_acceptance_summary,
    render_timeline,
    write_run_reports,
    write_timeline_report,
)
from rtscortex.memory import StoredEvent, read_event_log
from tests.helpers import make_observation


def _event(
    event_id: int,
    event_type: str,
    payload: dict[str, object],
    *,
    step_id: int = 0,
) -> StoredEvent:
    return StoredEvent(
        event_id=event_id,
        run_id="live-run",
        episode_id="episode-0",
        step_id=step_id,
        event_type=event_type,
        created_at=f"2026-01-01T00:00:{event_id:02d}+00:00",
        payload=payload,
    )


def _write_journal(run_dir: Path, events: list[StoredEvent]) -> None:
    run_dir.mkdir()
    (run_dir / "events.jsonl").write_text(
        "".join(json.dumps(asdict(event), sort_keys=True) + "\n" for event in events),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _qualification_evidence(
    tmp_path: Path,
    *,
    expected_git_sha: str,
    diagnostic_only: bool = False,
) -> Path:
    recovery = tmp_path / "recovery-canary.json"
    recovery.write_text(
        json.dumps(
            {
                "format_version": "1.1",
                "git_sha": expected_git_sha,
                "expected_git_sha": expected_git_sha,
                "passed": True,
                "recovery_evidence_present": True,
                "checkpoint_tail_recovery_bounded": True,
            }
        ),
        encoding="utf-8",
    )
    baseline = tmp_path / "engineering-baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "format_version": "1.0",
                "source_issue": "SCX-PT-034",
                "natural_run_bytes_per_game_loop": 10_000.0,
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "qualification-evidence.json"
    manifest.write_text(
        json.dumps(
            {
                "format_version": "1.0",
                "evidence_kind": "three-seed-qualification",
                "diagnostic_only": diagnostic_only,
                "expected_git_sha": expected_git_sha,
                "recovery_evidence": {
                    "path": str(recovery),
                    "sha256": _sha256(recovery),
                },
                "natural_run_baseline": {
                    "path": str(baseline),
                    "sha256": _sha256(baseline),
                },
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_qualification_evidence_reaches_engineering_report_call_path(
    tmp_path: Path,
) -> None:
    expected_git_sha = "a" * 40
    run_dir = tmp_path / "qualified"
    _write_journal(
        run_dir,
        [
            _event(
                1,
                "observation",
                make_observation(
                    run_id="live-run",
                    episode_id="episode-0",
                    game_loop=100,
                ).model_dump(mode="json"),
            ),
            _event(
                2,
                "postgame_review_completed",
                {"strategic_consequence_count": 0},
            ),
        ],
    )
    manifest = _qualification_evidence(
        tmp_path,
        expected_git_sha=expected_git_sha,
    )

    artifacts = write_run_reports(
        run_dir,
        qualification_evidence_path=manifest,
    )
    report = json.loads(artifacts.engineering_gates_path.read_text(encoding="utf-8"))

    assert report["metrics"]["recovery_evidence_present"] is True
    assert report["metrics"]["checkpoint_tail_recovery_bounded"] is True
    assert report["metrics"]["postgame_semantic_event_coverage"] == 1.0
    assert report["metrics"]["natural_run_disk_reduction_ratio"] is not None
    assert report["evidence"]["expected_git_sha"] == expected_git_sha
    assert report["evidence"]["recovery_evidence"]["sha256"] == _sha256(
        tmp_path / "recovery-canary.json"
    )
    assert report["evidence"]["natural_run_baseline"] == {
        "path": str((tmp_path / "engineering-baseline.json").resolve()),
        "sha256": _sha256(tmp_path / "engineering-baseline.json"),
        "source_issue": "SCX-PT-034",
        "bytes_per_game_loop": 10_000.0,
    }


def test_semantic_build_gate_is_serialized_by_run_report_writer(tmp_path: Path) -> None:
    expected_git_sha = "d" * 40
    run_dir = tmp_path / "semantic-build"
    operation_id = "operation:" + "a" * 64
    attempt_id = "attempt:" + "b" * 64
    _write_journal(
        run_dir,
        [
            _event(
                1,
                "command_lifecycle",
                {
                    "status": "dispatched",
                    "command": {
                        "command_id": "build-1",
                        "operation_id": operation_id,
                        "attempt_id": attempt_id,
                        "actor": "Builder/Builder-Probe-1",
                        "name": "Build_Pylon_Screen",
                        "arguments": [[20, 20]],
                        "created_game_loop": 0,
                        "source": "planner",
                    },
                },
            ),
            _event(
                2,
                "execution",
                {
                    "protocol_version": "1.1",
                    "run_id": "live-run",
                    "episode_id": "episode-0",
                    "step_id": 0,
                    "command_id": "build-1",
                    "operation_id": operation_id,
                    "attempt_id": attempt_id,
                    "action_name": "Build_Pylon_Screen",
                    "actor": "Builder/Builder-Probe-1",
                    "source": "planner",
                    "requested_arguments": [[20, 20]],
                    "resolved_arguments": [[20, 20]],
                    "success": False,
                    "status": "failed",
                    "execution_stage": "effect_verification",
                    "failure_code": "no_build_start_evidence",
                    "effect_evidence": {
                        "effect_kind": "build",
                        "failure_classification": "gameplay_no_start_unknown",
                        "classification_basis": ["no_authoritative_rejection_evidence"],
                    },
                },
            ),
        ],
    )
    manifest = _qualification_evidence(
        tmp_path,
        expected_git_sha=expected_git_sha,
    )

    artifacts = write_run_reports(
        run_dir,
        qualification_evidence_path=manifest,
    )
    report = json.loads(artifacts.engineering_gates_path.read_text(encoding="utf-8"))

    assert report["diagnostics"]["semantic_build_operation_max_failure_streak"] == 1
    assert report["diagnostics"]["semantic_build_failure_classification"] == {
        "gameplay_no_start_unknown": 1
    }
    assert report["gates"]["semantic_build_failure_streak_bounded"]["passed"] is True


def test_diagnostic_manifest_cannot_populate_formal_engineering_evidence(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "diagnostic"
    _write_journal(
        run_dir,
        [
            _event(
                1,
                "observation",
                make_observation(
                    run_id="live-run",
                    episode_id="episode-0",
                    game_loop=100,
                ).model_dump(mode="json"),
            )
        ],
    )
    manifest = _qualification_evidence(
        tmp_path,
        expected_git_sha="b" * 40,
        diagnostic_only=True,
    )

    artifacts = write_run_reports(run_dir, qualification_evidence_path=manifest)
    report = json.loads(artifacts.engineering_gates_path.read_text(encoding="utf-8"))

    assert report["metrics"]["recovery_evidence_present"] is False
    assert report["metrics"]["checkpoint_tail_recovery_bounded"] is None
    assert report["metrics"]["natural_run_disk_reduction_ratio"] is None
    assert report["accepted"] is False


def test_missing_recovery_claims_remain_fail_closed(tmp_path: Path) -> None:
    run_dir = tmp_path / "missing-recovery-claims"
    _write_journal(
        run_dir,
        [
            _event(
                1,
                "observation",
                make_observation(
                    run_id="live-run",
                    episode_id="episode-0",
                    game_loop=100,
                ).model_dump(mode="json"),
            )
        ],
    )
    manifest_path = _qualification_evidence(tmp_path, expected_git_sha="c" * 40)
    recovery_path = tmp_path / "recovery-canary.json"
    recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
    del recovery["recovery_evidence_present"]
    del recovery["checkpoint_tail_recovery_bounded"]
    recovery_path.write_text(json.dumps(recovery), encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["recovery_evidence"]["sha256"] = _sha256(recovery_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    artifacts = write_run_reports(
        run_dir,
        qualification_evidence_path=manifest_path,
    )
    report = json.loads(artifacts.engineering_gates_path.read_text(encoding="utf-8"))

    assert report["metrics"]["recovery_evidence_present"] is False
    assert report["metrics"]["checkpoint_tail_recovery_bounded"] is None
    assert report["accepted"] is False


def test_hard_acceptance_allows_not_applicable_gates() -> None:
    gates = (
        AcceptanceGate(
            name="meaningful_command_success_rate",
            value=1.0,
            comparison=">=",
            threshold=0.7,
            unit="ratio",
            passed=True,
        ),
        AcceptanceGate(
            name="production_effect_confirmed_rate",
            value=0.0,
            comparison=">=",
            threshold=0.9,
            unit="ratio",
            passed=None,
        ),
    )

    summary = _hard_acceptance_summary(gates, complete=True)

    assert summary["passed"] is True
    assert summary["passed_gates"] == 1
    assert summary["failed_gates"] == 0
    assert summary["not_applicable_gates"] == 1


def test_timeline_renders_live_state_reasoning_actions_and_execution(tmp_path: Path) -> None:
    observation = make_observation(
        run_id="live-run",
        episode_id="episode-0",
        step_id=0,
        game_loop=58,
        health=0.42,
    ).model_copy(update={"text_observation": "RAW OBSERVATION SHOULD STAY IN JSONL"})
    command = ActionCommand(
        command_id="command-1",
        actor="CombatGroupSmac/Stalker-1",
        name="Attack_Unit",
        arguments=["enemy-1"],
        priority=60,
        ttl_game_loops=32,
        created_game_loop=58,
        source=ActionSource.PLANNER,
    )
    batch = ActionBatch(
        run_id="live-run",
        episode_id="episode-0",
        step_id=0,
        decision_id="decision-0",
        strategic_goal="Focus the nearest enemy",
        summary="Keep the Stalker at range",
        commands=[command],
        rejected_commands=["command-old: actor is outside the action scope"],
    )
    result = EpisodeResult(
        run_id="live-run",
        episode_id="episode-0",
        scenario="2s3z",
        seed=7,
        outcome=EpisodeOutcome.VICTORY,
        score=1.0,
        steps=481,
    )
    summary = EpisodeSummary(
        run_id="live-run",
        episode_id="episode-0",
        scenario="2s3z",
        seed=7,
        outcome=EpisodeOutcome.VICTORY,
        summary="The focused target was removed.",
        lessons=["Preserve ranged units."],
        source_step_id=481,
    )
    events = [
        _event(1, "observation", observation.model_dump(mode="json")),
        _event(
            2,
            "plan_accepted",
            {
                "strategic_goal": "Focus the nearest enemy",
                "summary": "Keep the Stalker at range",
                "commands": [command.model_dump(mode="json")],
                "source_step_id": 0,
                "created_game_loop": 58,
                "source_game_loop": 50,
                "accepted_game_loop": 58,
                "plan_age_game_loops": 8,
                "fingerprint": "abc",
                "is_revision": True,
            },
        ),
        _event(
            3,
            "module_result",
            {
                "module": "reflection",
                "latency_ms": 12.5,
                "command_count": 0,
                "model_call": True,
                "provider": "openai_compatible",
                "model": "qwen3-8b",
                "usage": {"prompt_tokens": 80, "completion_tokens": 20, "total_tokens": 100},
                "output": {
                    "reflection": "The previous focus-fire action succeeded.",
                    "lessons": ["Keep focusing one target."],
                },
            },
        ),
        _event(
            4,
            "decision",
            {
                "batch": batch.model_dump(mode="json"),
                "planner_candidates": [command.model_dump(mode="json")],
                "reflex_candidates": [],
                "preemptions": [
                    {
                        "actor": command.actor,
                        "winner_command_id": "reflex-command",
                        "loser_command_id": command.command_id,
                    }
                ],
                "reflex_latency_ms": 1.0,
                "tick_latency_ms": 15.0,
            },
        ),
        _event(
            5,
            "command_lifecycle",
            {
                "command": command.model_dump(mode="json"),
                "status": "dispatched",
            },
        ),
        _event(
            6,
            "execution",
            ExecutionReport(
                run_id="live-run",
                episode_id="episode-0",
                step_id=0,
                command_id=command.command_id,
                success=True,
                action_name=command.name,
                actor=command.actor,
                source=command.source,
                requested_arguments=command.arguments,
                resolved_arguments=command.arguments,
                status=ExecutionStatus.SUCCEEDED,
                execution_stage=ExecutionStage.PYSC2_ACCEPTANCE,
                pysc2_function="select_control_group -> Attack_screen",
                latency_ms=2.5,
            ).model_dump(mode="json"),
        ),
        _event(7, "episode_result", result.model_dump(mode="json"), step_id=481),
        _event(8, "episode_summary", summary.model_dump(mode="json"), step_id=481),
    ]
    run_dir = tmp_path / "live"
    _write_journal(run_dir, events)

    artifacts = write_run_reports(run_dir)
    report = artifacts.timeline_path.read_text(encoding="utf-8")
    summary_report = artifacts.summary_path.read_text(encoding="utf-8")
    summary_payload = json.loads(summary_report)
    episode = summary_payload["runs"]["live-run"]["episodes"]["episode-0"]

    assert artifacts.timeline_path == run_dir / "timeline.md"
    assert artifacts.summary_path == run_dir / "summary.json"
    assert artifacts.engineering_gates_path == run_dir / "engineering-gates.json"
    assert artifacts.engineering_gates_path.is_file()
    assert episode["complete"] is True
    assert episode["result"]["outcome"] == "victory"
    assert episode["result"]["scenario"] == "2s3z"
    assert episode["result"]["seed"] == 7
    assert episode["result"]["steps"] == 481
    assert episode["metrics"]["model_requests"] == 1
    assert episode["metrics"]["prompt_tokens"] == 80
    assert episode["metrics"]["completion_tokens"] == 20
    assert episode["metrics"]["total_tokens"] == 100
    assert episode["metrics"]["execution"]["status_counts"] == {"succeeded": 1}
    assert episode["metrics"]["execution"]["terminal_backlog_rate"] == 0.0
    assert episode["classification_conservation"] == {
        "reported": 1,
        "succeeded": 1,
        "failed": 0,
        "cancelled": 0,
        "unconfirmed": 0,
        "classified": 1,
        "conserved": True,
    }
    assert episode["terminal_reports"]["exactly_once"] is True
    assert episode["hard_acceptance"]["gates"]["meaningful_command_success_rate"]["passed"] is True
    assert "`openai_compatible`/`qwen3-8b`" in report
    assert "`2s3z` | `victory` | 7 | 1.00 | 481" in report
    assert "`Adept` x1 (HP 42%)" in report
    assert "Reflection: The previous focus-fire action succeeded." in report
    assert "CombatGroupSmac/Stalker-1" in report
    assert "select_control_group -> Attack_screen" in report
    assert "actor is outside the action scope" in report
    assert "Reflex preemption" in report
    assert "source loop `50`, accepted loop `58`, age `8` loops" in report
    assert "### Decision activity" in report
    assert "### Meaningful outcomes" in report
    assert "Meaningful success: `1/1` (100.0%)" in report
    assert "### Build funnel" in report
    assert "### Safety and attribution invariants" in report
    assert "Terminal backlog rate: `0.0%` (`0/1`)" in report
    assert "### Hard acceptance gates" in report
    assert "insufficient samples (requires at least two accepted plans)" in report
    assert "### Failure taxonomy" in report
    assert "Legacy execution-report rate: `1/1` (100.0%) (deprecated)" in report
    assert "RAW OBSERVATION SHOULD STAY IN JSONL" not in report
    assert report.index("Event 2 · Plan revision") < report.index("Event 4 · Decision")
    assert report.index("Event 4 · Decision") < report.index("Event 6 · Execution")

    second = write_run_reports(run_dir)
    assert second.timeline_path.read_text(encoding="utf-8") == report
    assert second.summary_path.read_text(encoding="utf-8") == summary_report


def test_timeline_renders_module_context_compaction_statistics(tmp_path: Path) -> None:
    observation = make_observation(
        run_id="live-run",
        episode_id="episode-0",
        step_id=0,
        game_loop=58,
    )
    events = [
        _event(1, "observation", observation.model_dump(mode="json")),
        _event(
            2,
            "module_result",
            {
                "module": "reflection",
                "latency_ms": 12.5,
                "command_count": 0,
                "model_call": True,
                "provider": "openai_compatible",
                "model": "qwen3-8b",
                "usage": {"prompt_tokens": 800, "completion_tokens": 20},
                "output": {
                    "reflection": "Keep the current build order.",
                    "lessons": [],
                    "context_compaction": {
                        "budget_chars": 6000,
                        "original_chars": 14620,
                        "final_chars": 5732,
                        "compacted": True,
                        "aggregated_own_units": 188,
                        "aggregated_own_structures": 84,
                        "aggregated_visible_enemies": 134,
                        "dropped_recent_events": 21,
                        "dropped_lessons": 3,
                        "dropped_episode_summaries": 2,
                        "dropped_spatial_lines": 7,
                    },
                },
            },
        ),
    ]
    run_dir = tmp_path / "compacted"
    _write_journal(run_dir, events)

    report = write_timeline_report(run_dir).read_text(encoding="utf-8")

    assert "Context budget: `6000` chars; `14620` → `5732` chars" in report
    assert "compacted: `yes`" in report
    assert "own units `188`" in report
    assert "own structures `84`" in report
    assert "visible enemies `134`" in report
    assert "recent events `21`" in report
    assert "lessons `3`" in report
    assert "episode summaries `2`" in report
    assert "spatial lines `7`" in report


def test_report_gates_raw_planner_attack_proposals_and_dispatched_safety(
    tmp_path: Path,
) -> None:
    observation = make_observation(
        run_id="live-run",
        episode_id="episode-0",
        step_id=0,
        game_loop=58,
    )
    friendly_attack = ActionCommand(
        command_id="live-run:episode-0:0:planner:1",
        actor="CombatGroup/Zealot-1",
        name="Attack_Unit",
        arguments=["unit-1"],
        priority=50,
        ttl_game_loops=112,
        created_game_loop=58,
        source=ActionSource.PLANNER,
    )
    result = EpisodeResult(
        run_id="live-run",
        episode_id="episode-0",
        scenario="Simple64",
        seed=0,
        outcome=EpisodeOutcome.TRUNCATED,
        steps=1,
    )
    events = [
        _event(1, "observation", observation.model_dump(mode="json")),
        _event(
            2,
            "module_result",
            {
                "module": "planning",
                "model_call": True,
                "output": {
                    "plan": {
                        "strategic_goal": "Reject unsafe attacks",
                        "proposed_actions": [
                            {
                                "actor": "Builder/Probe-1",
                                "name": "Attack_Unit",
                                "arguments": ["0x1"],
                            },
                            {
                                "actor": "CombatGroup/Zealot-1",
                                "name": "Attack_Unit",
                                "arguments": ["unit-1"],
                            },
                        ],
                    }
                },
            },
        ),
        _event(
            3,
            "command_lifecycle",
            {
                "command": friendly_attack.model_dump(mode="json"),
                "status": "dispatched",
            },
        ),
        _event(4, "episode_result", result.model_dump(mode="json")),
    ]
    run_dir = tmp_path / "unsafe-proposals"
    _write_journal(run_dir, events)

    artifacts = write_run_reports(run_dir)
    report = artifacts.timeline_path.read_text(encoding="utf-8")
    summary = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))
    episode = summary["runs"]["live-run"]["episodes"]["episode-0"]
    execution = episode["metrics"]["execution"]
    gates = episode["hard_acceptance"]["gates"]

    assert execution["planner_unsafe_attack_proposals"] == 2
    assert execution["planner_unsafe_attack_rejected_before_dispatch"] == 1
    assert execution["planner_unsafe_attack_dispatched"] == 1
    assert gates["planner_builder_attack_proposals"]["passed"] is False
    assert gates["planner_friendly_target_attack_proposals"]["passed"] is False
    assert gates["friendly_target_attacks"]["passed"] is False
    assert "Planner unsafe Attack proposals: `2`" in report
    assert "rejected before dispatch `1`; dispatched `1`" in report


def test_report_exactly_once_gate_rejects_unexpected_terminal_report(
    tmp_path: Path,
) -> None:
    observation = make_observation(
        run_id="live-run",
        episode_id="episode-0",
        step_id=0,
        game_loop=58,
    )
    command = ActionCommand(
        command_id="known",
        actor="CombatGroup/Zealot-1",
        name="Attack_Unit",
        arguments=["0x1"],
        priority=50,
        ttl_game_loops=112,
        created_game_loop=58,
        source=ActionSource.PLANNER,
    )

    def report(command_id: str) -> ExecutionReport:
        return ExecutionReport(
            run_id="live-run",
            episode_id="episode-0",
            step_id=0,
            command_id=command_id,
            success=False,
            action_name=command.name,
            actor=command.actor,
            source=command.source,
            requested_arguments=command.arguments,
            status=ExecutionStatus.FAILED,
            execution_stage=ExecutionStage.PYSC2_ACCEPTANCE,
            failure_code="pysc2_rejected",
        )

    result = EpisodeResult(
        run_id="live-run",
        episode_id="episode-0",
        scenario="Simple64",
        seed=0,
        outcome=EpisodeOutcome.TRUNCATED,
        steps=1,
    )
    events = [
        _event(1, "observation", observation.model_dump(mode="json")),
        _event(
            2,
            "command_lifecycle",
            {"command": command.model_dump(mode="json"), "status": "dispatched"},
        ),
        _event(3, "execution", report("known").model_dump(mode="json")),
        _event(4, "execution", report("rogue").model_dump(mode="json")),
        _event(5, "episode_result", result.model_dump(mode="json")),
    ]
    run_dir = tmp_path / "unexpected-terminal"
    _write_journal(run_dir, events)

    artifacts = write_run_reports(run_dir)
    summary = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))
    episode = summary["runs"]["live-run"]["episodes"]["episode-0"]

    assert episode["terminal_reports"]["unexpected_reports"] == 1
    assert episode["terminal_reports"]["exactly_once"] is False
    assert episode["hard_acceptance"]["gates"]["unexpected_terminal_reports"]["passed"] is False


def test_report_exposes_production_funnel_provenance_and_acceptance_only_gate(
    tmp_path: Path,
) -> None:
    confirmed = ActionCommand(
        command_id="train-confirmed",
        actor="Gateway/0x1",
        name="Train_Adept",
        source=ActionSource.PLANNER,
        created_game_loop=100,
    )
    acceptance_only = ActionCommand(
        command_id="train-acceptance-only",
        actor="Gateway/0x2",
        name="Train_Zealot",
        source=ActionSource.PLANNER,
        created_game_loop=100,
    )
    batch = ActionBatch(
        run_id="live-run",
        episode_id="episode-0",
        step_id=0,
        decision_id="decision-production",
        commands=[confirmed, acceptance_only],
    )
    result = EpisodeResult(
        run_id="live-run",
        episode_id="episode-0",
        scenario="Simple64",
        seed=0,
        outcome=EpisodeOutcome.TRUNCATED,
        steps=1,
    )
    confirmed_report = ExecutionReport(
        run_id="live-run",
        episode_id="episode-0",
        step_id=0,
        command_id=confirmed.command_id,
        success=True,
        action_name=confirmed.name,
        actor=confirmed.actor,
        source=confirmed.source,
        status=ExecutionStatus.SUCCEEDED,
        execution_stage=ExecutionStage.EFFECT_VERIFICATION,
        effect_evidence=EffectEvidence(
            effect_kind="production",
            producer_tag="0x1",
            producer_type="Gateway",
            expected_unit_type="Adept",
            expected_order_id=54,
            baseline_producer_orders=[],
            producer_orders=[54],
            production_order_seen=True,
            confirmation_kind="producer_order",
            accepted_game_loop=100,
            confirmed_game_loop=108,
        ),
    )
    acceptance_report = ExecutionReport(
        run_id="live-run",
        episode_id="episode-0",
        step_id=0,
        command_id=acceptance_only.command_id,
        success=True,
        action_name=acceptance_only.name,
        actor=acceptance_only.actor,
        source=acceptance_only.source,
        status=ExecutionStatus.SUCCEEDED,
        execution_stage=ExecutionStage.PYSC2_ACCEPTANCE,
    )
    events = [
        _event(
            1,
            "decision",
            {
                "planner_candidates": [
                    confirmed.model_dump(mode="json"),
                    acceptance_only.model_dump(mode="json"),
                ],
                "validated_candidates": [
                    confirmed.model_dump(mode="json"),
                    acceptance_only.model_dump(mode="json"),
                ],
                "batch": batch.model_dump(mode="json"),
            },
        ),
        _event(
            2,
            "command_lifecycle",
            {"command": confirmed.model_dump(mode="json"), "status": "dispatched"},
        ),
        _event(
            3,
            "command_lifecycle",
            {"command": acceptance_only.model_dump(mode="json"), "status": "dispatched"},
        ),
        _event(4, "execution", confirmed_report.model_dump(mode="json")),
        _event(5, "execution", acceptance_report.model_dump(mode="json")),
        _event(6, "episode_result", result.model_dump(mode="json")),
    ]
    run_dir = tmp_path / "production-funnel"
    _write_journal(run_dir, events)

    artifacts = write_run_reports(run_dir)
    report = artifacts.timeline_path.read_text(encoding="utf-8")
    summary = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))
    episode = summary["runs"]["live-run"]["episodes"]["episode-0"]
    execution = episode["metrics"]["execution"]
    gates = episode["hard_acceptance"]["gates"]

    assert execution["production_funnel"]["pysc2_accepted"] == 2
    assert execution["production_funnel"]["effect_confirmed"] == 1
    assert execution["production_funnel"]["acceptance_only"] == 1
    assert execution["production_provenance_coverage"] == 0.5
    assert execution["production_by_action"] == {"Train_Adept": 1, "Train_Zealot": 1}
    assert execution["production_by_producer"] == {
        "Gateway / 0x1": 1,
        "Gateway/0x2": 1,
    }
    assert execution["confirmation_latency_game_loops_p50"] == 8.0
    assert gates["production_acceptance_only"]["passed"] is False
    assert gates["production_provenance_coverage"]["passed"] is False
    assert gates["production_effect_confirmed_rate"]["passed"] is False
    assert gates["production_timeout_rate"]["passed"] is True
    assert "### Production funnel" in report
    assert "Production confirmed by order on `Gateway` `0x1` for `Adept`" in report
    assert "Production acceptance only (deprecated)" in report
    assert "#### Production by action" in report
    assert "#### Production by producer" in report


def test_cortex_integrity_violations_fail_hard_acceptance(tmp_path: Path) -> None:
    result = EpisodeResult(
        run_id="live-run",
        episode_id="episode-0",
        scenario="mock",
        seed=0,
        outcome=EpisodeOutcome.DRAW,
        score=0.0,
        steps=1,
    )
    events = [
        _event(
            1,
            "candidate_set_built",
            {
                "intent_id": "intent-a",
                "candidates": [{"candidate_id": "candidate-a"}],
            },
        ),
        _event(
            2,
            "executor_selection",
            {
                "intent_id": "intent-b",
                "selection_id": "selection-b",
                "candidate_id": "candidate-a",
                "executor_id": "broken-executor",
            },
        ),
        _event(
            3,
            "command_lineage",
            {
                "command_id": "command-b",
                "intent_id": "intent-b",
                "selection_id": "selection-b",
                "candidate_id": "candidate-a",
            },
        ),
        _event(4, "command_lifecycle", {"command_id": "command-b", "status": "dispatched"}),
        _event(5, "episode_result", result.model_dump(mode="json")),
    ]
    run_dir = tmp_path / "cortex-integrity"
    _write_journal(run_dir, events)

    artifacts = write_run_reports(run_dir)
    summary = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))
    episode = summary["runs"]["live-run"]["episodes"]["episode-0"]
    gates = episode["hard_acceptance"]["gates"]

    assert gates["cortex_executor_candidate_violations"]["passed"] is False
    assert gates["cortex_lineage_integrity_violations"]["passed"] is False
    assert gates["cortex_missing_command_lineage"]["passed"] is False
    assert episode["hard_acceptance"]["passed"] is False


def test_real_legacy_full_match_report_freezes_baseline_lines() -> None:
    fixture = Path(__file__).parents[1] / "fixtures" / "legacy_full_match_characterization.jsonl"

    report = render_timeline(list(read_event_log(fixture)))

    assert "| 946 | 754 | 462 | 39 |" in report
    assert "Tracked control NoOps: `782` (`729` succeeded)" in report
    assert "Meaningful commands: `121` — `30` succeeded, `73` failed" in report
    assert "Meaningful success: `30/121` (24.8%)" in report
    assert "Completed execution success: `30/103` (29.1%)" in report
    assert "Terminal cancelled: `71`" in report
    assert "Legacy execution-report rate: `759/903` (84.1%) (deprecated)" in report


def test_timeline_renders_strategic_consequence_and_aggregate() -> None:
    report = render_timeline(
        [
            _event(
                1,
                "strategic_consequence_attributed",
                {
                    "consequence_type": "advantage_not_converted",
                    "role": "offense",
                    "semantic_action": None,
                    "start_game_loop": 8_000,
                    "end_game_loop": 8_896,
                    "explanation": (
                        "A verified army advantage persisted without offensive execution"
                    ),
                },
                step_id=100,
            )
        ]
    )

    assert "Strategic consequence `advantage_not_converted`" in report
    assert "loops `8000–8896`" in report
    assert "Strategic consequences attributed from completed matches: `1`" in report
    assert "#### Strategic consequences" in report
    assert "| `advantage_not_converted` | 1 |" in report


def test_legacy_incomplete_mock_journal_still_writes_report_and_cli_succeeds(
    tmp_path: Path,
) -> None:
    observation = make_observation(
        run_id="live-run",
        episode_id="episode-0",
        step_id=0,
        game_loop=0,
        include_enemy=False,
    )
    command = ActionCommand(
        command_id="fallback-0",
        actor="global",
        name="No_Operation",
        created_game_loop=0,
        source=ActionSource.FALLBACK,
    )
    batch = ActionBatch(
        protocol_version="1.0",
        run_id="live-run",
        episode_id="episode-0",
        step_id=0,
        decision_id="decision-0",
        commands=[command],
    )
    events = [
        _event(1, "observation", observation.model_dump(mode="json")),
        _event(
            2,
            "module_result",
            {
                "module": "planning",
                "latency_ms": 3.0,
                "command_count": 0,
                "model_call": True,
                "provider": "fake",
                "model": "fake-planner-v1",
                "usage": None,
            },
        ),
        _event(
            3,
            "decision",
            {
                "batch": batch.model_dump(mode="json"),
                "preemptions": [],
                "reflex_latency_ms": 0.1,
                "tick_latency_ms": 0.5,
            },
        ),
        _event(
            4,
            "execution",
            ExecutionReport(
                protocol_version="1.0",
                run_id="live-run",
                episode_id="episode-0",
                step_id=0,
                command_id=command.command_id,
                success=True,
                pysc2_function="mock.No_Operation",
            ).model_dump(mode="json"),
        ),
    ]
    run_dir = tmp_path / "mock"
    _write_journal(run_dir, events)

    result = CliRunner().invoke(cli_module.app, ["report", str(run_dir)])
    report = (run_dir / "timeline.md").read_text(encoding="utf-8")
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    episode = summary["runs"]["live-run"]["episodes"]["episode-0"]

    assert result.exit_code == 0, result.output
    assert f"Timeline: {run_dir / 'timeline.md'}" in result.output
    assert f"Summary: {run_dir / 'summary.json'}" in result.output
    assert episode["complete"] is False
    assert episode["result"] is None
    assert episode["metrics"]["execution"]["control_noops"] == 1
    assert episode["metrics"]["execution"]["meaningful_commands"] == 0
    assert episode["hard_acceptance"]["passed"] is False
    assert episode["hard_acceptance"]["gates"]["production_acceptance_only"]["passed"] is None
    assert "`incomplete`" in report
    assert "No terminal episode result was recorded" in report
    assert "Structured output not recorded (legacy event)." in report
    assert "mock.No_Operation" in report
    assert "`No_Operation` by `global` from `fallback`" in report
    assert "Tracked control NoOps: `1` (`1` succeeded)" in report
    assert "Meaningful commands: `0`" in report


def test_report_reads_legacy_planning_ttl(tmp_path: Path) -> None:
    observation = make_observation(
        run_id="live-run",
        episode_id="episode-0",
        step_id=0,
        game_loop=0,
    )
    events = [
        _event(1, "observation", observation.model_dump(mode="json")),
        _event(
            2,
            "module_result",
            {
                "module": "planning",
                "latency_ms": 3.0,
                "model_call": True,
                "provider": "fake",
                "model": "fake-planner-v1",
                "usage": None,
                "output": {
                    "plan": {
                        "strategic_goal": "Hold the ramp",
                        "steps": ["Target the visible enemy"],
                        "proposed_actions": [
                            {
                                "actor": "CombatGroup/Zealot-1",
                                "name": "Attack_Unit",
                                "arguments": ["0xabc"],
                                "priority": 60,
                                "ttl_game_loops": 32,
                            }
                        ],
                    }
                },
            },
        ),
    ]
    run_dir = tmp_path / "legacy-planning"
    _write_journal(run_dir, events)

    report = write_timeline_report(run_dir).read_text(encoding="utf-8")

    assert "Proposed goal: **Hold the ramp**" in report
    assert "`Attack_Unit`" in report
    assert "TTL" not in report


def test_report_rejects_missing_empty_and_malformed_journals(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    missing.mkdir()
    with pytest.raises(ReportError, match="does not contain events.jsonl"):
        write_timeline_report(missing)
    with pytest.raises(ReportError, match="does not contain events.jsonl"):
        write_run_reports(missing)

    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "events.jsonl").touch()
    with pytest.raises(ReportError, match="Event journal is empty"):
        write_timeline_report(empty)
    with pytest.raises(ReportError, match="Event journal is empty"):
        write_run_reports(empty)

    malformed = tmp_path / "malformed"
    malformed.mkdir()
    (malformed / "events.jsonl").write_text("{broken\n", encoding="utf-8")
    with pytest.raises(ReportError, match="Invalid event journal"):
        write_timeline_report(malformed)
    with pytest.raises(ReportError, match="Invalid event journal"):
        write_run_reports(malformed)

    result = CliRunner().invoke(cli_module.app, ["report", str(missing)])
    assert result.exit_code == 2
    assert "does not contain events.jsonl" in result.output
    assert not (missing / "timeline.md").exists()
    assert not (missing / "summary.json").exists()
    assert not (empty / "summary.json").exists()
    assert not (malformed / "summary.json").exists()


def test_best_effort_report_failure_does_not_escape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_dir = tmp_path / "run"
    observation = make_observation(run_id="live-run", episode_id="episode-0")
    _write_journal(run_dir, [_event(1, "observation", observation.model_dump(mode="json"))])

    def fail_report(_: Path) -> None:
        raise RuntimeError("derived artifact write failed")

    monkeypatch.setattr(cli_module, "write_run_reports", fail_report)

    cli_module._write_run_reports_best_effort(run_dir)

    assert (
        "Warning: could not generate run reports: derived artifact write failed"
        in capsys.readouterr().err
    )
