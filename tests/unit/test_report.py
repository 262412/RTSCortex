from __future__ import annotations

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
    EpisodeOutcome,
    EpisodeResult,
    EpisodeSummary,
    ExecutionReport,
)
from rtscortex.evaluation.report import ReportError, write_timeline_report
from rtscortex.memory import StoredEvent
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
            "execution",
            ExecutionReport(
                run_id="live-run",
                episode_id="episode-0",
                step_id=0,
                command_id=command.command_id,
                success=True,
                pysc2_function="select_control_group -> Attack_screen",
                latency_ms=2.5,
            ).model_dump(mode="json"),
        ),
        _event(6, "episode_result", result.model_dump(mode="json"), step_id=481),
        _event(7, "episode_summary", summary.model_dump(mode="json"), step_id=481),
    ]
    run_dir = tmp_path / "live"
    _write_journal(run_dir, events)

    output_path = write_timeline_report(run_dir)
    report = output_path.read_text(encoding="utf-8")

    assert output_path == run_dir / "timeline.md"
    assert "`openai_compatible`/`qwen3-8b`" in report
    assert "`2s3z` | `victory` | 7 | 1.00 | 481" in report
    assert "`Adept` x1 (HP 42%)" in report
    assert "Reflection: The previous focus-fire action succeeded." in report
    assert "CombatGroupSmac/Stalker-1" in report
    assert "select_control_group -> Attack_screen" in report
    assert "actor is outside the action scope" in report
    assert "Reflex preemption" in report
    assert "source loop `50`, accepted loop `58`, age `8` loops" in report
    assert "RAW OBSERVATION SHOULD STAY IN JSONL" not in report
    assert report.index("Event 2 · Plan revision") < report.index("Event 4 · Decision")
    assert report.index("Event 4 · Decision") < report.index("Event 5 · Execution")


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

    assert result.exit_code == 0, result.output
    assert f"Timeline: {run_dir / 'timeline.md'}" in result.output
    assert "`incomplete`" in report
    assert "No terminal episode result was recorded" in report
    assert "Structured output not recorded (legacy event)." in report
    assert "mock.No_Operation" in report


def test_report_rejects_missing_empty_and_malformed_journals(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    missing.mkdir()
    with pytest.raises(ReportError, match="does not contain events.jsonl"):
        write_timeline_report(missing)

    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "events.jsonl").touch()
    with pytest.raises(ReportError, match="Event journal is empty"):
        write_timeline_report(empty)

    malformed = tmp_path / "malformed"
    malformed.mkdir()
    (malformed / "events.jsonl").write_text("{broken\n", encoding="utf-8")
    with pytest.raises(ReportError, match="Invalid event journal"):
        write_timeline_report(malformed)

    result = CliRunner().invoke(cli_module.app, ["report", str(missing)])
    assert result.exit_code == 2
    assert "does not contain events.jsonl" in result.output
    assert not (missing / "timeline.md").exists()
