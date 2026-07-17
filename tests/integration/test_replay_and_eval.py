from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from rtscortex.evaluation import run_mock_episode, run_mock_suite
from rtscortex.evaluation.replay import replay_event_log
from rtscortex.runtime.factory import build_runtime
from tests.helpers import make_config, make_observation


def _write_journal(path: Path, records: list[tuple[str, dict[str, object]]]) -> None:
    encoded: list[str] = []
    for event_id, (event_type, payload) in enumerate(records, start=1):
        encoded.append(
            json.dumps(
                {
                    "event_id": event_id,
                    "run_id": "legacy-run",
                    "episode_id": "episode-0",
                    "step_id": 0,
                    "event_type": event_type,
                    "created_at": "2026-07-14T00:00:00+00:00",
                    "payload": payload,
                },
                sort_keys=True,
            )
            + "\n"
        )
    path.write_text("".join(encoded), encoding="utf-8")


def test_event_log_replays_to_identical_decisions(tmp_path: Path) -> None:
    config = make_config(tmp_path)

    async def execute() -> None:
        original_dir = tmp_path / "original"
        runtime = build_runtime(config, original_dir)
        try:
            await run_mock_episode(
                config=config,
                runtime=runtime,
                run_id="replay-run",
                episode_id="episode-0",
                seed=0,
            )
        finally:
            await runtime.close()
        replay = await replay_event_log(
            original_dir / "events.jsonl",
            config=config,
            output_dir=tmp_path / "replayed",
        )
        assert replay.observations == replay.matched_decisions
        assert replay.mismatched_steps == []
        assert replay.backfilled_legacy_executions == 0
        assert replay.skipped_legacy_executions == 0

    asyncio.run(execute())


def test_v1_0_execution_is_backfilled_from_matching_dispatched_command(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)

    async def execute() -> None:
        original_dir = tmp_path / "legacy-original"
        runtime = build_runtime(config, original_dir)
        try:
            await run_mock_episode(
                config=config,
                runtime=runtime,
                run_id="legacy-replay-run",
                episode_id="episode-0",
                seed=0,
            )
        finally:
            await runtime.close()

        journal = original_dir / "events.jsonl"
        records = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
        for record in records:
            payload = record["payload"]
            if record["event_type"] == "observation":
                payload["protocol_version"] = "1.0"
            batch = payload.get("batch")
            if isinstance(batch, dict):
                batch["protocol_version"] = "1.0"
            if record["event_type"] == "execution":
                record["payload"] = {
                    key: value
                    for key, value in payload.items()
                    if key
                    in {
                        "protocol_version",
                        "run_id",
                        "episode_id",
                        "step_id",
                        "command_id",
                        "success",
                        "failure_reason",
                        "pysc2_function",
                        "latency_ms",
                        "game_result",
                    }
                }
                record["payload"]["protocol_version"] = "1.0"
        journal.write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
            encoding="utf-8",
        )

        replay = await replay_event_log(
            journal,
            config=config,
            output_dir=tmp_path / "legacy-replayed",
        )

        assert replay.observations == replay.matched_decisions
        assert replay.mismatched_steps == []
        assert replay.backfilled_legacy_executions > 0
        assert replay.skipped_legacy_executions == 0

        replayed_events = [
            json.loads(line)
            for line in (tmp_path / "legacy-replayed" / "events.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        replayed_executions = [
            record["payload"] for record in replayed_events if record["event_type"] == "execution"
        ]
        assert replayed_executions
        assert all(report["protocol_version"] == "1.0" for report in replayed_executions)
        assert all(report["action_name"] for report in replayed_executions)
        assert all(report["actor"] for report in replayed_executions)
        assert all(report["source"] in {"planner", "reflex"} for report in replayed_executions)

    asyncio.run(execute())


@pytest.mark.parametrize("source", ["fallback", "planner"])
def test_real_v1_0_semantic_noop_is_normalized_to_untracked_idle(
    tmp_path: Path,
    source: str,
) -> None:
    config = make_config(tmp_path, variant="noop")
    observation = make_observation(
        run_id="legacy-run",
        episode_id="episode-0",
        step_id=0,
        game_loop=0,
    ).model_copy(update={"protocol_version": "1.0"})
    legacy_observation = observation.model_dump(mode="json")
    for action in legacy_observation["available_actions"]:
        action.pop("argument_candidates", None)
    command_id = f"legacy-run:episode-0:0:{source}:0"
    legacy_batch: dict[str, object] = {
        "protocol_version": "1.0",
        "run_id": "legacy-run",
        "episode_id": "episode-0",
        "step_id": 0,
        "decision_id": "legacy-run:episode-0:0:decision",
        "strategic_goal": "",
        "summary": "",
        "planner_pending": False,
        "commands": [
            {
                "command_id": command_id,
                "actor": "Developer/Empty",
                "name": "No_Operation",
                "arguments": [],
                "priority": 0,
                "ttl_game_loops": 1,
                "created_game_loop": 0,
                "source": source,
                "preconditions": {},
            }
        ],
        "rejected_commands": [],
    }
    legacy_execution: dict[str, object] = {
        "protocol_version": "1.0",
        "run_id": "legacy-run",
        "episode_id": "episode-0",
        "step_id": 0,
        "command_id": command_id,
        "success": True,
        "failure_reason": None,
        "pysc2_function": "no_op",
        "latency_ms": 0.0,
        "game_result": None,
    }
    journal = tmp_path / f"legacy-{source}.jsonl"
    _write_journal(
        journal,
        [
            ("observation", legacy_observation),
            ("decision", {"batch": legacy_batch}),
            ("execution", legacy_execution),
        ],
    )

    replay = asyncio.run(
        replay_event_log(
            journal,
            config=config,
            output_dir=tmp_path / f"legacy-{source}-replayed",
        )
    )

    assert replay.observations == 1
    assert replay.matched_decisions == 1
    assert replay.mismatched_steps == []
    assert replay.backfilled_legacy_executions == 1
    assert replay.skipped_legacy_executions == 1
    replayed_records = [
        json.loads(line)
        for line in (tmp_path / f"legacy-{source}-replayed" / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert all(record["event_type"] != "execution" for record in replayed_records)
    reference = next(
        record
        for record in replayed_records
        if record["event_type"] == "legacy_execution_reference"
    )
    assert reference["payload"]["report"]["action_name"] == "No_Operation"
    assert reference["payload"]["report"]["source"] == source


def test_v1_1_unknown_execution_still_fails_fast(tmp_path: Path) -> None:
    config = make_config(tmp_path)

    async def execute() -> None:
        original_dir = tmp_path / "current-original"
        runtime = build_runtime(config, original_dir)
        try:
            await run_mock_episode(
                config=config,
                runtime=runtime,
                run_id="current-replay-run",
                episode_id="episode-0",
                seed=0,
            )
        finally:
            await runtime.close()

        journal = original_dir / "events.jsonl"
        records = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
        execution = next(record for record in records if record["event_type"] == "execution")
        execution["payload"]["command_id"] = "unknown-current-command"
        journal.write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
            encoding="utf-8",
        )

        with pytest.raises(RuntimeError, match="unknown command"):
            await replay_event_log(
                journal,
                config=config,
                output_dir=tmp_path / "current-replayed",
            )

    asyncio.run(execute())


def test_evaluation_writes_all_report_formats(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    config.evaluation.seeds = [0]
    output = tmp_path / "evaluation"
    summary = asyncio.run(run_mock_suite(config, output))
    variants = summary["variants"]
    assert isinstance(variants, dict)
    assert set(variants) == {
        "noop",
        "reflex_only",
        "planner_only",
        "planner_reflection_memory_reflex",
    }
    assert (output / "summary.json").is_file()
    assert (output / "episodes.jsonl").is_file()
    assert (output / "report.md").is_file()
    assert (output / "config.yaml").is_file()
    assert (output / "provenance.json").is_file()

    provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["configuration"]["seeds"] == [0]
    assert provenance["provider"]["model"] == "fake-planner-v1"
    assert provenance["prompts"]["planning"]["version"] == "planning-v1"
    assert provenance["environment"]["adapter_version"] == "mock-v1"
    assert provenance["code"]["rtscortex_commit"]
    assert provenance["code"]["llm_pysc2_commit"] == ("551c863475c0c4a96a181080974d24b59589e9f3")

    run_provenance_path = (
        output / "runs" / "planner_reflection_memory_reflex" / "seed-0" / "provenance.json"
    )
    run_provenance = json.loads(run_provenance_path.read_text(encoding="utf-8"))
    assert run_provenance["configuration"]["seeds"] == [0]
    assert run_provenance["configuration"]["agent_variant"] == ("planner_reflection_memory_reflex")
    assert run_provenance_path.with_name("config.yaml").is_file()

    episodes = [
        json.loads(line)
        for line in (output / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(episodes) == 4
    assert all("evaluation_metrics" in episode for episode in episodes)
    assert all("tick_latency_ms_p95" in episode["evaluation_metrics"] for episode in episodes)
    assert all("failure_reason" in episode["evaluation_metrics"] for episode in episodes)
    assert all(
        "action_success_rate" in episode["evaluation_metrics"]["deprecated_fields"]
        for episode in episodes
    )

    full_metrics = variants["planner_reflection_memory_reflex"]
    assert "planner_latency_ms_p50" in full_metrics
    assert "reflex_preemptions" in full_metrics
    assert "plan_revision_rate" in full_metrics
    assert "model_cost_usd" in full_metrics
    assert "action_success_rate" in full_metrics["deprecated_fields"]
    assert "production_funnel" in full_metrics["execution"]
    assert "production_provenance_coverage" in full_metrics["execution"]
    assert "production_by_action" in full_metrics["execution"]
    assert "production_by_producer" in full_metrics["execution"]

    report = (output / "report.md").read_text(encoding="utf-8")
    assert "Planner p50/p95 ms" in report
    assert "## Decision activity" in report
    assert "## Meaningful outcomes" in report
    assert "## Build funnel" in report
    assert "## Production funnel" in report
    assert "### Production breakdown" in report
    assert "## Failure taxonomy" in report
    assert "Terminal coverage" in report
    assert "Failure classification" in report
    assert "Action success" not in report
    assert "LLM-PySC2 commit" in report

    with pytest.raises(FileExistsError, match="not empty"):
        asyncio.run(run_mock_suite(config, output))
