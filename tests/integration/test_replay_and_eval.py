from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from rtscortex.evaluation import run_mock_episode, run_mock_suite
from rtscortex.evaluation.replay import replay_event_log
from rtscortex.runtime.factory import build_runtime
from tests.helpers import make_config


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

    full_metrics = variants["planner_reflection_memory_reflex"]
    assert "planner_latency_ms_p50" in full_metrics
    assert "reflex_preemptions" in full_metrics
    assert "plan_revision_rate" in full_metrics
    assert "model_cost_usd" in full_metrics

    report = (output / "report.md").read_text(encoding="utf-8")
    assert "Planner p95 ms" in report
    assert "LLM-PySC2 commit" in report

    with pytest.raises(FileExistsError, match="not empty"):
        asyncio.run(run_mock_suite(config, output))
