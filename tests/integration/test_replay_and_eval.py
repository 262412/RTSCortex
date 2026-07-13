from __future__ import annotations

import asyncio
from pathlib import Path

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
