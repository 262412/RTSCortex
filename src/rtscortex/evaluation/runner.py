"""Reproducible offline evaluation runner."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

from rtscortex.adapters import MockSC2Adapter
from rtscortex.config import AgentSettings, AgentVariant, ExperimentConfig
from rtscortex.contracts import EpisodeOutcome, EpisodeResult
from rtscortex.runtime.engine import RuntimeEngine
from rtscortex.runtime.factory import build_runtime

VARIANTS: tuple[AgentVariant, ...] = (
    "noop",
    "reflex_only",
    "planner_only",
    "planner_reflection_memory_reflex",
)


async def run_mock_episode(
    *,
    config: ExperimentConfig,
    runtime: RuntimeEngine,
    run_id: str,
    episode_id: str,
    seed: int,
) -> EpisodeResult:
    adapter = MockSC2Adapter(
        scenario=config.environment.scenario,
        max_steps=config.environment.max_steps,
    )
    observation = await adapter.reset(run_id=run_id, episode_id=episode_id, seed=seed)
    steps = 0
    try:
        while not adapter.done and steps < config.environment.max_steps:
            batch = await runtime.tick(observation)
            observation, reports = await adapter.step(batch)
            for report in reports:
                runtime.record_execution(report)
            steps += 1
        success_rate = (
            adapter.action_successes / adapter.action_attempts if adapter.action_attempts else 0.0
        )
        result = EpisodeResult(
            run_id=run_id,
            episode_id=episode_id,
            scenario=config.environment.scenario,
            seed=seed,
            outcome=adapter.outcome,
            score=100.0 if adapter.outcome is EpisodeOutcome.VICTORY else 0.0,
            steps=steps,
            metrics={
                "action_attempts": float(adapter.action_attempts),
                "action_success_rate": success_rate,
            },
        )
        runtime.end_episode(result)
        return result
    finally:
        await adapter.close()


async def run_mock_suite(config: ExperimentConfig, output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[tuple[str, EpisodeResult]] = []
    for requested_variant in VARIANTS:
        variant_config = config.model_copy(
            update={"agent": AgentSettings(variant=requested_variant)},
            deep=True,
        )
        for seed in config.evaluation.seeds:
            run_id = f"mock-{requested_variant}-seed-{seed}"
            episode_id = "episode-0"
            run_dir = output_dir / "runs" / requested_variant / f"seed-{seed}"
            runtime = build_runtime(variant_config, run_dir)
            try:
                result = await run_mock_episode(
                    config=variant_config,
                    runtime=runtime,
                    run_id=run_id,
                    episode_id=episode_id,
                    seed=seed,
                )
                results.append((requested_variant, result))
            finally:
                await runtime.close()

    with (output_dir / "episodes.jsonl").open("w", encoding="utf-8") as stream:
        for result_variant, result in results:
            payload = result.model_dump(mode="json")
            payload["variant"] = result_variant
            stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

    variants: dict[str, dict[str, float]] = {}
    for report_variant in VARIANTS:
        selected = [
            result for result_variant, result in results if result_variant == report_variant
        ]
        variants[report_variant] = {
            "episodes": float(len(selected)),
            "success_rate": mean(result.outcome is EpisodeOutcome.VICTORY for result in selected),
            "mean_score": mean(result.score for result in selected),
            "mean_steps": mean(result.steps for result in selected),
            "action_success_rate": mean(
                result.metrics["action_success_rate"] for result in selected
            ),
        }
    summary: dict[str, object] = {
        "scenario": config.environment.scenario,
        "seeds": config.evaluation.seeds,
        "variants": variants,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# RTSCortex Offline Evaluation",
        "",
        f"Scenario: `{config.environment.scenario}`",
        "",
        "| Variant | Episodes | Success rate | Mean score | Mean steps |",
        "|---|---:|---:|---:|---:|",
    ]
    for display_variant, metrics in variants.items():
        lines.append(
            f"| `{display_variant}` | {int(metrics['episodes'])} | "
            f"{metrics['success_rate']:.3f} | {metrics['mean_score']:.2f} | "
            f"{metrics['mean_steps']:.2f} |"
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
