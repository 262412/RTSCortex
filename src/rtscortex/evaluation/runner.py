"""Reproducible offline evaluation runner."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

from rtscortex.adapters import MockSC2Adapter
from rtscortex.config import AgentSettings, AgentVariant, ExperimentConfig
from rtscortex.contracts import EpisodeOutcome, EpisodeResult
from rtscortex.evaluation.metrics import (
    EpisodeMetrics,
    aggregate_episode_metrics,
    compute_episode_metrics,
)
from rtscortex.evaluation.provenance import write_experiment_snapshot
from rtscortex.memory import read_event_log
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
            failure_reason=(
                "max_steps_reached" if adapter.outcome is EpisodeOutcome.TRUNCATED else None
            ),
        )
        runtime.end_episode(result)
        return result
    finally:
        await adapter.close()


async def run_mock_suite(config: ExperimentConfig, output_dir: Path) -> dict[str, object]:
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"Evaluation output directory is not empty: {output_dir}")
    provenance = write_experiment_snapshot(config, output_dir)
    results: list[tuple[str, EpisodeResult, EpisodeMetrics]] = []
    for requested_variant in VARIANTS:
        variant_config = config.model_copy(
            update={"agent": AgentSettings(variant=requested_variant)},
            deep=True,
        )
        for seed in config.evaluation.seeds:
            episode_config = variant_config.model_copy(
                update={
                    "run": variant_config.run.model_copy(update={"seed": seed}),
                    "evaluation": variant_config.evaluation.model_copy(update={"seeds": [seed]}),
                }
            )
            run_id = f"mock-{requested_variant}-seed-{seed}"
            episode_id = "episode-0"
            run_dir = output_dir / "runs" / requested_variant / f"seed-{seed}"
            write_experiment_snapshot(episode_config, run_dir)
            runtime = build_runtime(episode_config, run_dir)
            try:
                result = await run_mock_episode(
                    config=episode_config,
                    runtime=runtime,
                    run_id=run_id,
                    episode_id=episode_id,
                    seed=seed,
                )
                events = list(read_event_log(run_dir / "events.jsonl"))
                metrics = compute_episode_metrics(
                    events,
                    result,
                    prompt_cost_per_million_tokens=(
                        episode_config.provider.prompt_cost_per_million_tokens
                    ),
                    completion_cost_per_million_tokens=(
                        episode_config.provider.completion_cost_per_million_tokens
                    ),
                )
                results.append((requested_variant, result, metrics))
            finally:
                await runtime.close()

    with (output_dir / "episodes.jsonl").open("w", encoding="utf-8") as stream:
        for result_variant, result, metrics in results:
            payload = result.model_dump(mode="json")
            payload["variant"] = result_variant
            payload["evaluation_metrics"] = metrics.as_dict()
            stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

    variants: dict[str, dict[str, Any]] = {}
    for report_variant in VARIANTS:
        selected = [
            (result, metrics)
            for result_variant, result, metrics in results
            if result_variant == report_variant
        ]
        episode_metrics = [metrics for _, metrics in selected]
        variants[report_variant] = aggregate_episode_metrics(episode_metrics)
        variants[report_variant].update(
            {
                "episodes": len(selected),
                "success_rate": mean(
                    result.outcome is EpisodeOutcome.VICTORY for result, _ in selected
                ),
                "mean_score": mean(result.score for result, _ in selected),
                "mean_steps": mean(result.steps for result, _ in selected),
            }
        )
    summary: dict[str, object] = {
        "scenario": config.environment.scenario,
        "seeds": config.evaluation.seeds,
        "provenance": provenance,
        "variants": variants,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_markdown_report(output_dir, config, variants, provenance)
    return summary


def _write_markdown_report(
    output_dir: Path,
    config: ExperimentConfig,
    variants: dict[str, dict[str, Any]],
    provenance: dict[str, Any],
) -> None:
    lines = [
        "# RTSCortex Offline Evaluation",
        "",
        f"Scenario: `{config.environment.scenario}`",
        f"Seeds: `{', '.join(str(seed) for seed in config.evaluation.seeds)}`",
        "",
        "## Task results",
        "",
        "| Variant | Episodes | Success | Mean score | Mean steps | Requests | Tokens | Cost USD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for display_variant, metrics in variants.items():
        lines.append(
            f"| `{display_variant}` | {int(metrics['episodes'])} | "
            f"{metrics['success_rate']:.3f} | {metrics['mean_score']:.2f} | "
            f"{metrics['mean_steps']:.2f} | {metrics['model_requests']} | "
            f"{metrics['total_tokens']} | "
            f"{metrics['model_cost_usd']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Decision activity",
            "",
            (
                "| Variant | Decisions | Planner pending | Fallback | Idle reasons | "
                "Unique rejected IDs | Duplicate dispatch |"
            ),
            "|---|---:|---:|---:|---|---:|---:|",
        ]
    )
    for display_variant, metrics in variants.items():
        execution = metrics["execution"]
        lines.append(
            f"| `{display_variant}` | {execution['decision_count']} | "
            f"{execution['planner_pending_decisions']} | "
            f"{execution['fallback_decisions']} | "
            f"{_count_summary(execution['idle_reason_counts'])} | "
            f"{execution['unique_validation_rejected_command_ids']} | "
            f"{execution['duplicate_dispatches']} |"
        )
    lines.extend(
        [
            "",
            "## Meaningful outcomes",
            "",
            (
                "| Variant | Commands | Succeeded | Failed | Cancelled | Unconfirmed | "
                "Meaningful success | Completed success | Backlog | Terminal coverage | "
                "Unexpected terminal | Failure classification | Transport NoOps |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for display_variant, metrics in variants.items():
        execution = metrics["execution"]
        lines.append(
            f"| `{display_variant}` | {execution['meaningful_commands']} | "
            f"{execution['meaningful_successes']} | "
            f"{execution['meaningful_failures']} | "
            f"{execution['meaningful_cancelled']} | "
            f"{execution['meaningful_unconfirmed']} | "
            f"{execution['meaningful_action_success_rate']:.3f} | "
            f"{execution['completed_execution_success_rate']:.3f} | "
            f"{execution['terminal_backlog_rate']:.3f} | "
            f"{execution['terminal_report_coverage']:.3f} | "
            f"{execution['unexpected_terminal_reports']} | "
            f"{execution['failure_classification_coverage']:.3f} | "
            f"{execution['transport_noop_primitives']} |"
        )
    lines.extend(
        [
            "",
            "## Build funnel",
            "",
            (
                "| Variant | Raw Planner proposed | Candidate validated | Translator accepted | "
                "PySC2 accepted | Effect confirmed | Effect confirmed rate | "
                "Effect timeout rate | Pre-dispatch rejection rate |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for display_variant, metrics in variants.items():
        funnel = metrics["execution"]["build_funnel"]
        lines.append(
            f"| `{display_variant}` | {funnel['proposed']} | "
            f"{funnel['candidate_validated']} | {funnel['translator_accepted']} | "
            f"{funnel['pysc2_accepted']} | {funnel['effect_confirmed']} | "
            f"{metrics['execution']['build_effect_confirmed_rate']:.3f} | "
            f"{metrics['execution']['build_effect_timeout_rate']:.3f} | "
            f"{metrics['execution']['build_pre_dispatch_rejection_rate']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Production funnel",
            "",
            (
                "| Variant | Raw Planner proposed | Candidate validated | "
                "Translator accepted | PySC2 accepted | Order confirmed | "
                "Unit fallback confirmed | Effect confirmed | Acceptance only (deprecated) | "
                "Effect confirmed rate | Provenance coverage | Timeout rate | "
                "Confirmation latency p50/p95 loops |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for display_variant, metrics in variants.items():
        execution = metrics["execution"]
        funnel = execution["production_funnel"]
        lines.append(
            f"| `{display_variant}` | {funnel['proposed']} | "
            f"{funnel['candidate_validated']} | {funnel['translator_accepted']} | "
            f"{funnel['pysc2_accepted']} | {funnel['order_confirmed']} | "
            f"{funnel['unit_fallback_confirmed']} | {funnel['effect_confirmed']} | "
            f"{funnel['acceptance_only']} | "
            f"{execution['production_effect_confirmed_rate']:.3f} | "
            f"{execution['production_provenance_coverage']:.3f} | "
            f"{execution['production_timeout_rate']:.3f} | "
            f"{execution['confirmation_latency_game_loops_p50']:.1f} / "
            f"{execution['confirmation_latency_game_loops_p95']:.1f} |"
        )
    lines.extend(
        [
            "",
            "### Production breakdown",
            "",
            "| Variant | Actions | Producers |",
            "|---|---|---|",
        ]
    )
    for display_variant, metrics in variants.items():
        execution = metrics["execution"]
        lines.append(
            f"| `{display_variant}` | "
            f"{_count_summary(execution['production_by_action'])} | "
            f"{_count_summary(execution['production_by_producer'])} |"
        )
    lines.extend(
        [
            "",
            "## Failure taxonomy",
            "",
            (
                "| Variant | Stage | Code | Action | Actor | Action/actor commands | "
                "Action/stage/code failures | Dispatched Builder Attack | "
                "Dispatched friendly target Attack |"
            ),
            "|---|---|---|---|---|---|---|---:|---:|",
        ]
    )
    for display_variant, metrics in variants.items():
        execution = metrics["execution"]
        lines.append(
            f"| `{display_variant}` | {_count_summary(execution['failure_by_stage'])} | "
            f"{_count_summary(execution['failure_by_code'])} | "
            f"{_count_summary(execution['failure_by_action'])} | "
            f"{_count_summary(execution['failure_by_actor'])} | "
            f"{_count_summary(execution['command_by_action_actor'])} | "
            f"{_count_summary(execution['failure_by_action_stage_code'])} | "
            f"{execution['builder_attack_commands']} | "
            f"{execution['friendly_target_attacks']} |"
        )
    lines.extend(
        [
            "",
            "## Safety and attribution invariants",
            "",
            (
                "| Variant | Planner audit | Planner Builder Attack | Planner friendly Attack | "
                "Unsafe rejected | Unsafe dispatched | Planner NoOps | Generic translation | "
                "Placement rejection | Unattributed primitive | Candidate-external PySC2 | "
                "573 terminal |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for display_variant, metrics in variants.items():
        execution = metrics["execution"]
        lines.append(
            f"| `{display_variant}` | {execution['planner_proposal_audited_results']}/"
            f"{execution['planner_module_results']} | "
            f"{execution['planner_builder_attack_proposals']} | "
            f"{execution['planner_friendly_target_attack_proposals']} | "
            f"{execution['planner_unsafe_attack_rejected_before_dispatch']} | "
            f"{execution['planner_unsafe_attack_dispatched']} | "
            f"{execution['planner_noop_proposals']} | "
            f"{execution['generic_translation_failures']} | "
            f"{execution['upstream_placement_rejections']} | "
            f"{execution['unattributed_primitives']} | "
            f"{execution['candidate_outside_pysc2_dispatches']} | "
            f"{execution['orchestration_573_terminal_reports']} |"
        )
    lines.extend(
        [
            "",
            "## Runtime diagnostics",
            "",
            (
                "| Variant | Planner p50/p95 ms | Reflex p50/p95 ms | Tick p50/p95 ms | "
                "Preemptions | Plan revision | Mean duration s | Failure reasons |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for display_variant, metrics in variants.items():
        failures = ", ".join(
            f"{reason}: {count}" for reason, count in metrics["failure_reasons"].items()
        )
        lines.append(
            f"| `{display_variant}` | {metrics['planner_latency_ms_p50']:.3f} / "
            f"{metrics['planner_latency_ms_p95']:.3f} | "
            f"{metrics['reflex_latency_ms_p50']:.3f} / "
            f"{metrics['reflex_latency_ms_p95']:.3f} | "
            f"{metrics['tick_latency_ms_p50']:.3f} / "
            f"{metrics['tick_latency_ms_p95']:.3f} | "
            f"{metrics['reflex_preemptions']} | {metrics['plan_revision_rate']:.3f} | "
            f"{metrics['mean_episode_duration_seconds']:.3f} | {failures or 'none'} |"
        )
    code = provenance["code"]
    provider = provenance["provider"]
    lines.extend(
        [
            "",
            "## Reproducibility",
            "",
            f"- RTSCortex commit: `{code['rtscortex_commit']}`",
            f"- LLM-PySC2 commit: `{code['llm_pysc2_commit']}`",
            f"- Provider/model: `{provider['kind']}` / `{provider['model']}`",
            "- Full configuration: `config.yaml`",
            "- Full provenance: `provenance.json`",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _count_summary(counts: dict[str, int]) -> str:
    if not counts:
        return "none"
    return ", ".join(f"`{name}`: {count}" for name, count in sorted(counts.items()))
