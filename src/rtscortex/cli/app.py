"""RTSCortex command-line entry point."""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal

import typer
import uvicorn
import yaml

from rtscortex.api import create_app
from rtscortex.cli.doctor import run_doctor
from rtscortex.config import ExperimentConfig, load_config
from rtscortex.console import ConsoleSession, LiveConsoleHub, create_console_app
from rtscortex.cortex import (
    ExecutorCorpusError,
    ExecutorSplit,
    benchmark_executor_corpus,
    build_executor_corpus,
    verify_executor_corpus,
)
from rtscortex.evaluation import (
    ReportError,
    RunReportArtifacts,
    run_mock_episode,
    run_mock_suite,
    write_run_reports,
)
from rtscortex.evaluation.replay import replay_event_log
from rtscortex.memory import EventStore, read_event_log
from rtscortex.playbook import PlaybookStore
from rtscortex.policy import (
    LLMPlanningPolicySubagent,
    PolicyShadowComparison,
    PolicyShadowRunner,
    attach_goal_progress,
    build_protoss_opening_goal,
    default_shadow_registrations,
    load_historical_observations,
)
from rtscortex.policy.comparison import (
    load_policy_comparison_config,
    run_policy_comparison,
)
from rtscortex.policy.corpus import build_policy_corpus_from_file, verify_policy_corpus
from rtscortex.providers import OpenAICompatibleProvider
from rtscortex.runtime.factory import build_runtime
from rtscortex.runtime.live import (
    LiveEnvironmentError,
    LiveProcessSupervisor,
    LiveWorkerSpec,
    WorkerProcessError,
    ensure_console_port_available,
    live_socket_path,
    prepare_live_worker,
)

app = typer.Typer(no_args_is_help=True, help="RTSCortex agent runtime")
policy_corpus_app = typer.Typer(
    no_args_is_help=True,
    help="Build and verify immutable policy-comparison corpora.",
)
executor_corpus_app = typer.Typer(
    no_args_is_help=True,
    help="Build and verify privacy-minimized fast-executor corpora.",
)
playbook_app = typer.Typer(
    no_args_is_help=True,
    help="Inspect the persistent cross-episode CortexPlaybook.",
)
app.add_typer(policy_corpus_app, name="policy-corpus")
app.add_typer(executor_corpus_app, name="executor-corpus")
app.add_typer(playbook_app, name="playbook")
PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONSOLE_STATIC_DIR = Path(__file__).resolve().parents[1] / "console" / "static"


def _run_id(prefix: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{prefix}-{stamp}-{uuid.uuid4().hex[:8]}"


def _reserve_run_dir(output_root: Path, prefix: str) -> tuple[str, Path]:
    """Atomically reserve a collision-resistant artifact directory."""

    output_root.mkdir(parents=True, exist_ok=True)
    for _ in range(3):
        run_id = _run_id(prefix)
        run_dir = output_root / run_id
        try:
            run_dir.mkdir(exist_ok=False)
        except FileExistsError:
            continue
        return run_id, run_dir
    raise RuntimeError("could not reserve a unique run directory")


def _live_worker_environment(
    config: ExperimentConfig,
    live_worker: LiveWorkerSpec,
) -> dict[str, str]:
    environment = {
        "SC2PATH": str(live_worker.sc2_path),
        "RTSCORTEX_AGENT_RACE": config.environment.agent_race,
        "RTSCORTEX_PENDING_PLAN_STEP_DELAY_SECONDS": str(
            config.environment.pending_plan_step_delay_seconds
        ),
        "RTSCORTEX_PAUSE_UNTIL_FIRST_PLAN": str(config.environment.pause_until_first_plan).lower(),
        "RTSCORTEX_RUNTIME_REQUEST_TIMEOUT_SECONDS": str(
            config.runtime.planner_timeout_seconds + 5.0
        ),
        "RTSCORTEX_ACTION_EFFECT_TIMEOUT_GAME_LOOPS": str(
            config.environment.action_effect_timeout_game_loops
        ),
        "RTSCORTEX_OBSERVATION_GAP_WATCHDOG_GAME_LOOPS": str(
            config.environment.observation_gap_watchdog_game_loops
        ),
        "RTSCORTEX_OBSERVATION_GAP_HARD_LIMIT_GAME_LOOPS": str(
            config.environment.observation_gap_hard_limit_game_loops
        ),
        "RTSCORTEX_ORCHESTRATION_PRIMITIVE_BUDGET": str(
            config.environment.orchestration_primitive_budget
        ),
        "RTSCORTEX_EXPANSION_SCOUT_ENABLED": str(
            config.environment.expansion_scout_enabled
        ).lower(),
        "RTSCORTEX_EXPANSION_SCOUT_INTERVAL_GAME_LOOPS": str(
            config.environment.expansion_scout_interval_game_loops
        ),
        "RTSCORTEX_CONSOLE_ENABLED": str(config.console.enabled).lower(),
        "RTSCORTEX_CONSOLE_FRAME_FPS": str(config.console.frame_fps),
        "RTSCORTEX_CONSOLE_JPEG_QUALITY": str(config.console.jpeg_quality),
        "RTSCORTEX_CONSOLE_RGB_SCREEN_SIZE": str(config.console.rgb_screen_size),
        "RTSCORTEX_CONSOLE_RGB_MINIMAP_SIZE": str(config.console.rgb_minimap_size),
    }
    if config.environment.simulation_speed_multiplier is not None:
        environment["RTSCORTEX_SIMULATION_SPEED_MULTIPLIER"] = str(
            config.environment.simulation_speed_multiplier
        )
    return environment


def _active_model_label(config: ExperimentConfig) -> str:
    if config.agent.variant == "cortex" and config.cortex.macro.kind == "hima":
        suffix = config.cortex.macro.candidate.removeprefix("protoss-")
        return f"SNUMPR/Protoss-{suffix}"
    if config.agent.variant == "cortex" and config.cortex.macro.kind == "hima_ensemble":
        race = config.cortex.macro.ensemble_members[0].candidate.rsplit("-", 1)[0]
        return f"HIMA {race.title()} a/b/c Ensemble"
    if config.agent.variant == "cortex" and config.cortex.macro.kind == "scripted":
        return f"Scripted {config.environment.agent_race.title()} canary"
    return config.provider.model


@playbook_app.command("show")
def playbook_show(
    database: Annotated[
        Path,
        typer.Option("--database", dir_okay=False, help="CortexPlaybook SQLite path."),
    ] = Path("~/scratch/outputs/RTSCortex/cortex-playbook.sqlite3"),
    promoted_only: Annotated[
        bool,
        typer.Option("--promoted-only", help="Hide lessons that still need more evidence."),
    ] = False,
) -> None:
    """Print the current reusable tactical notebook as structured JSON."""

    path = database.expanduser()
    if not path.is_file():
        raise typer.BadParameter(
            f"playbook database does not exist: {path}",
            param_hint="--database",
        )
    store = PlaybookStore(path)
    try:
        lessons = store.lessons()
    finally:
        store.close()
    if promoted_only:
        lessons = [lesson for lesson in lessons if lesson.status.value == "promoted"]
    typer.echo(
        json.dumps(
            [lesson.model_dump(mode="json") for lesson in lessons],
            ensure_ascii=False,
            indent=2,
        )
    )


def _snapshot_config(config: ExperimentConfig, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = config.model_dump(mode="json")
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )


def _echo_report_artifacts(artifacts: RunReportArtifacts) -> None:
    typer.echo(f"Timeline: {artifacts.timeline_path}")
    typer.echo(f"Summary: {artifacts.summary_path}")


def _write_run_reports_best_effort(run_dir: Path) -> None:
    journal_path = run_dir / "events.jsonl"
    try:
        if not journal_path.is_file() or journal_path.stat().st_size == 0:
            return
        artifacts = write_run_reports(run_dir)
    except Exception as error:
        typer.echo(f"Warning: could not generate run reports: {error}", err=True)
        return
    _echo_report_artifacts(artifacts)


def _require_console_static_dir() -> Path:
    index = CONSOLE_STATIC_DIR / "index.html"
    if not index.is_file():
        raise typer.BadParameter(
            "Live Console frontend assets are missing; run `npm ci && npm run build` "
            "from web/live-console",
            param_hint="--console",
        )
    return CONSOLE_STATIC_DIR


@app.command()
def doctor(
    require_sc2: Annotated[
        bool, typer.Option(help="Treat a missing StarCraft II installation as an error.")
    ] = False,
    config_path: Annotated[
        Path | None,
        typer.Option("--config", exists=True, dir_okay=False, help="Live experiment config."),
    ] = None,
) -> None:
    """Check the core environment, pinned submodule, and optional SC2 installation."""

    config = load_config(config_path) if config_path is not None else None
    checks = run_doctor(PROJECT_ROOT, require_sc2=require_sc2, config=config)
    for check in checks:
        typer.echo(f"{check.status.upper():8} {check.name:16} {check.detail}")
    if any(check.status == "error" for check in checks):
        raise typer.Exit(code=1)


@app.command("run")
def run_experiment(
    config_path: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
    seed: Annotated[int | None, typer.Option("--seed", min=0)] = None,
    console: Annotated[
        bool,
        typer.Option("--console", help="Serve the read-only Live Console for this run."),
    ] = False,
    console_port: Annotated[
        int | None,
        typer.Option("--console-port", min=1, max=65_535),
    ] = None,
) -> None:
    """Run one configured episode."""

    config = load_config(config_path)
    if seed is not None:
        config = config.model_copy(update={"run": config.run.model_copy(update={"seed": seed})})
    console_enabled = config.console.enabled or console or console_port is not None
    effective_console_port = console_port or config.console.port
    config = config.model_copy(
        update={
            "console": config.console.model_copy(
                update={"enabled": console_enabled, "port": effective_console_port}
            )
        }
    )
    console_static_dir: Path | None = None
    if console_enabled:
        if config.environment.adapter != "llm_pysc2":
            raise typer.BadParameter(
                "Live Console streaming requires environment.adapter=llm_pysc2; "
                "use `rtscortex console RUN_DIR` for completed mock runs",
                param_hint="--console",
            )
        try:
            ensure_console_port_available(effective_console_port)
        except LiveEnvironmentError as error:
            raise typer.BadParameter(str(error), param_hint="--console-port") from error
        console_static_dir = _require_console_static_dir()
    run_id, run_dir = _reserve_run_dir(config.run.output_root, config.agent.variant)
    episode_id = "episode-0"
    live_worker: LiveWorkerSpec | None = None
    runtime_socket: Path | None = None
    if config.environment.adapter == "llm_pysc2":
        try:
            live_worker = prepare_live_worker(config, PROJECT_ROOT)
            runtime_socket = live_socket_path(config.run.runtime_root, run_id)
        except LiveEnvironmentError as error:
            raise typer.BadParameter(str(error), param_hint="--config") from error
    _snapshot_config(config, run_dir)

    async def execute() -> str:
        runtime = build_runtime(config, run_dir)
        if live_worker is not None:
            assert runtime_socket is not None
            worker_environment = _live_worker_environment(config, live_worker)
            console_hub: LiveConsoleHub | None = None
            console_api = None
            if config.console.enabled:
                session = ConsoleSession(
                    run_id=run_id,
                    episode_id=episode_id,
                    status="starting",
                    scenario=config.environment.scenario,
                    seed=config.run.seed,
                    model=_active_model_label(config),
                    stale_after_seconds=config.console.stale_after_seconds,
                    frontend_event_limit=config.console.frontend_event_limit,
                )
                console_hub = LiveConsoleHub(session)
                assert console_static_dir is not None
                console_api = create_console_app(
                    runtime.store,
                    session,
                    console_hub,
                    console_static_dir,
                    frontend_event_limit=config.console.frontend_event_limit,
                )
            supervisor = LiveProcessSupervisor(
                runtime=runtime,
                run_id=run_id,
                episode_id=episode_id,
                scenario=config.environment.scenario,
                seed=config.run.seed,
                socket_path=runtime_socket,
                worker_command=live_worker.command,
                worker_environment=worker_environment,
                console_hub=console_hub,
                console_app=console_api,
                console_port=(config.console.port if config.console.enabled else None),
                run_dir=run_dir,
                server_ready_timeout_seconds=(config.environment.server_ready_timeout_seconds),
                shutdown_timeout_seconds=config.environment.shutdown_timeout_seconds,
            )
            result = await supervisor.run()
            return result.model_dump_json(indent=2)
        try:
            result = await run_mock_episode(
                config=config,
                runtime=runtime,
                run_id=run_id,
                episode_id=episode_id,
                seed=config.run.seed,
            )
            return result.model_dump_json(indent=2)
        finally:
            await runtime.close()

    try:
        if config.console.enabled:
            typer.echo(f"Live Console: http://127.0.0.1:{config.console.port}")
            typer.echo(f"Run directory: {run_dir}")
        output = asyncio.run(execute())
    except WorkerProcessError as error:
        _write_run_reports_best_effort(run_dir)
        typer.echo(error.result.model_dump_json(indent=2), err=True)
        typer.echo(f"Artifacts: {run_dir}", err=True)
        raise typer.Exit(code=1) from error
    except LiveEnvironmentError as error:
        _write_run_reports_best_effort(run_dir)
        typer.echo(f"Live run failed: {error}", err=True)
        typer.echo(f"Artifacts: {run_dir}", err=True)
        raise typer.Exit(code=1) from error
    except BaseException:
        _write_run_reports_best_effort(run_dir)
        raise
    _write_run_reports_best_effort(run_dir)
    typer.echo(output)
    typer.echo(f"Artifacts: {run_dir}")


@app.command("console")
def serve_console(
    run_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    port: Annotated[int, typer.Option(min=1, max=65_535)] = 8765,
) -> None:
    """Serve the read-only event history for a completed run."""

    journal_path = run_dir / "events.jsonl"
    database_path = run_dir / "events.sqlite3"
    config_path = run_dir / "config.yaml"
    missing = [
        path.name for path in (journal_path, database_path, config_path) if not path.is_file()
    ]
    if missing:
        raise typer.BadParameter(
            f"run directory is missing: {', '.join(missing)}",
            param_hint="RUN_DIR",
        )
    try:
        first_event = next(iter(read_event_log(journal_path)))
    except StopIteration as error:
        raise typer.BadParameter("events.jsonl is empty", param_hint="RUN_DIR") from error
    config = load_config(config_path)
    try:
        ensure_console_port_available(port)
    except LiveEnvironmentError as error:
        raise typer.BadParameter(str(error), param_hint="--port") from error
    static_dir = _require_console_static_dir()
    store = EventStore(database_path, journal_path)
    session = ConsoleSession(
        run_id=first_event.run_id,
        episode_id=first_event.episode_id,
        status="historical",
        scenario=config.environment.scenario,
        seed=config.run.seed,
        model=_active_model_label(config),
        stale_after_seconds=config.console.stale_after_seconds,
        frontend_event_limit=config.console.frontend_event_limit,
    )
    hub = LiveConsoleHub(session)
    console_api = create_console_app(
        store,
        session,
        hub,
        static_dir,
        frontend_event_limit=config.console.frontend_event_limit,
    )
    typer.echo(f"Historical Console: http://127.0.0.1:{port}")
    typer.echo("RGB history is unavailable because frames are not persisted.")
    try:
        uvicorn.run(console_api, host="127.0.0.1", port=port, log_level="info")
    finally:
        store.close()


@app.command("eval")
def evaluate(
    config_path: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
    output_dir: Annotated[Path | None, typer.Option("--output-dir")] = None,
) -> None:
    """Run all four deterministic offline baselines and write reports."""

    config = load_config(config_path)
    target = output_dir or config.run.output_root / _run_id("evaluation")
    try:
        summary = asyncio.run(run_mock_suite(config, target.expanduser()))
    except FileExistsError as error:
        raise typer.BadParameter(str(error), param_hint="--output-dir") from error
    typer.echo(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    typer.echo(f"Artifacts: {target.expanduser()}")


@app.command()
def report(
    run_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
) -> None:
    """Generate readable Markdown and machine-readable JSON run reports."""

    try:
        artifacts = write_run_reports(run_dir)
    except ReportError as error:
        raise typer.BadParameter(str(error), param_hint="RUN_DIR") from error
    _echo_report_artifacts(artifacts)


@app.command()
def replay(
    journal: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    config_path: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
    output_dir: Annotated[Path | None, typer.Option("--output-dir")] = None,
) -> None:
    """Re-run recorded observations and compare their action batches."""

    config = load_config(config_path)
    target = output_dir or config.run.output_root / _run_id("replay")
    result = asyncio.run(replay_event_log(journal, config=config, output_dir=target.expanduser()))
    typer.echo(json.dumps(result.__dict__, indent=2, sort_keys=True))
    if result.mismatched_steps:
        raise typer.Exit(code=1)


@policy_corpus_app.command("build")
def policy_corpus_build(
    config_path: Annotated[
        Path,
        typer.Option("--config", exists=True, dir_okay=False),
    ],
    output_dir: Annotated[Path, typer.Option("--output-dir", file_okay=False)],
) -> None:
    """Build a deterministic, provenance-rich policy fixture corpus."""

    try:
        result = build_policy_corpus_from_file(
            config_path.expanduser().resolve(),
            output_dir.expanduser().resolve(),
        )
    except (OSError, ValueError) as error:
        raise typer.BadParameter(str(error), param_hint="--config") from error
    typer.echo(
        json.dumps(
            {
                "fixture_count": result.manifest.fixture_count,
                "stratum_counts": {
                    key.value: value for key, value in result.manifest.stratum_counts.items()
                },
                "seeds": result.manifest.seeds,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    typer.echo(f"Manifest: {result.manifest_path}")
    typer.echo(f"Fixtures: {result.fixtures_path}")


@executor_corpus_app.command("build")
def executor_corpus_build(
    sources: Annotated[
        list[Path],
        typer.Argument(
            exists=True,
            help="One or more run directories or events.jsonl journals.",
        ),
    ],
    output_dir: Annotated[Path, typer.Option("--output-dir", file_okay=False)],
    split_seed: Annotated[str, typer.Option("--split-seed")] = ("rtscortex-fast-executor-v0.1"),
) -> None:
    """Export labeled executor selections without RGB, prompts, tags, or coordinates."""

    try:
        result = build_executor_corpus(
            sources,
            output_dir.expanduser().resolve(),
            split_seed=split_seed,
        )
    except (OSError, ExecutorCorpusError, ValueError) as error:
        raise typer.BadParameter(str(error), param_hint="SOURCES") from error
    typer.echo(result.manifest.model_dump_json(indent=2))
    typer.echo(f"Manifest: {result.manifest_path}")


@executor_corpus_app.command("verify")
def executor_corpus_verify(
    manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    verify_sources: Annotated[
        bool,
        typer.Option(
            "--verify-sources",
            help="Also verify privacy-safe source journal fingerprints.",
        ),
    ] = False,
) -> None:
    """Verify executor corpus hashes, conservation, and episode split isolation."""

    verification = verify_executor_corpus(
        manifest.expanduser().resolve(),
        verify_sources=verify_sources,
    )
    typer.echo(verification.model_dump_json(indent=2))
    if not verification.valid:
        raise typer.Exit(code=1)


@app.command("executor-benchmark")
def executor_benchmark(
    manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    repetitions: Annotated[int, typer.Option(min=1)] = 100,
    split: Annotated[
        str,
        typer.Option(help="Corpus split to evaluate: test, validation, train, or all."),
    ] = "test",
) -> None:
    """Benchmark deterministic ranking over saved candidate features."""

    try:
        selected_split: ExecutorSplit | Literal["all"] = (
            "all" if split == "all" else ExecutorSplit(split)
        )
        result = benchmark_executor_corpus(
            manifest.expanduser().resolve(),
            repetitions=repetitions,
            split=selected_split,
        )
    except (OSError, ExecutorCorpusError, ValueError) as error:
        raise typer.BadParameter(str(error), param_hint="MANIFEST") from error
    typer.echo(result.model_dump_json(indent=2))


@policy_corpus_app.command("verify")
def policy_corpus_verify(
    manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    verify_sources: Annotated[
        bool,
        typer.Option(
            "--verify-sources",
            help="Also hash and cross-check every original journal.",
        ),
    ] = False,
) -> None:
    """Verify corpus hashes, balance, ordering, and optional source journals."""

    try:
        verification = verify_policy_corpus(
            manifest.expanduser().resolve(),
            verify_sources=verify_sources,
        )
    except (OSError, ValueError) as error:
        raise typer.BadParameter(str(error), param_hint="MANIFEST") from error
    typer.echo(verification.model_dump_json(indent=2))
    if not verification.valid:
        raise typer.Exit(code=1)


@app.command("policy-compare")
def policy_compare(
    config_path: Annotated[
        Path,
        typer.Option("--config", exists=True, dir_okay=False),
    ],
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            file_okay=False,
            help="Override the timestamped output directory from the config.",
        ),
    ] = None,
) -> None:
    """Compare Qwen, HIMA, and HierNet candidates on one immutable corpus."""

    try:
        config = load_policy_comparison_config(
            config_path.expanduser().resolve(),
            base_dir=PROJECT_ROOT,
        )
        artifacts = run_policy_comparison(config, output_dir=output_dir)
    except (OSError, ValueError) as error:
        raise typer.BadParameter(str(error), param_hint="--config") from error
    typer.echo(
        json.dumps(
            [summary.model_dump(mode="json") for summary in artifacts.comparison.summaries],
            ensure_ascii=False,
            indent=2,
        )
    )
    typer.echo(f"Comparison: {artifacts.reports.comparison_path}")
    typer.echo(f"Report: {artifacts.reports.report_path}")
    typer.echo(f"Artifacts: {artifacts.output_dir}")


async def _run_policy_shadow(
    *,
    config: ExperimentConfig | None,
    journal_path: Path,
    limit: int,
    stride: int,
    use_current_qwen: bool,
) -> PolicyShadowComparison:
    fixtures = load_historical_observations(
        journal_path,
        limit=limit,
        stride=stride,
    )
    if not fixtures:
        raise ValueError("events.jsonl contains no observation events")
    fixtures = attach_goal_progress(fixtures, build_protoss_opening_goal())

    provider: OpenAICompatibleProvider | None = None
    current_qwen = None
    if use_current_qwen:
        if config is None or config.provider.kind != "openai_compatible":
            raise ValueError(
                "current Qwen comparison requires provider.kind=openai_compatible "
                "in RUN_DIR/config.yaml"
            )
        provider = OpenAICompatibleProvider(
            base_url=config.provider.base_url,
            model=config.provider.model,
            api_key_env=config.provider.api_key_env,
            timeout_seconds=config.provider.timeout_seconds,
            max_tokens=config.provider.max_tokens,
            enable_thinking=config.provider.enable_thinking,
        )
        current_qwen = LLMPlanningPolicySubagent(provider)

    try:
        return await PolicyShadowRunner().compare(
            fixtures,
            default_shadow_registrations(current_qwen=current_qwen),
        )
    finally:
        if provider is not None:
            await provider.close()


@app.command("policy-shadow")
def policy_shadow(
    run_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    limit: Annotated[int, typer.Option(min=1, help="Maximum historical observations.")] = 10,
    stride: Annotated[
        int,
        typer.Option(min=1, help="Select every Nth observation in journal order."),
    ] = 1,
    output: Annotated[
        Path | None,
        typer.Option("--output", dir_okay=False, help="Comparison JSON artifact path."),
    ] = None,
    current_qwen: Annotated[
        bool,
        typer.Option(
            "--current-qwen/--no-current-qwen",
            help="Evaluate the OpenAI-compatible Qwen provider from the run config.",
        ),
    ] = True,
) -> None:
    """Compare shadow policies on the same historical Protoss observations."""

    journal_path = run_dir / "events.jsonl"
    if not journal_path.is_file():
        raise typer.BadParameter(
            "run directory is missing events.jsonl",
            param_hint="RUN_DIR",
        )
    config: ExperimentConfig | None = None
    if current_qwen:
        config_path = run_dir / "config.yaml"
        if not config_path.is_file():
            raise typer.BadParameter(
                "run directory is missing config.yaml",
                param_hint="RUN_DIR",
            )
        config = load_config(config_path)

    try:
        comparison = asyncio.run(
            _run_policy_shadow(
                config=config,
                journal_path=journal_path,
                limit=limit,
                stride=stride,
                use_current_qwen=current_qwen,
            )
        )
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="RUN_DIR") from error

    target = (output or run_dir / "policy-shadow-comparison.json").expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(comparison.model_dump_json(indent=2) + "\n", encoding="utf-8")
    typer.echo(
        json.dumps(
            [summary.model_dump(mode="json") for summary in comparison.summaries],
            ensure_ascii=False,
            indent=2,
        )
    )
    typer.echo(f"Artifact: {target}")


@app.command()
def serve(
    config_path: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
    socket_path: Annotated[Path | None, typer.Option("--socket")] = None,
    tcp: Annotated[bool, typer.Option(help="Use loopback TCP instead of a Unix socket.")] = False,
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option(min=1, max=65535)] = 8765,
) -> None:
    """Serve the versioned runtime API over a Unix socket or loopback TCP."""

    config = load_config(config_path)
    if socket_path is not None and tcp:
        raise typer.BadParameter("--socket and --tcp cannot be used together")
    run_id, run_dir = _reserve_run_dir(config.run.output_root, "server")
    _snapshot_config(config, run_dir)
    runtime = build_runtime(config, run_dir)
    api = create_app(runtime, manage_runtime_lifecycle=True)
    use_socket = socket_path is not None or (os.name != "nt" and not tcp)
    if use_socket:
        resolved_socket = (
            socket_path.expanduser()
            if socket_path is not None
            else config.run.runtime_root / run_id / "runtime.sock"
        )
        resolved_socket.parent.mkdir(parents=True, exist_ok=True)
        typer.echo(f"Runtime socket: {resolved_socket}")
        uvicorn.run(api, uds=str(resolved_socket), log_level="info")
    else:
        typer.echo(f"Runtime endpoint: http://{host}:{port}")
        uvicorn.run(api, host=host, port=port, log_level="info")


if __name__ == "__main__":
    app()
