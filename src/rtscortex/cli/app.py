"""RTSCortex command-line entry point."""

from __future__ import annotations

import asyncio
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer
import uvicorn
import yaml

from rtscortex.api import create_app
from rtscortex.cli.doctor import run_doctor
from rtscortex.config import ExperimentConfig, load_config
from rtscortex.evaluation import run_mock_episode, run_mock_suite
from rtscortex.evaluation.replay import replay_event_log
from rtscortex.runtime.factory import build_runtime
from rtscortex.runtime.live import (
    LiveEnvironmentError,
    LiveProcessSupervisor,
    LiveWorkerSpec,
    WorkerProcessError,
    live_socket_path,
    prepare_live_worker,
)

app = typer.Typer(no_args_is_help=True, help="RTSCortex agent runtime")
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _run_id(prefix: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{stamp}"


def _snapshot_config(config: ExperimentConfig, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = config.model_dump(mode="json")
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )


@app.command()
def doctor(
    require_sc2: Annotated[
        bool, typer.Option(help="Treat a missing StarCraft II installation as an error.")
    ] = False,
) -> None:
    """Check the core environment, pinned submodule, and optional SC2 installation."""

    checks = run_doctor(PROJECT_ROOT, require_sc2=require_sc2)
    for check in checks:
        typer.echo(f"{check.status.upper():8} {check.name:16} {check.detail}")
    if any(check.status == "error" for check in checks):
        raise typer.Exit(code=1)


@app.command("run")
def run_experiment(
    config_path: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
) -> None:
    """Run one configured episode."""

    config = load_config(config_path)
    run_id = _run_id(config.agent.variant)
    episode_id = "episode-0"
    run_dir = config.run.output_root / run_id
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
            supervisor = LiveProcessSupervisor(
                runtime=runtime,
                run_id=run_id,
                episode_id=episode_id,
                scenario=config.environment.scenario,
                seed=config.run.seed,
                socket_path=runtime_socket,
                worker_command=live_worker.command,
                worker_environment={"SC2PATH": str(live_worker.sc2_path)},
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
        output = asyncio.run(execute())
    except WorkerProcessError as error:
        typer.echo(error.result.model_dump_json(indent=2), err=True)
        typer.echo(f"Artifacts: {run_dir}", err=True)
        raise typer.Exit(code=1) from error
    except LiveEnvironmentError as error:
        typer.echo(f"Live run failed: {error}", err=True)
        typer.echo(f"Artifacts: {run_dir}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(output)
    typer.echo(f"Artifacts: {run_dir}")


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
    run_id = _run_id("server")
    run_dir = config.run.output_root / run_id
    _snapshot_config(config, run_dir)
    runtime = build_runtime(config, run_dir)
    api = create_app(runtime)
    try:
        if socket_path is not None and tcp:
            raise typer.BadParameter("--socket and --tcp cannot be used together")
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
    finally:
        asyncio.run(runtime.close())


if __name__ == "__main__":
    app()
