from __future__ import annotations

import asyncio
import json
import os
import socket
import sys
import time
from pathlib import Path
from threading import Thread
from typing import Any

import httpx
import pytest
import uvicorn
import yaml
from rtscortex_llm_pysc2.protocol import RuntimeClient
from typer.testing import CliRunner

import rtscortex.cli.app as cli_module
from rtscortex.api import create_app
from rtscortex.console import ConsoleSession, LiveConsoleHub, create_console_app
from rtscortex.contracts import EpisodeOutcome
from rtscortex.memory import EventStore, read_event_log
from rtscortex.runtime.factory import build_runtime
from rtscortex.runtime.live import (
    LiveEnvironmentError,
    LiveProcessSupervisor,
    LiveWorkerSpec,
    WorkerProcessError,
    ensure_console_port_available,
)
from tests.helpers import make_config

SUCCESS_WORKER = """
import os
import httpx

socket_path = os.environ["RTSCORTEX_RUNTIME_SOCKET"]
assert socket_path == os.environ["RTSCORTEX_SOCKET"]
transport = httpx.HTTPTransport(uds=socket_path)
with httpx.Client(transport=transport, base_url=os.environ["RTSCORTEX_RUNTIME_URL"]) as client:
    health = client.get("/healthz")
    health.raise_for_status()
    response = client.post(
        "/v1/episode/end",
        json={
            "protocol_version": "1.1",
            "run_id": os.environ["RTSCORTEX_RUN_ID"],
            "episode_id": os.environ["RTSCORTEX_EPISODE_ID"],
            "scenario": os.environ["RTSCORTEX_SCENARIO"],
            "seed": int(os.environ["RTSCORTEX_SEED"]),
            "outcome": "victory",
            "score": 1.0,
            "steps": 3,
            "metrics": {},
            "failure_reason": None,
        },
    )
    response.raise_for_status()
"""

HEALTH_WORKER = """
import os
import httpx

transport = httpx.HTTPTransport(uds=os.environ["RTSCORTEX_RUNTIME_SOCKET"])
with httpx.Client(transport=transport, base_url="http://rtscortex") as client:
    response = client.get("/healthz")
    response.raise_for_status()
"""

METRICS_THEN_EXIT_WORKER = """
import json
import os
from pathlib import Path

Path(os.environ["RTSCORTEX_WORKER_METRICS_PATH"]).write_text(
    json.dumps({
        "unattributed_primitives": 2,
        "candidate_outside_pysc2_dispatches": 1,
    }),
    encoding="utf-8",
)
raise SystemExit(7)
"""

TICK_THEN_EXIT_WORKER = """
import os
import httpx

transport = httpx.HTTPTransport(uds=os.environ["RTSCORTEX_RUNTIME_SOCKET"])
with httpx.Client(transport=transport, base_url="http://rtscortex") as client:
    response = client.post(
        "/v1/tick",
        json={
            "protocol_version": "1.1",
            "run_id": os.environ["RTSCORTEX_RUN_ID"],
            "episode_id": os.environ["RTSCORTEX_EPISODE_ID"],
            "step_id": 0,
            "game_loop": 0,
            "state": {
                "economy": {},
                "production_queue": [],
                "own_units": [],
                "own_structures": [],
                "visible_enemies": [
                    {
                        "unit_id": "0x1",
                        "unit_type": "Zergling",
                        "alliance": "enemy",
                    }
                ],
                "upgrades": [],
            },
            "available_actions": [
                {
                    "name": "Attack_Unit",
                    "argument_names": ["target"],
                    "argument_types": ["tag"],
                    "actor_scopes": ["army"],
                    "argument_candidates": [["0x1"]],
                }
            ],
        },
    )
    response.raise_for_status()
    assert len(response.json()["commands"]) == 1
"""

PACED_SUCCESS_WORKER = (
    """
import os

assert os.environ["RTSCORTEX_SIMULATION_SPEED_MULTIPLIER"] == "0.25"
assert os.environ["RTSCORTEX_PAUSE_UNTIL_FIRST_PLAN"] == "true"
assert os.environ["RTSCORTEX_ACTION_EFFECT_TIMEOUT_GAME_LOOPS"] == "96"
"""
    + SUCCESS_WORKER
)

CONSOLE_SUCCESS_WORKER = (
    """
import os

assert os.environ["RTSCORTEX_CONSOLE_ENABLED"] == "true"
assert os.environ["RTSCORTEX_CONSOLE_FRAME_FPS"] == "2.0"
assert os.environ["RTSCORTEX_CONSOLE_JPEG_QUALITY"] == "75"
assert os.environ["RTSCORTEX_CONSOLE_RGB_SCREEN_SIZE"] == "256"
assert os.environ["RTSCORTEX_CONSOLE_RGB_MINIMAP_SIZE"] == "128"
"""
    + SUCCESS_WORKER
)


def _supervisor(
    tmp_path: Path,
    worker_code: str,
    *,
    console_port: int | None = None,
) -> LiveProcessSupervisor:
    config = make_config(tmp_path)
    run_dir = tmp_path / "artifacts"
    runtime = build_runtime(config, run_dir)
    console_hub = None
    console_app = None
    if console_port is not None:
        session = ConsoleSession(
            run_id="live-run",
            episode_id="episode-0",
            status="starting",
            scenario="pvz_task1_level1",
            seed=7,
            model=config.provider.model,
        )
        console_hub = LiveConsoleHub(session)
        console_app = create_console_app(runtime.store, session, console_hub)
    return LiveProcessSupervisor(
        runtime=runtime,
        run_id="live-run",
        episode_id="episode-0",
        scenario="pvz_task1_level1",
        seed=7,
        socket_path=tmp_path / "runtime" / "live.sock",
        worker_command=(sys.executable, "-c", worker_code),
        worker_environment={"SC2PATH": str(tmp_path / "StarCraftII")},
        console_hub=console_hub,
        console_app=console_app,
        console_port=console_port,
        run_dir=run_dir,
        server_ready_timeout_seconds=3.0,
        shutdown_timeout_seconds=3.0,
    )


def _unused_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _start_uds_server(app: Any, socket_path: Path) -> tuple[uvicorn.Server, Thread]:
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    server = uvicorn.Server(
        uvicorn.Config(app, uds=str(socket_path), log_level="error", access_log=False)
    )
    thread = Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 3
    while not server.started and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert server.started
    return server, thread


def _stop_uds_server(server: uvicorn.Server, thread: Thread, socket_path: Path) -> None:
    server.should_exit = True
    thread.join(timeout=3)
    assert not thread.is_alive()
    socket_path.unlink(missing_ok=True)


def test_supervisor_waits_for_server_and_returns_reported_result(tmp_path: Path) -> None:
    supervisor = _supervisor(tmp_path, SUCCESS_WORKER)

    result = asyncio.run(supervisor.run())

    assert result.outcome is EpisodeOutcome.VICTORY
    assert result.steps == 3
    assert not supervisor.socket_path.exists()
    assert not supervisor._ownership_path.exists()
    assert (supervisor.run_dir / "worker.stdout.log").is_file()
    events = list(read_event_log(supervisor.run_dir / "events.jsonl"))
    assert [event.event_type for event in events[-2:]] == [
        "episode_result",
        "episode_summary",
    ]


def test_supervisor_ownership_collision_fails_before_runtime_start(
    tmp_path: Path,
) -> None:
    supervisor = _supervisor(tmp_path, SUCCESS_WORKER)
    supervisor._ownership_path.parent.mkdir(parents=True, exist_ok=True)
    supervisor._ownership_path.write_text("pid=someone-else\n", encoding="utf-8")
    started = False
    original_start = supervisor.runtime.start

    async def tracked_start() -> None:
        nonlocal started
        started = True
        await original_start()

    supervisor.runtime.start = tracked_start  # type: ignore[method-assign]

    with pytest.raises(LiveEnvironmentError, match="already owned"):
        asyncio.run(supervisor.run())

    assert started is False
    assert supervisor._ownership_path.read_text(encoding="utf-8") == "pid=someone-else\n"


def test_supervisor_serves_read_only_console_without_exposing_control_api(
    tmp_path: Path,
) -> None:
    port = _unused_loopback_port()
    supervisor = _supervisor(
        tmp_path,
        SUCCESS_WORKER + "\nimport time\ntime.sleep(0.5)\n",
        console_port=port,
    )

    async def execute() -> None:
        task = asyncio.create_task(supervisor.run())
        async with httpx.AsyncClient(timeout=1.0) as client:
            for _ in range(100):
                try:
                    health = await client.get(f"http://127.0.0.1:{port}/console/api/v1/health")
                    if health.status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(0.01)
            else:
                raise AssertionError("Live Console did not become reachable")
            assert health.json()["read_only"] is True
            control = await client.post(f"http://127.0.0.1:{port}/v1/tick", json={})
            assert control.status_code == 404
        result = await task
        assert result.outcome is EpisodeOutcome.VICTORY

    asyncio.run(execute())

    assert supervisor.console_hub is not None
    assert supervisor.console_hub.session().status == "completed"


def test_console_server_stopping_does_not_stop_worker_or_runtime(tmp_path: Path) -> None:
    port = _unused_loopback_port()
    supervisor = _supervisor(
        tmp_path,
        SUCCESS_WORKER + "\nimport time\ntime.sleep(1.0)\n",
        console_port=port,
    )

    async def execute() -> None:
        run_task = asyncio.create_task(supervisor.run())
        async with httpx.AsyncClient(timeout=1.0) as client:
            for _ in range(100):
                try:
                    response = await client.get(f"http://127.0.0.1:{port}/console/api/v1/health")
                    if response.status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(0.01)
            else:
                raise AssertionError("Live Console did not become reachable")

        assert supervisor._console_server is not None
        assert supervisor._console_server_task is not None
        supervisor._console_server.should_exit = True
        await asyncio.wait_for(supervisor._console_server_task, timeout=2.0)
        assert run_task.done() is False

        result = await run_task
        assert result.outcome is EpisodeOutcome.VICTORY

    asyncio.run(execute())


def test_console_port_preflight_rejects_an_occupied_loopback_port() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        port = int(listener.getsockname()[1])
        with pytest.raises(LiveEnvironmentError, match="Live Console port"):
            ensure_console_port_available(port)


def test_console_port_preflight_reuses_a_recently_closed_listener() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        port = int(listener.getsockname()[1])

    ensure_console_port_available(port)


@pytest.mark.skipif(os.name == "nt", reason="Unix-domain socket test")
def test_runtime_client_reconnects_after_unix_socket_restart(tmp_path: Path) -> None:
    runtime = build_runtime(make_config(tmp_path), tmp_path / "reconnect-artifacts")
    socket_path = tmp_path / "runtime" / "reconnect.sock"
    app = create_app(runtime)
    client = RuntimeClient(unix_socket=str(socket_path))
    first_server, first_thread = _start_uds_server(app, socket_path)

    try:
        assert client.health()["status"] == "ok"
        _stop_uds_server(first_server, first_thread, socket_path)
        with pytest.raises(httpx.TransportError):
            client.health()

        second_server, second_thread = _start_uds_server(app, socket_path)
        try:
            assert client.health()["status"] == "ok"
        finally:
            _stop_uds_server(second_server, second_thread, socket_path)
    finally:
        client.close()
        if first_thread.is_alive():
            _stop_uds_server(first_server, first_thread, socket_path)
        asyncio.run(runtime.close())


def test_supervisor_records_truncation_when_clean_worker_has_no_result(tmp_path: Path) -> None:
    supervisor = _supervisor(tmp_path, HEALTH_WORKER)

    result = asyncio.run(supervisor.run())

    assert result.outcome is EpisodeOutcome.TRUNCATED
    assert result.failure_reason == (
        "worker exited with status 0 before reporting an episode result"
    )
    assert not supervisor.socket_path.exists()


def test_supervisor_synthetic_result_cancels_dispatched_command(tmp_path: Path) -> None:
    supervisor = _supervisor(tmp_path, TICK_THEN_EXIT_WORKER)

    result = asyncio.run(supervisor.run())

    assert result.outcome is EpisodeOutcome.TRUNCATED
    events = list(read_event_log(supervisor.run_dir / "events.jsonl"))
    executions = [event.payload for event in events if event.event_type == "execution"]
    assert len(executions) == 1
    assert executions[0]["status"] == "cancelled"
    assert executions[0]["execution_stage"] == "episode_end"
    assert executions[0]["failure_code"] == ("worker_terminated_before_execution_report")


def test_supervisor_records_error_and_cleans_up_after_nonzero_exit(tmp_path: Path) -> None:
    supervisor = _supervisor(tmp_path, "raise SystemExit(7)")

    with pytest.raises(WorkerProcessError) as captured:
        asyncio.run(supervisor.run())

    assert captured.value.return_code == 7
    assert captured.value.result.outcome is EpisodeOutcome.ERROR
    assert not supervisor.socket_path.exists()
    events = list(read_event_log(supervisor.run_dir / "events.jsonl"))
    result_events = [event for event in events if event.event_type == "episode_result"]
    assert result_events[-1].payload["outcome"] == "error"


def test_supervisor_preserves_worker_metrics_in_synthetic_result(tmp_path: Path) -> None:
    supervisor = _supervisor(tmp_path, METRICS_THEN_EXIT_WORKER)

    with pytest.raises(WorkerProcessError) as captured:
        asyncio.run(supervisor.run())

    assert captured.value.result.metrics == {
        "unattributed_primitives": 2.0,
        "candidate_outside_pysc2_dispatches": 1.0,
    }
    events = list(read_event_log(supervisor.run_dir / "events.jsonl"))
    result = next(event.payload for event in events if event.event_type == "episode_result")
    assert result["metrics"] == {
        "unattributed_primitives": 2.0,
        "candidate_outside_pysc2_dispatches": 1.0,
    }


def test_supervisor_records_truncation_and_stops_worker_when_cancelled(tmp_path: Path) -> None:
    worker_code = """
import os
import time
from pathlib import Path

Path(os.environ["TEST_MARKER"]).touch()
time.sleep(30)
"""
    supervisor = _supervisor(tmp_path, worker_code)
    marker = tmp_path / "worker-started"
    supervisor.worker_environment["TEST_MARKER"] = str(marker)

    async def execute() -> None:
        task = asyncio.create_task(supervisor.run())
        for _ in range(100):
            if marker.exists():
                break
            await asyncio.sleep(0.02)
        assert marker.exists()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(execute())

    assert not supervisor.socket_path.exists()
    events = list(read_event_log(supervisor.run_dir / "events.jsonl"))
    result_events = [event for event in events if event.event_type == "episode_result"]
    assert result_events[-1].payload["outcome"] == "truncated"


def test_run_command_uses_live_supervisor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "live.yaml"
    runtime_root = Path("/tmp") / f"rtscortex-test-{os.getpid()}"
    config_path.write_text(
        f"""
run:
  output_root: {tmp_path / "outputs"}
  runtime_root: {runtime_root}
  seed: 7
environment:
  adapter: llm_pysc2
  scenario: pvz_task1_level1
  max_steps: 4
  simulation_speed_multiplier: 0.25
  pause_until_first_plan: true
  action_effect_timeout_game_loops: 96
provider:
  kind: fake
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        cli_module,
        "prepare_live_worker",
        lambda config, project_root: LiveWorkerSpec(
            command=(sys.executable, "-c", PACED_SUCCESS_WORKER),
            sc2_path=tmp_path / "StarCraftII",
        ),
    )

    result = CliRunner().invoke(
        cli_module.app,
        ["run", "--config", str(config_path), "--seed", "9"],
    )

    assert result.exit_code == 0, result.output
    assert '"outcome": "victory"' in result.output
    assert '"seed": 9' in result.output
    assert "Artifacts:" in result.output
    run_dirs = list((tmp_path / "outputs").iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    episode = next(iter(summary["runs"].values()))["episodes"]["episode-0"]
    assert (run_dir / "timeline.md").is_file()
    assert episode["result"]["outcome"] == "victory"
    assert episode["result"]["seed"] == 9
    assert episode["result"]["steps"] == 3
    assert f"Timeline: {run_dir / 'timeline.md'}" in result.output
    assert f"Summary: {run_dir / 'summary.json'}" in result.output
    runtime_root.rmdir()


def test_run_command_enables_console_and_snapshots_effective_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "live-console.yaml"
    runtime_root = Path("/tmp") / f"rtscortex-console-test-{os.getpid()}"
    config_path.write_text(
        f"""
run:
  output_root: {tmp_path / "outputs"}
  runtime_root: {runtime_root}
environment:
  adapter: llm_pysc2
  scenario: Simple64
provider:
  kind: fake
""",
        encoding="utf-8",
    )
    static_dir = tmp_path / "static"
    static_dir.mkdir()
    (static_dir / "index.html").write_text("<html>console</html>", encoding="utf-8")
    monkeypatch.setattr(cli_module, "CONSOLE_STATIC_DIR", static_dir)
    port = _unused_loopback_port()

    def prepare(config: Any, project_root: Path) -> LiveWorkerSpec:
        del project_root
        assert config.console.enabled is True
        assert config.console.port == port
        return LiveWorkerSpec(
            command=(sys.executable, "-c", CONSOLE_SUCCESS_WORKER),
            sc2_path=tmp_path / "StarCraftII",
        )

    monkeypatch.setattr(cli_module, "prepare_live_worker", prepare)

    result = CliRunner().invoke(
        cli_module.app,
        [
            "run",
            "--config",
            str(config_path),
            "--console",
            "--console-port",
            str(port),
        ],
    )

    assert result.exit_code == 0, result.output
    assert f"Live Console: http://127.0.0.1:{port}" in result.output
    run_dir = next((tmp_path / "outputs").iterdir())
    snapshot = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    assert snapshot["console"]["enabled"] is True
    assert snapshot["console"]["port"] == port
    runtime_root.rmdir()


def test_console_command_serves_completed_event_history_read_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "completed-run"
    run_dir.mkdir()
    (run_dir / "config.yaml").write_text(
        "environment:\n  scenario: Simple64\nprovider:\n  kind: fake\n",
        encoding="utf-8",
    )
    store = EventStore(run_dir / "events.sqlite3", run_dir / "events.jsonl")
    store.append_event(
        run_id="completed-run",
        episode_id="episode-0",
        step_id=0,
        event_type="observation",
        payload={"game_loop": 0},
    )
    store.close()
    static_dir = tmp_path / "static-history"
    static_dir.mkdir()
    (static_dir / "index.html").write_text("<html>console</html>", encoding="utf-8")
    monkeypatch.setattr(cli_module, "CONSOLE_STATIC_DIR", static_dir)
    captured: dict[str, Any] = {}

    def run(app: Any, *, host: str, port: int, log_level: str) -> None:
        captured.update(app=app, host=host, port=port, log_level=log_level)

    monkeypatch.setattr("rtscortex.cli.app.uvicorn.run", run)
    port = _unused_loopback_port()

    result = CliRunner().invoke(
        cli_module.app,
        ["console", str(run_dir), "--port", str(port)],
    )

    assert result.exit_code == 0, result.output
    assert f"Historical Console: http://127.0.0.1:{port}" in result.output
    assert "RGB history is unavailable" in result.output
    assert captured["host"] == "127.0.0.1"
    routes = {route.path for route in captured["app"].routes}
    assert "/console/api/v1/events" in routes
    assert "/v1/tick" not in routes


def test_run_command_writes_reports_after_worker_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "live-error.yaml"
    runtime_root = Path("/tmp") / f"rtscortex-error-test-{os.getpid()}"
    config_path.write_text(
        f"""
run:
  output_root: {tmp_path / "outputs"}
  runtime_root: {runtime_root}
  seed: 7
environment:
  adapter: llm_pysc2
  scenario: pvz_task1_level1
  max_steps: 4
provider:
  kind: fake
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        cli_module,
        "prepare_live_worker",
        lambda config, project_root: LiveWorkerSpec(
            command=(sys.executable, "-c", "raise SystemExit(7)"),
            sc2_path=tmp_path / "StarCraftII",
        ),
    )

    result = CliRunner().invoke(cli_module.app, ["run", "--config", str(config_path)])

    assert result.exit_code == 1
    assert "worker exited with status 7 before reporting an episode result" in result.output
    run_dirs = list((tmp_path / "outputs").iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    episode = next(iter(summary["runs"].values()))["episodes"]["episode-0"]
    assert (run_dir / "timeline.md").is_file()
    assert episode["result"]["outcome"] == "error"
    assert episode["result"]["failure_reason"] == (
        "worker exited with status 7 before reporting an episode result"
    )
    assert f"Timeline: {run_dir / 'timeline.md'}" in result.output
    assert f"Summary: {run_dir / 'summary.json'}" in result.output
    runtime_root.rmdir()
