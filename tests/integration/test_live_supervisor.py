from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from threading import Thread
from typing import Any

import httpx
import pytest
import uvicorn
from rtscortex_llm_pysc2.protocol import RuntimeClient
from typer.testing import CliRunner

import rtscortex.cli.app as cli_module
from rtscortex.api import create_app
from rtscortex.contracts import EpisodeOutcome
from rtscortex.memory import read_event_log
from rtscortex.runtime.factory import build_runtime
from rtscortex.runtime.live import LiveProcessSupervisor, LiveWorkerSpec, WorkerProcessError
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


def _supervisor(tmp_path: Path, worker_code: str) -> LiveProcessSupervisor:
    config = make_config(tmp_path)
    run_dir = tmp_path / "artifacts"
    runtime = build_runtime(config, run_dir)
    return LiveProcessSupervisor(
        runtime=runtime,
        run_id="live-run",
        episode_id="episode-0",
        scenario="pvz_task1_level1",
        seed=7,
        socket_path=tmp_path / "runtime" / "live.sock",
        worker_command=(sys.executable, "-c", worker_code),
        worker_environment={"SC2PATH": str(tmp_path / "StarCraftII")},
        run_dir=run_dir,
        server_ready_timeout_seconds=3.0,
        shutdown_timeout_seconds=3.0,
    )


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
    assert (supervisor.run_dir / "worker.stdout.log").is_file()
    events = list(read_event_log(supervisor.run_dir / "events.jsonl"))
    assert [event.event_type for event in events[-2:]] == [
        "episode_result",
        "episode_summary",
    ]


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
