"""Lifecycle supervision for the isolated LLM-PySC2 environment worker."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import signal
import socket
import subprocess
from collections.abc import Generator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import httpx
import uvicorn
from fastapi import FastAPI

from rtscortex.api import create_app
from rtscortex.config import ExperimentConfig
from rtscortex.console import LiveConsoleHub
from rtscortex.contracts import EpisodeOutcome, EpisodeResult
from rtscortex.runtime.engine import RuntimeEngine

WORKER_AGENT = "rtscortex_llm_pysc2.worker.RTSCortexMainAgent"
PVZ_TASK1_MINIMUM_SC2_BUILD = 92440


class LiveEnvironmentError(RuntimeError):
    """Raised when a live episode cannot safely be started."""


class WorkerProcessError(RuntimeError):
    """Raised after a worker exits unsuccessfully and its result is recorded."""

    def __init__(self, return_code: int, result: EpisodeResult) -> None:
        self.return_code = return_code
        self.result = result
        super().__init__(f"LLM-PySC2 worker exited with status {return_code}")


@dataclass(frozen=True)
class LiveScenarioSpec:
    """SC2 installation requirements for one supported live scenario."""

    map_directory: str
    map_filename: str | None = None
    source_map_required: bool = True
    minimum_sc2_build: int | None = None

    def map_relative_path(self, scenario: str) -> Path:
        """Return the map path below the SC2 ``Maps`` directory."""

        filename = self.map_filename or scenario
        return Path(self.map_directory) / f"{filename}.SC2Map"


LIVE_SCENARIOS = {
    "pvz_task1_level1": LiveScenarioSpec(
        map_directory="llm_pysc2",
        minimum_sc2_build=PVZ_TASK1_MINIMUM_SC2_BUILD,
    ),
    "2s3z": LiveScenarioSpec(map_directory="llm_smac"),
    "Simple64": LiveScenarioSpec(map_directory="Melee", source_map_required=False),
}


@dataclass(frozen=True)
class LiveWorkerSpec:
    """Validated inputs needed to start the pinned environment worker."""

    command: tuple[str, ...]
    sc2_path: Path


class _EmbeddedUvicornServer(uvicorn.Server):
    """Run Uvicorn without replacing the CLI's cancellation handlers."""

    @contextlib.contextmanager
    def capture_signals(self) -> Generator[None, None, None]:
        yield


def prepare_live_worker(
    config: ExperimentConfig,
    project_root: Path,
    *,
    environment: Mapping[str, str] | None = None,
) -> LiveWorkerSpec:
    """Validate live dependencies and build the fixed v0.1 worker command."""

    if config.environment.adapter != "llm_pysc2":
        raise LiveEnvironmentError("live supervision requires environment.adapter=llm_pysc2")
    if os.name == "nt":
        raise LiveEnvironmentError("the v0.1 live worker requires Unix-domain sockets")
    scenario = live_scenario_spec(config.environment.scenario)

    values = os.environ if environment is None else environment
    worker_python = _worker_python(config, values)
    sc2_path = _sc2_path(config, values)
    errors: list[str] = []

    if not worker_python.is_file() or not os.access(worker_python, os.X_OK):
        errors.append(f"worker Python is missing or not executable: {worker_python}")
    else:
        version = subprocess.run(
            [str(worker_python), "--version"],
            check=False,
            capture_output=True,
            text=True,
        )
        version_text = (version.stdout or version.stderr).strip()
        if version.returncode != 0 or not version_text.startswith("Python 3.9."):
            errors.append(f"worker Python must be Python 3.9, found: {version_text or 'unknown'}")
        else:
            probe_environment = dict(values)
            probe_environment["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
            probe = subprocess.run(
                [
                    str(worker_python),
                    "-c",
                    "import pysc2.bin.agent; import rtscortex_llm_pysc2.worker",
                ],
                check=False,
                capture_output=True,
                text=True,
                env=probe_environment,
            )
            if probe.returncode != 0:
                detail = probe.stderr.strip().splitlines()
                errors.append(
                    "worker packages are unavailable" + (f": {detail[-1]}" if detail else "")
                )

    if sc2_path is None:
        errors.append("SC2PATH is unset and environment.sc2_path is not configured")
    else:
        executable = _find_sc2_executable(sc2_path)
        if executable is None:
            errors.append(f"no SC2_x64 executable found below {sc2_path}")
        elif scenario.minimum_sc2_build is not None:
            build = sc2_build(executable)
            if build is None:
                errors.append(f"cannot determine the SC2 build from {executable}")
            elif build < scenario.minimum_sc2_build:
                errors.append(
                    f"SC2 build {build} is older than required build "
                    f"{scenario.minimum_sc2_build} for {config.environment.scenario}"
                )
        map_path = sc2_path / "Maps" / scenario.map_relative_path(config.environment.scenario)
        if not map_path.is_file():
            errors.append(f"scenario map is missing: {map_path}")

    if not waiting_response_patch_is_applied(project_root):
        errors.append(
            "the LLM-PySC2 waiting-response patch is not applied; see "
            "integrations/llm_pysc2/patches/README.md"
        )
    if not random_seed_patch_is_applied(project_root):
        errors.append(
            "the PySC2 random-seed patch is not applied; see "
            "integrations/llm_pysc2/patches/README.md"
        )
    if not build_feature_plane_patch_is_applied(project_root):
        errors.append(
            "the LLM-PySC2 build-coordinate patch is not applied; see "
            "integrations/llm_pysc2/patches/README.md"
        )
    if not translation_result_patch_is_applied(project_root):
        errors.append(
            "the LLM-PySC2 translation-result patch is not applied; see "
            "integrations/llm_pysc2/patches/README.md"
        )
    if not near_placement_patch_is_applied(project_root):
        errors.append(
            "the LLM-PySC2 near-placement patch is not applied; see "
            "integrations/llm_pysc2/patches/README.md"
        )
    if not pretranslation_abort_patch_is_applied(project_root):
        errors.append(
            "the LLM-PySC2 pre-translation abort patch is not applied; see "
            "integrations/llm_pysc2/patches/README.md"
        )
    if not transient_unit_grace_patch_is_applied(project_root):
        errors.append(
            "the LLM-PySC2 transient-unit grace patch is not applied; see "
            "integrations/llm_pysc2/patches/README.md"
        )
    if not nexus_resource_clearance_patch_is_applied(project_root):
        errors.append(
            "the LLM-PySC2 Nexus resource-clearance patch is not applied; see "
            "integrations/llm_pysc2/patches/README.md"
        )
    if not nexus_exact_screen_scale_patch_is_applied(project_root):
        errors.append(
            "the LLM-PySC2 Nexus exact-screen-scale patch is not applied; see "
            "integrations/llm_pysc2/patches/README.md"
        )
    if not max_frames_episode_hook_patch_is_applied(project_root):
        errors.append(
            "the PySC2 max-frame episode hook patch is not applied; see "
            "integrations/llm_pysc2/patches/README.md"
        )
    if not atomic_log_directory_patch_is_applied(project_root):
        errors.append(
            "the LLM-PySC2 concurrent log-directory patch is not applied; see "
            "integrations/llm_pysc2/patches/README.md"
        )
    if not gas_rebalance_worker_management_patch_is_applied(project_root):
        errors.append(
            "the LLM-PySC2 gas-rebalance worker-management patch is not applied; see "
            "integrations/llm_pysc2/patches/README.md"
        )

    if errors:
        raise LiveEnvironmentError("Live environment validation failed:\n- " + "\n- ".join(errors))
    assert sc2_path is not None
    command = [
        str(worker_python),
        "-m",
        "pysc2.bin.agent",
        "--map",
        config.environment.scenario,
        "--agent",
        WORKER_AGENT,
        "--agent_race",
        config.environment.agent_race,
        "--agent2",
        "Bot",
        "--agent2_race",
        config.environment.opponent_race,
        "--difficulty",
        config.environment.opponent_difficulty,
        "--bot_build",
        config.environment.opponent_build,
        "--step_mul",
        str(config.environment.step_mul),
    ]
    if config.environment.game_steps_per_episode is not None:
        command.extend(
            [
                "--game_steps_per_episode",
                str(config.environment.game_steps_per_episode),
            ]
        )
    if config.console.enabled:
        command.extend(
            [
                "--rgb_screen_size",
                str(config.console.rgb_screen_size),
                "--rgb_minimap_size",
                str(config.console.rgb_minimap_size),
                "--action_space",
                "FEATURES",
            ]
        )
    command.extend(
        [
            "--parallel",
            "1",
            "--render=false",
            "--save_replay=false",
            "--max_agent_steps",
            str(config.environment.max_steps),
            "--random_seed",
            str(config.run.seed),
        ]
    )
    return LiveWorkerSpec(command=tuple(command), sc2_path=sc2_path)


def live_scenario_spec(scenario: str) -> LiveScenarioSpec:
    """Return the declared requirements for a supported live scenario."""

    try:
        return LIVE_SCENARIOS[scenario]
    except KeyError as error:
        supported = ", ".join(sorted(LIVE_SCENARIOS))
        raise LiveEnvironmentError(
            f"unsupported live scenario {scenario!r}; supported scenarios: {supported}"
        ) from error


def live_socket_path(runtime_root: Path, run_id: str) -> Path:
    """Return a unique socket path that stays below Linux's UDS length limit."""

    run_key = hashlib.sha256(run_id.encode()).hexdigest()[:16]
    path = runtime_root.expanduser() / f"runtime-{run_key}.sock"
    if len(os.fsencode(path)) > 107:
        raise LiveEnvironmentError(f"runtime socket path exceeds the Unix limit: {path}")
    return path


def ensure_console_port_available(port: int) -> None:
    """Fail before SC2 starts when the loopback console port is occupied."""

    if not 1 <= port <= 65_535:
        raise LiveEnvironmentError(f"invalid console port: {port}")
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", port))
    except OSError as error:
        raise LiveEnvironmentError(
            f"Live Console port 127.0.0.1:{port} is unavailable: {error}"
        ) from error


class LiveProcessSupervisor:
    """Own the API server, worker process, and runtime for one live episode."""

    def __init__(
        self,
        *,
        runtime: RuntimeEngine,
        run_id: str,
        episode_id: str,
        scenario: str,
        seed: int,
        socket_path: Path,
        worker_command: Sequence[str],
        run_dir: Path,
        worker_environment: Mapping[str, str] | None = None,
        console_hub: LiveConsoleHub | None = None,
        console_app: FastAPI | None = None,
        console_port: int | None = None,
        server_ready_timeout_seconds: float = 15.0,
        shutdown_timeout_seconds: float = 10.0,
    ) -> None:
        if not worker_command:
            raise ValueError("worker_command cannot be empty")
        console_values = (console_hub, console_app, console_port)
        if any(value is not None for value in console_values) and not all(
            value is not None for value in console_values
        ):
            raise ValueError("console_hub, console_app, and console_port must be provided together")
        self.runtime = runtime
        self.run_id = run_id
        self.episode_id = episode_id
        self.scenario = scenario
        self.seed = seed
        self.socket_path = socket_path
        self.worker_command = tuple(worker_command)
        self.run_dir = run_dir
        self.worker_environment = dict(worker_environment or {})
        self.console_hub = console_hub
        self.console_app = console_app
        self.console_port = console_port
        self.server_ready_timeout_seconds = server_ready_timeout_seconds
        self.shutdown_timeout_seconds = shutdown_timeout_seconds
        self._server: uvicorn.Server | None = None
        self._server_task: asyncio.Task[None] | None = None
        self._console_server: uvicorn.Server | None = None
        self._console_server_task: asyncio.Task[None] | None = None
        self._worker: asyncio.subprocess.Process | None = None
        self._stdout: BinaryIO | None = None
        self._stderr: BinaryIO | None = None
        self._worker_metrics_path = self.run_dir / "worker.metrics.json"
        self._ownership_path = self.socket_path.with_name(
            f"{self.socket_path.name}.lock"
        )
        self._owns_run = False
        self._owns_runtime_socket = False

    async def run(self) -> EpisodeResult:
        """Run until the worker exits, always closing every owned resource."""

        try:
            self._acquire_run_ownership()
            if self.console_port is not None:
                ensure_console_port_available(self.console_port)
            await self.runtime.start()
            await self._start_server()
            await self._start_console_server()
            if self.console_hub is not None:
                self.console_hub.set_status("running", episode_id=self.episode_id)
            await self._start_worker()
            return_code = await self._wait_for_worker()
            result = self._terminal_result()
            if result is None:
                outcome = EpisodeOutcome.TRUNCATED if return_code == 0 else EpisodeOutcome.ERROR
                result = self._record_synthetic_result(
                    outcome,
                    f"worker exited with status {return_code} before reporting an episode result",
                )
            if self.console_hub is not None:
                if result.outcome is EpisodeOutcome.ERROR:
                    self.console_hub.set_status("failed", episode_id=self.episode_id)
                else:
                    self.console_hub.set_status("completed", episode_id=self.episode_id)
            if return_code != 0:
                raise WorkerProcessError(return_code, result)
            return result
        except asyncio.CancelledError:
            if self.console_hub is not None:
                self.console_hub.set_status("failed", episode_id=self.episode_id)
            await self._stop_worker()
            if self._terminal_result() is None:
                self._record_synthetic_result(
                    EpisodeOutcome.TRUNCATED,
                    "live run was cancelled before the worker reported an episode result",
                )
            raise
        except BaseException as error:
            if self.console_hub is not None:
                self.console_hub.set_status("failed", episode_id=self.episode_id)
            await self._stop_worker()
            if self._terminal_result() is None:
                self._record_synthetic_result(
                    EpisodeOutcome.ERROR,
                    f"live run failed: {type(error).__name__}: {error}",
                )
            raise
        finally:
            try:
                await self._stop_worker()
                await self._stop_console_server()
                await self._stop_server()
            finally:
                try:
                    await self.runtime.close()
                finally:
                    self._release_run_ownership()

    def _acquire_run_ownership(self) -> None:
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(
                self._ownership_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
            )
        except FileExistsError as error:
            raise LiveEnvironmentError(
                f"live run is already owned: {self._ownership_path}"
            ) from error
        try:
            os.write(
                descriptor,
                f"pid={os.getpid()} run_id={self.run_id}\n".encode(),
            )
        finally:
            os.close(descriptor)
        self._owns_run = True
        if self.socket_path.exists():
            raise LiveEnvironmentError(
                f"runtime socket already exists: {self.socket_path}"
            )

    def _release_run_ownership(self) -> None:
        if not self._owns_run:
            return
        with contextlib.suppress(OSError):
            self._ownership_path.unlink()
        self._owns_run = False

    async def _start_server(self) -> None:
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        if self.socket_path.exists():
            raise LiveEnvironmentError(f"runtime socket already exists: {self.socket_path}")
        self._owns_runtime_socket = True
        config = uvicorn.Config(
            create_app(self.runtime, console_hub=self.console_hub),
            uds=str(self.socket_path),
            log_level="warning",
            lifespan="off",
            access_log=False,
            timeout_graceful_shutdown=max(1, round(self.shutdown_timeout_seconds)),
        )
        self._server = _EmbeddedUvicornServer(config)
        self._server_task = asyncio.create_task(self._server.serve())
        await self._wait_until_ready()

    async def _start_console_server(self) -> None:
        if self.console_app is None or self.console_port is None:
            return
        ensure_console_port_available(self.console_port)
        config = uvicorn.Config(
            self.console_app,
            host="127.0.0.1",
            port=self.console_port,
            log_level="warning",
            lifespan="on",
            access_log=False,
            timeout_graceful_shutdown=max(1, round(self.shutdown_timeout_seconds)),
        )
        self._console_server = _EmbeddedUvicornServer(config)
        self._console_server_task = asyncio.create_task(self._console_server.serve())
        await self._wait_until_console_ready()

    async def _wait_until_ready(self) -> None:
        assert self._server_task is not None
        transport = httpx.AsyncHTTPTransport(uds=str(self.socket_path))
        deadline = asyncio.get_running_loop().time() + self.server_ready_timeout_seconds
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://rtscortex",
            timeout=0.5,
        ) as client:
            while asyncio.get_running_loop().time() < deadline:
                if self._server_task.done():
                    await self._server_task
                    raise LiveEnvironmentError("runtime API stopped before becoming ready")
                try:
                    response = await client.get("/healthz")
                    if response.status_code == 200:
                        return
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(0.05)
        raise LiveEnvironmentError(
            f"runtime API did not become ready within {self.server_ready_timeout_seconds}s"
        )

    async def _wait_until_console_ready(self) -> None:
        assert self._console_server_task is not None
        assert self.console_port is not None
        deadline = asyncio.get_running_loop().time() + self.server_ready_timeout_seconds
        async with httpx.AsyncClient(timeout=0.5) as client:
            while asyncio.get_running_loop().time() < deadline:
                if self._console_server_task.done():
                    await self._console_server_task
                    raise LiveEnvironmentError("Live Console stopped before becoming ready")
                try:
                    response = await client.get(
                        f"http://127.0.0.1:{self.console_port}/console/api/v1/health"
                    )
                    if response.status_code == 200:
                        return
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(0.05)
        raise LiveEnvironmentError(
            f"Live Console did not become ready within {self.server_ready_timeout_seconds}s"
        )

    async def _start_worker(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._worker_metrics_path.unlink(missing_ok=True)
        environment = dict(os.environ)
        environment.update(self.worker_environment)
        environment.update(
            {
                "RTSCORTEX_RUNTIME_SOCKET": str(self.socket_path),
                "RTSCORTEX_SOCKET": str(self.socket_path),
                "RTSCORTEX_RUNTIME_URL": "http://rtscortex",
                "RTSCORTEX_RUN_ID": self.run_id,
                "RTSCORTEX_EPISODE_ID": self.episode_id,
                "RTSCORTEX_SCENARIO": self.scenario,
                "RTSCORTEX_SEED": str(self.seed),
                "RTSCORTEX_WORKER_METRICS_PATH": str(self._worker_metrics_path),
                "PYGAME_HIDE_SUPPORT_PROMPT": "1",
            }
        )
        self._stdout = (self.run_dir / "worker.stdout.log").open("wb")
        self._stderr = (self.run_dir / "worker.stderr.log").open("wb")
        try:
            self._worker = await asyncio.create_subprocess_exec(
                *self.worker_command,
                stdout=self._stdout,
                stderr=self._stderr,
                env=environment,
                start_new_session=os.name != "nt",
            )
        except BaseException:
            self._close_worker_logs()
            raise

    async def _wait_for_worker(self) -> int:
        assert self._worker is not None
        assert self._server_task is not None
        worker_wait = asyncio.create_task(self._worker.wait())
        try:
            done, _ = await asyncio.wait(
                {worker_wait, self._server_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if worker_wait in done:
                return worker_wait.result()
            if self._server_task in done:
                await self._server_task
                raise LiveEnvironmentError("runtime API stopped while the worker was running")
            raise RuntimeError("process supervisor wait returned without a completed task")
        finally:
            if not worker_wait.done():
                worker_wait.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await worker_wait

    async def _stop_worker(self) -> None:
        worker = self._worker
        if worker is not None and worker.returncode is None:
            _signal_process(worker, signal.SIGTERM)
            try:
                await asyncio.wait_for(worker.wait(), timeout=self.shutdown_timeout_seconds)
            except TimeoutError:
                _signal_process(worker, signal.SIGKILL)
                await worker.wait()
        self._close_worker_logs()

    async def _stop_server(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._server_task is not None:
            try:
                await asyncio.wait_for(
                    self._server_task,
                    timeout=self.shutdown_timeout_seconds,
                )
            except TimeoutError:
                self._server_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._server_task
            except (Exception, asyncio.CancelledError):
                pass
        if self._owns_runtime_socket:
            self.socket_path.unlink(missing_ok=True)
            self._owns_runtime_socket = False

    async def _stop_console_server(self) -> None:
        if self._console_server is not None:
            self._console_server.should_exit = True
        if self._console_server_task is not None:
            try:
                await asyncio.wait_for(
                    self._console_server_task,
                    timeout=self.shutdown_timeout_seconds,
                )
            except TimeoutError:
                self._console_server_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._console_server_task
            except (Exception, asyncio.CancelledError):
                pass

    def _close_worker_logs(self) -> None:
        if self._stdout is not None:
            self._stdout.close()
            self._stdout = None
        if self._stderr is not None:
            self._stderr.close()
            self._stderr = None

    def _terminal_result(self) -> EpisodeResult | None:
        event = self.runtime.store.last_event(self.run_id, self.episode_id, "episode_result")
        if event is None:
            return None
        return EpisodeResult.model_validate(event.payload)

    def _record_synthetic_result(
        self,
        outcome: EpisodeOutcome,
        failure_reason: str,
    ) -> EpisodeResult:
        observation = self.runtime.store.last_event(
            self.run_id,
            self.episode_id,
            "observation",
        )
        steps = 0 if observation is None else observation.step_id + 1
        result = EpisodeResult(
            run_id=self.run_id,
            episode_id=self.episode_id,
            scenario=self.scenario,
            seed=self.seed,
            outcome=outcome,
            steps=steps,
            metrics=self._worker_metrics(),
            failure_reason=failure_reason,
        )
        self.runtime.end_episode(result)
        return result

    def _worker_metrics(self) -> dict[str, float]:
        if not self._worker_metrics_path.is_file():
            return {}
        payload = json.loads(self._worker_metrics_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise LiveEnvironmentError("worker metrics sidecar must contain a JSON object")
        metrics: dict[str, float] = {}
        for key in ("unattributed_primitives", "candidate_outside_pysc2_dispatches"):
            value = payload.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise LiveEnvironmentError(f"worker metrics sidecar has invalid {key!r}")
            metrics[key] = float(value)
        return metrics


def _worker_python(config: ExperimentConfig, environment: Mapping[str, str]) -> Path:
    override = environment.get("RTSCORTEX_LLM_PYSC2_PYTHON")
    value = Path(override) if override else config.environment.worker_python
    # Keep the virtual-environment interpreter path intact. Resolving this symlink
    # selects the base interpreter and drops the worker environment's site-packages.
    return value.expanduser().absolute()


def _sc2_path(
    config: ExperimentConfig,
    environment: Mapping[str, str],
) -> Path | None:
    value = config.environment.sc2_path
    if value is None:
        raw = environment.get("SC2PATH")
        value = None if raw is None else Path(raw)
    return None if value is None else value.expanduser().resolve()


def _find_sc2_executable(sc2_path: Path) -> Path | None:
    candidates = [
        candidate
        for candidate in (sc2_path / "Versions").glob("Base*/SC2_x64")
        if candidate.is_file() and os.access(candidate, os.X_OK)
    ]
    if candidates:
        return max(candidates, key=lambda candidate: sc2_build(candidate) or -1)
    direct = sc2_path / "SC2_x64"
    return direct if direct.is_file() and os.access(direct, os.X_OK) else None


def sc2_build(executable: Path) -> int | None:
    """Read the numeric SC2 build from a standard ``Versions/Base*`` path."""

    name = executable.parent.name
    if not name.startswith("Base") or not name[4:].isdigit():
        return None
    return int(name[4:])


def waiting_response_patch_is_applied(project_root: Path) -> bool:
    """Return whether the reviewed asynchronous wait hook is present upstream."""

    source = project_root / "third_party/LLM-PySC2/llm_pysc2/agents/llm_pysc2_agent_main.py"
    if not source.is_file():
        return False
    text = source.read_text(encoding="utf-8")
    branch_start = text.find("elif not self._all_agent_waiting_response_finished():")
    if branch_start < 0:
        return False
    branch_end = text.find("elif not self._all_agent_executing_finished():", branch_start)
    branch = text[branch_start:branch_end]
    return "actions.FUNCTIONS.no_op()" in branch and "return func_call" in branch


def random_seed_patch_is_applied(project_root: Path) -> bool:
    """Return whether the PySC2 runner forwards its seed into ``SC2Env``."""

    source = project_root / "third_party/LLM-PySC2/pysc2/bin/agent.py"
    if not source.is_file():
        return False
    text = source.read_text(encoding="utf-8")
    return (
        'flags.DEFINE_integer("random_seed", None' in text
        and "random_seed=FLAGS.random_seed" in text
    )


def build_feature_plane_patch_is_applied(project_root: Path) -> bool:
    """Return whether build validation reads PySC2 planes as ``[y][x]``."""

    source = project_root / "third_party/LLM-PySC2/llm_pysc2/lib/llm_action.py"
    if not source.is_file():
        return False
    text = source.read_text(encoding="utf-8")
    return all(
        marker in text
        for marker in (
            "feature_screen.power[y0][x0]",
            "feature_screen.creep[y0][x0]",
            "feature_screen.buildable[y][x]",
            "feature_screen.pathable[y][x]",
            "feature_screen.player_relative[y][x]",
        )
    )


def translation_result_patch_is_applied(project_root: Path) -> bool:
    """Return whether translator attempts expose structured primitive provenance."""

    source = project_root / "third_party/LLM-PySC2/llm_pysc2/agents/llm_pysc2_agent.py"
    if not source.is_file():
        return False
    text = source.read_text(encoding="utf-8")
    return all(
        marker in text
        for marker in (
            "self.last_translation_result",
            "'requested_function_id': requested_function_id",
            "'emitted_function_id': func_id",
            "'ordinal': translation_ordinal",
            "'total': self._rtscortex_translation_total",
            "all_args_valid = all_args_valid and func_valid",
        )
    )


def near_placement_patch_is_applied(project_root: Path) -> bool:
    """Return whether Near building helpers use exact anchors and full footprints."""

    source = project_root / "third_party/LLM-PySC2/llm_pysc2/lib/llm_action.py"
    main_source = project_root / "third_party/LLM-PySC2/llm_pysc2/agents/llm_pysc2_agent_main.py"
    if not source.is_file() or not main_source.is_file():
        return False
    text = source.read_text(encoding="utf-8") + main_source.read_text(encoding="utf-8")
    return all(
        marker in text
        for marker in (
            "(0, F.no_op, ())",
            "if unit.tag == tag:",
            "not unit.is_on_screen or not (0 < unit.x < size_screen",
            "is not a neutral resource anchor",
            "def full_footprint_valid(center_x, center_y):",
            "feature_screen.player_relative[y][x] != 0",
            "complete footprint",
            "return translator settlement no_op",
        )
    )


def pretranslation_abort_patch_is_applied(project_root: Path) -> bool:
    """Return whether upstream reports command-owned pre-translation aborts."""

    agent_source = project_root / "third_party/LLM-PySC2/llm_pysc2/agents/llm_pysc2_agent.py"
    main_source = project_root / "third_party/LLM-PySC2/llm_pysc2/agents/llm_pysc2_agent_main.py"
    if not agent_source.is_file() or not main_source.is_file():
        return False
    text = agent_source.read_text(encoding="utf-8") + main_source.read_text(encoding="utf-8")
    return all(
        marker in text
        for marker in (
            "agent.last_execution_abort = {",
            "'failure_code': 'actor_not_available'",
            "team head unit is unavailable before action translation",
        )
    )


def transient_unit_grace_patch_is_applied(project_root: Path) -> bool:
    """Return whether transient gas/transport absence uses confirmed death state."""

    agent_source = project_root / "third_party/LLM-PySC2/llm_pysc2/agents/llm_pysc2_agent.py"
    main_source = project_root / "third_party/LLM-PySC2/llm_pysc2/agents/llm_pysc2_agent_main.py"
    funcs_source = project_root / "third_party/LLM-PySC2/llm_pysc2/agents/main_agent_funcs.py"
    if not agent_source.is_file() or not main_source.is_file() or not funcs_source.is_file():
        return False
    agent_text = agent_source.read_text(encoding="utf-8")
    main_text = main_source.read_text(encoding="utf-8")
    funcs_text = funcs_source.read_text(encoding="utf-8")
    return (
        all(
            marker in agent_text
            for marker in (
                "MainAgent confirms unit death across observations",
                "team['unit_tags'] = list(dict.fromkeys(team['unit_tags']))",
            )
        )
        and all(
            marker in main_text
            for marker in (
                "if tag not in self.unit_uid_disappear:",
                "agent.curr_action_name != 'No_Operation'",
                "wait for confirmed disappearance",
                "keep the action pending",
            )
        )
        and all(
            marker in funcs_text
            for marker in (
                "Advance confirmed-death state on every observation",
                "tag for tag, steps in self.unit_disappear_steps.items() if steps >= 40",
                "tag for tag in agent.unit_tag_list if tag not in self.unit_uid_disappear",
                "tag for tag in team['unit_tags'] if tag not in self.unit_uid_disappear",
            )
        )
    )


def nexus_resource_clearance_patch_is_applied(project_root: Path) -> bool:
    """Return whether Nexus placement enforces resource-ring geometry."""

    source = project_root / "third_party/LLM-PySC2/llm_pysc2/lib/llm_action.py"
    if not source.is_file():
        return False
    text = source.read_text(encoding="utf-8")
    return all(
        marker in text
        for marker in (
            "def resource_clearance_score(center_x, center_y, resources):",
            "has no complete footprint with valid resource clearance",
            "candidates.append((clearance_score, centroid_distance, candidate_x, candidate_y))",
        )
    )


def nexus_exact_screen_scale_patch_is_applied(project_root: Path) -> bool:
    """Return whether Nexus screen geometry uses exact scale and current visibility."""

    source = project_root / "third_party/LLM-PySC2/llm_pysc2/lib/llm_action.py"
    if not source.is_file():
        return False
    text = source.read_text(encoding="utf-8")
    return all(
        marker in text
        for marker in (
            "pixel_scale = size_screen / SCREEN_WORLD_GRID",
            "sample_stride = max(1, int(pixel_scale))",
            "minimum, ideal, maximum = 7 * pixel_scale, 8.5 * pixel_scale, 10 * pixel_scale",
            "minimum, ideal, maximum = 6 * pixel_scale, 7.5 * pixel_scale, 9 * pixel_scale",
            "feature_screen.visibility_map[y][x] != features.Visibility.VISIBLE",
            "unit.display_type != 1 or not unit.is_on_screen",
        )
    )


def max_frames_episode_hook_patch_is_applied(project_root: Path) -> bool:
    """Return whether PySC2 reports a clean max-frame truncation to the agent."""

    source = project_root / "third_party/LLM-PySC2/pysc2/env/run_loop.py"
    if not source.is_file():
        return False
    text = source.read_text(encoding="utf-8")
    return all(
        marker in text
        for marker in (
            'getattr(agent, "on_episode_truncated", None)',
            "on_episode_truncated(total_frames)",
        )
    )


def atomic_log_directory_patch_is_applied(project_root: Path) -> bool:
    """Return whether concurrent workers allocate upstream log directories atomically."""

    source = project_root / "third_party/LLM-PySC2/llm_pysc2/agents/llm_pysc2_agent_main.py"
    if not source.is_file():
        return False
    text = source.read_text(encoding="utf-8")
    return all(
        marker in text
        for marker in (
            "except FileExistsError:",
            "llm_pysc2_global_log_id = max(llm_pysc2_global_log_id, self.log_id)",
        )
    )


def gas_rebalance_worker_management_patch_is_applied(project_root: Path) -> bool:
    """Return whether gas rebalancing is governed by the worker-management flag."""

    source = project_root / "third_party/LLM-PySC2/llm_pysc2/agents/main_agent_funcs.py"
    if not source.is_file():
        return False
    text = source.read_text(encoding="utf-8")
    return (
        "self.config.ENABLE_AUTO_WORKER_MANAGE and self.is_all_nexus_full is False"
        in text
    )


def reserved_builder_worker_patch_is_applied(project_root: Path) -> bool:
    """Return whether automatic worker assignment preserves Builder actors."""

    source = project_root / "third_party/LLM-PySC2/llm_pysc2/agents/main_agent_funcs.py"
    if not source.is_file():
        return False
    text = source.read_text(encoding="utf-8")
    return all(
        marker in text
        for marker in (
            "_rtscortex_reserved_worker_tags",
            "HoldPosition_quick('now')",
            "Reserved worker",
        )
    )


def _signal_process(worker: asyncio.subprocess.Process, sig: signal.Signals) -> None:
    with contextlib.suppress(ProcessLookupError):
        if os.name != "nt":
            os.killpg(worker.pid, sig)
        elif sig is signal.SIGTERM:
            worker.terminate()
        else:
            worker.kill()
