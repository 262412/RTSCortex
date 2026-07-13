"""Lifecycle supervision for the isolated LLM-PySC2 environment worker."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import os
import signal
import subprocess
from collections.abc import Generator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import httpx
import uvicorn

from rtscortex.api import create_app
from rtscortex.config import ExperimentConfig
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
        elif config.environment.scenario == "pvz_task1_level1":
            build = sc2_build(executable)
            if build is None:
                errors.append(f"cannot determine the SC2 build from {executable}")
            elif build < PVZ_TASK1_MINIMUM_SC2_BUILD:
                errors.append(
                    f"SC2 build {build} is older than required build "
                    f"{PVZ_TASK1_MINIMUM_SC2_BUILD} for pvz_task1_level1"
                )
        map_path = sc2_path / "Maps" / "llm_pysc2" / f"{config.environment.scenario}.SC2Map"
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

    if errors:
        raise LiveEnvironmentError("Live environment validation failed:\n- " + "\n- ".join(errors))
    assert sc2_path is not None
    command = (
        str(worker_python),
        "-m",
        "pysc2.bin.agent",
        "--map",
        config.environment.scenario,
        "--agent",
        WORKER_AGENT,
        "--agent_race",
        "protoss",
        "--parallel",
        "1",
        "--render=false",
        "--save_replay=false",
        "--max_agent_steps",
        str(config.environment.max_steps),
        "--random_seed",
        str(config.run.seed),
    )
    return LiveWorkerSpec(command=command, sc2_path=sc2_path)


def live_socket_path(runtime_root: Path, run_id: str) -> Path:
    """Return a unique socket path that stays below Linux's UDS length limit."""

    run_key = hashlib.sha256(run_id.encode()).hexdigest()[:16]
    path = runtime_root.expanduser() / f"runtime-{run_key}.sock"
    if len(os.fsencode(path)) > 107:
        raise LiveEnvironmentError(f"runtime socket path exceeds the Unix limit: {path}")
    return path


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
        server_ready_timeout_seconds: float = 15.0,
        shutdown_timeout_seconds: float = 10.0,
    ) -> None:
        if not worker_command:
            raise ValueError("worker_command cannot be empty")
        self.runtime = runtime
        self.run_id = run_id
        self.episode_id = episode_id
        self.scenario = scenario
        self.seed = seed
        self.socket_path = socket_path
        self.worker_command = tuple(worker_command)
        self.run_dir = run_dir
        self.worker_environment = dict(worker_environment or {})
        self.server_ready_timeout_seconds = server_ready_timeout_seconds
        self.shutdown_timeout_seconds = shutdown_timeout_seconds
        self._server: uvicorn.Server | None = None
        self._server_task: asyncio.Task[None] | None = None
        self._worker: asyncio.subprocess.Process | None = None
        self._stdout: BinaryIO | None = None
        self._stderr: BinaryIO | None = None

    async def run(self) -> EpisodeResult:
        """Run until the worker exits, always closing every owned resource."""

        try:
            await self._start_server()
            await self._start_worker()
            return_code = await self._wait_for_worker()
            result = self._terminal_result()
            if result is None:
                outcome = EpisodeOutcome.TRUNCATED if return_code == 0 else EpisodeOutcome.ERROR
                result = self._record_synthetic_result(
                    outcome,
                    f"worker exited with status {return_code} before reporting an episode result",
                )
            if return_code != 0:
                raise WorkerProcessError(return_code, result)
            return result
        except asyncio.CancelledError:
            await self._stop_worker()
            if self._terminal_result() is None:
                self._record_synthetic_result(
                    EpisodeOutcome.TRUNCATED,
                    "live run was cancelled before the worker reported an episode result",
                )
            raise
        except BaseException as error:
            await self._stop_worker()
            if self._terminal_result() is None:
                self._record_synthetic_result(
                    EpisodeOutcome.ERROR,
                    f"live run failed: {type(error).__name__}: {error}",
                )
            raise
        finally:
            await self._stop_worker()
            await self._stop_server()
            await self.runtime.close()

    async def _start_server(self) -> None:
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        if self.socket_path.exists():
            raise LiveEnvironmentError(f"runtime socket already exists: {self.socket_path}")
        config = uvicorn.Config(
            create_app(self.runtime),
            uds=str(self.socket_path),
            log_level="warning",
            lifespan="off",
            access_log=False,
            timeout_graceful_shutdown=max(1, round(self.shutdown_timeout_seconds)),
        )
        self._server = _EmbeddedUvicornServer(config)
        self._server_task = asyncio.create_task(self._server.serve())
        await self._wait_until_ready()

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

    async def _start_worker(self) -> None:
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
                "PYGAME_HIDE_SUPPORT_PROMPT": "1",
            }
        )
        self.run_dir.mkdir(parents=True, exist_ok=True)
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
        self.socket_path.unlink(missing_ok=True)

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
            failure_reason=failure_reason,
        )
        self.runtime.end_episode(result)
        return result


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


def _signal_process(worker: asyncio.subprocess.Process, sig: signal.Signals) -> None:
    with contextlib.suppress(ProcessLookupError):
        if os.name != "nt":
            os.killpg(worker.pid, sig)
        elif sig is signal.SIGTERM:
            worker.terminate()
        else:
            worker.kill()
