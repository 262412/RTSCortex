"""Lifecycle supervision for the isolated live HIMA policy process."""

from __future__ import annotations

import asyncio
import contextlib
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import BinaryIO, Final

from rtscortex.policy.hima.live import (
    HIMALiveHealth,
    HIMALivePolicyClient,
    HIMALiveProtocolError,
)
from rtscortex.policy.hima.models import (
    HIMA_ADAPTER_VERSION,
    HIMA_PARSER_VERSION,
    HIMA_VOCABULARY_VERSION,
)
from rtscortex.policy.hima.subagent import HIMA_PINNED_REVISIONS


class HIMASidecarError(RuntimeError):
    """Raised when the pinned HIMA worker cannot become ready."""


HIMA_CANDIDATE_MODEL_IDS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "protoss-a": "SNUMPR/Protoss-a",
        "protoss-b": "SNUMPR/Protoss-b",
        "protoss-c": "SNUMPR/Protoss-c",
    }
)


@dataclass(frozen=True)
class ValidatedHIMACheckpoint:
    """Exact local checkpoint identity accepted for live inference."""

    candidate: str
    model_id: str
    revision: str
    model_path: Path


def validate_local_hima_checkpoint(
    *,
    candidate: str,
    model_path: Path,
    allow_unlicensed_weights: bool,
) -> ValidatedHIMACheckpoint:
    """Validate license acknowledgement and exact Hub snapshot provenance.

    This function only inspects the local filesystem. It deliberately does not
    call the Hugging Face Hub or resolve a branch such as ``main``.
    """

    model_id = HIMA_CANDIDATE_MODEL_IDS.get(candidate)
    if model_id is None:
        raise HIMASidecarError(f"unsupported HIMA candidate: {candidate}")
    if not allow_unlicensed_weights:
        raise HIMASidecarError(
            "HIMA weights have no declared license; set "
            "cortex.macro.allow_unlicensed_weights=true only after explicitly "
            "accepting that risk"
        )
    expanded = model_path.expanduser()
    if not expanded.is_absolute():
        raise HIMASidecarError("HIMA model_path must be an absolute local path")
    resolved = expanded.resolve()
    if not resolved.is_dir():
        raise HIMASidecarError(f"HIMA model directory does not exist: {resolved}")

    revision = HIMA_PINNED_REVISIONS[model_id]
    repo_token = f"models--{model_id.replace('/', '--')}"
    if not (
        resolved.name == revision
        and resolved.parent.name == "snapshots"
        and resolved.parent.parent.name == repo_token
    ):
        raise HIMASidecarError(
            "HIMA checkpoint provenance mismatch: expected exact local snapshot "
            f"{repo_token}/snapshots/{revision}, received {resolved}"
        )
    return ValidatedHIMACheckpoint(
        candidate=candidate,
        model_id=model_id,
        revision=revision,
        model_path=resolved,
    )


@dataclass(frozen=True)
class HIMASidecarSpec:
    """Validated process inputs for one run-scoped HIMA worker."""

    command: tuple[str, ...]
    socket_path: Path
    stdout_path: Path
    stderr_path: Path
    ready_timeout_seconds: float = 60.0
    shutdown_timeout_seconds: float = 10.0
    environment: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        if not self.command:
            raise ValueError("HIMA sidecar command cannot be empty")
        if not self.socket_path.is_absolute():
            raise ValueError("HIMA sidecar socket path must be absolute")
        if self.ready_timeout_seconds <= 0:
            raise ValueError("HIMA sidecar ready timeout must be positive")
        if self.shutdown_timeout_seconds <= 0:
            raise ValueError("HIMA sidecar shutdown timeout must be positive")


def hima_sidecar_command(
    python_executable: Path,
    *,
    socket_path: Path,
    model_id: str,
    model_path: Path,
    device: str,
    max_new_tokens: int,
    allow_unlicensed_weights: bool,
) -> tuple[str, ...]:
    """Build the explicit, local-only HIMA live worker command."""

    command = [
        str(python_executable),
        "-m",
        "rtscortex.policy.hima.live_worker",
        "--socket",
        str(socket_path),
        "--model-id",
        model_id,
        "--model-path",
        str(model_path),
        "--device",
        device,
        "--max-new-tokens",
        str(max_new_tokens),
    ]
    if allow_unlicensed_weights:
        command.append("--allow-unlicensed-weights")
    return tuple(command)


class HIMASidecarProcess:
    """Start one HIMA worker, verify its typed health, and stop it cleanly."""

    def __init__(
        self,
        spec: HIMASidecarSpec,
        client: HIMALivePolicyClient,
        *,
        expected_model_id: str,
        expected_model_revision: str,
    ) -> None:
        self.spec = spec
        self.client = client
        self.expected_model_id = expected_model_id
        self.expected_model_revision = expected_model_revision
        self._process: asyncio.subprocess.Process | None = None
        self._stdout: BinaryIO | None = None
        self._stderr: BinaryIO | None = None
        self._health: HIMALiveHealth | None = None
        self._start_lock = asyncio.Lock()
        self._ownership_path = self.spec.socket_path.with_name(
            f"{self.spec.socket_path.name}.lock"
        )
        self._owns_socket = False
        self._owns_lock = False

    @property
    def health(self) -> HIMALiveHealth | None:
        return self._health

    async def start(self) -> HIMALiveHealth:
        """Start once and wait until the exact configured checkpoint is ready."""

        async with self._start_lock:
            return await self._start_once()

    async def _start_once(self) -> HIMALiveHealth:
        """Perform one serialized startup attempt."""

        if self._health is not None:
            return self._health
        if self._process is not None:
            raise HIMASidecarError("HIMA sidecar startup is already in progress")

        self.spec.socket_path.parent.mkdir(parents=True, exist_ok=True)
        self.spec.stdout_path.parent.mkdir(parents=True, exist_ok=True)
        self.spec.stderr_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._acquire_ownership()
            if self.spec.socket_path.exists():
                raise HIMASidecarError(
                    f"HIMA sidecar socket already exists: {self.spec.socket_path}"
                )
            self._owns_socket = True
            self._stdout = self.spec.stdout_path.open("xb")
            self._stderr = self.spec.stderr_path.open("xb")
            environment = dict(os.environ)
            environment.update(self.spec.environment or {})
            environment["HF_HUB_OFFLINE"] = "1"
            environment["TRANSFORMERS_OFFLINE"] = "1"
            environment.setdefault("TOKENIZERS_PARALLELISM", "false")
            self._process = await asyncio.create_subprocess_exec(
                *self.spec.command,
                stdout=self._stdout,
                stderr=self._stderr,
                env=environment,
            )
            health = await self._wait_until_ready()
        except BaseException:
            await self.close()
            raise
        self._health = health
        return health

    def _acquire_ownership(self) -> None:
        try:
            descriptor = os.open(
                self._ownership_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
            )
        except FileExistsError as error:
            raise HIMASidecarError(
                f"HIMA sidecar is already owned: {self._ownership_path}"
            ) from error
        try:
            os.write(descriptor, f"pid={os.getpid()}\n".encode())
        finally:
            os.close(descriptor)
        self._owns_lock = True

    async def _wait_until_ready(self) -> HIMALiveHealth:
        assert self._process is not None
        deadline = asyncio.get_running_loop().time() + self.spec.ready_timeout_seconds
        last_error: Exception | None = None
        while asyncio.get_running_loop().time() < deadline:
            if self._process.returncode is not None:
                raise HIMASidecarError(
                    "HIMA sidecar exited before becoming ready with status "
                    f"{self._process.returncode}; see {self.spec.stderr_path}"
                )
            if self.spec.socket_path.exists():
                try:
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        break
                    health = await asyncio.wait_for(
                        self.client.health(),
                        timeout=remaining,
                    )
                    self._validate_health(health)
                    return health
                except HIMASidecarError:
                    raise
                except HIMALiveProtocolError as error:
                    raise HIMASidecarError(str(error)) from error
                except Exception as error:  # readiness probes intentionally retry
                    last_error = error
            if asyncio.get_running_loop().time() >= deadline:
                break
            await asyncio.sleep(0.05)
        detail = "" if last_error is None else f": {last_error}"
        raise HIMASidecarError(
            f"HIMA sidecar did not become ready within "
            f"{self.spec.ready_timeout_seconds:.1f}s{detail}"
        )

    def _validate_health(self, health: HIMALiveHealth) -> None:
        if health.status != "ready":
            raise HIMASidecarError(f"HIMA sidecar is not ready: {health.status}")
        if health.model_id != self.expected_model_id:
            raise HIMASidecarError(
                f"HIMA sidecar model mismatch: {health.model_id!r} != "
                f"{self.expected_model_id!r}"
            )
        if health.model_revision != self.expected_model_revision:
            raise HIMASidecarError(
                "HIMA sidecar revision mismatch: "
                f"{health.model_revision!r} != {self.expected_model_revision!r}"
            )
        versions = {
            "adapter": (health.adapter_version, HIMA_ADAPTER_VERSION),
            "parser": (health.parser_version, HIMA_PARSER_VERSION),
            "vocabulary": (health.vocabulary_version, HIMA_VOCABULARY_VERSION),
        }
        for component, (actual, expected) in versions.items():
            if actual != expected:
                raise HIMASidecarError(
                    f"HIMA sidecar {component} version mismatch: "
                    f"{actual!r} != {expected!r}"
                )

    async def close(self) -> None:
        """Stop the worker and release its client, logs, and socket."""

        process = self._process
        self._process = None
        self._health = None
        if process is not None and process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                process.terminate()
            try:
                await asyncio.wait_for(
                    process.wait(),
                    timeout=self.spec.shutdown_timeout_seconds,
                )
            except TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    process.kill()
                await process.wait()
        await self.client.close()
        for stream in (self._stdout, self._stderr):
            if stream is not None:
                stream.close()
        self._stdout = None
        self._stderr = None
        if self._owns_socket:
            with contextlib.suppress(OSError):
                self.spec.socket_path.unlink()
            self._owns_socket = False
        if self._owns_lock:
            with contextlib.suppress(OSError):
                self._ownership_path.unlink()
            self._owns_lock = False


def validate_sidecar_executable(command: Sequence[str]) -> None:
    """Fail early when the configured HIMA Python cannot be executed."""

    if not command:
        raise HIMASidecarError("HIMA sidecar command cannot be empty")
    executable = Path(command[0]).expanduser()
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise HIMASidecarError(
            f"HIMA Python is missing or not executable: {executable}"
        )
