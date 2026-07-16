"""Runtime construction from experiment configuration."""

from __future__ import annotations

import hashlib
from pathlib import Path

from rtscortex.config import ExperimentConfig
from rtscortex.contracts import LLMProvider
from rtscortex.memory import EventStore
from rtscortex.policy.hima.live import HIMALivePolicyClient
from rtscortex.providers import FakeProvider, OpenAICompatibleProvider
from rtscortex.runtime.engine import RuntimeEngine
from rtscortex.runtime.hima_sidecar import (
    HIMASidecarError,
    HIMASidecarProcess,
    HIMASidecarSpec,
    hima_sidecar_command,
    validate_local_hima_checkpoint,
    validate_sidecar_executable,
)

_UDS_PATH_LIMIT = 100


def build_runtime(config: ExperimentConfig, run_dir: Path) -> RuntimeEngine:
    store = EventStore(run_dir / "events.sqlite3", run_dir / "events.jsonl")
    if config.agent.variant == "cortex":
        try:
            return _build_cortex_runtime(config, run_dir, store)
        except BaseException:
            store.close()
            raise

    provider = _build_legacy_provider(config)
    return RuntimeEngine(config=config, store=store, provider=provider)


def _build_legacy_provider(config: ExperimentConfig) -> LLMProvider:
    provider: LLMProvider
    if config.provider.kind == "fake":
        provider = FakeProvider()
    else:
        provider = OpenAICompatibleProvider(
            base_url=config.provider.base_url,
            model=config.provider.model,
            api_key_env=config.provider.api_key_env,
            timeout_seconds=config.provider.timeout_seconds,
            max_tokens=config.provider.max_tokens,
            enable_thinking=config.provider.enable_thinking,
        )
    return provider


def _build_cortex_runtime(
    config: ExperimentConfig,
    run_dir: Path,
    store: EventStore,
) -> RuntimeEngine:
    from rtscortex.runtime.cortex_engine import CortexRuntimeEngine

    macro_client: HIMALivePolicyClient | None = None
    macro_sidecar: HIMASidecarProcess | None = None
    macro_startup_failure: Exception | None = None
    if config.cortex.macro.kind == "hima":
        try:
            macro_client, macro_sidecar = _build_hima_sidecar(config, run_dir)
        except Exception as error:
            if config.cortex.macro.required:
                raise
            macro_startup_failure = error
    return CortexRuntimeEngine(
        config=config,
        store=store,
        provider=FakeProvider(),
        macro_client=macro_client,
        macro_sidecar=macro_sidecar,
        macro_startup_failure=macro_startup_failure,
    )


def _build_hima_sidecar(
    config: ExperimentConfig,
    run_dir: Path,
) -> tuple[HIMALivePolicyClient, HIMASidecarProcess]:
    macro = config.cortex.macro
    if macro.model_path is None:
        raise HIMASidecarError("cortex HIMA macro policy requires model_path")
    checkpoint = validate_local_hima_checkpoint(
        candidate=macro.candidate,
        model_path=macro.model_path,
        allow_unlicensed_weights=macro.allow_unlicensed_weights,
    )

    runtime_root = config.run.runtime_root.expanduser().resolve()
    socket_path = _hima_socket_path(runtime_root, run_dir)
    command = hima_sidecar_command(
        macro.python_executable.expanduser(),
        socket_path=socket_path,
        model_id=checkpoint.model_id,
        model_path=checkpoint.model_path,
        device=macro.device,
        max_new_tokens=macro.max_new_tokens,
        allow_unlicensed_weights=macro.allow_unlicensed_weights,
    )
    validate_sidecar_executable(command)
    client = HIMALivePolicyClient.for_unix_socket(
        socket_path,
        timeout_seconds=macro.timeout_seconds,
        expected_model_id=checkpoint.model_id,
    )
    spec = HIMASidecarSpec(
        command=command,
        socket_path=socket_path,
        stdout_path=run_dir / "hima-sidecar.stdout.log",
        stderr_path=run_dir / "hima-sidecar.stderr.log",
        shutdown_timeout_seconds=config.environment.shutdown_timeout_seconds,
    )
    sidecar = HIMASidecarProcess(
        spec,
        client,
        expected_model_id=checkpoint.model_id,
        expected_model_revision=checkpoint.revision,
    )
    return client, sidecar


def _hima_socket_path(runtime_root: Path, run_dir: Path) -> Path:
    """Return a deterministic short UDS path scoped to one run directory."""

    run_identity = str(run_dir.expanduser().resolve()).encode()
    digest = hashlib.sha256(run_identity).hexdigest()[:16]
    socket_path = runtime_root / "hima" / f"{digest}.sock"
    if len(str(socket_path).encode()) > _UDS_PATH_LIMIT:
        raise HIMASidecarError(
            f"HIMA Unix socket path is too long ({len(str(socket_path).encode())} bytes): "
            f"{socket_path}; configure a shorter run.runtime_root"
        )
    return socket_path
