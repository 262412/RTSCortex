from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import pytest

from rtscortex.config import (
    AgentSettings,
    CortexMacroSettings,
    CortexSettings,
    ExperimentConfig,
    ProviderSettings,
    RunSettings,
    load_config,
)
from rtscortex.contracts import EconomyState, ObservationEnvelope, SC2State
from rtscortex.policy.hima import HIMA_PINNED_REVISIONS
from rtscortex.providers import FakeProvider
from rtscortex.runtime import factory
from rtscortex.runtime.cortex_engine import CortexRuntimeEngine
from rtscortex.runtime.hima_sidecar import HIMASidecarError

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _snapshot_path(root: Path, model_id: str) -> Path:
    revision = HIMA_PINNED_REVISIONS[model_id]
    path = (
        root
        / f"models--{model_id.replace('/', '--')}"
        / "snapshots"
        / revision
    )
    path.mkdir(parents=True)
    return path


def test_legacy_runtime_still_uses_configured_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = FakeProvider()
    calls: list[dict[str, Any]] = []

    def build_provider(**kwargs: Any) -> FakeProvider:
        calls.append(kwargs)
        return provider

    monkeypatch.setattr(factory, "OpenAICompatibleProvider", build_provider)
    config = ExperimentConfig(
        provider=ProviderSettings(
            kind="openai_compatible",
            base_url="http://127.0.0.1:9999/v1",
            model="local-test-model",
        )
    )

    runtime = factory.build_runtime(config, tmp_path / "run")

    assert runtime.provider is provider
    assert calls == [
        {
            "base_url": "http://127.0.0.1:9999/v1",
            "model": "local-test-model",
            "api_key_env": "RTSCORTEX_LLM_API_KEY",
            "timeout_seconds": 30.0,
            "max_tokens": None,
            "enable_thinking": None,
        }
    ]
    asyncio.run(runtime.close())


def test_cortex_disabled_uses_fake_provider_and_no_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_openai(**_: Any) -> None:
        raise AssertionError("cortex must not construct an OpenAI provider")

    monkeypatch.setattr(factory, "OpenAICompatibleProvider", reject_openai)
    config = ExperimentConfig(
        agent=AgentSettings(variant="cortex"),
        provider=ProviderSettings(kind="openai_compatible"),
    )

    runtime = factory.build_runtime(config, tmp_path / "run")

    assert isinstance(runtime, CortexRuntimeEngine)
    assert isinstance(runtime.provider, FakeProvider)
    assert runtime._macro_client is None
    assert runtime._macro_sidecar is None
    asyncio.run(runtime.close())


def test_build_hima_sidecar_uses_pinned_local_snapshot_and_short_uds(
    tmp_path: Path,
) -> None:
    model_id = "SNUMPR/Protoss-a"
    model_path = _snapshot_path(tmp_path / "hub", model_id)
    runtime_root = Path("/tmp") / tmp_path.name
    config = ExperimentConfig(
        run=RunSettings(runtime_root=runtime_root),
        agent=AgentSettings(variant="cortex"),
        cortex=CortexSettings(
            macro=CortexMacroSettings(
                kind="hima",
                candidate="protoss-a",
                python_executable=Path(sys.executable),
                model_path=model_path,
                allow_unlicensed_weights=True,
            )
        ),
    )

    client, sidecar = factory._build_hima_sidecar(config, tmp_path / "run")

    assert sidecar.expected_model_id == model_id
    assert sidecar.expected_model_revision == HIMA_PINNED_REVISIONS[model_id]
    assert sidecar.spec.socket_path.parent == runtime_root / "hima"
    assert len(str(sidecar.spec.socket_path).encode()) <= 100
    assert sidecar.spec.command[0] == sys.executable
    assert str(model_path.resolve()) in sidecar.spec.command
    assert sidecar.spec.environment is None
    asyncio.run(client.close())


def test_hima_sidecar_build_fails_closed_without_license_ack(
    tmp_path: Path,
) -> None:
    model_path = _snapshot_path(tmp_path / "hub", "SNUMPR/Protoss-a")
    config = ExperimentConfig(
        run=RunSettings(runtime_root=tmp_path / "runtime"),
        agent=AgentSettings(variant="cortex"),
        cortex=CortexSettings(
            macro=CortexMacroSettings(
                kind="hima",
                python_executable=Path(sys.executable),
                model_path=model_path,
                allow_unlicensed_weights=False,
            )
        ),
    )

    with pytest.raises(HIMASidecarError, match="no declared license"):
        factory._build_hima_sidecar(config, tmp_path / "run")


def test_optional_hima_factory_failure_builds_degraded_reflex_runtime(
    tmp_path: Path,
) -> None:
    model_path = _snapshot_path(tmp_path / "hub", "SNUMPR/Protoss-a")
    config = ExperimentConfig(
        run=RunSettings(runtime_root=tmp_path / "runtime"),
        agent=AgentSettings(variant="cortex"),
        cortex=CortexSettings(
            macro=CortexMacroSettings(
                kind="hima",
                python_executable=Path(sys.executable),
                model_path=model_path,
                allow_unlicensed_weights=False,
                required=False,
            )
        ),
    )
    run_dir = tmp_path / "run"

    runtime = factory.build_runtime(config, run_dir)

    assert isinstance(runtime, CortexRuntimeEngine)
    assert runtime._macro_client is None
    assert runtime._macro_sidecar is None
    assert isinstance(runtime._macro_startup_failure, HIMASidecarError)

    async def exercise() -> None:
        await runtime.start()
        batch = await runtime.tick(
            ObservationEnvelope(
                run_id="factory-degraded",
                episode_id="episode-1",
                step_id=0,
                game_loop=0,
                state=SC2State(economy=EconomyState()),
            )
        )
        assert batch.commands == []
        assert batch.planner_pending is False

    asyncio.run(exercise())
    failures = runtime.store.events_of_type(
        "factory-degraded",
        "episode-1",
        "specialist_failed",
    )
    assert len(failures) == 1
    assert failures[0].payload["stage"] == "startup"
    assert failures[0].payload["fallback"] == "deterministic_reflex"
    assert "no declared license" in failures[0].payload["message"]
    asyncio.run(runtime.close())


def test_required_hima_factory_failure_remains_fail_closed(tmp_path: Path) -> None:
    model_path = _snapshot_path(tmp_path / "hub", "SNUMPR/Protoss-a")
    config = ExperimentConfig(
        run=RunSettings(runtime_root=tmp_path / "runtime"),
        agent=AgentSettings(variant="cortex"),
        cortex=CortexSettings(
            macro=CortexMacroSettings(
                kind="hima",
                python_executable=Path(sys.executable),
                model_path=model_path,
                allow_unlicensed_weights=False,
                required=True,
            )
        ),
    )

    with pytest.raises(HIMASidecarError, match="no declared license"):
        factory.build_runtime(config, tmp_path / "run")


def test_hima_socket_path_rejects_an_overlong_runtime_root(tmp_path: Path) -> None:
    runtime_root = tmp_path / ("x" * 100)

    with pytest.raises(HIMASidecarError, match="too long"):
        factory._hima_socket_path(runtime_root, tmp_path / "run")


def test_live_hima_cortex_example_is_safe_by_default() -> None:
    config = load_config(
        PROJECT_ROOT / "configs/experiments/live_simple64_hima_a_cortex.yaml"
    )

    assert config.agent.variant == "cortex"
    assert config.cortex.macro.kind == "hima"
    assert config.cortex.macro.candidate == "protoss-a"
    assert config.cortex.macro.allow_unlicensed_weights is False


def test_live_hima_cortex_canary_records_explicit_license_acceptance() -> None:
    config = load_config(
        PROJECT_ROOT / "configs/experiments/live_simple64_hima_a_cortex_canary.yaml"
    )

    assert config.agent.variant == "cortex"
    assert config.cortex.macro.kind == "hima"
    assert config.cortex.macro.candidate == "protoss-a"
    assert config.cortex.macro.allow_unlicensed_weights is True
    assert config.environment.max_steps == 2_500
    assert config.environment.game_steps_per_episode == 2_500
    assert config.provider.kind == "fake"
