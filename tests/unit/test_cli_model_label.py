from pathlib import Path

import pytest

from rtscortex.cli.app import _active_model_label, _live_worker_environment
from rtscortex.config import (
    AgentSettings,
    AuthoritativeBuildCircuitCanarySettings,
    CortexHIMAEnsembleMemberSettings,
    CortexMacroSettings,
    CortexSettings,
    EnvironmentSettings,
    EvaluationSettings,
    ExperimentConfig,
    ProviderSettings,
    ReflexSettings,
    RuntimeSettings,
)
from rtscortex.runtime.live import LiveWorkerSpec


def test_console_model_label_uses_the_active_cortex_specialist() -> None:
    config = ExperimentConfig(
        agent=AgentSettings(variant="cortex"),
        cortex=CortexSettings(
            macro=CortexMacroSettings(
                kind="hima",
                candidate="protoss-b",
                model_path=Path("/tmp/hima-b"),
            )
        ),
        provider=ProviderSettings(model="unused-generic-provider"),
    )

    assert _active_model_label(config) == "SNUMPR/Protoss-b"


def test_console_model_label_keeps_the_legacy_provider_model() -> None:
    config = ExperimentConfig(provider=ProviderSettings(model="Qwen/Qwen3-8B"))

    assert _active_model_label(config) == "Qwen/Qwen3-8B"


def test_console_model_label_identifies_the_race_brain_ensemble() -> None:
    members = (
        CortexHIMAEnsembleMemberSettings(candidate="protoss-a", model_path=Path("/tmp/protoss-a")),
        CortexHIMAEnsembleMemberSettings(candidate="protoss-b", model_path=Path("/tmp/protoss-b")),
        CortexHIMAEnsembleMemberSettings(candidate="protoss-c", model_path=Path("/tmp/protoss-c")),
    )
    config = ExperimentConfig(
        agent=AgentSettings(variant="cortex"),
        cortex=CortexSettings(
            macro=CortexMacroSettings(
                kind="hima_ensemble",
                ensemble_members=list(members),
            )
        ),
    )

    assert _active_model_label(config) == "HIMA Protoss a/b/c Ensemble"


def test_live_worker_environment_propagates_the_configured_agent_race(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = ExperimentConfig(
        environment=EnvironmentSettings(
            adapter="llm_pysc2",
            scenario="Simple64",
            agent_race="terran",
        )
    )
    monkeypatch.setenv("PYTHONPATH", "/existing/python/path")
    worker = LiveWorkerSpec(
        command=("python",),
        sc2_path=Path("/tmp/StarCraftII"),
        python_path=(Path("/tmp/reviewed/LLM-PySC2"),),
    )

    environment = _live_worker_environment(
        config,
        worker,
        placement_outbox_path=Path("/tmp/run/placement-outbox.sqlite3"),
    )

    assert environment["RTSCORTEX_AGENT_RACE"] == "terran"
    assert environment["SC2PATH"] == "/tmp/StarCraftII"
    assert environment["RTSCORTEX_PLACEMENT_OUTBOX_PATH"] == "/tmp/run/placement-outbox.sqlite3"
    assert environment["PYTHONPATH"] == "/tmp/reviewed/LLM-PySC2:/existing/python/path"


def test_live_worker_environment_propagates_authoritative_canary_contract() -> None:
    config = ExperimentConfig(
        environment=EnvironmentSettings(
            adapter="llm_pysc2",
            execution_action_space="raw",
            agent_race="protoss",
            max_steps=1024,
            expansion_scout_enabled=False,
        ),
        runtime=RuntimeSettings(max_actions=1),
        agent=AgentSettings(variant="cortex"),
        cortex=CortexSettings(
            macro=CortexMacroSettings(kind="scripted", scripted_actions=["Pylon"]),
        ),
        reflex=ReflexSettings(enabled=False),
        evaluation=EvaluationSettings(
            seeds=[7],
            authoritative_build_circuit_canary=(
                AuthoritativeBuildCircuitCanarySettings(enabled=True)
            ),
        ),
    )
    worker = LiveWorkerSpec(command=("python",), sc2_path=Path("/tmp/StarCraftII"))

    environment = _live_worker_environment(
        config,
        worker,
        authoritative_circuit_canary_journal_path=Path("/tmp/run/circuit.jsonl"),
    )

    assert environment["RTSCORTEX_AUTHORITATIVE_BUILD_CIRCUIT_CANARY"] == "true"
    assert (
        environment["RTSCORTEX_AUTHORITATIVE_BUILD_CIRCUIT_CANARY_MODE"]
        == "stale_candidate_then_builder_rebind"
    )
    assert environment["RTSCORTEX_AUTHORITATIVE_BUILD_CIRCUIT_CANARY_FAILURE_ATTEMPTS"] == "3"
    assert environment["RTSCORTEX_AUTHORITATIVE_BUILD_CIRCUIT_CANARY_HOLD_OBSERVATIONS"] == "1"
    assert environment["RTSCORTEX_AUTHORITATIVE_BUILD_CIRCUIT_CANARY_JOURNAL"] == (
        "/tmp/run/circuit.jsonl"
    )
