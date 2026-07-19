from pathlib import Path

from rtscortex.cli.app import _active_model_label, _live_worker_environment
from rtscortex.config import (
    AgentSettings,
    CortexHIMAEnsembleMemberSettings,
    CortexMacroSettings,
    CortexSettings,
    EnvironmentSettings,
    ExperimentConfig,
    ProviderSettings,
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


def test_live_worker_environment_propagates_the_configured_agent_race() -> None:
    config = ExperimentConfig(
        environment=EnvironmentSettings(
            adapter="llm_pysc2",
            scenario="Simple64",
            agent_race="terran",
        )
    )
    worker = LiveWorkerSpec(command=("python",), sc2_path=Path("/tmp/StarCraftII"))

    environment = _live_worker_environment(config, worker)

    assert environment["RTSCORTEX_AGENT_RACE"] == "terran"
    assert environment["SC2PATH"] == "/tmp/StarCraftII"
