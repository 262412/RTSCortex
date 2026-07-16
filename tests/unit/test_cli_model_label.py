from pathlib import Path

from rtscortex.cli.app import _active_model_label
from rtscortex.config import (
    AgentSettings,
    CortexMacroSettings,
    CortexSettings,
    ExperimentConfig,
    ProviderSettings,
)


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
