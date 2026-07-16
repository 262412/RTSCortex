from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from rtscortex.config import (
    AgentSettings,
    ConsoleSettings,
    CortexMacroSettings,
    CortexSettings,
    EnvironmentSettings,
    ExperimentConfig,
    RuntimeSettings,
)


def test_environment_settings_accept_melee_runtime_controls() -> None:
    settings = EnvironmentSettings(
        adapter="llm_pysc2",
        scenario="Simple64",
        agent_race="protoss",
        opponent_race="zerg",
        opponent_difficulty="easy",
        opponent_build="macro",
        step_mul=1,
        game_steps_per_episode=28_800,
        simulation_speed_multiplier=0.25,
        pause_until_first_plan=True,
        action_effect_timeout_game_loops=96,
    )

    assert settings.opponent_race == "zerg"
    assert settings.opponent_difficulty == "easy"
    assert settings.opponent_build == "macro"
    assert settings.step_mul == 1
    assert settings.game_steps_per_episode == 28_800
    assert settings.simulation_speed_multiplier == 0.25
    assert settings.pause_until_first_plan is True
    assert settings.action_effect_timeout_game_loops == 96


@pytest.mark.parametrize("multiplier", [0.0, -0.1, 1.01])
def test_environment_settings_reject_invalid_simulation_speed(multiplier: float) -> None:
    with pytest.raises(ValidationError):
        EnvironmentSettings(simulation_speed_multiplier=multiplier)


def test_environment_settings_reject_invalid_action_effect_timeout() -> None:
    with pytest.raises(ValidationError):
        EnvironmentSettings(action_effect_timeout_game_loops=0)


def test_runtime_command_ttl_defaults_to_planning_interval() -> None:
    settings = RuntimeSettings(planning_interval_game_loops=112)

    assert settings.planner_command_ttl_game_loops == 112


def test_runtime_command_ttl_can_be_configured_independently() -> None:
    settings = RuntimeSettings(
        planning_interval_game_loops=112,
        planner_command_ttl_game_loops=64,
    )

    assert settings.planner_command_ttl_game_loops == 64


def test_console_settings_are_disabled_and_bounded_by_default() -> None:
    settings = ExperimentConfig().console

    assert settings == ConsoleSettings()
    assert settings.enabled is False
    assert settings.port == 8765
    assert settings.frame_fps == 2.0
    assert settings.frontend_event_limit == 5_000


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("port", 0),
        ("frame_fps", 0),
        ("jpeg_quality", 96),
        ("stale_after_seconds", 0),
        ("frontend_event_limit", 99),
    ],
)
def test_console_settings_reject_invalid_values(field: str, value: int) -> None:
    with pytest.raises(ValidationError):
        ConsoleSettings.model_validate({field: value})


def test_cortex_settings_are_opt_in_and_do_not_change_legacy_default() -> None:
    config = ExperimentConfig()

    assert config.agent.variant == "planner_reflection_memory_reflex"
    assert config.cortex.macro.kind == "disabled"
    assert config.cortex.executor.kind == "deterministic"
    assert config.cortex.explanation.enabled is False
    assert AgentSettings(variant="cortex").variant == "cortex"


def test_hima_cortex_macro_requires_an_explicit_model_path() -> None:
    with pytest.raises(ValidationError, match="requires model_path"):
        CortexMacroSettings(kind="hima")


def test_expanded_config_expands_cortex_runtime_paths() -> None:
    config = ExperimentConfig(
        agent=AgentSettings(variant="cortex"),
        cortex=CortexSettings(
            macro=CortexMacroSettings(
                kind="hima",
                python_executable=Path("~/fastscratch/envs/rtscortex-hima/bin/python"),
                model_path=Path("~/fastscratch/models/hima-a"),
            )
        ),
    ).expanded()

    assert config.cortex.macro.python_executable.is_absolute()
    assert config.cortex.macro.model_path is not None
    assert config.cortex.macro.model_path.is_absolute()
