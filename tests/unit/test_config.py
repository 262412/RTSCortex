from __future__ import annotations

import pytest
from pydantic import ValidationError

from rtscortex.config import EnvironmentSettings


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
