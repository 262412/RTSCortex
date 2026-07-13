from __future__ import annotations

from pathlib import Path

from rtscortex.config import (
    AgentSettings,
    AgentVariant,
    ExperimentConfig,
    ProviderSettings,
    RunSettings,
    RuntimeSettings,
)
from rtscortex.contracts import (
    AvailableAction,
    EconomyState,
    ObservationEnvelope,
    SC2State,
    UnitState,
)


def make_config(
    tmp_path: Path,
    *,
    variant: AgentVariant = "planner_reflection_memory_reflex",
    deterministic: bool = True,
    planner_timeout_seconds: float = 1.0,
    planning_interval_game_loops: int = 16,
) -> ExperimentConfig:
    return ExperimentConfig(
        run=RunSettings(output_root=tmp_path, runtime_root=tmp_path / "runtime"),
        runtime=RuntimeSettings(
            deterministic=deterministic,
            planner_timeout_seconds=planner_timeout_seconds,
            planning_interval_game_loops=planning_interval_game_loops,
        ),
        agent=AgentSettings(variant=variant),
        provider=ProviderSettings(kind="fake"),
    )


def make_observation(
    *,
    run_id: str = "run-1",
    episode_id: str = "episode-1",
    step_id: int = 0,
    game_loop: int = 0,
    alerts: list[str] | None = None,
    health: float = 0.8,
    include_enemy: bool = True,
) -> ObservationEnvelope:
    enemies = (
        [
            UnitState(
                unit_id="enemy-1",
                unit_type="Zergling",
                alliance="enemy",
                health_fraction=1.0,
            )
        ]
        if include_enemy
        else []
    )
    return ObservationEnvelope(
        run_id=run_id,
        episode_id=episode_id,
        step_id=step_id,
        game_loop=game_loop,
        state=SC2State(
            economy=EconomyState(minerals=50, supply_used=2, supply_cap=15),
            own_units=[
                UnitState(
                    unit_id="unit-1",
                    unit_type="Adept",
                    alliance="self",
                    health_fraction=health,
                )
            ],
            visible_enemies=enemies,
        ),
        text_observation="A compact test observation.",
        available_actions=[
            AvailableAction(name="Attack_Unit", argument_names=["target"], actor_scopes=["army"]),
            AvailableAction(name="Retreat"),
            AvailableAction(name="No_Operation", actor_scopes=["global"]),
        ],
        alerts=alerts or [],
    )
