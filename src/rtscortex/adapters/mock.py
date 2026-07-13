"""Deterministic SC2-like environment used for offline development."""

from __future__ import annotations

from rtscortex.contracts import (
    ActionBatch,
    AvailableAction,
    EconomyState,
    EpisodeOutcome,
    ExecutionReport,
    ObservationEnvelope,
    SC2State,
    UnitState,
)


class MockSC2Adapter:
    def __init__(self, *, scenario: str, max_steps: int) -> None:
        self.scenario = scenario
        self.max_steps = max_steps
        self.run_id = ""
        self.episode_id = ""
        self.seed = 0
        self.step_id = 0
        self.enemy_health = 1.0
        self.own_health = 0.8
        self.done = False
        self.outcome = EpisodeOutcome.TRUNCATED
        self.action_attempts = 0
        self.action_successes = 0

    async def reset(self, *, run_id: str, episode_id: str, seed: int) -> ObservationEnvelope:
        self.run_id = run_id
        self.episode_id = episode_id
        self.seed = seed
        self.step_id = 0
        self.enemy_health = 1.0
        self.own_health = 0.8
        self.done = False
        self.outcome = EpisodeOutcome.TRUNCATED
        self.action_attempts = 0
        self.action_successes = 0
        return self._observation()

    async def step(self, actions: ActionBatch) -> tuple[ObservationEnvelope, list[ExecutionReport]]:
        reports: list[ExecutionReport] = []
        for command in actions.commands:
            self.action_attempts += 1
            success = command.name in {"Attack_Unit", "Retreat", "No_Operation"}
            if command.name == "Attack_Unit" and self.enemy_health > 0:
                self.enemy_health = max(0.0, self.enemy_health - 0.55)
            elif command.name == "Retreat":
                self.own_health = min(1.0, self.own_health + 0.2)
            if success:
                self.action_successes += 1
            reports.append(
                ExecutionReport(
                    run_id=self.run_id,
                    episode_id=self.episode_id,
                    step_id=actions.step_id,
                    command_id=command.command_id,
                    success=success,
                    failure_reason=None if success else "unsupported mock action",
                    pysc2_function=f"mock.{command.name}",
                    latency_ms=0.1,
                )
            )

        self.step_id += 1
        if self.enemy_health <= 0:
            self.done = True
            self.outcome = EpisodeOutcome.VICTORY
        elif self.step_id >= self.max_steps:
            self.done = True
            self.outcome = EpisodeOutcome.TRUNCATED
        elif self.step_id == 1:
            self.own_health = 0.2
        return self._observation(), reports

    def _observation(self) -> ObservationEnvelope:
        enemies = []
        if self.enemy_health > 0:
            enemies.append(
                UnitState(
                    unit_id="zergling-1",
                    unit_type="Zergling",
                    alliance="enemy",
                    position=(20.0, 20.0),
                    health_fraction=self.enemy_health,
                )
            )
        return ObservationEnvelope(
            run_id=self.run_id,
            episode_id=self.episode_id,
            step_id=self.step_id,
            game_loop=self.step_id * 8,
            state=SC2State(
                economy=EconomyState(
                    minerals=50,
                    supply_used=4,
                    supply_cap=15,
                    workers=0,
                    army_supply=4,
                ),
                own_units=[
                    UnitState(
                        unit_id="adept-1",
                        unit_type="Adept",
                        alliance="self",
                        position=(10.0, 10.0),
                        health_fraction=self.own_health,
                    )
                ],
                visible_enemies=enemies,
            ),
            text_observation=(
                f"Mock {self.scenario}: one Adept sees {len(enemies)} hostile unit(s)."
            ),
            available_actions=[
                AvailableAction(
                    name="Attack_Unit", argument_names=["target_id"], actor_scopes=["army"]
                ),
                AvailableAction(name="Retreat", argument_names=[]),
                AvailableAction(name="No_Operation", argument_names=[], actor_scopes=["global"]),
            ],
            alerts=["under_attack"] if enemies else [],
        )

    async def close(self) -> None:
        return None
