"""Condition-response policies inspired by SwarmBrain's ReflexNet."""

from __future__ import annotations

from rtscortex.contracts import ActionCommand, ActionSource, ObservationEnvelope


class ReflexEngine:
    def __init__(self, *, enabled: bool, low_health_threshold: float) -> None:
        self.enabled = enabled
        self.low_health_threshold = low_health_threshold

    def evaluate(self, observation: ObservationEnvelope) -> list[ActionCommand]:
        if not self.enabled:
            return []
        available = {action.name: action for action in observation.available_actions}
        commands: list[ActionCommand] = []

        if "Retreat" in available:
            for unit in observation.state.own_units:
                if unit.health_fraction <= self.low_health_threshold:
                    commands.append(
                        self._command(
                            observation,
                            index=len(commands),
                            actor=unit.unit_id,
                            name="Retreat",
                            arguments=[],
                            priority=100,
                            ttl_game_loops=4,
                        )
                    )

        under_attack = any(
            alert.casefold() in {"under_attack", "building_under_attack", "unit_under_attack"}
            for alert in observation.alerts
        )
        if under_attack and observation.state.visible_enemies and "Attack_Unit" in available:
            attack = available["Attack_Unit"]
            actors = attack.actor_scopes or ["army"]
            for actor in actors:
                commands.append(
                    self._command(
                        observation,
                        index=len(commands),
                        actor=actor,
                        name="Attack_Unit",
                        arguments=[observation.state.visible_enemies[0].unit_id],
                        priority=90,
                        ttl_game_loops=8,
                    )
                )
        return commands

    @staticmethod
    def _command(
        observation: ObservationEnvelope,
        *,
        index: int,
        actor: str,
        name: str,
        arguments: list[object],
        priority: int,
        ttl_game_loops: int,
    ) -> ActionCommand:
        return ActionCommand(
            command_id=(
                f"{observation.run_id}:{observation.episode_id}:"
                f"{observation.step_id}:reflex:{index}"
            ),
            actor=actor,
            name=name,
            arguments=arguments,
            priority=priority,
            ttl_game_loops=ttl_game_loops,
            created_game_loop=observation.game_loop,
            source=ActionSource.REFLEX,
        )
