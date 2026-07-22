"""Condition-response policies inspired by SwarmBrain's ReflexNet."""

from __future__ import annotations

from rtscortex.contracts import ActionCommand, ActionSource, ObservationEnvelope
from rtscortex.targeting import living_targetable_enemies


class ReflexEngine:
    def __init__(self, *, enabled: bool, low_health_threshold: float) -> None:
        self.enabled = enabled
        self.low_health_threshold = low_health_threshold

    def evaluate(self, observation: ObservationEnvelope) -> list[ActionCommand]:
        if not self.enabled:
            return []
        available = {action.name: action for action in observation.available_actions}
        attack_actions = [
            action for action in observation.available_actions if action.name == "Attack_Unit"
        ]
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
        if not under_attack:
            controller_action = (
                available.get("Effect_InjectLarva")
                or available.get("Build_CreepTumor_Queen_Screen")
                or available.get("Build_CreepTumor_Tumor_Screen")
                or available.get("Morph_OrbitalCommand")
                or available.get("Effect_CalldownMULE_Screen")
                or (
                    available.get("Train_SCV")
                    if _automatic_scv_is_needed(observation)
                    else None
                )
            )
            if controller_action is not None and (
                not controller_action.argument_names or controller_action.argument_candidates
            ):
                actor = next(
                    (
                        scope
                        for scope in controller_action.actor_scopes
                        if scope.startswith(("CombatGroup", "Developer"))
                    ),
                    None,
                )
                if actor is not None:
                    commands.append(
                        self._command(
                            observation,
                            index=len(commands),
                            actor=actor,
                            name=controller_action.name,
                            arguments=(
                                []
                                if not controller_action.argument_names
                                else list((controller_action.argument_candidates or [])[0])
                            ),
                            priority=_controller_priority(controller_action.name),
                            ttl_game_loops=8,
                        )
                    )
        enemy_ids = {
            _normalize_tag(enemy.unit_id): enemy.unit_id
            for enemy in living_targetable_enemies(observation.state.visible_enemies)
        }
        if under_attack and enemy_ids:
            dispatched_actors: set[str] = set()
            for attack in attack_actions:
                candidates = attack.argument_candidates or [
                    [enemy_id] for enemy_id in enemy_ids.values()
                ]
                target = next(
                    (
                        enemy_ids[_normalize_tag(candidate[0])]
                        for candidate in candidates
                        if candidate and _normalize_tag(candidate[0]) in enemy_ids
                    ),
                    None,
                )
                if target is None:
                    continue
                actors = [
                    actor
                    for actor in attack.actor_scopes
                    if actor not in dispatched_actors
                    and (actor == "army" or actor.startswith("CombatGroup"))
                ]
                for actor in actors:
                    dispatched_actors.add(actor)
                    commands.append(
                        self._command(
                            observation,
                            index=len(commands),
                            actor=actor,
                            name="Attack_Unit",
                            arguments=[target],
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


def _automatic_scv_is_needed(observation: ObservationEnvelope) -> bool:
    townhall_types = {"CommandCenter", "OrbitalCommand", "PlanetaryFortress"}
    completed_townhalls = sum(
        structure.unit_type in townhall_types and structure.status != "constructing"
        for structure in observation.state.own_structures
    )
    if completed_townhalls <= 0:
        return False
    target_workers = min(80, completed_townhalls * 22)
    return observation.state.economy.workers < target_workers


def _controller_priority(action_name: str) -> int:
    return {
        "Effect_InjectLarva": 80,
        "Morph_OrbitalCommand": 78,
        "Effect_CalldownMULE_Screen": 75,
        "Train_SCV": 65,
        "Build_CreepTumor_Queen_Screen": 45,
        "Build_CreepTumor_Tumor_Screen": 40,
    }[action_name]

def _normalize_tag(value: object) -> str:
    if isinstance(value, int) and not isinstance(value, bool):
        return hex(value)
    return str(value).casefold()
