"""Deterministic goal progress evaluation for the Protoss Simple64 MVP."""

from __future__ import annotations

from dataclasses import dataclass

from rtscortex.contracts import ActivePlanSnapshot, ObservationEnvelope, SC2State
from rtscortex.progress.models import (
    GoalBlockerKind,
    GoalProgressBlocker,
    GoalProgressItem,
    GoalProgressReport,
    GoalProgressStatus,
    GoalRequirement,
    GoalRequirementKind,
    GoalSpec,
)


@dataclass(frozen=True)
class StatePrerequisite:
    kind: GoalRequirementKind
    target: str
    count: int = 1


@dataclass(frozen=True)
class ProgressActionSpec:
    """State effect, cost, and prerequisites of one supported action."""

    name: str
    effect_kind: GoalRequirementKind
    effect_target: str
    minerals: int = 0
    vespene: int = 0
    supply: int = 0
    prerequisites: tuple[StatePrerequisite, ...] = ()


PROTOSS_SIMPLE64_ACTION_SPECS: tuple[ProgressActionSpec, ...] = (
    ProgressActionSpec(
        "Build_Pylon_Screen",
        GoalRequirementKind.STRUCTURE,
        "Pylon",
        minerals=100,
    ),
    ProgressActionSpec(
        "Build_Gateway_Screen",
        GoalRequirementKind.STRUCTURE,
        "Gateway",
        minerals=150,
        prerequisites=(
            StatePrerequisite(GoalRequirementKind.STRUCTURE, "Pylon"),
        ),
    ),
    ProgressActionSpec(
        "Build_Forge_Screen",
        GoalRequirementKind.STRUCTURE,
        "Forge",
        minerals=150,
        prerequisites=(
            StatePrerequisite(GoalRequirementKind.STRUCTURE, "Pylon"),
        ),
    ),
    ProgressActionSpec(
        "Build_CyberneticsCore_Screen",
        GoalRequirementKind.STRUCTURE,
        "CyberneticsCore",
        minerals=150,
        prerequisites=(
            StatePrerequisite(GoalRequirementKind.STRUCTURE, "Pylon"),
            StatePrerequisite(GoalRequirementKind.STRUCTURE, "Gateway"),
        ),
    ),
    ProgressActionSpec(
        "Build_Assimilator_Near",
        GoalRequirementKind.STRUCTURE,
        "Assimilator",
        minerals=75,
        prerequisites=(
            StatePrerequisite(GoalRequirementKind.STRUCTURE, "Nexus"),
        ),
    ),
    ProgressActionSpec(
        "Build_Nexus_Near",
        GoalRequirementKind.STRUCTURE,
        "Nexus",
        minerals=400,
    ),
    ProgressActionSpec(
        "Build_Stargate_Screen",
        GoalRequirementKind.STRUCTURE,
        "Stargate",
        minerals=150,
        vespene=150,
        prerequisites=(
            StatePrerequisite(GoalRequirementKind.STRUCTURE, "CyberneticsCore"),
        ),
    ),
    ProgressActionSpec(
        "Build_ShieldBattery_Screen",
        GoalRequirementKind.STRUCTURE,
        "ShieldBattery",
        minerals=100,
        prerequisites=(
            StatePrerequisite(GoalRequirementKind.STRUCTURE, "CyberneticsCore"),
        ),
    ),
    ProgressActionSpec(
        "Train_Probe",
        GoalRequirementKind.UNIT,
        "Probe",
        minerals=50,
        supply=1,
        prerequisites=(
            StatePrerequisite(GoalRequirementKind.STRUCTURE, "Nexus"),
        ),
    ),
    ProgressActionSpec(
        "Train_Zealot",
        GoalRequirementKind.UNIT,
        "Zealot",
        minerals=100,
        supply=2,
        prerequisites=(
            StatePrerequisite(GoalRequirementKind.STRUCTURE, "Gateway"),
        ),
    ),
    ProgressActionSpec(
        "Train_Stalker",
        GoalRequirementKind.UNIT,
        "Stalker",
        minerals=125,
        vespene=50,
        supply=2,
        prerequisites=(
            StatePrerequisite(GoalRequirementKind.STRUCTURE, "Gateway"),
            StatePrerequisite(GoalRequirementKind.STRUCTURE, "CyberneticsCore"),
        ),
    ),
    ProgressActionSpec(
        "Train_Adept",
        GoalRequirementKind.UNIT,
        "Adept",
        minerals=100,
        vespene=25,
        supply=2,
        prerequisites=(
            StatePrerequisite(GoalRequirementKind.STRUCTURE, "Gateway"),
            StatePrerequisite(GoalRequirementKind.STRUCTURE, "CyberneticsCore"),
        ),
    ),
    ProgressActionSpec(
        "Train_VoidRay",
        GoalRequirementKind.UNIT,
        "VoidRay",
        minerals=250,
        vespene=150,
        supply=4,
        prerequisites=(
            StatePrerequisite(GoalRequirementKind.STRUCTURE, "Stargate"),
        ),
    ),
    ProgressActionSpec(
        "Train_Oracle",
        GoalRequirementKind.UNIT,
        "Oracle",
        minerals=150,
        vespene=150,
        supply=3,
        prerequisites=(
            StatePrerequisite(GoalRequirementKind.STRUCTURE, "Stargate"),
        ),
    ),
    ProgressActionSpec(
        "Train_Phoenix",
        GoalRequirementKind.UNIT,
        "Phoenix",
        minerals=150,
        vespene=100,
        supply=2,
        prerequisites=(
            StatePrerequisite(GoalRequirementKind.STRUCTURE, "Stargate"),
        ),
    ),
    ProgressActionSpec(
        "Research_WarpGate",
        GoalRequirementKind.UPGRADE,
        "WarpGate",
        minerals=50,
        vespene=50,
        prerequisites=(
            StatePrerequisite(GoalRequirementKind.STRUCTURE, "CyberneticsCore"),
        ),
    ),
)

_DEFENSIVE_ALERTS = frozenset(
    {"under_attack", "building_under_attack", "unit_under_attack"}
)
_INCOMPLETE_STATUSES = frozenset(
    {"constructing", "in_progress", "pending", "queued", "warping_in"}
)


class GoalProgressVerifier:
    """Evaluate explicit state requirements without interpreting natural language."""

    def __init__(
        self,
        action_specs: tuple[ProgressActionSpec, ...] = PROTOSS_SIMPLE64_ACTION_SPECS,
    ) -> None:
        self._actions_by_name = {spec.name: spec for spec in action_specs}
        self._actions_by_effect = {
            (spec.effect_kind, _canonical(spec.effect_target)): spec for spec in action_specs
        }
        if len(self._actions_by_name) != len(action_specs):
            raise ValueError("progress action names must be unique")
        if len(self._actions_by_effect) != len(action_specs):
            raise ValueError("each progress action must have a unique state effect")

    def goal_from_action_names(
        self,
        *,
        strategic_goal: str,
        action_names: list[str],
        observation: ObservationEnvelope | None = None,
        goal_id: str = "active_goal",
    ) -> GoalSpec:
        """Build an ordered, measurable goal from explicit state-changing actions."""

        if not action_names:
            raise ValueError("at least one state-changing action is required")
        requirements: list[GoalRequirement] = []
        effect_counts: dict[tuple[GoalRequirementKind, str], int] = {}
        effect_baselines: dict[tuple[GoalRequirementKind, str], int] = {}
        previous_id: str | None = None
        for index, action_name in enumerate(action_names, start=1):
            try:
                action = self._actions_by_name[action_name]
            except KeyError as error:
                raise ValueError(f"unsupported goal action: {action_name}") from error
            effect_key = (action.effect_kind, _canonical(action.effect_target))
            effect_counts[effect_key] = effect_counts.get(effect_key, 0) + 1
            if effect_key not in effect_baselines:
                effect_baselines[effect_key] = self._effect_baseline(observation, action)
            target_count = effect_baselines[effect_key] + effect_counts[effect_key]
            requirement_id = f"step-{index}:{action_name}"
            requirements.append(
                GoalRequirement(
                    requirement_id=requirement_id,
                    kind=action.effect_kind,
                    target=action.effect_target,
                    count=target_count,
                    action_name=action.name,
                    depends_on=[] if previous_id is None else [previous_id],
                    description=(
                        f"Have at least {target_count} {action.effect_target}"
                    ),
                )
            )
            previous_id = requirement_id
        return GoalSpec(
            goal_id=goal_id,
            strategic_goal=strategic_goal,
            requirements=requirements,
        )

    def goal_from_active_plan(
        self,
        active_plan: ActivePlanSnapshot,
        *,
        observation: ObservationEnvelope | None = None,
        goal_id: str = "active_goal",
    ) -> GoalSpec:
        """Project registered state-changing commands from an active plan."""

        action_names = [
            command.name
            for command in active_plan.commands
            if command.name in self._actions_by_name
        ]
        if not action_names:
            raise ValueError("active plan has no measurable state-changing commands")
        return self.goal_from_action_names(
            strategic_goal=active_plan.strategic_goal,
            action_names=action_names,
            observation=observation,
            goal_id=goal_id,
        )

    @staticmethod
    def _effect_baseline(
        observation: ObservationEnvelope | None,
        action: ProgressActionSpec,
    ) -> int:
        if observation is None:
            return 0
        completed, state_in_progress = _state_counts(
            observation.state,
            action.effect_kind,
            action.effect_target,
        )
        queued = _queued_count(
            observation.state,
            target=action.effect_target,
            action_name=action.name,
        )
        return completed + max(state_in_progress, queued)

    def verify(
        self,
        observation: ObservationEnvelope,
        goal: GoalSpec,
    ) -> GoalProgressReport:
        """Return measured progress and the currently legal goal-advancing actions."""

        items = {
            requirement.requirement_id: self._measure_requirement(
                observation.state,
                requirement,
            )
            for requirement in goal.requirements
        }
        achieved = [
            items[requirement.requirement_id]
            for requirement in goal.requirements
            if _is_achieved(items[requirement.requirement_id])
        ]
        missing = [
            items[requirement.requirement_id]
            for requirement in goal.requirements
            if not _is_achieved(items[requirement.requirement_id])
        ]

        advancing_actions: list[str] = []
        blockers: list[GoalProgressBlocker] = []
        blocker_keys: set[tuple[str, GoalBlockerKind, str | None]] = set()
        achieved_ids = {item.requirement_id for item in achieved}
        requirements_by_id = {
            requirement.requirement_id: requirement for requirement in goal.requirements
        }
        for requirement in goal.requirements:
            item = items[requirement.requirement_id]
            if _is_achieved(item):
                continue
            self._resolve_requirement(
                observation=observation,
                requirement=requirement,
                item=item,
                requirements_by_id=requirements_by_id,
                items=items,
                achieved_ids=achieved_ids,
                advancing_actions=advancing_actions,
                blockers=blockers,
                blocker_keys=blocker_keys,
                resolving=set(),
            )

        advancing_actions = list(dict.fromkeys(advancing_actions))
        if not missing:
            status = GoalProgressStatus.ACHIEVED
        elif advancing_actions:
            status = GoalProgressStatus.ACTIONABLE
        elif any(
            blocker.kind
            in {GoalBlockerKind.EFFECT_IN_PROGRESS, GoalBlockerKind.PREREQUISITE_IN_PROGRESS}
            for blocker in blockers
        ):
            status = GoalProgressStatus.IN_PROGRESS
        else:
            status = GoalProgressStatus.BLOCKED

        defensive_hold_required = any(
            alert.casefold() in _DEFENSIVE_ALERTS for alert in observation.alerts
        )
        return GoalProgressReport(
            run_id=observation.run_id,
            episode_id=observation.episode_id,
            step_id=observation.step_id,
            game_loop=observation.game_loop,
            goal_id=goal.goal_id,
            strategic_goal=goal.strategic_goal,
            status=status,
            achieved=achieved,
            missing=missing,
            blockers=blockers,
            advancing_actions=advancing_actions,
            unique_next_action=(
                advancing_actions[0] if len(advancing_actions) == 1 else None
            ),
            defensive_hold_required=defensive_hold_required,
        )

    def _measure_requirement(
        self,
        state: SC2State,
        requirement: GoalRequirement,
    ) -> GoalProgressItem:
        current_count, state_in_progress = _state_counts(
            state,
            requirement.kind,
            requirement.target,
        )
        action = self._action_for(requirement)
        queued_count = _queued_count(
            state,
            target=requirement.target,
            action_name=action.name if action is not None else requirement.action_name,
        )
        return GoalProgressItem(
            requirement_id=requirement.requirement_id,
            kind=requirement.kind,
            target=requirement.target,
            required_count=requirement.count,
            current_count=current_count,
            in_progress_count=max(state_in_progress, queued_count),
            description=requirement.description,
        )

    def _resolve_requirement(
        self,
        *,
        observation: ObservationEnvelope,
        requirement: GoalRequirement,
        item: GoalProgressItem,
        requirements_by_id: dict[str, GoalRequirement],
        items: dict[str, GoalProgressItem],
        achieved_ids: set[str],
        advancing_actions: list[str],
        blockers: list[GoalProgressBlocker],
        blocker_keys: set[tuple[str, GoalBlockerKind, str | None]],
        resolving: set[tuple[GoalRequirementKind, str]],
    ) -> None:
        if item.current_count + item.in_progress_count >= item.required_count:
            self._add_blocker(
                blockers,
                blocker_keys,
                requirement.requirement_id,
                GoalBlockerKind.EFFECT_IN_PROGRESS,
                f"{requirement.target} is already in progress",
                requirement.action_name,
            )
            return

        unresolved_dependencies = [
            dependency_id
            for dependency_id in requirement.depends_on
            if dependency_id not in achieved_ids
        ]
        if unresolved_dependencies:
            for dependency_id in unresolved_dependencies:
                dependency = requirements_by_id[dependency_id]
                self._add_blocker(
                    blockers,
                    blocker_keys,
                    requirement.requirement_id,
                    GoalBlockerKind.GOAL_DEPENDENCY,
                    f"waiting for goal requirement {dependency_id}",
                    requirement.action_name,
                )
                dependency_item = items[dependency_id]
                self._resolve_requirement(
                    observation=observation,
                    requirement=dependency,
                    item=dependency_item,
                    requirements_by_id=requirements_by_id,
                    items=items,
                    achieved_ids=achieved_ids,
                    advancing_actions=advancing_actions,
                    blockers=blockers,
                    blocker_keys=blocker_keys,
                    resolving=resolving,
                )
            return

        action = self._action_for(requirement)
        if action is None:
            self._add_blocker(
                blockers,
                blocker_keys,
                requirement.requirement_id,
                GoalBlockerKind.NO_PROGRESS_ACTION,
                f"no registered action creates {requirement.target}",
                requirement.action_name,
            )
            return

        effect_key = (requirement.kind, _canonical(requirement.target))
        if effect_key in resolving:
            raise RuntimeError(f"cyclic action prerequisites for {requirement.target}")
        next_resolving = resolving | {effect_key}
        prerequisites_ready = True
        for prerequisite in action.prerequisites:
            current_count, in_progress_count = _state_counts(
                observation.state,
                prerequisite.kind,
                prerequisite.target,
            )
            if current_count >= prerequisite.count:
                continue
            prerequisites_ready = False
            prerequisite_action = self._actions_by_effect.get(
                (prerequisite.kind, _canonical(prerequisite.target))
            )
            queued_count = _queued_count(
                observation.state,
                target=prerequisite.target,
                action_name=(prerequisite_action.name if prerequisite_action else None),
            )
            if current_count + max(in_progress_count, queued_count) >= prerequisite.count:
                self._add_blocker(
                    blockers,
                    blocker_keys,
                    requirement.requirement_id,
                    GoalBlockerKind.PREREQUISITE_IN_PROGRESS,
                    f"{prerequisite.target} prerequisite is in progress",
                    action.name,
                )
                continue
            self._add_blocker(
                blockers,
                blocker_keys,
                requirement.requirement_id,
                GoalBlockerKind.MISSING_PREREQUISITE,
                f"requires {prerequisite.count} completed {prerequisite.target}",
                action.name,
            )
            synthetic = GoalRequirement(
                requirement_id=f"tech:{prerequisite.kind}:{prerequisite.target}",
                kind=prerequisite.kind,
                target=prerequisite.target,
                count=prerequisite.count,
                action_name=prerequisite_action.name if prerequisite_action else None,
            )
            synthetic_item = self._measure_requirement(observation.state, synthetic)
            self._resolve_requirement(
                observation=observation,
                requirement=synthetic,
                item=synthetic_item,
                requirements_by_id=requirements_by_id,
                items=items,
                achieved_ids=achieved_ids,
                advancing_actions=advancing_actions,
                blockers=blockers,
                blocker_keys=blocker_keys,
                resolving=next_resolving,
            )
        if not prerequisites_ready:
            return

        if observation.state.economy.minerals < action.minerals:
            self._add_blocker(
                blockers,
                blocker_keys,
                requirement.requirement_id,
                GoalBlockerKind.INSUFFICIENT_MINERALS,
                f"requires {action.minerals} minerals",
                action.name,
            )
            return
        if observation.state.economy.vespene < action.vespene:
            self._add_blocker(
                blockers,
                blocker_keys,
                requirement.requirement_id,
                GoalBlockerKind.INSUFFICIENT_VESPENE,
                f"requires {action.vespene} vespene",
                action.name,
            )
            return
        supply_free = (
            observation.state.economy.supply_cap
            - observation.state.economy.supply_used
        )
        if supply_free < action.supply:
            self._add_blocker(
                blockers,
                blocker_keys,
                requirement.requirement_id,
                GoalBlockerKind.INSUFFICIENT_SUPPLY,
                f"requires {action.supply} free supply",
                action.name,
            )
            return
        available = next(
            (
                candidate
                for candidate in observation.available_actions
                if candidate.name == action.name and candidate.actor_scopes
            ),
            None,
        )
        if available is None:
            self._add_blocker(
                blockers,
                blocker_keys,
                requirement.requirement_id,
                GoalBlockerKind.ACTION_UNAVAILABLE,
                f"{action.name} is not dispatchable in this observation",
                action.name,
            )
            return
        advancing_actions.append(action.name)

    def _action_for(self, requirement: GoalRequirement) -> ProgressActionSpec | None:
        if requirement.action_name is not None:
            action = self._actions_by_name.get(requirement.action_name)
            if action is None:
                return None
            if (
                action.effect_kind != requirement.kind
                or _canonical(action.effect_target) != _canonical(requirement.target)
            ):
                return None
            return action
        return self._actions_by_effect.get(
            (requirement.kind, _canonical(requirement.target))
        )

    @staticmethod
    def _add_blocker(
        blockers: list[GoalProgressBlocker],
        blocker_keys: set[tuple[str, GoalBlockerKind, str | None]],
        requirement_id: str,
        kind: GoalBlockerKind,
        detail: str,
        action_name: str | None,
    ) -> None:
        key = (requirement_id, kind, action_name)
        if key in blocker_keys:
            return
        blocker_keys.add(key)
        blockers.append(
            GoalProgressBlocker(
                requirement_id=requirement_id,
                kind=kind,
                detail=detail,
                action_name=action_name,
            )
        )


def _state_counts(
    state: SC2State,
    kind: GoalRequirementKind,
    target: str,
) -> tuple[int, int]:
    canonical_target = _canonical(target)
    if kind == GoalRequirementKind.STRUCTURE:
        matching = [
            structure
            for structure in state.own_structures
            if _canonical(structure.unit_type) == canonical_target
        ]
        completed = sum(1 for structure in matching if _is_complete(structure.status))
        return completed, len(matching) - completed
    if kind == GoalRequirementKind.UNIT:
        matching = [
            unit for unit in state.own_units if _canonical(unit.unit_type) == canonical_target
        ]
        completed = sum(1 for unit in matching if _is_complete(unit.status))
        return completed, len(matching) - completed
    canonical_upgrade_target = _canonical_upgrade(target)
    completed = sum(
        1
        for upgrade in state.upgrades
        if _canonical_upgrade(upgrade) == canonical_upgrade_target
    )
    return completed, 0


def _queued_count(
    state: SC2State,
    *,
    target: str,
    action_name: str | None,
) -> int:
    aliases = {_canonical(target)}
    if action_name is not None:
        aliases.add(_canonical(action_name))
        aliases.add(_canonical(_action_effect_stem(action_name)))
    return sum(1 for item in state.production_queue if _canonical(item.name) in aliases)


def _action_effect_stem(action_name: str) -> str:
    stem = action_name
    for prefix in ("Build_", "Train_", "Research_"):
        if stem.startswith(prefix):
            stem = stem.removeprefix(prefix)
            break
    for suffix in ("_Screen", "_Near"):
        if stem.endswith(suffix):
            stem = stem.removesuffix(suffix)
            break
    return stem


def _is_complete(status: str | None) -> bool:
    return status is None or status.casefold() not in _INCOMPLETE_STATUSES


def _is_achieved(item: GoalProgressItem) -> bool:
    return item.current_count >= item.required_count


def _canonical(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def _canonical_upgrade(value: str) -> str:
    canonical = _canonical(value)
    return "warpgate" if canonical in {"warpgate", "warpgateresearch"} else canonical
