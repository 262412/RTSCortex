"""Structured outputs produced by deliberative modules."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    create_model,
    model_validator,
)

from rtscortex.contracts import ActionArgumentType, AvailableAction, ObservationEnvelope


class AgentOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ReflectionOutput(AgentOutput):
    summary: str = Field(max_length=500)
    lessons: list[str] = Field(default_factory=list, max_length=3)


PositionArgument = Annotated[list[StrictInt], Field(min_length=2, max_length=2)]
ActionArgument = StrictStr | StrictInt | StrictFloat | StrictBool | PositionArgument


class ActionProposal(AgentOutput):
    actor: str
    name: str
    arguments: list[ActionArgument] = Field(default_factory=list)
    priority: int = Field(default=50, ge=0, le=89)


class PlanningOutput(AgentOutput):
    strategic_goal: str = Field(max_length=200)
    steps: list[str] = Field(default_factory=list, max_length=3)
    proposed_actions: list[ActionProposal] = Field(default_factory=list, max_length=3)


def project_planning_observation(observation: ObservationEnvelope) -> ObservationEnvelope:
    """Expose only actions that are legal for the current typed opening state."""

    state = observation.state
    supply_free = state.economy.supply_cap - state.economy.supply_used
    has_pending_pylon = any(
        structure.unit_type == "Pylon" and structure.status == "constructing"
        for structure in state.own_structures
    )
    has_gateway = any(structure.unit_type == "Gateway" for structure in state.own_structures)
    has_completed_gateway = any(
        structure.unit_type == "Gateway" and structure.status != "constructing"
        for structure in state.own_structures
    )
    has_completed_core = any(
        structure.unit_type == "CyberneticsCore" and structure.status != "constructing"
        for structure in state.own_structures
    )

    enemy_ids = list(
        dict.fromkeys(_normalize_tag(enemy.unit_id) for enemy in state.visible_enemies)
    )[:8]
    available_actions: list[AvailableAction] = []
    for action in observation.available_actions:
        if action.name == "No_Operation":
            continue
        if action.name == "Attack_Unit":
            actor_scopes = [actor for actor in action.actor_scopes if _is_combat_actor(actor)]
            if not enemy_ids or not actor_scopes:
                continue
            enemy_id_set = set(enemy_ids)
            candidates = action.argument_candidates or []
            candidates = [
                candidate
                for candidate in candidates
                if candidate and _normalize_tag(candidate[0]) in enemy_id_set
            ][:8]
            if not candidates:
                continue
            action = action.model_copy(
                update={
                    "actor_scopes": actor_scopes,
                    "argument_candidates": candidates,
                }
            )
        elif _requires_argument_candidates(action) and not action.argument_candidates:
            continue
        if (
            (action.name == "Build_Pylon_Screen" and (supply_free > 4 or has_pending_pylon))
            or (action.name == "Build_Gateway_Screen" and has_gateway)
            or (
                action.name == "Train_Stalker"
                and (not has_completed_core or state.economy.vespene < 50)
            )
        ):
            continue
        available_actions.append(action)
    available_names = {action.name for action in available_actions}
    if (
        has_completed_gateway
        and state.economy.army_supply == 0
        and "Train_Zealot" in available_names
    ):
        available_actions = [
            action for action in available_actions if action.name == "Train_Zealot"
        ]

    exposed_build_actions = {
        action.name for action in available_actions if action.name.startswith("Build_")
    }
    text_observation = "\n".join(
        line
        for line in observation.text_observation.splitlines()
        if not _is_hidden_build_candidate(line, exposed_build_actions)
    )
    return observation.model_copy(
        update={
            "available_actions": available_actions,
            "text_observation": text_observation,
        }
    )


def _is_combat_actor(actor: str) -> bool:
    return actor == "army" or actor.startswith("CombatGroup")


def _normalize_tag(value: object) -> str:
    if isinstance(value, int) and not isinstance(value, bool):
        return hex(value)
    return str(value).casefold()


def _requires_argument_candidates(action: AvailableAction) -> bool:
    return any(
        argument_type in {ActionArgumentType.POSITION, ActionArgumentType.TAG}
        for argument_type in action.argument_types
    )


def _is_hidden_build_candidate(line: str, exposed_build_actions: set[str]) -> bool:
    stripped = line.strip()
    action_name, marker, _ = stripped.partition(" candidates:")
    return bool(
        marker and action_name.startswith("Build_") and action_name not in exposed_build_actions
    )


def planning_output_model(observation: ObservationEnvelope) -> type[PlanningOutput]:
    """Constrain each structured proposal to one action visible this tick."""

    observation = project_planning_observation(observation)
    proposal_models = [
        proposal_model
        for index, action in enumerate(observation.available_actions)
        if action.actor_scopes
        for proposal_model in _action_proposal_models(index, action)
    ]
    if not proposal_models:
        output_model = create_model(
            "AvailablePlanningOutput",
            __base__=PlanningOutput,
            proposed_actions=(
                list[ActionProposal],
                Field(default_factory=list, max_length=0),
            ),
        )
        return output_model

    proposal_type = (
        proposal_models[0]
        if len(proposal_models) == 1
        else cast(Any, Union)[tuple(proposal_models)]
    )
    proposal_list: Any = list.__class_getitem__(proposal_type)
    output_model = create_model(
        "AvailablePlanningOutput",
        __base__=PlanningOutput,
        proposed_actions=(
            proposal_list,
            Field(default_factory=list, max_length=3),
        ),
    )
    return output_model


def _action_proposal_models(
    index: int,
    action: AvailableAction,
) -> list[type[ActionProposal]]:
    if action.argument_candidates is None:
        return [_action_proposal_model(index, action, None, 0)]
    return [
        _action_proposal_model(index, action, candidate, candidate_index)
        for candidate_index, candidate in enumerate(action.argument_candidates)
    ]


def _action_proposal_model(
    index: int,
    action: AvailableAction,
    candidate: list[Any] | None,
    candidate_index: int,
) -> type[ActionProposal]:
    argument_types = action.argument_types or [
        ActionArgumentType.ANY for _ in action.argument_names
    ]
    python_types = (
        tuple(_argument_python_type(argument_type) for argument_type in argument_types)
        if candidate is None
        else tuple(
            _candidate_argument_python_type(value, argument_type)
            for value, argument_type in zip(candidate, argument_types, strict=True)
        )
    )
    arguments_type = tuple.__class_getitem__(python_types)
    name_literal = cast(Any, Literal)[(action.name,)]
    actor_literal = cast(Any, Literal)[tuple(dict.fromkeys(action.actor_scopes))]
    validators = (
        {}
        if candidate is None
        else {
            "validate_exact_candidate": _candidate_arguments_validator(
                candidate,
                argument_types,
            )
        }
    )
    return create_model(
        f"AvailableActionProposal{index}Candidate{candidate_index}",
        __base__=ActionProposal,
        __validators__=validators,
        actor=(actor_literal, ...),
        name=(name_literal, ...),
        arguments=(arguments_type, ...),
    )


def _candidate_arguments_validator(
    candidate: list[Any],
    argument_types: list[ActionArgumentType],
) -> Any:
    @model_validator(mode="before")
    def validate_exact_candidate(value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        arguments = value.get("arguments")
        if not isinstance(arguments, (list, tuple)) or not _strict_candidate_match(
            list(arguments),
            candidate,
            argument_types,
        ):
            raise ValueError("arguments must exactly match one available argument candidate")
        return value

    return validate_exact_candidate


def _strict_candidate_match(
    arguments: list[Any],
    candidate: list[Any],
    argument_types: list[ActionArgumentType],
) -> bool:
    if len(arguments) != len(candidate):
        return False
    for argument, expected, argument_type in zip(
        arguments,
        candidate,
        argument_types,
        strict=True,
    ):
        if argument_type is ActionArgumentType.POSITION:
            if not isinstance(argument, (list, tuple)) or list(argument) != expected:
                return False
            if any(type(coordinate) is not int for coordinate in argument):
                return False
        elif argument_type is ActionArgumentType.TAG:
            if type(argument) is not str or argument.casefold() != str(expected).casefold():
                return False
        elif argument_type is ActionArgumentType.INTEGER:
            if type(argument) is not int or argument != expected:
                return False
        elif argument_type is ActionArgumentType.NUMBER:
            if type(argument) not in {int, float} or type(argument) is not type(expected):
                return False
            if argument != expected:
                return False
        elif argument_type is ActionArgumentType.BOOLEAN:
            if type(argument) is not bool or argument is not expected:
                return False
        elif type(argument) is not type(expected) or argument != expected:
            return False
    return True


def _candidate_argument_python_type(value: Any, argument_type: ActionArgumentType) -> Any:
    if argument_type is ActionArgumentType.POSITION:
        x, y = value
        x_literal = cast(Any, Literal)[(x,)]
        y_literal = cast(Any, Literal)[(y,)]
        return tuple.__class_getitem__((x_literal, y_literal))
    try:
        hash(value)
    except TypeError:
        return _argument_python_type(argument_type)
    return cast(Any, Literal)[(value,)]


def _argument_python_type(argument_type: ActionArgumentType) -> Any:
    return {
        ActionArgumentType.STRING: StrictStr,
        ActionArgumentType.INTEGER: StrictInt,
        ActionArgumentType.NUMBER: StrictInt | StrictFloat,
        ActionArgumentType.BOOLEAN: StrictBool,
        ActionArgumentType.POSITION: PositionArgument,
        ActionArgumentType.TAG: StrictStr | StrictInt,
        ActionArgumentType.ANY: ActionArgument,
    }[argument_type]
