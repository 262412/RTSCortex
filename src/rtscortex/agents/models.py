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
    ttl_game_loops: int = Field(default=32, ge=1)


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

    available_actions = [
        action
        for action in observation.available_actions
        if not (
            (action.name == "Build_Pylon_Screen" and (supply_free > 4 or has_pending_pylon))
            or (action.name == "Build_Gateway_Screen" and has_gateway)
            or (
                action.name == "Train_Stalker"
                and (not has_completed_core or state.economy.vespene < 50)
            )
        )
    ]
    available_names = {action.name for action in available_actions}
    if (
        has_completed_gateway
        and state.economy.army_supply == 0
        and "Train_Zealot" in available_names
    ):
        available_actions = [
            action
            for action in available_actions
            if action.name in {"Train_Zealot", "No_Operation"}
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
        _action_proposal_model(index, action)
        for index, action in enumerate(observation.available_actions)
        if action.actor_scopes
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


def _action_proposal_model(index: int, action: AvailableAction) -> type[ActionProposal]:
    argument_types = action.argument_types or [
        ActionArgumentType.ANY for _ in action.argument_names
    ]
    arguments_type = tuple.__class_getitem__(
        tuple(_argument_python_type(argument_type) for argument_type in argument_types)
    )
    name_literal = cast(Any, Literal)[(action.name,)]
    actor_literal = cast(Any, Literal)[tuple(dict.fromkeys(action.actor_scopes))]
    return create_model(
        f"AvailableActionProposal{index}",
        __base__=ActionProposal,
        actor=(actor_literal, ...),
        name=(name_literal, ...),
        arguments=(arguments_type, ...),
    )


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
