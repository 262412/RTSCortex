"""Route RTSCortex actions into LLM-PySC2's positional team protocol."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from rtscortex_llm_pysc2.observation import split_actor


@dataclass(frozen=True)
class RoutedCommand:
    command_id: str
    actor: str
    team_name: str
    name: str
    rendered_action: str

    def to_dict(self) -> dict[str, str]:
        return {
            "command_id": self.command_id,
            "actor": self.actor,
            "team_name": self.team_name,
            "name": self.name,
            "rendered_action": self.rendered_action,
        }


@dataclass(frozen=True)
class RoutedActionBatch:
    protocol_version: str
    run_id: str
    episode_id: str
    step_id: int
    decision_id: str
    agent_name: str
    team_order: tuple[str, ...]
    commands: tuple[RoutedCommand, ...]
    action_text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "run_id": self.run_id,
            "episode_id": self.episode_id,
            "step_id": self.step_id,
            "decision_id": self.decision_id,
            "agent_name": self.agent_name,
            "team_order": list(self.team_order),
            "commands": [command.to_dict() for command in self.commands],
            "action_text": self.action_text,
        }


class ActionRouter:
    """Render commands in the exact order used by ``team_unit_team_list``."""

    def route(
        self,
        batch: Mapping[str, Any],
        *,
        agent_name: str,
        team_order: Sequence[str],
        available_actions: Sequence[Mapping[str, Any]],
    ) -> RoutedActionBatch:
        order = tuple(team_order)
        if len(order) != len(set(order)):
            raise ValueError("team_order cannot contain duplicates")

        specifications = _action_specifications(available_actions)
        commands_by_team: dict[str, list[RoutedCommand]] = {team: [] for team in order}
        for value in batch["commands"]:
            command = _mapping(value, "command")
            actor = str(command["actor"])
            command_agent, team_name = split_actor(actor)
            if command_agent != agent_name:
                continue
            if team_name not in commands_by_team:
                raise ValueError(f"actor {actor!r} is absent from the current team order")

            name = str(command["name"])
            specification_key = (name, actor)
            if specification_key not in specifications:
                raise ValueError(f"action {name!r} is unavailable for actor {actor!r}")
            argument_names, argument_types = specifications[specification_key]
            arguments = _list(command["arguments"], "command arguments")
            if len(arguments) != len(argument_names):
                raise ValueError(
                    f"action {name!r} expects {len(argument_names)} arguments, got {len(arguments)}"
                )
            rendered_arguments = ", ".join(
                _format_argument(argument_types[index], arguments[index])
                for index in range(len(argument_names))
            )
            commands_by_team[team_name].append(
                RoutedCommand(
                    command_id=str(command["command_id"]),
                    actor=actor,
                    team_name=team_name,
                    name=name,
                    rendered_action=f"<{name}({rendered_arguments})>",
                )
            )

        lines = ["Actions:"]
        routed_commands: list[RoutedCommand] = []
        for team_name in order:
            lines.append(f"    Team {team_name}:")
            team_commands = commands_by_team[team_name]
            if not team_commands:
                lines.append("        <No_Operation()>")
                continue
            for routed_command in team_commands:
                lines.append(f"        {routed_command.rendered_action}")
                routed_commands.append(routed_command)

        return RoutedActionBatch(
            protocol_version=str(batch["protocol_version"]),
            run_id=str(batch["run_id"]),
            episode_id=str(batch["episode_id"]),
            step_id=int(batch["step_id"]),
            decision_id=str(batch["decision_id"]),
            agent_name=agent_name,
            team_order=order,
            commands=tuple(routed_commands),
            action_text="\n".join(lines),
        )


def _action_specifications(
    available_actions: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], tuple[tuple[str, ...], tuple[str, ...]]]:
    specifications: dict[tuple[str, str], tuple[tuple[str, ...], tuple[str, ...]]] = {}
    for action in available_actions:
        name = str(action["name"])
        arguments = tuple(str(item) for item in _list(action["argument_names"], "argument_names"))
        argument_types = tuple(
            str(item) for item in _list(action["argument_types"], "argument_types")
        )
        if len(arguments) != len(argument_types):
            raise ValueError("argument_types must match argument_names")
        for actor_value in _list(action["actor_scopes"], "actor_scopes"):
            actor = str(actor_value)
            split_actor(actor)
            specifications[(name, actor)] = (arguments, argument_types)
    return specifications


def _format_argument(argument_type: str, value: Any) -> str:
    if argument_type == "tag":
        return _format_tag(value)
    if argument_type == "position":
        return json.dumps(value, separators=(",", ":"))
    if isinstance(value, (list, dict)):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


def _format_tag(value: Any) -> str:
    if isinstance(value, bool):
        raise ValueError("action tag cannot be boolean")
    if isinstance(value, int):
        return hex(value)
    if isinstance(value, str):
        return hex(int(value, 0))
    raise ValueError(f"unsupported action tag {value!r}")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    return value
