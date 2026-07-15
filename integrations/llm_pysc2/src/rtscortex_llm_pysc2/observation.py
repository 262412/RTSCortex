"""Convert JSON-safe LLM-PySC2 snapshots into RTSCortex observations."""

from __future__ import annotations

import json
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from typing import Any, cast


@dataclass(frozen=True)
class ScreenCandidateMetadata:
    """Bridge-only world target associated with one wire-level screen candidate."""

    world_target: tuple[float, float]
    anchor_tag: int


class BridgeAvailableAction(dict[str, Any]):
    """A JSON-compatible action carrying metadata that never enters the wire payload."""

    def __init__(self, value: Mapping[str, Any]) -> None:
        super().__init__(value)
        self.screen_metadata_by_actor: dict[
            str, dict[tuple[int, int], ScreenCandidateMetadata]
        ] = {}

    def add_screen_metadata(
        self,
        actor: str,
        metadata: Mapping[tuple[int, int], ScreenCandidateMetadata],
    ) -> None:
        if metadata:
            self.screen_metadata_by_actor[actor] = dict(metadata)

    def screen_metadata(
        self,
        actor: str,
        screen_target: tuple[int, int],
    ) -> ScreenCandidateMetadata | None:
        return self.screen_metadata_by_actor.get(actor, {}).get(screen_target)


class ObservationMapper:
    """Map a worker snapshot to the versioned ``ObservationEnvelope`` wire shape.

    The live worker is responsible for extracting this JSON-safe snapshot from
    PySC2 objects. Keeping that extraction boundary explicit lets this mapper and
    its contract tests run without importing PySC2.
    """

    def map(self, snapshot: Mapping[str, Any]) -> dict[str, Any]:
        player = _mapping(snapshot["player_common"], "player_common")
        units = [_mapping(item, "units item") for item in _list(snapshot["units"], "units")]

        own_units: list[dict[str, Any]] = []
        own_structures: list[dict[str, Any]] = []
        visible_enemies: list[dict[str, Any]] = []
        for unit in units:
            mapped = _map_unit(unit)
            if unit["alliance"] == "enemy":
                visible_enemies.append(mapped)
            elif unit["alliance"] == "self" and bool(unit["is_structure"]):
                own_structures.append(mapped)
            elif unit["alliance"] == "self":
                own_units.append(mapped)

        return {
            "protocol_version": "1.1",
            "run_id": str(snapshot["run_id"]),
            "episode_id": str(snapshot["episode_id"]),
            "step_id": int(snapshot["step_id"]),
            "game_loop": int(snapshot["game_loop"]),
            "observed_at": str(snapshot["observed_at"]),
            "state": {
                "economy": {
                    "minerals": int(player["minerals"]),
                    "vespene": int(player["vespene"]),
                    "supply_used": int(player["food_used"]),
                    "supply_cap": int(player["food_cap"]),
                    "workers": int(player["food_workers"]),
                    "army_supply": int(player["food_army"]),
                },
                "production_queue": [
                    _map_production_item(_mapping(item, "production_queue item"))
                    for item in _list(snapshot["production_queue"], "production_queue")
                ],
                "own_units": own_units,
                "own_structures": own_structures,
                "visible_enemies": visible_enemies,
                "upgrades": [str(item) for item in _list(snapshot["upgrades"], "upgrades")],
            },
            "text_observation": str(snapshot["text_observation"]),
            "available_actions": _map_available_actions(snapshot["teams"]),
            "alerts": [str(item) for item in _list(snapshot["alerts"], "alerts")],
            "image_uri": cast(Any, snapshot["image_uri"]),
        }


def canonical_actor(agent_name: str, team_name: str) -> str:
    """Return the stable actor scope shared by observations and actions."""

    if not agent_name or not team_name or "/" in agent_name or "/" in team_name:
        raise ValueError("actor components must be non-empty and cannot contain '/'")
    return f"{agent_name}/{team_name}"


def split_actor(actor: str) -> tuple[str, str]:
    """Split and validate a canonical ``agent/team`` actor scope."""

    parts = actor.split("/")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"invalid actor scope {actor!r}; expected 'agent/team'")
    return parts[0], parts[1]


def _map_unit(unit: Mapping[str, Any]) -> dict[str, Any]:
    health = float(unit["health"])
    health_max = float(unit["health_max"])
    health_fraction = health / health_max if health_max > 0 else 0.0
    position = _list(unit["position"], "unit position")
    if len(position) != 2:
        raise ValueError("unit position must contain exactly two coordinates")
    return {
        "unit_id": _format_tag(unit["tag"]),
        "unit_type": str(unit["unit_type"]),
        "alliance": str(unit["alliance"]),
        "position": [float(position[0]), float(position[1])],
        "health_fraction": health_fraction,
        "energy": None if unit["energy"] is None else float(unit["energy"]),
        "status": None if unit["status"] is None else str(unit["status"]),
    }


def _map_production_item(item: Mapping[str, Any]) -> dict[str, Any]:
    producer_id = item["producer_tag"]
    return {
        "name": str(item["name"]),
        "producer_id": None if producer_id is None else _format_tag(producer_id),
        "progress": float(item["progress"]),
    }


def _map_available_actions(value: Any) -> list[dict[str, Any]]:
    teams = [_mapping(item, "teams item") for item in _list(value, "teams")]
    actions: MutableMapping[tuple[str, tuple[str, ...], tuple[str, ...], str], dict[str, Any]] = {}
    for team in teams:
        actor = canonical_actor(str(team["agent_name"]), str(team["team_name"]))
        for item in _list(team["available_actions"], "available_actions"):
            action = _mapping(item, "available_actions item")
            argument_names = tuple(
                str(name) for name in _list(action["argument_names"], "argument_names")
            )
            argument_types = tuple(
                str(name) for name in _list(action["argument_types"], "argument_types")
            )
            if len(argument_names) != len(argument_types):
                raise ValueError("argument_types must match argument_names")
            argument_candidates = _map_argument_candidates(
                action.get("argument_candidates"),
                argument_types,
            )
            candidates_key = json.dumps(argument_candidates, separators=(",", ":"))
            key = (str(action["name"]), argument_names, argument_types, candidates_key)
            if key not in actions:
                actions[key] = BridgeAvailableAction(
                    {
                        "name": key[0],
                        "argument_names": list(argument_names),
                        "argument_types": list(argument_types),
                        "argument_candidates": argument_candidates,
                        "actor_scopes": [],
                    }
                )
            mapped_action = actions[key]
            cast(list[str], mapped_action["actor_scopes"]).append(actor)
            if isinstance(mapped_action, BridgeAvailableAction):
                mapped_action.add_screen_metadata(
                    actor,
                    _map_screen_provenance(action.get("bridge_screen_provenance")),
                )
    return list(actions.values())


def _map_screen_provenance(
    value: Any,
) -> dict[tuple[int, int], ScreenCandidateMetadata]:
    if value is None:
        return {}
    result: dict[tuple[int, int], ScreenCandidateMetadata] = {}
    for item_value in _list(value, "bridge_screen_provenance"):
        item = _mapping(item_value, "bridge_screen_provenance item")
        screen = _list(item["screen_target"], "screen_target")
        world = _list(item["world_target"], "world_target")
        if (
            len(screen) != 2
            or not all(isinstance(coordinate, int) for coordinate in screen)
            or len(world) != 2
        ):
            raise ValueError("screen provenance must contain two screen and world coordinates")
        target = (int(screen[0]), int(screen[1]))
        if target in result:
            raise ValueError("screen provenance cannot contain duplicate candidates")
        result[target] = ScreenCandidateMetadata(
            world_target=(float(world[0]), float(world[1])),
            anchor_tag=int(item["anchor_tag"]),
        )
    return result


def _map_argument_candidates(
    value: Any,
    argument_types: tuple[str, ...],
) -> list[list[Any]] | None:
    if value is None:
        return None
    candidates = _list(value, "argument_candidates")
    mapped = []
    for candidate_value in candidates:
        candidate = _list(candidate_value, "argument candidate")
        if len(candidate) != len(argument_types):
            raise ValueError("argument candidate length must match argument_types")
        mapped.append(
            [
                _map_candidate_argument(argument_types[index], argument)
                for index, argument in enumerate(candidate)
            ]
        )
    return mapped


def _map_candidate_argument(argument_type: str, value: Any) -> Any:
    if argument_type == "tag":
        return _format_tag(value)
    if argument_type == "position":
        position = _list(value, "position candidate")
        if len(position) != 2 or not all(
            isinstance(coordinate, int) and not isinstance(coordinate, bool)
            for coordinate in position
        ):
            raise ValueError("position candidate must contain exactly two integers")
        return position
    return value


def _format_tag(value: Any) -> str:
    if isinstance(value, bool):
        raise ValueError("unit tag cannot be boolean")
    if isinstance(value, int):
        return hex(value)
    if isinstance(value, str):
        return hex(int(value, 0))
    raise ValueError(f"unsupported unit tag {value!r}")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return cast(Mapping[str, Any], value)


def _list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    return value
