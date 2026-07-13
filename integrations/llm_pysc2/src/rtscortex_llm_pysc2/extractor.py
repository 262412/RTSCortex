"""Extract a JSON-safe RTSCortex snapshot from an upstream PySC2 timestep."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any, Optional

SUPPORTED_ARGUMENTS = frozenset({"minimap", "screen", "tag"})
ALLIANCES = {1: "self", 2: "ally", 3: "neutral", 4: "enemy"}
SC2_ALERT_NAMES = {6: "building_under_attack", 19: "unit_under_attack"}


class TimeStepExtractor:
    """Read only the stable fields used by the v1 observation contract.

    PySC2 and LLM-PySC2 are intentionally not imported by this module. The live
    worker passes their objects in, while unit names and building types can be
    injected for deterministic tests.
    """

    def __init__(
        self,
        run_id: str,
        episode_id: str,
        *,
        unit_names: Optional[Mapping[int, str]] = None,
        building_types: Sequence[int] = (),
    ) -> None:
        self.run_id = run_id
        self.episode_id = episode_id
        self.unit_names = dict(unit_names or {})
        self.building_types = frozenset(int(value) for value in building_types)

    def extract(
        self,
        timestep: Any,
        agents: Mapping[str, Any],
        text_observations: Mapping[str, str],
        *,
        step_id: int,
    ) -> dict[str, Any]:
        observation = timestep.observation
        player = _value(observation, "player_common", _value(observation, "player", None))
        if player is None:
            raise ValueError("PySC2 observation has no player data")

        raw_units = list(_value(observation, "raw_units", ()))
        return {
            "run_id": self.run_id,
            "episode_id": self.episode_id,
            "step_id": int(step_id),
            "game_loop": int(_scalar(_value(observation, "game_loop", 0))),
            "observed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "player_common": _extract_player(player),
            "production_queue": _extract_production_queue(observation),
            "units": [self._extract_unit(unit) for unit in raw_units],
            "upgrades": [f"upgrade:{int(value)}" for value in _value(observation, "upgrades", ())],
            "teams": _extract_team_actions(agents),
            "text_observation": "\n\n".join(
                f"[{name}]\n{text_observations[name]}" for name in sorted(text_observations)
            ),
            "alerts": [_alert_name(value) for value in _value(observation, "alerts", ())],
            "image_uri": None,
        }

    def _extract_unit(self, unit: Any) -> dict[str, Any]:
        unit_type = int(_value(unit, "unit_type", 0))
        health = float(_value(unit, "health", 0.0))
        health_ratio = float(_value(unit, "health_ratio", 0.0)) / 255.0
        health_max = float(_value(unit, "health_max", 0.0))
        if health_max <= 0:
            health_max = health / health_ratio if health_ratio > 0 else max(health, 1.0)
        order_length = int(_value(unit, "order_length", 0))
        return {
            "tag": int(_value(unit, "tag", 0)),
            "unit_type": self.unit_names.get(unit_type, f"unit:{unit_type}"),
            "alliance": ALLIANCES.get(int(_value(unit, "alliance", 0)), "neutral"),
            "is_structure": unit_type in self.building_types,
            "position": [float(_value(unit, "x", 0.0)), float(_value(unit, "y", 0.0))],
            "health": health,
            "health_max": health_max,
            "energy": float(_value(unit, "energy", 0.0)),
            "status": "idle" if order_length == 0 else "active",
        }


def _extract_player(player: Any) -> dict[str, int]:
    return {
        "minerals": int(_value(player, "minerals", 0)),
        "vespene": int(_value(player, "vespene", 0)),
        "food_used": int(_value(player, "food_used", 0)),
        "food_cap": int(_value(player, "food_cap", 0)),
        "food_workers": int(_value(player, "food_workers", 0)),
        "food_army": int(_value(player, "food_army", 0)),
    }


def _extract_production_queue(observation: Any) -> list[dict[str, Any]]:
    result = []
    for item in _value(observation, "production_queue", ()):
        ability_id = int(_value(item, "ability_id", 0))
        progress = float(_value(item, "build_progress", 0.0))
        if progress > 1.0:
            progress /= 100.0
        result.append(
            {
                "name": f"ability:{ability_id}",
                "producer_tag": None,
                "progress": min(max(progress, 0.0), 1.0),
            }
        )
    return result


def _extract_team_actions(agents: Mapping[str, Any]) -> list[dict[str, Any]]:
    teams = []
    for agent_name in sorted(agents):
        agent = agents[agent_name]
        team_definitions = {
            str(team["name"]): team for team in agent.config.AGENTS[agent_name]["team"]
        }
        team_observations = list(agent.team_unit_obs_list)
        for index, team_name in enumerate(agent.team_unit_team_list):
            team = team_definitions[str(team_name)]
            available_ids = _available_function_ids(team_observations, index)
            actions = _available_team_actions(agent, team, available_ids)
            teams.append(
                {
                    "agent_name": agent_name,
                    "team_name": str(team_name),
                    "available_actions": actions,
                }
            )
    return teams


def _available_team_actions(
    agent: Any, team: Mapping[str, Any], available_ids: Optional[frozenset[int]]
) -> list[dict[str, Any]]:
    action_space = agent.config.AGENTS[agent.name]["action"]
    unit_types = list(team.get("unit_type", ())) or ["EmptyGroup"]
    candidates = [action for unit_type in unit_types for action in action_space.get(unit_type, ())]
    result: list[dict[str, Any]] = []
    seen = set()
    for action in candidates:
        argument_names = tuple(str(value) for value in action.get("arg", ()))
        if not set(argument_names).issubset(SUPPORTED_ARGUMENTS):
            continue
        function_ids = [int(triple[0]) for triple in action.get("func", ())]
        if available_ids is not None and any(value not in available_ids for value in function_ids):
            continue
        key = (str(action["name"]), argument_names)
        if key in seen:
            continue
        seen.add(key)
        result.append(
            {
                "name": key[0],
                "argument_names": list(argument_names),
                "argument_types": [
                    "tag" if name == "tag" else "position" for name in argument_names
                ],
            }
        )

    if not any(action["name"] == "No_Operation" for action in result):
        result.insert(0, {"name": "No_Operation", "argument_names": [], "argument_types": []})
    return result


def _available_function_ids(
    team_observations: Sequence[Any], index: int
) -> Optional[frozenset[int]]:
    if index >= len(team_observations):
        return None
    values = _value(team_observations[index].observation, "available_actions", ())
    return frozenset(int(value) for value in values)


def _value(value: Any, name: str, default: Any) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _scalar(value: Any) -> Any:
    if isinstance(value, (str, bytes)):
        return value
    try:
        if len(value) == 1:
            return value[0]
    except (TypeError, IndexError):
        pass
    return value


def _alert_name(value: Any) -> str:
    alert = int(value)
    return SC2_ALERT_NAMES.get(alert, f"alert:{alert}")
