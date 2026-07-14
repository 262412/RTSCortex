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
        action_source_types: Optional[Mapping[int, int]] = None,
    ) -> None:
        self.run_id = run_id
        self.episode_id = episode_id
        self.unit_names = dict(unit_names or {})
        self.building_types = frozenset(int(value) for value in building_types)
        self.action_source_types = {
            int(function_id): int(unit_type)
            for function_id, unit_type in (action_source_types or {}).items()
        }

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
        text_observation = "\n\n".join(
            f"[{name}]\n{text_observations[name]}" for name in sorted(text_observations)
        )
        build_candidates = _build_screen_candidate_lines(observation)
        if build_candidates:
            text_observation = "\n\n".join(
                [text_observation, "[RTSCortex Build Candidates]", *build_candidates]
            )
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
            "teams": _extract_team_actions(
                agents,
                owned_unit_types={
                    int(_value(unit, "unit_type", 0))
                    for unit in raw_units
                    if int(_value(unit, "alliance", 0)) == 1
                },
                action_source_types=self.action_source_types,
            ),
            "text_observation": text_observation,
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


def _extract_team_actions(
    agents: Mapping[str, Any],
    *,
    owned_unit_types: set[int],
    action_source_types: Mapping[int, int],
) -> list[dict[str, Any]]:
    teams = []
    for agent_name in sorted(agents):
        agent = agents[agent_name]
        team_definitions = {
            str(team["name"]): team for team in agent.config.AGENTS[agent_name]["team"]
        }
        team_observations = list(agent.team_unit_obs_list)
        for index, team_name in enumerate(current_team_order(agent)):
            team = team_definitions[str(team_name)]
            available_ids = (
                None
                if str(team_name) == "Empty"
                else _available_function_ids(team_observations, index)
            )
            actions = _available_team_actions(
                agent,
                team,
                available_ids,
                owned_unit_types=owned_unit_types,
                action_source_types=action_source_types,
            )
            teams.append(
                {
                    "agent_name": agent_name,
                    "team_name": str(team_name),
                    "available_actions": actions,
                }
            )
    return teams


def current_team_order(agent: Any) -> tuple[str, ...]:
    """Return the positional team order, including upstream's implicit Empty team."""

    team_names = [str(value) for value in agent.team_unit_team_list]
    if getattr(agent, "flag_enable_empty_unit_group", False):
        configured_teams = agent.config.AGENTS[agent.name]["team"]
        for team in configured_teams:
            name = str(team["name"])
            if name == "Empty" and name not in team_names:
                team_names.append(name)
    return tuple(team_names)


def _available_team_actions(
    agent: Any,
    team: Mapping[str, Any],
    available_ids: Optional[frozenset[int]],
    *,
    owned_unit_types: set[int],
    action_source_types: Mapping[int, int],
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
        required_sources = {
            action_source_types[function_id]
            for function_id in function_ids
            if function_id in action_source_types
        }
        if required_sources and not required_sources.issubset(owned_unit_types):
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


def _build_screen_candidate_lines(observation: Any) -> list[str]:
    pylon = build_screen_candidates(observation, "Build_Pylon_Screen")
    gateway = build_screen_candidates(observation, "Build_Gateway_Screen")
    result = []
    if pylon:
        result.append(f"Build_Pylon_Screen candidates: {pylon}")
    if gateway:
        result.append(f"Build_Gateway_Screen candidates: {gateway}")
    return result


def build_screen_candidates(observation: Any, action_name: str) -> list[list[int]]:
    specifications = {
        "Build_Pylon_Screen": (2, False),
        "Build_Gateway_Screen": (3, True),
    }
    if action_name not in specifications:
        return []
    feature_screen = _value(observation, "feature_screen", None)
    if feature_screen is None:
        return []
    buildable = _value(feature_screen, "buildable", None)
    pathable = _value(feature_screen, "pathable", None)
    player_relative = _value(feature_screen, "player_relative", None)
    power = _value(feature_screen, "power", None)
    shape = getattr(buildable, "shape", ())
    if not shape or buildable is None or pathable is None or player_relative is None:
        return []
    screen_size = int(shape[0])
    building_size, require_power = specifications[action_name]
    return _valid_build_positions(
        buildable,
        pathable,
        player_relative,
        power,
        screen_size=screen_size,
        building_size=building_size,
        require_power=require_power,
    )


def _valid_build_positions(
    buildable: Any,
    pathable: Any,
    player_relative: Any,
    power: Any,
    *,
    screen_size: int,
    building_size: int,
    require_power: bool,
) -> list[list[int]]:
    ratio = max(1, int(screen_size / 24))
    stride = max(4, ratio)
    candidates: list[tuple[float, int, int]] = []
    for x0 in range(stride, screen_size, stride):
        for y0 in range(stride, screen_size, stride):
            if require_power and (
                power is None or power[x0][y0] != 1 or power[y0][x0] != 1
            ):
                continue
            x1 = int(x0 - ratio * (building_size - 1) / 2)
            y1 = int(y0 - ratio * (building_size - 1) / 2)
            valid = True
            for i in range(building_size):
                for j in range(building_size):
                    x = int(x1 + i * ratio)
                    y = int(y1 + j * ratio)
                    if not (0 < x < screen_size and 0 < y < screen_size):
                        valid = False
                    elif not _build_cell_is_valid_in_both_coordinate_orders(
                        buildable,
                        pathable,
                        player_relative,
                        x,
                        y,
                    ):
                        valid = False
            if valid:
                distance = (x0 - screen_size / 2) ** 2 + (y0 - screen_size / 2) ** 2
                candidates.append((distance, x0, y0))
    candidates.sort()
    return [[x, y] for _, x, y in candidates[:8]]


def _build_cell_is_valid_in_both_coordinate_orders(
    buildable: Any,
    pathable: Any,
    player_relative: Any,
    x: int,
    y: int,
) -> bool:
    """Satisfy PySC2's row-major plane and LLM-PySC2's transposed validator."""

    for row, column in ((y, x), (x, y)):
        if buildable[row][column] != 1 or pathable[row][column] != 1:
            return False
        if player_relative[row][column] not in (0, 1):
            return False
    return True


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
