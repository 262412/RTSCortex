"""Extract a JSON-safe RTSCortex snapshot from an upstream PySC2 timestep."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from rtscortex_llm_pysc2.production import (
    production_spec,
    production_spec_for_order,
)

SUPPORTED_ARGUMENTS = frozenset({"minimap", "screen", "tag"})
SCREEN_WORLD_GRID = 24.0
ALLIANCES = {1: "self", 2: "ally", 3: "neutral", 4: "enemy"}
SC2_ALERT_NAMES = {6: "building_under_attack", 19: "unit_under_attack"}
TOWNHALL_NAMES = frozenset(
    {
        "nexus",
        "hatchery",
        "lair",
        "hive",
        "commandcenter",
        "orbitalcommand",
        "planetaryfortress",
    }
)


@dataclass(frozen=True)
class BuildSpec:
    target_structure: str
    placement_kind: str
    footprint: int
    requires_power: bool
    mineral_cost: int
    vespene_cost: int = 0
    prerequisites: tuple[str, ...] = ()


@dataclass(frozen=True)
class ScreenCandidateProvenance:
    """Bridge-private link from an observed screen candidate to world space."""

    screen_target: tuple[int, int]
    world_target: tuple[float, float]
    anchor_tag: int


BUILD_SPECS = {
    "Build_Pylon_Screen": BuildSpec("Pylon", "screen", 2, False, 100),
    "Build_Gateway_Screen": BuildSpec("Gateway", "screen", 3, True, 150),
    "Build_CyberneticsCore_Screen": BuildSpec(
        "CyberneticsCore",
        "screen",
        3,
        True,
        150,
        prerequisites=("Gateway",),
    ),
    "Build_Assimilator_Near": BuildSpec("Assimilator", "geyser", 3, False, 75),
    "Build_Nexus_Near": BuildSpec("Nexus", "expansion", 5, False, 400),
    "Build_Stargate_Screen": BuildSpec(
        "Stargate",
        "screen",
        3,
        True,
        150,
        vespene_cost=150,
        prerequisites=("CyberneticsCore",),
    ),
    "Build_ShieldBattery_Screen": BuildSpec(
        "ShieldBattery",
        "screen",
        2,
        True,
        100,
        prerequisites=("CyberneticsCore",),
    ),
}

SCREEN_POINT_ACTIONS = frozenset({"Move_Screen", "Ability_Blink_Screen"})
MINIMAP_POINT_ACTIONS = frozenset({"Move_Minimap"})
SELECT_BLINK_ACTION = "Select_Unit_Blink_Screen"
PRODUCTION_ACTION_PREFIXES = ("Train_", "Research_")
RESEARCH_PREREQUISITES = {"Research_WarpGate": ("CyberneticsCore",)}
RESEARCH_COSTS = {"Research_WarpGate": (50, 50, 0)}


def semantic_argument_candidates(
    observation: Any,
    action_name: str,
    *,
    unit_names: Mapping[int, str],
) -> Optional[list[list[Any]]]:
    """Return the single semantic candidate domain used at observe and dispatch time."""

    return _argument_candidates(
        observation,
        action_name,
        unit_names=unit_names,
        include_home_minimap=True,
    )


def is_production_action(action_name: str) -> bool:
    return action_name.startswith(PRODUCTION_ACTION_PREFIXES)


def production_source_tag(
    observation: Any,
    action: Mapping[str, Any],
    *,
    unit_names: Mapping[int, str],
    action_source_types: Mapping[int, int],
) -> Optional[int]:
    """Resolve the completed idle structure that can execute a production action."""

    action_name = str(action.get("name", ""))
    if not is_production_action(action_name):
        return None
    spec = production_spec(action_name)
    cost = (
        None
        if spec is None
        else (spec.minerals, spec.vespene, spec.supply)
    ) or RESEARCH_COSTS.get(action_name)
    if cost is not None and not _production_cost_is_available(observation, *cost):
        return None
    source_types = {
        int(action_source_types[function_id])
        for function_id in _action_function_ids(action)
        if function_id in action_source_types
    }
    if len(source_types) != 1:
        return None
    raw_units = list(_value(observation, "raw_units", ()))
    prerequisites = (
        spec.prerequisites if spec is not None else RESEARCH_PREREQUISITES.get(action_name, ())
    )
    completed_structures = {
        _unit_name(unit, unit_names)
        for unit in raw_units
        if int(_value(unit, "alliance", 0)) == 1 and _build_progress(unit) >= 1.0
    }
    if not set(prerequisites).issubset(completed_structures):
        return None
    source_type = next(iter(source_types))
    candidates = [
        unit
        for unit in raw_units
        if int(_value(unit, "alliance", 0)) == 1
        and int(_value(unit, "unit_type", 0)) == source_type
        and _build_progress(unit) >= 1.0
        and int(_value(unit, "active", 0)) == 0
        and int(_value(unit, "tag", 0)) > 0
    ]
    if not candidates:
        return None
    # Upstream ``find_idle_unit_tag`` selects the first matching raw unit. Do
    # not sort or skip ahead, otherwise the cached producer provenance can
    # silently refer to a different building than translator primitive 573.
    selected = candidates[0]
    if int(_value(selected, "order_length", 0)) != 0:
        return None
    return int(_value(selected, "tag", 0))


def _production_cost_is_available(
    observation: Any,
    minerals: int,
    vespene: int,
    supply: int,
) -> bool:
    player = _value(observation, "player_common", _value(observation, "player", None))
    if player is None:
        return False
    free_supply = int(_value(player, "food_cap", 0)) - int(_value(player, "food_used", 0))
    return (
        int(_value(player, "minerals", 0)) >= minerals
        and int(_value(player, "vespene", 0)) >= vespene
        and free_supply >= supply
    )


def nexus_placement_footprint_is_visible(
    observation: Any,
    position: Sequence[int | float],
) -> bool:
    """Return whether the exact translated Nexus footprint is currently visible."""

    if len(position) != 2:
        return False
    feature_screen = _value(observation, "feature_screen", None)
    visibility = _value(feature_screen, "visibility_map", None)
    dimensions = _plane_dimensions(visibility)
    if dimensions is None:
        return False
    height, width = dimensions
    center_x, center_y = float(position[0]), float(position[1])
    footprint = BUILD_SPECS["Build_Nexus_Near"].footprint
    half_width = footprint * width / SCREEN_WORLD_GRID / 2
    half_height = footprint * height / SCREEN_WORLD_GRID / 2
    minimum_x, maximum_x = math.ceil(center_x - half_width), math.floor(center_x + half_width)
    minimum_y, maximum_y = math.ceil(center_y - half_height), math.floor(center_y + half_height)
    if minimum_x < 0 or minimum_y < 0 or maximum_x >= width or maximum_y >= height:
        return False
    return all(
        int(visibility[y][x]) == 2
        for y in range(minimum_y, maximum_y + 1)
        for x in range(minimum_x, maximum_x + 1)
    )


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
        upgrade_names: Optional[Mapping[int, str]] = None,
        building_types: Sequence[int] = (),
        action_source_types: Optional[Mapping[int, int]] = None,
    ) -> None:
        self.run_id = run_id
        self.episode_id = episode_id
        self.unit_names = dict(unit_names or {})
        self.upgrade_names = {
            int(upgrade_id): str(name)
            for upgrade_id, name in (upgrade_names or {}).items()
        }
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
        teams = _extract_team_actions(
            agents,
            fallback_observation=observation,
            unit_names=self.unit_names,
            owned_unit_types={
                int(_value(unit, "unit_type", 0))
                for unit in raw_units
                if int(_value(unit, "alliance", 0)) == 1
            },
            action_source_types=self.action_source_types,
        )
        text_observation = "\n\n".join(
            f"[{name}]\n{text_observations[name]}" for name in sorted(text_observations)
        )
        return {
            "run_id": self.run_id,
            "episode_id": self.episode_id,
            "step_id": int(step_id),
            "game_loop": int(_scalar(_value(observation, "game_loop", 0))),
            "observed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "player_common": _extract_player(player),
            "production_queue": _extract_production_queue(
                observation,
                unit_names=self.unit_names,
            ),
            "units": [self._extract_unit(unit) for unit in raw_units],
            "upgrades": [
                self.upgrade_names.get(int(value), f"upgrade:{int(value)}")
                for value in _value(observation, "upgrades", ())
            ],
            "teams": teams,
            "text_observation": text_observation,
            "alerts": [_alert_name(value) for value in _value(observation, "alerts", ())],
            "image_uri": None,
        }

    def _extract_unit(self, unit: Any) -> dict[str, Any]:
        unit_type = int(_value(unit, "unit_type", 0))
        is_structure = unit_type in self.building_types
        health = float(_value(unit, "health", 0.0))
        health_ratio = float(_value(unit, "health_ratio", 0.0)) / 255.0
        health_max = float(_value(unit, "health_max", 0.0))
        if health_max <= 0:
            health_max = health / health_ratio if health_ratio > 0 else max(health, 1.0)
        order_length = int(_value(unit, "order_length", 0))
        status = "idle" if order_length == 0 else "active"
        build_progress = _value(unit, "build_progress", None)
        if is_structure and build_progress is not None:
            normalized_progress = float(build_progress)
            if normalized_progress > 1.0:
                normalized_progress /= 100.0
            if normalized_progress < 1.0:
                status = "constructing"
        return {
            "tag": int(_value(unit, "tag", 0)),
            "unit_type": self.unit_names.get(unit_type, f"unit:{unit_type}"),
            "alliance": ALLIANCES.get(int(_value(unit, "alliance", 0)), "neutral"),
            "is_structure": is_structure,
            "position": [float(_value(unit, "x", 0.0)), float(_value(unit, "y", 0.0))],
            "health": health,
            "health_max": health_max,
            "energy": float(_value(unit, "energy", 0.0)),
            "status": status,
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


def _extract_production_queue(
    observation: Any,
    *,
    unit_names: Mapping[int, str],
) -> list[dict[str, Any]]:
    """Project known unit orders with their exact producer tags.

    PySC2's top-level ``production_queue`` does not identify its producer. Raw
    unit orders do, so supported direct-training actions are projected from
    those first. Unknown top-level queue entries remain available as legacy
    ``ability:<id>`` diagnostics.
    """

    result: list[dict[str, Any]] = []
    producer_counts: Counter[str] = Counter()
    for unit in _value(observation, "raw_units", ()):
        if int(_value(unit, "alliance", 0)) != 1:
            continue
        producer_type = _unit_name(unit, unit_names)
        producer_tag = int(_value(unit, "tag", 0))
        if producer_tag <= 0:
            continue
        for order_id, progress in _unit_order_entries(unit):
            spec = production_spec_for_order(order_id)
            if spec is None or spec.producer_type != producer_type:
                continue
            producer_counts[spec.action_name] += 1
            result.append(
                {
                    "name": spec.action_name,
                    "producer_tag": producer_tag,
                    "progress": progress,
                }
            )

    for item in _value(observation, "production_queue", ()):
        ability_id = int(_value(item, "ability_id", 0))
        spec = production_spec_for_order(ability_id)
        if spec is not None and producer_counts[spec.action_name] > 0:
            producer_counts[spec.action_name] -= 1
            continue
        result.append(
            {
                "name": spec.action_name if spec is not None else f"ability:{ability_id}",
                "producer_tag": None,
                "progress": _normalized_progress(_value(item, "build_progress", 0.0)),
            }
        )
    return result


def _unit_order_entries(unit: Any) -> tuple[tuple[int, float], ...]:
    explicit = _value(unit, "orders", None)
    if explicit is not None:
        return tuple(
            (
                int(_value(order, "ability_id", _value(order, "order_id", order))),
                _normalized_progress(_value(order, "progress", 0.0)),
            )
            for order in explicit
        )
    count = min(max(int(_value(unit, "order_length", 0)), 0), 4)
    return tuple(
        (
            int(_value(unit, f"order_id_{index}", 0)),
            _normalized_progress(_value(unit, f"order_progress_{index}", 0.0)),
        )
        for index in range(count)
    )


def _normalized_progress(value: Any) -> float:
    progress = float(value)
    if progress > 1.0:
        progress /= 100.0
    return min(max(progress, 0.0), 1.0)


def _extract_team_actions(
    agents: Mapping[str, Any],
    *,
    fallback_observation: Any,
    unit_names: Mapping[int, str],
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
            team_observation = (
                fallback_observation
                if index >= len(team_observations)
                else team_observations[index].observation
            )
            actions = _available_team_actions(
                agent,
                team,
                available_ids,
                observation=team_observation,
                unit_names=unit_names,
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
    observation: Any,
    unit_names: Mapping[int, str],
    owned_unit_types: set[int],
    action_source_types: Mapping[int, int],
) -> list[dict[str, Any]]:
    action_space = agent.config.AGENTS[agent.name]["action"]
    unit_types = list(team.get("unit_type", ())) or ["EmptyGroup"]
    candidates = [action for unit_type in unit_types for action in action_space.get(unit_type, ())]
    result: list[dict[str, Any]] = []
    seen = set()
    completed_builder_structures = (
        _completed_own_structures(observation, unit_names) if agent.name == "Builder" else set()
    )
    for action in candidates:
        action_name = str(action["name"])
        if agent.name == "Builder" and _builder_movement_is_locked(
            action_name,
            completed_builder_structures,
        ):
            continue
        argument_names = tuple(str(value) for value in action.get("arg", ()))
        if not set(argument_names).issubset(SUPPORTED_ARGUMENTS):
            continue
        argument_types = tuple("tag" if name == "tag" else "position" for name in argument_names)
        function_ids = [int(triple[0]) for triple in action.get("func", ())]
        build_spec = BUILD_SPECS.get(action_name)
        if (
            build_spec is None
            and available_ids is not None
            and any(value not in available_ids for value in function_ids)
        ):
            continue
        required_sources = {
            action_source_types[function_id]
            for function_id in function_ids
            if function_id in action_source_types
        }
        if is_production_action(action_name):
            if (
                production_source_tag(
                    observation,
                    action,
                    unit_names=unit_names,
                    action_source_types=action_source_types,
                )
                is None
            ):
                continue
        elif required_sources and not required_sources.issubset(owned_unit_types):
            continue
        argument_candidates = _argument_candidates(
            observation,
            action_name,
            unit_names=unit_names,
            include_home_minimap=agent.name.startswith("CombatGroup"),
        )
        if (
            agent.name == "Builder"
            and action_name == "Move_Screen"
            and builder_move_requires_power(observation, unit_names)
        ):
            argument_candidates = [
                [candidate]
                for candidate in _movement_screen_candidates(
                    observation,
                    blink=False,
                    require_power=True,
                )
            ]
        needs_screen_provenance = action_name in SCREEN_POINT_ACTIONS or (
            build_spec is not None and build_spec.placement_kind == "screen"
        )
        screen_provenance = (
            screen_candidate_provenance(
                observation,
                [candidate[0] for candidate in argument_candidates or []],
            )
            if needs_screen_provenance
            else []
        )
        if argument_candidates and needs_screen_provenance and not screen_provenance:
            continue
        if (
            any(argument_type in {"tag", "position"} for argument_type in argument_types)
            and not argument_candidates
        ):
            continue
        key = (action_name, argument_names)
        if key in seen:
            continue
        seen.add(key)
        action_snapshot: dict[str, Any] = {
            "name": key[0],
            "argument_names": list(argument_names),
            "argument_types": list(argument_types),
            "argument_candidates": argument_candidates,
        }
        if screen_provenance:
            action_snapshot["bridge_screen_provenance"] = [
                {
                    "screen_target": list(item.screen_target),
                    "world_target": list(item.world_target),
                    "anchor_tag": item.anchor_tag,
                }
                for item in screen_provenance
            ]
        result.append(action_snapshot)

    if not any(action["name"] == "No_Operation" for action in result):
        result.insert(
            0,
            {
                "name": "No_Operation",
                "argument_names": [],
                "argument_types": [],
                "argument_candidates": None,
            },
        )
    return result


def _completed_own_structures(observation: Any, unit_names: Mapping[int, str]) -> set[str]:
    return {
        _unit_name(unit, unit_names)
        for unit in _value(observation, "raw_units", ())
        if int(_value(unit, "alliance", 0)) == 1 and _build_progress(unit) >= 1.0
    }


def _builder_movement_is_locked(action_name: str, completed_structures: set[str]) -> bool:
    if action_name == "Move_Screen":
        return "Pylon" not in completed_structures
    if action_name == "Move_Minimap":
        return not {"Pylon", "Gateway"}.issubset(completed_structures)
    return False


def builder_move_requires_power(
    observation: Any,
    unit_names: Mapping[int, str],
) -> bool:
    """Keep the opening Probe inside Pylon power until a Gateway completes."""

    return "Gateway" not in _completed_own_structures(observation, unit_names)


def _argument_candidates(
    observation: Any,
    action_name: str,
    *,
    unit_names: Mapping[int, str],
    include_home_minimap: bool,
) -> Optional[list[list[Any]]]:
    if action_name == "Attack_Unit":
        return [
            [int(_value(unit, "tag", 0))]
            for unit in _value(observation, "feature_units", ())
            if int(_value(unit, "alliance", 0)) == 4
            and bool(_value(unit, "is_on_screen", True))
            and int(_value(unit, "tag", 0)) > 0
        ]
    if action_name in SCREEN_POINT_ACTIONS:
        return [
            [candidate]
            for candidate in _movement_screen_candidates(
                observation,
                blink=action_name == "Ability_Blink_Screen",
            )
        ]
    if action_name in MINIMAP_POINT_ACTIONS:
        return [
            [candidate]
            for candidate in _movement_minimap_candidates(
                observation,
                include_home=include_home_minimap,
            )
        ]
    if action_name == SELECT_BLINK_ACTION:
        positions = _movement_screen_candidates(observation, blink=True)
        stalker_tags = sorted(
            {
                int(_value(unit, "tag", 0))
                for unit in _value(observation, "feature_units", ())
                if int(_value(unit, "alliance", 0)) == 1
                and bool(_value(unit, "is_on_screen", True))
                and _unit_name(unit, unit_names) == "Stalker"
                and int(_value(unit, "tag", 0)) > 0
            }
        )
        return [[tag, position] for tag in stalker_tags for position in positions][:8]
    spec = BUILD_SPECS.get(action_name)
    if spec is None:
        return None
    if not _build_prerequisites_satisfied(observation, spec, unit_names):
        return []
    if spec.placement_kind == "screen":
        return [[candidate] for candidate in build_screen_candidates(observation, action_name)]
    if spec.placement_kind == "geyser":
        return [[tag] for tag in _assimilator_candidates(observation, unit_names)]
    return [[tag] for tag in _expansion_anchor_candidates(observation, unit_names)]


def _action_function_ids(action: Mapping[str, Any]) -> tuple[int, ...]:
    functions = action.get("func", ())
    if not isinstance(functions, (list, tuple)):
        return ()
    return tuple(
        int(triple[0])
        for triple in functions
        if isinstance(triple, (list, tuple)) and len(triple) == 3
    )


def _movement_screen_candidates(
    observation: Any,
    *,
    blink: bool,
    require_power: bool = False,
) -> list[list[int]]:
    feature_screen = _value(observation, "feature_screen", None)
    pathable = _value(feature_screen, "pathable", None)
    power = _value(feature_screen, "power", None)
    shape: Sequence[Any] = getattr(pathable, "shape", ())
    if (
        pathable is None
        or len(shape) != 2
        or require_power
        and getattr(power, "shape", ()) != shape
    ):
        return []
    height, width = int(shape[0]), int(shape[1])
    if height <= 0 or width <= 0:
        return []

    own_positions = sorted(
        (
            float(_value(unit, "x", 0.0)),
            float(_value(unit, "y", 0.0)),
        )
        for unit in _value(observation, "feature_units", ())
        if int(_value(unit, "alliance", 0)) == 1 and bool(_value(unit, "is_on_screen", True))
    )
    if blink and not own_positions:
        return []
    anchor = (
        (
            sum(position[0] for position in own_positions) / len(own_positions),
            sum(position[1] for position in own_positions) / len(own_positions),
        )
        if blink
        else (width / 2, height / 2)
    )
    stride = max(4, min(width, height) // 8)
    blink_radius = min(width, height) / 3
    candidates: list[tuple[float, int, int]] = []
    for y in range(stride, height, stride):
        for x in range(stride, width, stride):
            if pathable[y][x] != 1:
                continue
            if require_power and power[y][x] != 1:
                continue
            distance = (x - anchor[0]) ** 2 + (y - anchor[1]) ** 2
            if blink and distance > blink_radius**2:
                continue
            candidates.append((distance, y, x))
    candidates.sort()
    return [[x, y] for _, y, x in candidates[:8]]


def _movement_minimap_candidates(
    observation: Any,
    *,
    limit: int = 8,
    include_home: bool = False,
) -> list[list[int]]:
    """Return stable pathable scouting targets across the minimap.

    Neutral minimap clusters are preferred because they normally identify melee resource
    bases. Fixed map-spanning points fill the remaining slots so scouting remains possible
    before remote resources have appeared in the minimap feature layers.
    """

    feature_minimap = _value(observation, "feature_minimap", None)
    pathable = _value(feature_minimap, "pathable", None)
    dimensions = _plane_dimensions(pathable)
    if dimensions is None:
        return []
    height, width = dimensions
    player_relative = _value(feature_minimap, "player_relative", None)
    visibility = _value(feature_minimap, "visibility_map", None)
    visibility_dimensions = _plane_dimensions(visibility)

    resource_targets: list[tuple[float, int, int]] = []
    own_points: list[tuple[int, int]] = []
    if _plane_dimensions(player_relative) == dimensions:
        neutral_points = {
            (x, y) for y in range(height) for x in range(width) if int(player_relative[y][x]) == 3
        }
        own_points = [
            (x, y) for y in range(height) for x in range(width) if int(player_relative[y][x]) == 1
        ]
        link_radius = max(2, min(height, width) // 16)
        own_clearance = max(4, min(height, width) // 8)
        for cluster in _cluster_minimap_points(neutral_points, link_radius=link_radius):
            if len(cluster) < 3:
                continue
            center_x = int(round(sum(point[0] for point in cluster) / len(cluster)))
            center_y = int(round(sum(point[1] for point in cluster) / len(cluster)))
            own_distance = min(
                ((center_x - x) ** 2 + (center_y - y) ** 2 for x, y in own_points),
                default=math.inf,
            )
            if own_distance < own_clearance**2:
                continue
            if visibility_dimensions == dimensions and int(visibility[center_y][center_x]) != 0:
                continue
            resource_targets.append((own_distance, center_y, center_x))

    resource_targets.sort()
    desired = [(x, y) for _, y, x in resource_targets]

    if not desired:
        last_x, last_y = width - 1, height - 1
        desired.extend(
            (round(last_x * x_fraction / 8), round(last_y * y_fraction / 8))
            for x_fraction, y_fraction in (
                (1, 1),
                (4, 1),
                (7, 1),
                (1, 4),
                (7, 4),
                (1, 7),
                (4, 7),
                (7, 7),
            )
        )

    search_radius = max(4, min(height, width) // 8)
    candidates: list[list[int]] = []
    offensive_limit = max(0, limit - 1 if include_home else limit)
    for target_x, target_y in desired if offensive_limit else ():
        candidate = _nearest_pathable_minimap_point(
            pathable,
            target_x,
            target_y,
            width=width,
            height=height,
            search_radius=search_radius,
        )
        if (
            candidate is None
            or candidate in candidates
            or (
                visibility_dimensions == dimensions
                and int(visibility[candidate[1]][candidate[0]]) != 0
            )
        ):
            continue
        candidates.append(candidate)
        if len(candidates) >= offensive_limit:
            break
    if include_home and own_points:
        home_x = int(round(sum(point[0] for point in own_points) / len(own_points)))
        home_y = int(round(sum(point[1] for point in own_points) / len(own_points)))
        home = _nearest_pathable_minimap_point(
            pathable,
            home_x,
            home_y,
            width=width,
            height=height,
            search_radius=search_radius,
        )
        if home is not None and home not in candidates:
            # The final candidate is the stable home/retreat target. Tactical
            # compilation excludes it for advances and selects it for retreats.
            candidates.append(home)
    return candidates


def _cluster_minimap_points(
    points: set[tuple[int, int]],
    *,
    link_radius: int,
) -> list[list[tuple[int, int]]]:
    remaining = set(points)
    clusters: list[list[tuple[int, int]]] = []
    offsets = [
        (offset_x, offset_y)
        for offset_y in range(-link_radius, link_radius + 1)
        for offset_x in range(-link_radius, link_radius + 1)
        if offset_x**2 + offset_y**2 <= link_radius**2
    ]
    while remaining:
        start = min(remaining, key=lambda point: (point[1], point[0]))
        remaining.remove(start)
        queue = [start]
        cluster: list[tuple[int, int]] = []
        while queue:
            point = queue.pop()
            cluster.append(point)
            for offset_x, offset_y in offsets:
                neighbor = (point[0] + offset_x, point[1] + offset_y)
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    queue.append(neighbor)
        clusters.append(cluster)
    return clusters


def _nearest_pathable_minimap_point(
    pathable: Any,
    target_x: int,
    target_y: int,
    *,
    width: int,
    height: int,
    search_radius: int,
) -> Optional[list[int]]:
    minimum_x, maximum_x = (
        max(0, target_x - search_radius),
        min(width - 1, target_x + search_radius),
    )
    minimum_y, maximum_y = (
        max(0, target_y - search_radius),
        min(height - 1, target_y + search_radius),
    )
    candidates = [
        ((x - target_x) ** 2 + (y - target_y) ** 2, y, x)
        for y in range(minimum_y, maximum_y + 1)
        for x in range(minimum_x, maximum_x + 1)
        if int(pathable[y][x]) == 1
    ]
    if not candidates:
        return None
    _, y, x = min(candidates)
    return [x, y]


def _plane_dimensions(plane: Any) -> Optional[tuple[int, int]]:
    shape: Sequence[Any] = getattr(plane, "shape", ())
    if plane is None or len(shape) != 2:
        return None
    height, width = int(shape[0]), int(shape[1])
    return (height, width) if height > 0 and width > 0 else None


def _build_prerequisites_satisfied(
    observation: Any,
    spec: BuildSpec,
    unit_names: Mapping[int, str],
) -> bool:
    player = _value(observation, "player_common", _value(observation, "player", None))
    if player is not None:
        if int(_value(player, "minerals", 0)) < spec.mineral_cost:
            return False
        if int(_value(player, "vespene", 0)) < spec.vespene_cost:
            return False
    completed = {
        _unit_name(unit, unit_names)
        for unit in _value(observation, "raw_units", ())
        if int(_value(unit, "alliance", 0)) == 1 and _build_progress(unit) >= 1.0
    }
    return all(prerequisite in completed for prerequisite in spec.prerequisites)


def _assimilator_candidates(
    observation: Any,
    unit_names: Mapping[int, str],
) -> list[int]:
    raw_units = list(_value(observation, "raw_units", ()))
    raw_by_tag = {int(_value(unit, "tag", 0)): unit for unit in raw_units}
    nexuses = [
        unit
        for unit in raw_units
        if int(_value(unit, "alliance", 0)) == 1
        and _unit_name(unit, unit_names) == "Nexus"
        and _build_progress(unit) >= 1.0
    ]
    assimilators = [
        unit
        for unit in raw_units
        if int(_value(unit, "alliance", 0)) == 1 and _unit_name(unit, unit_names) == "Assimilator"
    ]
    candidates = []
    for unit in _value(observation, "feature_units", ()):
        tag = int(_value(unit, "tag", 0))
        raw = raw_by_tag.get(tag, unit)
        if (
            tag <= 0
            or int(_value(unit, "alliance", 0)) != 3
            or not bool(_value(unit, "is_on_screen", True))
            or not _is_gas(_unit_name(raw, unit_names))
        ):
            continue
        if not any(_distance(raw, nexus) < 10.0 for nexus in nexuses):
            continue
        if any(_distance(raw, assimilator) < 2.0 for assimilator in assimilators):
            continue
        candidates.append(tag)
    return sorted(set(candidates))


def _expansion_anchor_candidates(
    observation: Any,
    unit_names: Mapping[int, str],
) -> list[int]:
    raw_units = list(_value(observation, "raw_units", ()))
    visible_resource_tags = {
        int(_value(unit, "tag", 0))
        for unit in _value(observation, "feature_units", ())
        if int(_value(unit, "alliance", 0)) == 3
        and bool(_value(unit, "is_on_screen", True))
        and int(_value(unit, "display_type", 1)) == 1
    }
    resources = [
        unit
        for unit in raw_units
        if int(_value(unit, "alliance", 0)) == 3
        and (_is_gas(_unit_name(unit, unit_names)) or _is_mineral(_unit_name(unit, unit_names)))
        and int(_value(unit, "tag", 0)) > 0
    ]
    if not resources:
        return []
    parent = list(range(len(resources)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left in range(len(resources)):
        for right in range(left + 1, len(resources)):
            if _distance(resources[left], resources[right]) <= 12.0:
                union(left, right)

    clusters: dict[int, list[Any]] = {}
    for index, resource in enumerate(resources):
        clusters.setdefault(find(index), []).append(resource)
    townhalls = [
        unit
        for unit in raw_units
        if _unit_name(unit, unit_names).casefold() in TOWNHALL_NAMES
        and int(_value(unit, "alliance", 0)) in {1, 2, 4}
    ]
    own_nexuses = [
        unit
        for unit in townhalls
        if _unit_name(unit, unit_names) == "Nexus" and int(_value(unit, "alliance", 0)) == 1
    ]
    ranked: list[tuple[float, int]] = []
    for cluster in clusters.values():
        if sum(_is_mineral(_unit_name(unit, unit_names)) for unit in cluster) < 5:
            continue
        if (
            sum(
                _is_mineral(_unit_name(unit, unit_names))
                and int(_value(unit, "tag", 0)) in visible_resource_tags
                for unit in cluster
            )
            < 5
        ):
            continue
        center_x = sum(float(_value(unit, "x", 0.0)) for unit in cluster) / len(cluster)
        center_y = sum(float(_value(unit, "y", 0.0)) for unit in cluster) / len(cluster)
        if any(_point_distance(center_x, center_y, townhall) < 12.0 for townhall in townhalls):
            continue
        anchor = min(
            cluster,
            key=lambda unit: (
                _point_distance(center_x, center_y, unit),
                int(_value(unit, "tag", 0)),
            ),
        )
        anchor_tag = int(_value(anchor, "tag", 0))
        if anchor_tag not in visible_resource_tags:
            continue
        if not _nexus_anchor_has_legal_screen_placement(
            observation,
            anchor_tag,
            unit_names,
        ):
            continue
        base_distance = min(
            (_distance(anchor, nexus) for nexus in own_nexuses),
            default=math.inf,
        )
        ranked.append((base_distance, anchor_tag))
    ranked.sort()
    return [tag for _, tag in ranked[:4]]


def _nexus_anchor_has_legal_screen_placement(
    observation: Any,
    anchor_tag: int,
    unit_names: Mapping[int, str],
) -> bool:
    feature_screen = _value(observation, "feature_screen", None)
    visibility = _value(feature_screen, "visibility_map", None)
    buildable = _value(feature_screen, "buildable", None)
    pathable = _value(feature_screen, "pathable", None)
    player_relative = _value(feature_screen, "player_relative", None)
    dimensions = _plane_dimensions(buildable)
    if (
        dimensions is None
        or dimensions[0] != dimensions[1]
        or any(
            _plane_dimensions(plane) != dimensions
            for plane in (visibility, pathable, player_relative)
        )
    ):
        return False
    screen_size = dimensions[0]
    pixel_scale = screen_size / SCREEN_WORLD_GRID
    sample_stride = max(1, int(pixel_scale))
    visible_resources = [
        unit
        for unit in _value(observation, "feature_units", ())
        if int(_value(unit, "alliance", 0)) == 3
        and int(_value(unit, "display_type", 1)) == 1
        and bool(_value(unit, "is_on_screen", True))
        and 0 < float(_value(unit, "x", 0.0)) < screen_size
        and 0 < float(_value(unit, "y", 0.0)) < screen_size
        and (_is_gas(_unit_name(unit, unit_names)) or _is_mineral(_unit_name(unit, unit_names)))
    ]
    anchor = next(
        (unit for unit in visible_resources if int(_value(unit, "tag", 0)) == anchor_tag),
        None,
    )
    if anchor is None:
        return False
    nearby_resources = [
        unit for unit in visible_resources if _distance(unit, anchor) < 16 * pixel_scale
    ]
    if sum(_is_mineral(_unit_name(unit, unit_names)) for unit in nearby_resources) < 5:
        return False
    center_x = sum(float(_value(unit, "x", 0.0)) for unit in nearby_resources) / len(
        nearby_resources
    )
    center_y = sum(float(_value(unit, "y", 0.0)) for unit in nearby_resources) / len(
        nearby_resources
    )
    townhalls = [
        unit
        for unit in _value(observation, "feature_units", ())
        if int(_value(unit, "alliance", 0)) in {1, 2, 4}
        and _unit_name(unit, unit_names).casefold() in TOWNHALL_NAMES
    ]
    invalid_footprint_prefix = _nexus_invalid_footprint_prefix(
        visibility,
        buildable,
        pathable,
        player_relative,
        screen_size,
    )
    search_radius = 12 * pixel_scale
    for candidate_y in range(sample_stride, screen_size, sample_stride):
        for candidate_x in range(sample_stride, screen_size, sample_stride):
            if (candidate_x - center_x) ** 2 + (candidate_y - center_y) ** 2 > search_radius**2:
                continue
            if not _nexus_footprint_is_legal(
                invalid_footprint_prefix,
                center_x=candidate_x,
                center_y=candidate_y,
                screen_size=screen_size,
                pixel_scale=pixel_scale,
            ):
                continue
            if not _nexus_resource_clearance_is_legal(
                candidate_x,
                candidate_y,
                nearby_resources,
                unit_names,
                pixel_scale,
            ):
                continue
            if any(
                _point_distance(candidate_x, candidate_y, townhall) < 12 * pixel_scale
                for townhall in townhalls
            ):
                continue
            return True
    return False


def _nexus_footprint_is_legal(
    invalid_footprint_prefix: list[list[int]],
    *,
    center_x: int,
    center_y: int,
    screen_size: int,
    pixel_scale: float,
) -> bool:
    half_extent = BUILD_SPECS["Build_Nexus_Near"].footprint * pixel_scale / 2
    min_x = math.ceil(center_x - half_extent)
    max_x = math.floor(center_x + half_extent)
    min_y = math.ceil(center_y - half_extent)
    max_y = math.floor(center_y + half_extent)
    if min_x <= 0 or min_y <= 0 or max_x >= screen_size or max_y >= screen_size:
        return False
    return (
        _rectangle_sum(
            invalid_footprint_prefix,
            min_x,
            max_x,
            min_y,
            max_y,
        )
        == 0
    )


def _nexus_invalid_footprint_prefix(
    visibility: Any,
    buildable: Any,
    pathable: Any,
    player_relative: Any,
    screen_size: int,
) -> list[list[int]]:
    prefix = [[0] * (screen_size + 1) for _ in range(screen_size + 1)]
    for y in range(screen_size):
        row_total = 0
        previous_row = prefix[y]
        current_row = prefix[y + 1]
        for x in range(screen_size):
            row_total += not (
                visibility[y][x] == 2
                and buildable[y][x] == 1
                and pathable[y][x] == 1
                and player_relative[y][x] == 0
            )
            current_row[x + 1] = previous_row[x + 1] + row_total
    return prefix


def _nexus_resource_clearance_is_legal(
    center_x: int,
    center_y: int,
    resources: Sequence[Any],
    unit_names: Mapping[int, str],
    pixel_scale: float,
) -> bool:
    for resource in resources:
        distance = _point_distance(center_x, center_y, resource)
        minimum, maximum = (
            (7 * pixel_scale, 10 * pixel_scale)
            if _is_gas(_unit_name(resource, unit_names))
            else (6 * pixel_scale, 9 * pixel_scale)
        )
        if not minimum < distance < maximum:
            return False
    return True


def _unit_name(unit: Any, unit_names: Mapping[int, str]) -> str:
    value = _value(unit, "unit_type", "")
    return value if isinstance(value, str) else unit_names.get(int(value), f"unit:{int(value)}")


def _is_gas(name: str) -> bool:
    return "geyser" in name.casefold()


def _is_mineral(name: str) -> bool:
    return "mineralfield" in name.casefold()


def _build_progress(unit: Any) -> float:
    progress = float(_value(unit, "build_progress", 0.0))
    return progress / 100.0 if progress > 1.0 else progress


def _distance(left: Any, right: Any) -> float:
    return math.hypot(
        float(_value(left, "x", 0.0)) - float(_value(right, "x", 0.0)),
        float(_value(left, "y", 0.0)) - float(_value(right, "y", 0.0)),
    )


def _point_distance(x: float, y: float, unit: Any) -> float:
    return math.hypot(
        x - float(_value(unit, "x", 0.0)),
        y - float(_value(unit, "y", 0.0)),
    )


def _available_function_ids(
    team_observations: Sequence[Any], index: int
) -> Optional[frozenset[int]]:
    if index >= len(team_observations):
        return None
    values = _value(team_observations[index].observation, "available_actions", ())
    return frozenset(int(value) for value in values)


def build_screen_candidates(observation: Any, action_name: str) -> list[list[int]]:
    return _build_screen_candidates(observation, action_name, limit=8)


def screen_to_world_target(
    observation: Any,
    screen_target: Sequence[int | float],
    *,
    preferred_anchor_tag: Optional[int] = None,
) -> Optional[ScreenCandidateProvenance]:
    """Project a feature-screen point through a same-tag raw/feature anchor."""

    if len(screen_target) != 2:
        return None
    anchor = _projection_anchor(observation, preferred_anchor_tag)
    dimensions = _screen_dimensions(observation)
    if anchor is None or dimensions is None:
        return None
    anchor_tag, raw_x, raw_y, feature_x, feature_y = anchor
    height, width = dimensions
    screen_x, screen_y = float(screen_target[0]), float(screen_target[1])
    return ScreenCandidateProvenance(
        screen_target=(int(screen_target[0]), int(screen_target[1])),
        world_target=(
            raw_x + (screen_x - feature_x) / (width / SCREEN_WORLD_GRID),
            raw_y + (screen_y - feature_y) / (height / SCREEN_WORLD_GRID),
        ),
        anchor_tag=anchor_tag,
    )


def screen_candidate_provenance(
    observation: Any,
    screen_targets: Sequence[Sequence[int | float]],
) -> list[ScreenCandidateProvenance]:
    """Return world-space provenance for every candidate exposed to Runtime."""

    provenance: list[ScreenCandidateProvenance] = []
    for candidate in screen_targets:
        projected = screen_to_world_target(observation, candidate)
        if projected is None:
            return []
        provenance.append(projected)
    return provenance


def resolve_screen_build_world_target(
    observation: Any,
    action_name: str,
    world_target: tuple[float, float],
    *,
    preferred_anchor_tag: Optional[int] = None,
    excluded_positions: Collection[tuple[int, int]] = (),
    force_resample: bool = False,
) -> Optional[list[int]]:
    """Reproject a routed build target and validate it against the current camera."""

    spec = BUILD_SPECS.get(action_name)
    if spec is None or spec.placement_kind != "screen":
        return None
    projected = _world_to_screen_target(
        observation,
        world_target,
        preferred_anchor_tag=preferred_anchor_tag,
    )
    dimensions = _screen_dimensions(observation)
    if projected is None or dimensions is None:
        return None
    height, width = dimensions
    if not (0 <= projected[0] < width and 0 <= projected[1] < height):
        return None
    excluded = set(excluded_positions)
    if force_resample:
        excluded.add((projected[0], projected[1]))
    if (
        not force_resample
        and tuple(projected) not in excluded
        and _build_screen_position_is_legal(observation, spec, projected)
    ):
        return projected

    candidates = [
        candidate
        for candidate in _build_screen_candidates(observation, action_name, limit=None)
        if tuple(candidate) not in excluded
    ]
    stride = max(4, int(height / SCREEN_WORLD_GRID))
    ranked = sorted(
        (
            (candidate[0] - projected[0]) ** 2 + (candidate[1] - projected[1]) ** 2,
            candidate[0],
            candidate[1],
            candidate,
        )
        for candidate in candidates
    )
    if not ranked or ranked[0][0] > (6 * stride) ** 2:
        return None
    return ranked[0][3]


def resolve_screen_point_world_target(
    observation: Any,
    action_name: str,
    world_target: tuple[float, float],
    *,
    preferred_anchor_tag: Optional[int] = None,
    require_power: bool = False,
) -> Optional[list[int]]:
    """Reproject a movement target into the current legal screen candidate domain."""

    if action_name not in SCREEN_POINT_ACTIONS:
        return None
    projected = _world_to_screen_target(
        observation,
        world_target,
        preferred_anchor_tag=preferred_anchor_tag,
    )
    dimensions = _screen_dimensions(observation)
    if projected is None or dimensions is None:
        return None
    height, width = dimensions
    if not (0 <= projected[0] < width and 0 <= projected[1] < height):
        return None
    candidates = _movement_screen_candidates(
        observation,
        blink=action_name == "Ability_Blink_Screen",
        require_power=require_power,
    )
    if projected in candidates:
        return projected
    stride = max(4, min(width, height) // 8)
    ranked = sorted(
        (
            (candidate[0] - projected[0]) ** 2 + (candidate[1] - projected[1]) ** 2,
            candidate[0],
            candidate[1],
            candidate,
        )
        for candidate in candidates
    )
    if not ranked or ranked[0][0] > (2 * stride) ** 2:
        return None
    return ranked[0][3]


def screen_build_position_is_legal(
    observation: Any,
    action_name: str,
    position: Sequence[int],
    *,
    unit_names: Optional[Mapping[int, str]] = None,
) -> bool:
    """Validate one exact screen position against the action's full footprint."""

    spec = BUILD_SPECS.get(action_name)
    return (
        spec is not None
        and spec.placement_kind == "screen"
        and _build_prerequisites_satisfied(observation, spec, unit_names or {})
        and _build_screen_position_is_legal(observation, spec, position)
    )


def _build_screen_candidates(
    observation: Any,
    action_name: str,
    *,
    limit: Optional[int],
) -> list[list[int]]:
    spec = BUILD_SPECS.get(action_name)
    if spec is None or spec.placement_kind != "screen":
        return []
    feature_screen = _value(observation, "feature_screen", None)
    if feature_screen is None:
        return []
    buildable = _value(feature_screen, "buildable", None)
    pathable = _value(feature_screen, "pathable", None)
    player_relative = _value(feature_screen, "player_relative", None)
    power = _value(feature_screen, "power", None)
    feature_units = _value(observation, "feature_units", ())
    shape = getattr(buildable, "shape", ())
    if not shape or buildable is None or pathable is None or player_relative is None:
        return []
    screen_size = int(shape[0])
    own_positions = [
        (float(_value(unit, "x", 0.0)), float(_value(unit, "y", 0.0)))
        for unit in feature_units
        if int(_value(unit, "alliance", 0)) == 1 and bool(_value(unit, "is_on_screen", True))
    ]
    semantic_anchor = (
        (
            sum(position[0] for position in own_positions) / len(own_positions),
            sum(position[1] for position in own_positions) / len(own_positions),
        )
        if own_positions
        else (screen_size / 2, screen_size / 2)
    )
    candidates = _valid_build_positions(
        buildable,
        pathable,
        player_relative,
        power,
        occupied_positions=tuple(
            (
                int(_value(unit, "x", 0)),
                int(_value(unit, "y", 0)),
                max(0.0, float(_value(unit, "radius", 0.5))),
            )
            for unit in feature_units
            if bool(_value(unit, "is_on_screen", True))
        ),
        screen_size=screen_size,
        building_size=spec.footprint,
        require_power=spec.requires_power,
        semantic_anchor=semantic_anchor,
    )
    return candidates if limit is None else candidates[:limit]


def _projection_anchor(
    observation: Any,
    preferred_anchor_tag: Optional[int],
) -> Optional[tuple[int, float, float, float, float]]:
    raw_by_tag = {
        int(_value(unit, "tag", 0)): unit
        for unit in _value(observation, "raw_units", ())
        if int(_value(unit, "tag", 0)) > 0
    }
    feature_by_tag = {
        int(_value(unit, "tag", 0)): unit
        for unit in _value(observation, "feature_units", ())
        if int(_value(unit, "tag", 0)) > 0 and bool(_value(unit, "is_on_screen", True))
    }
    shared_tags = sorted(raw_by_tag.keys() & feature_by_tag.keys())
    if not shared_tags:
        return None
    tag = preferred_anchor_tag if preferred_anchor_tag in shared_tags else shared_tags[0]
    raw = raw_by_tag[tag]
    feature = feature_by_tag[tag]
    return (
        tag,
        float(_value(raw, "x", 0.0)),
        float(_value(raw, "y", 0.0)),
        float(_value(feature, "x", 0.0)),
        float(_value(feature, "y", 0.0)),
    )


def _screen_dimensions(observation: Any) -> Optional[tuple[int, int]]:
    feature_screen = _value(observation, "feature_screen", None)
    plane = _value(feature_screen, "buildable", None)
    if plane is None:
        plane = _value(feature_screen, "pathable", None)
    shape: Sequence[Any] = getattr(plane, "shape", ())
    if plane is None or len(shape) != 2:
        return None
    height, width = int(shape[0]), int(shape[1])
    return (height, width) if height > 0 and width > 0 else None


def _world_to_screen_target(
    observation: Any,
    world_target: tuple[float, float],
    *,
    preferred_anchor_tag: Optional[int],
) -> Optional[list[int]]:
    anchor = _projection_anchor(observation, preferred_anchor_tag)
    dimensions = _screen_dimensions(observation)
    if anchor is None or dimensions is None:
        return None
    _, raw_x, raw_y, feature_x, feature_y = anchor
    height, width = dimensions
    return [
        int(round(feature_x + (world_target[0] - raw_x) * width / SCREEN_WORLD_GRID)),
        int(round(feature_y + (world_target[1] - raw_y) * height / SCREEN_WORLD_GRID)),
    ]


def _build_screen_position_is_legal(
    observation: Any,
    spec: BuildSpec,
    position: Sequence[int],
) -> bool:
    feature_screen = _value(observation, "feature_screen", None)
    buildable = _value(feature_screen, "buildable", None)
    pathable = _value(feature_screen, "pathable", None)
    player_relative = _value(feature_screen, "player_relative", None)
    power = _value(feature_screen, "power", None)
    dimensions = _screen_dimensions(observation)
    if (
        dimensions is None
        or dimensions[0] != dimensions[1]
        or pathable is None
        or player_relative is None
    ):
        return False
    screen_size = dimensions[0]
    ratio = max(1, int(screen_size / SCREEN_WORLD_GRID))
    prefix = _invalid_build_cell_prefix(
        buildable,
        pathable,
        player_relative,
        power,
        screen_size,
        require_power=spec.requires_power,
    )
    return _build_footprint_is_clear(
        prefix,
        _occupied_positions(observation),
        _build_footprint_bounds(int(position[0]), int(position[1]), ratio, spec.footprint),
        screen_size,
    )


def _occupied_positions(observation: Any) -> tuple[tuple[int, int, float], ...]:
    return tuple(
        (
            int(_value(unit, "x", 0)),
            int(_value(unit, "y", 0)),
            max(0.0, float(_value(unit, "radius", 0.5))),
        )
        for unit in _value(observation, "feature_units", ())
        if bool(_value(unit, "is_on_screen", True))
    )


def _valid_build_positions(
    buildable: Any,
    pathable: Any,
    player_relative: Any,
    power: Any,
    *,
    occupied_positions: tuple[tuple[int, int, float], ...],
    screen_size: int,
    building_size: int,
    require_power: bool,
    semantic_anchor: tuple[float, float],
) -> list[list[int]]:
    ratio = max(1, int(screen_size / 24))
    stride = max(4, ratio)
    invalid_cell_prefix = _invalid_build_cell_prefix(
        buildable,
        pathable,
        player_relative,
        power,
        screen_size,
        require_power=require_power,
    )
    candidates: list[tuple[float, float, int, int]] = []
    for x0 in range(stride, screen_size, stride):
        for y0 in range(stride, screen_size, stride):
            bounds = _build_footprint_bounds(x0, y0, ratio, building_size)
            if _build_footprint_is_clear(
                invalid_cell_prefix,
                occupied_positions,
                bounds,
                screen_size,
            ):
                anchor_distance = (x0 - semantic_anchor[0]) ** 2 + (y0 - semantic_anchor[1]) ** 2
                center_distance = (x0 - screen_size / 2) ** 2 + (y0 - screen_size / 2) ** 2
                candidates.append((anchor_distance, center_distance, x0, y0))
    candidates.sort()
    return [[x, y] for _, _, x, y in candidates]


def _build_cell_is_valid(
    buildable: Any,
    pathable: Any,
    player_relative: Any,
    x: int,
    y: int,
) -> bool:
    """Check PySC2 feature planes using their row-major ``[y][x]`` layout."""

    if buildable[y][x] != 1 or pathable[y][x] != 1:
        return False
    if player_relative[y][x] != 0:
        return False
    return True


def _build_footprint_bounds(
    center_x: int,
    center_y: int,
    ratio: int,
    building_size: int,
) -> tuple[int, int, int, int]:
    first_x = int(center_x - ratio * (building_size - 1) / 2)
    first_y = int(center_y - ratio * (building_size - 1) / 2)
    last_x = first_x + ratio * (building_size - 1)
    last_y = first_y + ratio * (building_size - 1)
    cell_radius = (ratio + 1) // 2
    return (
        first_x - cell_radius,
        last_x + cell_radius,
        first_y - cell_radius,
        last_y + cell_radius,
    )


def _build_footprint_is_clear(
    invalid_cell_prefix: list[list[int]],
    occupied_positions: tuple[tuple[int, int, float], ...],
    bounds: tuple[int, int, int, int],
    screen_size: int,
) -> bool:
    min_x, max_x, min_y, max_y = bounds
    if min_x <= 0 or min_y <= 0 or max_x >= screen_size or max_y >= screen_size:
        return False
    if any(
        min_x - math.ceil(radius) <= unit_x <= max_x + math.ceil(radius)
        and min_y - math.ceil(radius) <= unit_y <= max_y + math.ceil(radius)
        for unit_x, unit_y, radius in occupied_positions
    ):
        return False
    return (
        _rectangle_sum(
            invalid_cell_prefix,
            min_x,
            max_x,
            min_y,
            max_y,
        )
        == 0
    )


def _invalid_build_cell_prefix(
    buildable: Any,
    pathable: Any,
    player_relative: Any,
    power: Any,
    screen_size: int,
    *,
    require_power: bool,
) -> list[list[int]]:
    prefix = [[0] * (screen_size + 1) for _ in range(screen_size + 1)]
    for y in range(screen_size):
        row_total = 0
        previous_row = prefix[y]
        current_row = prefix[y + 1]
        for x in range(screen_size):
            row_total += (
                not _build_cell_is_valid(
                    buildable,
                    pathable,
                    player_relative,
                    x,
                    y,
                )
                or require_power
                and (power is None or power[y][x] != 1)
            )
            current_row[x + 1] = previous_row[x + 1] + row_total
    return prefix


def _rectangle_sum(
    prefix: list[list[int]],
    min_x: int,
    max_x: int,
    min_y: int,
    max_y: int,
) -> int:
    return (
        prefix[max_y + 1][max_x + 1]
        - prefix[min_y][max_x + 1]
        - prefix[max_y + 1][min_x]
        + prefix[min_y][min_x]
    )


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
