"""Single world-space placement authority for PySC2 raw build actions."""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class RawPlacement:
    action_name: str
    world_target: tuple[float, float]
    anchor_tag: int | None = None


@dataclass(frozen=True)
class RawPlacementCandidates:
    argument_candidates: list[list[Any]]
    screen_provenance: list[Any]
    unavailable_reason: str | None = None


class RawPlacementFailure(RuntimeError):
    def __init__(self, code: str, reason: str) -> None:
        super().__init__(reason)
        self.code = code


class RawPlacementService:
    """Generate, resolve, execute and verify one canonical placement identity.

    The wire protocol may still carry a feature-screen coordinate for backward
    compatibility, but raw execution and effect verification only consume the
    world target retained here.
    """

    def __init__(self, *, unit_names: Mapping[int, str]) -> None:
        self.unit_names = {int(key): str(value) for key, value in unit_names.items()}
        self._known_resources: dict[int, dict[str, Any]] = {}
        self._suppressed_clusters: set[int] = set()
        self._suppressed_resource_tags: set[int] = set()
        self._quarantined_targets: dict[str, list[tuple[float, float]]] = {}
        self._command_targets: dict[str, RawPlacement] = {}
        self._placement_diagnostics: dict[str, str] = {}

    @property
    def known_resources(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._known_resources[tag] for tag in sorted(self._known_resources))

    @property
    def suppressed_anchors(self) -> frozenset[int]:
        return frozenset((*self._suppressed_clusters, *self._suppressed_resource_tags))

    @property
    def quarantined_targets(self) -> Mapping[str, Sequence[tuple[float, float]]]:
        return self._quarantined_targets

    @property
    def placement_alerts(self) -> tuple[str, ...]:
        return tuple(
            f"placement_unavailable:{action}:{reason}"
            for action, reason in sorted(self._placement_diagnostics.items())
        )

    def reset_diagnostics(self) -> None:
        self._placement_diagnostics.clear()

    def observe(self, observation: Any, *, require_feature_visibility: bool) -> None:
        self.observe_units(
            _value(observation, "raw_units", ()),
            _value(observation, "feature_units", ()),
            require_feature_visibility=require_feature_visibility,
        )

    def observe_units(
        self,
        raw_units: Sequence[Any],
        feature_units: Sequence[Any],
        *,
        require_feature_visibility: bool,
    ) -> None:
        visible_feature_tags = {
            int(_value(unit, "tag", 0))
            for unit in feature_units
            if int(_value(unit, "alliance", 0)) == 3
            and bool(_value(unit, "is_on_screen", True))
            and int(_value(unit, "display_type", 1)) == 1
        }
        for unit in raw_units:
            tag = int(_value(unit, "tag", 0))
            name = _unit_name(unit, self.unit_names)
            if (
                tag <= 0
                or int(_value(unit, "alliance", 0)) != 3
                or int(_value(unit, "display_type", 1)) != 1
                or not _is_resource(name)
                or require_feature_visibility
                and tag not in visible_feature_tags
            ):
                continue
            self._known_resources[tag] = {
                "tag": tag,
                "unit_type": name,
                "alliance": 3,
                "x": float(_value(unit, "x", 0.0)),
                "y": float(_value(unit, "y", 0.0)),
                "display_type": 1,
            }

    def candidates(
        self,
        observation: Any,
        action_name: str,
        *,
        builder_tags: Collection[int] = (),
    ) -> RawPlacementCandidates:
        # Lazy import avoids making the extractor depend on itself while all
        # public raw placement entry points remain centralized in this service.
        from rtscortex_llm_pysc2.extractor import (
            BUILD_SPECS,
            _gas_structure_candidates,
            build_screen_candidates,
            screen_candidate_provenance,
        )

        spec = BUILD_SPECS[action_name]
        if spec.placement_kind == "screen":
            screen_candidates = build_screen_candidates(
                observation,
                action_name,
                unit_names=self.unit_names,
                builder_tags=builder_tags,
            )
            provenance = screen_candidate_provenance(observation, screen_candidates)
            radius = max(1.5, spec.footprint * 0.75)
            kept = [
                (candidate, item)
                for candidate, item in zip(screen_candidates, provenance)  # noqa: B905 - Python 3.9
                if not self.is_quarantined(
                    action_name,
                    item.world_target,
                    radius=radius,
                )
            ]
            reason = None if kept else self._screen_unavailable_reason(observation, spec)
            self._remember_diagnostic(action_name, reason)
            return RawPlacementCandidates(
                argument_candidates=[[candidate] for candidate, _ in kept],
                screen_provenance=[item for _, item in kept],
                unavailable_reason=reason,
            )
        if spec.placement_kind == "geyser":
            candidates = _gas_structure_candidates(
                observation,
                self.unit_names,
                target_structure=spec.target_structure,
            )
            reason = None if candidates else "no_unoccupied_geyser"
            self._remember_diagnostic(action_name, reason)
            return RawPlacementCandidates(
                argument_candidates=[[tag] for tag in candidates],
                screen_provenance=[],
                unavailable_reason=reason,
            )
        candidates = self._expansion_candidates(observation)
        reason = None if candidates else "no_unoccupied_resource_cluster"
        self._remember_diagnostic(action_name, reason)
        return RawPlacementCandidates(
            argument_candidates=[[tag] for tag in candidates],
            screen_provenance=[],
            unavailable_reason=reason,
        )

    def resolve(
        self,
        *,
        command_id: str,
        action_name: str,
        requested_arguments: Sequence[Any],
        observation: Any,
        world_target: tuple[float, float] | None,
        preferred_anchor_tag: int | None = None,
        builder_tags: Collection[int] = (),
    ) -> RawPlacement:
        from rtscortex_llm_pysc2.extractor import (
            BUILD_SPECS,
            resolve_screen_build_world_target,
            screen_to_world_target,
        )

        spec = BUILD_SPECS[action_name]
        if spec.placement_kind == "screen":
            if world_target is None:
                raise RawPlacementFailure(
                    "no_legal_placement",
                    f"{action_name} has no validated world-space target",
                )
            target = (float(world_target[0]), float(world_target[1]))
            if self.is_quarantined(
                action_name,
                target,
                radius=max(1.5, spec.footprint * 0.75),
            ):
                raise RawPlacementFailure(
                    "no_legal_placement",
                    f"{action_name} target {target} is permanently quarantined",
                )
            if _value(observation, "feature_screen", None) is not None:
                resolved_screen = resolve_screen_build_world_target(
                    observation,
                    action_name,
                    target,
                    preferred_anchor_tag=preferred_anchor_tag,
                    unit_names=self.unit_names,
                    builder_tags=builder_tags,
                )
                resolved = (
                    None
                    if resolved_screen is None
                    else screen_to_world_target(
                        observation,
                        resolved_screen,
                        preferred_anchor_tag=preferred_anchor_tag,
                    )
                )
                if resolved is None or math.dist(target, resolved.world_target) > 0.25:
                    raise RawPlacementFailure(
                        "no_legal_placement",
                        f"{action_name} world target {target} is no longer legal",
                    )
            placement = RawPlacement(action_name, target)
        elif spec.placement_kind == "geyser":
            anchor = _tag_argument(requested_arguments, action_name=action_name)
            target_unit = _unit_by_tag(observation, anchor)
            if (
                target_unit is None
                or int(_value(target_unit, "alliance", 0)) != 3
                or not _is_gas(_unit_name(target_unit, self.unit_names))
            ):
                raise RawPlacementFailure(
                    "invalid_geyser_tag",
                    f"{action_name} target {hex(anchor)} is not a visible neutral geyser",
                )
            placement = RawPlacement(
                action_name,
                (
                    float(_value(target_unit, "x", 0.0)),
                    float(_value(target_unit, "y", 0.0)),
                ),
                anchor,
            )
        else:
            anchor = _tag_argument(requested_arguments, action_name=action_name)
            cluster_id = self._cluster_id(anchor)
            if cluster_id is None or self._cluster_is_suppressed(
                self._resource_cluster(cluster_id)
            ):
                raise RawPlacementFailure(
                    "invalid_expansion_anchor",
                    f"{action_name} anchor {hex(anchor)} is permanently suppressed",
                )
            expansion_target = self._expansion_target(cluster_id, observation)
            if expansion_target is None:
                raise RawPlacementFailure(
                    "no_legal_placement",
                    f"{action_name} anchor {hex(anchor)} has no legal world placement",
                )
            placement = RawPlacement(action_name, expansion_target, cluster_id)
        self._command_targets[command_id] = placement
        return placement

    def command_target(self, command_id: str) -> RawPlacement | None:
        return self._command_targets.get(command_id)

    def quarantine_command(
        self,
        *,
        command_id: str,
        action_name: str,
        requested_arguments: Sequence[Any],
        world_target: tuple[float, float] | None,
    ) -> None:
        placement = self._command_targets.get(command_id)
        target = None if placement is None else placement.world_target
        anchor = None if placement is None else placement.anchor_tag
        if target is None and world_target is not None:
            target = (float(world_target[0]), float(world_target[1]))
        if anchor is None and action_name.endswith("_Near") and requested_arguments:
            try:
                anchor = _tag_argument(requested_arguments, action_name=action_name)
            except RawPlacementFailure:
                anchor = None
        if target is not None:
            self.suppress_world_target(action_name, target)
        if anchor is not None and any(
            townhall in action_name for townhall in ("Nexus", "CommandCenter", "Hatchery")
        ):
            self.suppress_anchor(anchor)

    def suppress_anchor(self, tag: int) -> None:
        cluster = self._resource_cluster(int(tag))
        cluster_id = self._cluster_id(int(tag))
        self._suppressed_clusters.add(int(tag) if cluster_id is None else cluster_id)
        self._suppressed_resource_tags.update(int(unit["tag"]) for unit in cluster)

    def confirm_command(self, command_id: str) -> None:
        placement = self._command_targets.get(command_id)
        if placement is None or placement.anchor_tag is None:
            return
        if any(
            townhall in placement.action_name
            for townhall in ("Nexus", "CommandCenter", "Hatchery")
        ):
            self.suppress_anchor(placement.anchor_tag)
            self.suppress_world_target(placement.action_name, placement.world_target)

    def suppress_world_target(
        self,
        action_name: str,
        target: Sequence[int | float],
    ) -> None:
        if len(target) != 2:
            return
        point = (float(target[0]), float(target[1]))
        stored = self._quarantined_targets.setdefault(action_name, [])
        if not any(math.dist(point, previous) < 0.25 for previous in stored):
            stored.append(point)

    def is_quarantined(
        self,
        action_name: str,
        target: tuple[float, float],
        *,
        radius: float,
    ) -> bool:
        return any(
            math.dist(target, previous) <= radius
            for previous in self._quarantined_targets.get(action_name, ())
        )

    def _expansion_target(
        self,
        anchor_tag: int,
        observation: Any,
    ) -> Optional[tuple[float, float]]:
        resources = self._resource_cluster(anchor_tag)
        if sum(_is_mineral(str(unit["unit_type"])) for unit in resources) < 5:
            return None
        center = (
            sum(float(unit["x"]) for unit in resources) / len(resources),
            sum(float(unit["y"]) for unit in resources) / len(resources),
        )
        occupied = [
            unit
            for unit in _value(observation, "raw_units", ())
            if int(_value(unit, "alliance", 0)) in {1, 2, 4} and _build_progress(unit) > 0.0
        ]
        candidates: list[tuple[float, float, int, int]] = []
        for y in range(math.floor(center[1] - 12), math.ceil(center[1] + 12) + 1):
            for x in range(math.floor(center[0] - 12), math.ceil(center[0] + 12) + 1):
                point = (float(x), float(y))
                if any(
                    math.dist(
                        point,
                        (
                            float(_value(unit, "x", 0.0)),
                            float(_value(unit, "y", 0.0)),
                        ),
                    )
                    < 4.0
                    for unit in occupied
                ):
                    continue
                distances = [
                    math.dist(point, (float(unit["x"]), float(unit["y"]))) for unit in resources
                ]
                if any(
                    distance < (6.5 if _is_gas(str(unit["unit_type"])) else 5.5)
                    for unit, distance in zip(resources, distances)  # noqa: B905 - Python 3.9
                ):
                    continue
                nearby_minerals = sum(
                    _is_mineral(str(unit["unit_type"])) and distance <= 12.0
                    for unit, distance in zip(resources, distances)  # noqa: B905 - Python 3.9
                )
                if nearby_minerals < 5:
                    continue
                score = sum(
                    abs(distance - (8.5 if _is_gas(str(unit["unit_type"])) else 7.5))
                    for unit, distance in zip(resources, distances)  # noqa: B905 - Python 3.9
                )
                candidates.append((score, math.dist(point, center), x, y))
        if not candidates:
            return None
        _, _, x, y = min(candidates)
        return float(x), float(y)

    def _resource_cluster(self, anchor_tag: int) -> list[dict[str, Any]]:
        anchor = self._known_resources.get(anchor_tag)
        if anchor is None:
            return []
        resources = [anchor]
        tags = {anchor_tag}
        while True:
            connected = [
                unit
                for tag, unit in self._known_resources.items()
                if tag not in tags
                and any(
                    math.dist(
                        (float(unit["x"]), float(unit["y"])),
                        (float(existing["x"]), float(existing["y"])),
                    )
                    <= 12.0
                    for existing in resources
                )
            ]
            if not connected:
                return resources
            resources.extend(connected)
            tags.update(int(unit["tag"]) for unit in connected)

    def _cluster_id(self, anchor_tag: int) -> int | None:
        cluster = self._resource_cluster(anchor_tag)
        if not cluster:
            return None
        return min(int(unit["tag"]) for unit in cluster)

    def _resource_clusters(self) -> list[tuple[int, list[dict[str, Any]]]]:
        remaining = set(self._known_resources)
        clusters: list[tuple[int, list[dict[str, Any]]]] = []
        while remaining:
            cluster = self._resource_cluster(min(remaining))
            tags = {int(unit["tag"]) for unit in cluster}
            remaining.difference_update(tags)
            cluster_id = min(tags)
            clusters.append((cluster_id, cluster))
        return sorted(clusters, key=lambda item: item[0])

    def _expansion_candidates(self, observation: Any) -> list[int]:
        own_townhalls = [
            unit
            for unit in _value(observation, "raw_units", ())
            if int(_value(unit, "alliance", 0)) == 1
            and _unit_name(unit, self.unit_names).casefold()
            in {
                "commandcenter",
                "hatchery",
                "hive",
                "lair",
                "nexus",
                "orbitalcommand",
                "planetaryfortress",
            }
        ]
        ranked: list[tuple[float, int]] = []
        for cluster_id, cluster in self._resource_clusters():
            if self._cluster_is_suppressed(cluster):
                continue
            if sum(_is_mineral(str(unit["unit_type"])) for unit in cluster) < 5:
                continue
            center = (
                sum(float(unit["x"]) for unit in cluster) / len(cluster),
                sum(float(unit["y"]) for unit in cluster) / len(cluster),
            )
            if any(
                math.dist(
                    center,
                    (
                        float(_value(townhall, "x", 0.0)),
                        float(_value(townhall, "y", 0.0)),
                    ),
                )
                < 12.0
                for townhall in own_townhalls
            ):
                continue
            if self._expansion_target(cluster_id, observation) is None:
                continue
            base_distance = min(
                (
                    math.dist(
                        center,
                        (
                            float(_value(townhall, "x", 0.0)),
                            float(_value(townhall, "y", 0.0)),
                        ),
                    )
                    for townhall in own_townhalls
                ),
                default=math.inf,
            )
            ranked.append((base_distance, cluster_id))
        ranked.sort()
        return [cluster_id for _, cluster_id in ranked[:4]]

    def _cluster_is_suppressed(self, cluster: Sequence[Mapping[str, Any]]) -> bool:
        tags = {int(unit["tag"]) for unit in cluster}
        return bool(
            tags.intersection(self._suppressed_resource_tags)
            or tags.intersection(self._suppressed_clusters)
        )

    def _remember_diagnostic(self, action_name: str, reason: str | None) -> None:
        if reason is None:
            self._placement_diagnostics.pop(action_name, None)
        else:
            self._placement_diagnostics[action_name] = reason

    @staticmethod
    def _screen_unavailable_reason(observation: Any, spec: Any) -> str:
        feature_screen = _value(observation, "feature_screen", None)
        if feature_screen is None:
            return "out_of_view"
        if spec.requires_power and not _plane_has_positive(_value(feature_screen, "power", None)):
            return "need_power"
        if spec.requires_creep and not _plane_has_positive(_value(feature_screen, "creep", None)):
            return "need_creep"
        if not _plane_has_positive(_value(feature_screen, "buildable", None)):
            return "not_buildable"
        if not _plane_has_positive(_value(feature_screen, "pathable", None)):
            return "not_pathable"
        return "occupied_or_unreachable"


def _tag_argument(arguments: Sequence[Any], *, action_name: str) -> int:
    if not arguments:
        raise RawPlacementFailure(
            "candidate_invalidated",
            f"{action_name} has no target tag",
        )
    try:
        value = int(arguments[0], 0) if isinstance(arguments[0], str) else int(arguments[0])
    except (TypeError, ValueError) as error:
        raise RawPlacementFailure(
            "candidate_invalidated",
            f"{action_name} target is not a tag",
        ) from error
    if value <= 0:
        raise RawPlacementFailure(
            "candidate_invalidated",
            f"{action_name} tag must be positive",
        )
    return value


def _unit_by_tag(observation: Any, tag: int) -> Optional[Any]:
    return next(
        (
            unit
            for unit in _value(observation, "raw_units", ())
            if int(_value(unit, "tag", -1)) == tag
        ),
        None,
    )


def _unit_name(unit: Any, unit_names: Mapping[int, str]) -> str:
    value = _value(unit, "unit_type", "")
    if isinstance(value, str):
        return value
    return unit_names.get(int(value), f"unit:{int(value)}")


def _is_resource(name: str) -> bool:
    return _is_mineral(name) or _is_gas(name)


def _is_mineral(name: str) -> bool:
    return "mineral" in name.casefold()


def _is_gas(name: str) -> bool:
    normalized = name.casefold()
    return "vespene" in normalized or "geyser" in normalized


def _build_progress(unit: Any) -> float:
    value = float(_value(unit, "build_progress", 0.0))
    return value / 100.0 if value > 1.0 else value


def _plane_has_positive(plane: Any) -> bool:
    shape: Sequence[Any] = getattr(plane, "shape", ())
    if plane is None or len(shape) != 2:
        return False
    return any(
        int(plane[y][x]) > 0
        for y in range(int(shape[0]))
        for x in range(int(shape[1]))
    )


def _value(value: Any, name: str, default: Any) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


__all__ = [
    "RawPlacement",
    "RawPlacementCandidates",
    "RawPlacementFailure",
    "RawPlacementService",
]
