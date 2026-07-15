"""Projection of RTSCortex observations into HIMA's checkpoint interface."""

from __future__ import annotations

import json
from collections import Counter

from rtscortex.policy.hima.models import HIMAObservationSnapshot
from rtscortex.policy.hima.vocabulary import HIMA_PROTOSS_ACTIONS, resolve_hima_action
from rtscortex.policy.models import PolicyObservationFixture

_INCOMPLETE_STATUSES = frozenset({"constructing", "pending", "under_construction"})
_TRAIN_NAMES = {
    action.upstream_name.casefold(): action.upstream_name
    for action in HIMA_PROTOSS_ACTIONS
    if action.category == "train"
}
_BUILD_NAMES = {
    action.upstream_name.casefold(): action.upstream_name
    for action in HIMA_PROTOSS_ACTIONS
    if action.category == "build"
}
_RESEARCH_NAMES = {
    action.upstream_name.casefold(): action.upstream_name
    for action in HIMA_PROTOSS_ACTIONS
    if action.category == "research"
}
_UNIT_ORDER = tuple(
    action.upstream_name
    for action in HIMA_PROTOSS_ACTIONS
    if action.category in {"train", "build"}
)
_RESEARCH_ORDER = tuple(
    action.upstream_name
    for action in HIMA_PROTOSS_ACTIONS
    if action.category == "research"
)


class HIMAObservationAdapter:
    """Build the exact upstream five-field, own-state-only JSON payload."""

    def adapt(self, fixture: PolicyObservationFixture) -> HIMAObservationSnapshot:
        state = fixture.observation.state
        unit_counts: Counter[str] = Counter()
        for unit in state.own_units:
            name = _TRAIN_NAMES.get(unit.unit_type.casefold())
            if name is not None:
                unit_counts[name] += 1

        for structure in state.own_structures:
            if not _is_completed_structure(structure.status):
                continue
            lookup_name = structure.unit_type.casefold()
            # Upstream deliberately reports completed Warp Gates as Gateways.
            if lookup_name == "warpgate":
                lookup_name = "gateway"
            name = _BUILD_NAMES.get(lookup_name)
            if name is not None:
                unit_counts[name] += 1

        visible_units = {
            name: unit_counts[name]
            for name in _UNIT_ORDER
            if unit_counts[name] > 0
        }
        completed_research = {
            name
            for raw_name in state.upgrades
            if (name := _RESEARCH_NAMES.get(raw_name.casefold())) is not None
        }
        previous_action = tuple(
            _normalize_previous_action(raw_action)
            for raw_action in fixture.previous_actions
        )
        return HIMAObservationSnapshot(
            supply_used=state.economy.supply_used,
            supply_capacity=state.economy.supply_cap,
            unit=visible_units,
            research=tuple(
                name for name in _RESEARCH_ORDER if name in completed_research
            ),
            previous_action=previous_action,
        )

    def serialize(self, snapshot: HIMAObservationSnapshot) -> str:
        """Match upstream's single ``json.dumps(query_input)`` user message."""

        return json.dumps(snapshot.upstream_payload(), ensure_ascii=False)

    def prepare(
        self,
        fixture: PolicyObservationFixture,
    ) -> tuple[HIMAObservationSnapshot, str]:
        snapshot = self.adapt(fixture)
        return snapshot, self.serialize(snapshot)


def _normalize_previous_action(raw_action: str) -> str:
    action = resolve_hima_action(raw_action)
    if action is None:
        raise ValueError(
            f"previous HIMA action is not in the pinned Protoss vocabulary: {raw_action}"
        )
    return action.upstream_name


def _is_completed_structure(status: str | None) -> bool:
    if status is None:
        return True
    normalized = status.strip().lower().replace("-", "_").replace(" ", "_")
    return normalized not in _INCOMPLETE_STATUSES and not normalized.startswith(
        "constructing"
    )
