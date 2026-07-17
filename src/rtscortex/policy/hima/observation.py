"""Projection of RTSCortex observations into HIMA's checkpoint interface."""

from __future__ import annotations

import json
from collections import Counter

from rtscortex.policy.hima.models import HIMAInputContext, HIMAObservationSnapshot
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
_RESEARCH_ALIASES = {
    "blink": "BlinkTech",
    "resonatingglaives": "AdeptPiercingAttack",
    "psistorm": "PsiStormTech",
    "shadowstrike": "DarkTemplarBlinkUpgrade",
    "graviticbooster": "ObserverGraviticBooster",
    "anionpulsecrystals": "PhoenixRangeUpgrade",
    "fluxvanes": "VoidRaySpeedUpgrade",
}
_RESEARCH_IDS = {
    39: "ProtossGroundWeaponsLevel1",
    40: "ProtossGroundWeaponsLevel2",
    41: "ProtossGroundWeaponsLevel3",
    42: "ProtossGroundArmorsLevel1",
    43: "ProtossGroundArmorsLevel2",
    44: "ProtossGroundArmorsLevel3",
    45: "ProtossShieldsLevel1",
    46: "ProtossShieldsLevel2",
    47: "ProtossShieldsLevel3",
    48: "ObserverGraviticBooster",
    49: "GraviticDrive",
    50: "ExtendedThermalLance",
    52: "PsiStormTech",
    78: "ProtossAirWeaponsLevel1",
    79: "ProtossAirWeaponsLevel2",
    80: "ProtossAirWeaponsLevel3",
    81: "ProtossAirArmorsLevel1",
    82: "ProtossAirArmorsLevel2",
    83: "ProtossAirArmorsLevel3",
    84: "WarpGateResearch",
    86: "Charge",
    87: "BlinkTech",
    99: "PhoenixRangeUpgrade",
    130: "AdeptPiercingAttack",
    141: "DarkTemplarBlinkUpgrade",
}
_UNIT_ORDER = tuple(
    action.upstream_name for action in HIMA_PROTOSS_ACTIONS if action.category in {"train", "build"}
)
_RESEARCH_ORDER = tuple(
    action.upstream_name for action in HIMA_PROTOSS_ACTIONS if action.category == "research"
)


class HIMAObservationAdapter:
    """Build the exact upstream five-field, own-state-only JSON payload."""

    def adapt(self, fixture: PolicyObservationFixture) -> HIMAObservationSnapshot:
        """Compatibility entrypoint for immutable comparison fixtures."""

        return self.adapt_context(
            HIMAInputContext(
                observation=fixture.observation,
                previous_actions=tuple(fixture.previous_actions),
            )
        )

    def adapt_context(self, context: HIMAInputContext) -> HIMAObservationSnapshot:
        """Project one offline or live context through the same HIMA contract."""

        state = context.observation.state
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

        visible_units = {name: unit_counts[name] for name in _UNIT_ORDER if unit_counts[name] > 0}
        completed_research = {
            name
            for raw_name in state.upgrades
            if (name := _normalize_research(raw_name)) is not None
        }
        previous_action = tuple(
            _normalize_previous_action(raw_action) for raw_action in context.previous_actions
        )
        return HIMAObservationSnapshot(
            supply_used=state.economy.supply_used,
            supply_capacity=state.economy.supply_cap,
            unit=visible_units,
            research=tuple(name for name in _RESEARCH_ORDER if name in completed_research),
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

    def prepare_context(
        self,
        context: HIMAInputContext,
    ) -> tuple[HIMAObservationSnapshot, str]:
        """Return the exact snapshot and message for a live or offline context."""

        snapshot = self.adapt_context(context)
        return snapshot, self.serialize(snapshot)


def _normalize_previous_action(raw_action: str) -> str:
    action = resolve_hima_action(raw_action)
    if action is None:
        raise ValueError(
            f"previous HIMA action is not in the pinned Protoss vocabulary: {raw_action}"
        )
    return action.upstream_name


def _normalize_research(raw_name: str) -> str | None:
    normalized = raw_name.casefold()
    direct = _RESEARCH_NAMES.get(normalized)
    if direct is not None:
        return direct
    alias = _RESEARCH_ALIASES.get(normalized)
    if alias is not None:
        return alias
    prefix = "upgrade:"
    if not normalized.startswith(prefix):
        return None
    try:
        upgrade_id = int(normalized.removeprefix(prefix))
    except ValueError:
        return None
    return _RESEARCH_IDS.get(upgrade_id)


def _is_completed_structure(status: str | None) -> bool:
    if status is None:
        return True
    normalized = status.strip().lower().replace("-", "_").replace(" ", "_")
    return normalized not in _INCOMPLETE_STATUSES and not normalized.startswith("constructing")
