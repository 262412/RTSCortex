from __future__ import annotations

import json

import pytest

from rtscortex.contracts import (
    AvailableAction,
    EconomyState,
    ProductionItem,
    UnitState,
)
from rtscortex.policy.hima import HIMA_ADAPTER_VERSION, HIMAObservationAdapter
from rtscortex.policy.models import PolicyObservationFixture
from tests.helpers import make_observation


def _fixture() -> PolicyObservationFixture:
    base = make_observation(game_loop=2_016)
    observation = base.model_copy(
        update={
            "text_observation": "SECRET_ENEMY_CONTEXT at (91, 22)",
            "alerts": ["SECRET_ALERT"],
            "available_actions": [AvailableAction(name="SECRET_RUNTIME_ACTION")],
            "state": base.state.model_copy(
                update={
                    "economy": EconomyState(
                        minerals=999,
                        vespene=888,
                        supply_used=21,
                        supply_cap=31,
                        workers=19,
                        army_supply=2,
                    ),
                    "own_units": [
                        UnitState(
                            unit_id="probe-secret-1",
                            unit_type="Probe",
                            alliance="self",
                        ),
                        UnitState(
                            unit_id="probe-secret-2",
                            unit_type="Probe",
                            alliance="self",
                        ),
                        UnitState(
                            unit_id="zealot-secret",
                            unit_type="Zealot",
                            alliance="self",
                        ),
                        UnitState(
                            unit_id="unknown-secret",
                            unit_type="RuntimeOnlyUnit",
                            alliance="self",
                        ),
                    ],
                    "own_structures": [
                        UnitState(
                            unit_id="nexus-secret",
                            unit_type="Nexus",
                            alliance="self",
                            position=(20, 20),
                            status="idle",
                        ),
                        UnitState(
                            unit_id="gateway-complete-secret",
                            unit_type="Gateway",
                            alliance="self",
                        ),
                        UnitState(
                            unit_id="warpgate-complete-secret",
                            unit_type="WarpGate",
                            alliance="self",
                        ),
                        UnitState(
                            unit_id="gateway-building-secret",
                            unit_type="Gateway",
                            alliance="self",
                            status="constructing",
                        ),
                        UnitState(
                            unit_id="pylon-building-secret",
                            unit_type="Pylon",
                            alliance="self",
                            status="pending",
                        ),
                    ],
                    "production_queue": [
                        ProductionItem(
                            name="Train_Zealot",
                            producer_id="gateway-complete-secret",
                            progress=0.5,
                        )
                    ],
                    "upgrades": [
                        "WarpGateResearch",
                        "ProtossGroundWeaponsLevel1",
                        "upgrade:999",
                    ],
                }
            ),
        }
    )
    return PolicyObservationFixture(
        fixture_id="technology-1",
        observation=observation,
        previous_actions=["Probe", "Pylon", "WarpGateResearch"],
    )


def test_adapter_emits_exact_upstream_five_field_payload() -> None:
    snapshot, rendered = HIMAObservationAdapter().prepare(_fixture())
    payload = json.loads(rendered)

    assert snapshot.adapter_version == HIMA_ADAPTER_VERSION
    assert snapshot.unit == {"Probe": 2, "Zealot": 1, "Nexus": 1, "Gateway": 2}
    assert snapshot.research == (
        "ProtossGroundWeaponsLevel1",
        "WarpGateResearch",
    )
    assert snapshot.previous_action == ("Probe", "Pylon", "WarpGateResearch")
    assert payload == {
        "supply_used": 21,
        "supply_capacity": 31,
        "unit": {"Probe": 2, "Zealot": 1, "Nexus": 1, "Gateway": 2},
        "research": ["ProtossGroundWeaponsLevel1", "WarpGateResearch"],
        "previous_action": ["Probe", "Pylon", "WarpGateResearch"],
    }
    assert list(payload) == [
        "supply_used",
        "supply_capacity",
        "unit",
        "research",
        "previous_action",
    ]

    forbidden = (
        "SECRET",
        "RuntimeOnlyUnit",
        "upgrade:999",
        "999",
        "888",
        "probe-secret",
        "gateway-complete-secret",
        "goal_progress",
        "available_actions",
        "position",
        "game_time",
        "production",
    )
    assert all(value not in rendered for value in forbidden)


def test_adapter_is_byte_deterministic_and_hashes_only_visible_projection() -> None:
    adapter = HIMAObservationAdapter()
    first_snapshot, first = adapter.prepare(_fixture())
    second_snapshot, second = adapter.prepare(_fixture())

    assert first == second
    assert first_snapshot.projection_hash == second_snapshot.projection_hash
    changed = first_snapshot.model_copy(update={"supply_used": 22})
    assert changed.projection_hash != first_snapshot.projection_hash


def test_adapter_excludes_in_progress_structures_instead_of_double_counting() -> None:
    snapshot = HIMAObservationAdapter().adapt(_fixture())

    assert snapshot.unit["Gateway"] == 2
    assert "Pylon" not in snapshot.unit
    assert "Train_Zealot" not in snapshot.unit


def test_adapter_normalizes_live_pysc2_upgrade_names_and_legacy_ids() -> None:
    base = _fixture()
    fixture = base.model_copy(
        update={
            "observation": base.observation.model_copy(
                update={
                    "state": base.observation.state.model_copy(
                        update={"upgrades": ["upgrade:84", "Blink"]}
                    )
                }
            )
        }
    )

    snapshot = HIMAObservationAdapter().adapt(fixture)

    assert snapshot.research == ("WarpGateResearch", "BlinkTech")


def test_adapter_normalizes_previous_actions_and_rejects_unknown_tokens() -> None:
    fixture = _fixture().model_copy(
        update={"previous_actions": ["TRAIN PROBE", "RESEARCH WARPGATE"]}
    )
    snapshot = HIMAObservationAdapter().adapt(fixture)
    assert snapshot.previous_action == ("Probe", "WarpGateResearch")

    invalid = fixture.model_copy(update={"previous_actions": ["Orthotomist"]})
    with pytest.raises(ValueError, match="not in the pinned Protoss vocabulary"):
        HIMAObservationAdapter().adapt(invalid)
