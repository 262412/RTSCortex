from __future__ import annotations

import pytest

from rtscortex.contracts import (
    ActionArgumentType,
    AvailableAction,
    EconomyState,
    ObservationEnvelope,
    SC2State,
    UnitState,
)
from rtscortex.policy.hima import (
    HIMA_TERRAN_ACTIONS,
    HIMA_ZERG_ACTIONS,
    HIMAMacroActionMapper,
    HIMAObservationAdapter,
    HIMAProposalParser,
    hima_actions_for_race,
)
from rtscortex.policy.hima.models import HIMAInputContext
from rtscortex.policy.models import PolicyActionClassification, PolicyObservationFixture
from rtscortex.races import race_profile


@pytest.mark.parametrize(
    ("race", "expected_count", "last_id"),
    (("protoss", 60, 325), ("terran", 69, 330), ("zerg", 63, 328)),
)
def test_official_hima_race_vocabulary_contract(
    race: str,
    expected_count: int,
    last_id: int,
) -> None:
    actions = hima_actions_for_race(race)

    assert len(actions) == expected_count
    assert len({action.upstream_action_id for action in actions}) == expected_count
    assert actions[0].upstream_action_id == 100
    assert actions[-1].upstream_action_id == last_id


def test_official_hima_race_tokens_are_not_cross_resolved() -> None:
    assert HIMA_TERRAN_ACTIONS[0].upstream_name == "SCV"
    assert HIMA_ZERG_ACTIONS[0].upstream_name == "Drone"
    assert HIMAProposalParser(race="terran").parse('Actions: ["SCV", "SupplyDepot"]').steps[
        1
    ].canonical_action == "BUILD SUPPLYDEPOT"
    proposal = HIMAProposalParser(race="zerg").parse('Actions: ["SCV"]')
    assert proposal.steps == []
    assert proposal.diagnostics[0].code == "unknown_action_token"


def test_terran_observation_adapter_keeps_official_five_fields() -> None:
    observation = ObservationEnvelope(
        run_id="run",
        episode_id="episode",
        step_id=0,
        game_loop=0,
        state=SC2State(
            economy=EconomyState(supply_used=14, supply_cap=23),
            own_units=[UnitState(unit_id="scv", unit_type="SCV", alliance="self")],
            own_structures=[
                UnitState(unit_id="cc", unit_type="CommandCenter", alliance="self"),
                UnitState(unit_id="rax", unit_type="Barracks", alliance="self"),
            ],
            upgrades=["Stimpack"],
        ),
    )
    snapshot = HIMAObservationAdapter(race="terran").adapt_context(
        HIMAInputContext(
            observation=observation,
            previous_actions=("BUILD SUPPLYDEPOT",),
        )
    )

    assert snapshot.upstream_payload() == {
        "supply_used": 14,
        "supply_capacity": 23,
        "unit": {"SCV": 1, "CommandCenter": 1, "Barracks": 1},
        "research": ["Stimpack"],
        "previous_action": ["SupplyDepot"],
    }


def test_terran_mapper_uses_terran_runtime_profile() -> None:
    observation = ObservationEnvelope(
        run_id="run",
        episode_id="episode",
        step_id=0,
        game_loop=0,
        state=SC2State(economy=EconomyState(minerals=100, supply_cap=15)),
        available_actions=[
            AvailableAction(
                name="Build_SupplyDepot_Screen",
                argument_names=["screen"],
                argument_types=[ActionArgumentType.POSITION],
                actor_scopes=["Builder/SCV-1"],
                argument_candidates=[[[40, 40]]],
            )
        ],
    )
    proposal = HIMAProposalParser(race="terran").parse('Actions: ["SupplyDepot"]')
    fixture = PolicyObservationFixture(fixture_id="terran", observation=observation)

    assessment = HIMAMacroActionMapper(race_profile("terran").data).assess(
        proposal,
        fixture,
    )[0]

    assert assessment.runtime_action == "Build_SupplyDepot_Screen"
    assert assessment.classification is PolicyActionClassification.MAPPED_LEGAL_NOW
