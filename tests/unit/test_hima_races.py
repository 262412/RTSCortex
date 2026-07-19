from __future__ import annotations

import asyncio

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
    HIMA_PARSER_VERSIONS,
    HIMA_PINNED_REVISIONS,
    HIMA_TERRAN_ACTIONS,
    HIMA_VOCABULARY_VERSIONS,
    HIMA_ZERG_ACTIONS,
    HIMALiveHealth,
    HIMALivePolicyService,
    HIMALiveProposalRequest,
    HIMALiveProposalResponse,
    HIMAMacroActionMapper,
    HIMAObservationAdapter,
    HIMAProposalParser,
    hima_actions_for_race,
)
from rtscortex.policy.hima.models import HIMAInputContext
from rtscortex.policy.models import PolicyActionClassification, PolicyObservationFixture
from rtscortex.policy.subagents import HIMA_RACE_SPECS
from rtscortex.races import race_profile

_OFFICIAL_TERRAN_ACTIONS = (
    *(
        (action_id, name, "train")
        for action_id, name in enumerate(
            (
                "SCV",
                "MULE",
                "Marine",
                "Reaper",
                "Marauder",
                "Ghost",
                "Hellion",
                "WidowMine",
                "Cyclone",
                "SiegeTank",
                "Thor",
                "VikingFighter",
                "Medivac",
                "Liberator",
                "Banshee",
                "Raven",
                "Battlecruiser",
            ),
            start=100,
        )
    ),
    *(
        (action_id, name, "build")
        for action_id, name in enumerate(
            (
                "CommandCenter",
                "Refinery",
                "OrbitalCommand",
                "PlanetaryFortress",
                "Barracks",
                "Factory",
                "Starport",
                "BarracksReactor",
                "BarracksTechLab",
                "FactoryReactor",
                "FactoryTechLab",
                "StarportReactor",
                "StarportTechLab",
                "SupplyDepot",
                "EngineeringBay",
                "Bunker",
                "MissileTurret",
                "SensorTower",
                "GhostAcademy",
                "Armory",
                "FusionCore",
            ),
            start=200,
        )
    ),
    *(
        (action_id, name, "research")
        for action_id, name in enumerate(
            (
                "TerranInfantryWeaponsLevel1",
                "TerranInfantryWeaponsLevel2",
                "TerranInfantryWeaponsLevel3",
                "TerranInfantryArmorsLevel1",
                "TerranInfantryArmorsLevel2",
                "TerranInfantryArmorsLevel3",
                "TerranVehicleWeaponsLevel1",
                "TerranVehicleWeaponsLevel2",
                "TerranVehicleWeaponsLevel3",
                "TerranShipWeaponsLevel1",
                "TerranShipWeaponsLevel2",
                "TerranShipWeaponsLevel3",
                "TerranVehicleAndShipArmorsLevel1",
                "TerranVehicleAndShipArmorsLevel2",
                "TerranVehicleAndShipArmorsLevel3",
                "TerranBuildingArmor",
                "HiSecAutoTracking",
                "Stimpack",
                "ShieldWall",
                "PunisherGrenades",
                "PersonalCloaking",
                "SmartServos",
                "HighCapacityBarrels",
                "DrillClaws",
                "CycloneLockOnDamageUpgrade",
                "MedivacIncreaseSpeedBoost",
                "LiberatorAGRangeUpgrade",
                "BansheeCloak",
                "BansheeSpeed",
                "InterferenceMatrix",
                "BattlecruiserEnableSpecializations",
            ),
            start=300,
        )
    ),
)


class _FakeTerranGenerator:
    def __init__(self) -> None:
        self.load_calls = 0
        self.user_messages: list[str] = []

    async def load(self) -> None:
        self.load_calls += 1

    async def generate(self, *, user_message: str) -> str:
        self.user_messages.append(user_message)
        return 'Reason: establish production. Actions: ["SCV", "SupplyDepot", "Barracks"]'


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
    assert (
        HIMAProposalParser(race="terran")
        .parse('Actions: ["SCV", "SupplyDepot"]')
        .steps[1]
        .canonical_action
        == "BUILD SUPPLYDEPOT"
    )
    proposal = HIMAProposalParser(race="zerg").parse('Actions: ["SCV"]')
    assert proposal.steps == []
    assert proposal.diagnostics[0].code == "unknown_action_token"


def test_official_hima_terran_vocabulary_matches_pinned_constants_exactly() -> None:
    assert (
        tuple(
            (action.upstream_action_id, action.upstream_name, action.category)
            for action in HIMA_TERRAN_ACTIONS
        )
        == _OFFICIAL_TERRAN_ACTIONS
    )


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


def test_terran_live_policy_uses_race_specific_contract_end_to_end() -> None:
    observation = ObservationEnvelope(
        run_id="run-terran",
        episode_id="episode-terran",
        step_id=7,
        game_loop=224,
        state=SC2State(
            economy=EconomyState(supply_used=14, supply_cap=23),
            own_units=[UnitState(unit_id="scv", unit_type="SCV", alliance="self")],
            own_structures=[
                UnitState(unit_id="cc", unit_type="CommandCenter", alliance="self"),
            ],
        ),
    )
    adapter = HIMAObservationAdapter(race="terran")
    snapshot = adapter.adapt_context(
        HIMAInputContext(observation=observation, previous_actions=("BUILD SUPPLYDEPOT",))
    )
    generator = _FakeTerranGenerator()
    service = HIMALivePolicyService(HIMA_RACE_SPECS["terran"][0], generator)
    request = HIMALiveProposalRequest(
        request_id="terran-live",
        run_id=observation.run_id,
        episode_id=observation.episode_id,
        step_id=observation.step_id,
        game_loop=observation.game_loop,
        snapshot=snapshot,
    )

    async def execute() -> tuple[HIMALiveHealth, HIMALiveProposalResponse]:
        await service.start()
        return service.health(), await service.propose(request)

    health, response = asyncio.run(execute())

    assert generator.load_calls == 1
    assert health.model_id == "SNUMPR/Terran-a"
    assert health.model_revision == HIMA_PINNED_REVISIONS["SNUMPR/Terran-a"]
    assert health.parser_version == HIMA_PARSER_VERSIONS["terran"]
    assert health.vocabulary_version == HIMA_VOCABULARY_VERSIONS["terran"]
    assert [step.canonical_action for step in response.proposal.steps] == [
        "TRAIN SCV",
        "BUILD SUPPLYDEPOT",
        "BUILD BARRACKS",
    ]
