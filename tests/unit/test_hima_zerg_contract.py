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
    HIMA_VOCABULARY_VERSIONS,
    HIMA_ZERG_ACTIONS,
    HIMALiveHealth,
    HIMALivePolicyService,
    HIMALiveProposalRequest,
    HIMALiveProposalResponse,
    HIMAMacroActionMapper,
    HIMAObservationAdapter,
    HIMAProposalParser,
)
from rtscortex.policy.hima.models import HIMAInputContext
from rtscortex.policy.models import PolicyActionClassification, PolicyObservationFixture
from rtscortex.policy.subagents import HIMA_RACE_SPECS
from rtscortex.races import race_profile

_OFFICIAL_ZERG_ACTIONS = (
    *((action_id, name, "train") for action_id, name in enumerate(
        (
            "Drone", "Overlord", "Zergling", "Queen", "Roach", "Baneling", "Ravager",
            "Overseer", "Hydralisk", "Mutalisk", "Corruptor", "Infestor", "SwarmHostMP",
            "LurkerMP", "Viper", "BroodLord", "Ultralisk",
        ),
        start=100,
    )),
    *((action_id, name, "build") for action_id, name in enumerate(
        (
            "Hatchery", "Extractor", "Lair", "Hive", "SpawningPool",
            "EvolutionChamber", "RoachWarren", "BanelingNest", "SpineCrawler",
            "SporeCrawler", "HydraliskDen", "InfestationPit", "LurkerDenMP", "Spire",
            "NydusNetwork", "UltraliskCavern", "GreaterSpire",
        ),
        start=200,
    )),
    *((action_id, name, "research") for action_id, name in enumerate(
        (
            "ZergMeleeWeaponsLevel1", "ZergMeleeWeaponsLevel2", "ZergMeleeWeaponsLevel3",
            "ZergMissileWeaponsLevel1", "ZergMissileWeaponsLevel2",
            "ZergMissileWeaponsLevel3", "ZergGroundArmorsLevel1",
            "ZergGroundArmorsLevel2", "ZergGroundArmorsLevel3",
            "ZergFlyerWeaponsLevel1", "ZergFlyerWeaponsLevel2", "ZergFlyerWeaponsLevel3",
            "ZergFlyerArmorsLevel1", "ZergFlyerArmorsLevel2", "ZergFlyerArmorsLevel3",
            "Burrow", "overlordspeed", "zerglingmovementspeed", "zerglingattackspeed",
            "GlialReconstitution", "TunnelingClaws", "CentrificalHooks",
            "EvolveMuscularAugments", "EvolveGroovedSpines", "NeuralParasite",
            "DiggingClaws", "LurkerRange", "ChitinousPlating", "AnabolicSynthesis",
        ),
        start=300,
    )),
)


class _FakeZergGenerator:
    def __init__(self) -> None:
        self.load_calls = 0

    async def load(self) -> None:
        self.load_calls += 1

    async def generate(self, *, user_message: str) -> str:
        assert '"unit": {"Drone": 1' in user_message
        return 'Reason: establish larvae production. Actions: ["Drone", "Overlord"]'


def _zerg_observation(*, available: bool = False) -> ObservationEnvelope:
    actions = []
    if available:
        actions.append(
            AvailableAction(
                name="Build_SpawningPool_Screen",
                argument_names=["screen"],
                argument_types=[ActionArgumentType.POSITION],
                actor_scopes=["Builder/Drone-1"],
                argument_candidates=[[[40, 40]]],
            )
        )
    return ObservationEnvelope(
        run_id="run-zerg",
        episode_id="episode-zerg",
        step_id=3,
        game_loop=96,
        state=SC2State(
            economy=EconomyState(
                minerals=200,
                vespene=100,
                supply_used=18,
                supply_cap=22,
            ),
            own_units=[
                UnitState(unit_id="drone", unit_type="Drone", alliance="self"),
                UnitState(unit_id="ling", unit_type="Zergling", alliance="self"),
                UnitState(unit_id="larva", unit_type="Larva", alliance="self"),
            ],
            own_structures=[
                UnitState(unit_id="hatch", unit_type="Hatchery", alliance="self"),
                UnitState(unit_id="pool", unit_type="SpawningPool", alliance="self"),
                UnitState(
                    unit_id="warren",
                    unit_type="RoachWarren",
                    alliance="self",
                    status="constructing",
                ),
            ],
            upgrades=["ZergMissileWeaponsLevel1", "zerglingmovementspeed"],
        ),
        available_actions=actions,
    )


def test_official_hima_zerg_vocabulary_matches_pinned_constants_exactly() -> None:
    assert tuple(
        (action.upstream_action_id, action.upstream_name, action.category)
        for action in HIMA_ZERG_ACTIONS
    ) == _OFFICIAL_ZERG_ACTIONS


def test_zerg_observation_adapter_keeps_official_five_fields() -> None:
    snapshot = HIMAObservationAdapter(race="zerg").adapt_context(
        HIMAInputContext(
            observation=_zerg_observation(),
            previous_actions=("TRAIN OVERLORD", "BUILD SPAWNINGPOOL"),
        )
    )

    assert snapshot.upstream_payload() == {
        "supply_used": 18,
        "supply_capacity": 22,
        "unit": {"Drone": 1, "Zergling": 1, "Hatchery": 1, "SpawningPool": 1},
        "research": ["ZergMissileWeaponsLevel1", "zerglingmovementspeed"],
        "previous_action": ["Overlord", "SpawningPool"],
    }


@pytest.mark.parametrize(
    ("raw_output", "expected"),
    [
        (
            'Reason: take gas. Actions: ["Drone", "Extractor"]',
            ["TRAIN DRONE", "BUILD EXTRACTOR"],
        ),
        (
            "Final Actions Summary: <SpawningPool><Zergling> x2",
            ["BUILD SPAWNINGPOOL", "TRAIN ZERGLING"],
        ),
        (
            "So my advice is <overlordspeed>",
            ["RESEARCH OVERLORDSPEED"],
        ),
    ],
)
def test_zerg_parser_accepts_only_pinned_output_formats(
    raw_output: str,
    expected: list[str],
) -> None:
    proposal = HIMAProposalParser(race="zerg").parse(raw_output)

    assert [step.canonical_action for step in proposal.steps] == expected
    assert proposal.diagnostics == []


def test_zerg_parser_does_not_fuzzy_match_unknown_tokens() -> None:
    proposal = HIMAProposalParser(race="zerg").parse('Actions: ["Zerglings"]')

    assert proposal.steps == []
    assert [diagnostic.code for diagnostic in proposal.diagnostics] == [
        "unknown_action_token"
    ]


def test_zerg_mapper_uses_zerg_runtime_profile() -> None:
    observation = _zerg_observation(available=True)
    proposal = HIMAProposalParser(race="zerg").parse('Actions: ["SpawningPool"]')
    fixture = PolicyObservationFixture(fixture_id="zerg", observation=observation)

    assessment = HIMAMacroActionMapper(race_profile("zerg").data).assess(
        proposal,
        fixture,
    )[0]

    assert assessment.runtime_action == "Build_SpawningPool_Screen"
    assert assessment.classification is PolicyActionClassification.MAPPED_LEGAL_NOW


def test_zerg_live_policy_sidecar_uses_the_pinned_contract_end_to_end() -> None:
    observation = _zerg_observation()
    snapshot = HIMAObservationAdapter(race="zerg").adapt_context(
        HIMAInputContext(observation=observation)
    )
    generator = _FakeZergGenerator()
    service = HIMALivePolicyService(HIMA_RACE_SPECS["zerg"][0], generator)
    request = HIMALiveProposalRequest(
        request_id="zerg-live",
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
    assert health.model_id == "SNUMPR/Zerg-a"
    assert health.model_revision == HIMA_PINNED_REVISIONS["SNUMPR/Zerg-a"]
    assert health.parser_version == HIMA_PARSER_VERSIONS["zerg"]
    assert health.vocabulary_version == HIMA_VOCABULARY_VERSIONS["zerg"]
    assert [step.canonical_action for step in response.proposal.steps] == [
        "TRAIN DRONE",
        "TRAIN OVERLORD",
    ]
