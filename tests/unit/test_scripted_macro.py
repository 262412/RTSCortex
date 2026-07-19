from __future__ import annotations

import asyncio

import pytest

from rtscortex.config import CortexMacroSettings
from rtscortex.contracts import EconomyState, ObservationEnvelope, SC2State
from rtscortex.policy.hima import HIMAInputContext
from rtscortex.runtime.scripted_macro import (
    SCRIPTED_MACRO_REVISION,
    ScriptedMacroPolicyClient,
)


def test_scripted_macro_config_requires_a_nonempty_exclusive_sequence() -> None:
    with pytest.raises(ValueError, match="requires scripted_actions"):
        CortexMacroSettings(kind="scripted")
    with pytest.raises(ValueError, match="only valid for kind=scripted"):
        CortexMacroSettings(kind="disabled", scripted_actions=["SupplyDepot"])


def test_scripted_macro_client_uses_pinned_race_parser_and_projection() -> None:
    asyncio.run(_assert_scripted_macro_client_uses_pinned_race_parser_and_projection())


async def _assert_scripted_macro_client_uses_pinned_race_parser_and_projection() -> None:
    client = ScriptedMacroPolicyClient(
        race="terran",
        actions=["SupplyDepot", "Barracks", "Refinery", "BarracksTechLab"],
        objective="Verify the Terran add-on path.",
    )
    observation = ObservationEnvelope(
        run_id="run-scripted",
        episode_id="episode-scripted",
        step_id=3,
        game_loop=112,
        state=SC2State(
            economy=EconomyState(
                minerals=500,
                vespene=100,
                supply_used=12,
                supply_cap=23,
                workers=12,
                army_supply=0,
            )
        ),
    )

    health = await client.health()
    response = await client.propose(
        HIMAInputContext(observation=observation),
        request_id="request-scripted",
    )

    assert health.model_revision == SCRIPTED_MACRO_REVISION
    assert response.request_id == "request-scripted"
    assert response.projection_hash != "0" * 64
    assert [step.canonical_action for step in response.proposal.steps] == [
        "BUILD SUPPLYDEPOT",
        "BUILD BARRACKS",
        "BUILD REFINERY",
        "BUILD BARRACKSTECHLAB",
    ]
    assert response.proposal.strategic_objective == "Verify the Terran add-on path."

    suffix = await client.propose(
        HIMAInputContext(
            observation=observation.model_copy(update={"game_loop": 224}),
            previous_actions=("SupplyDepot", "Barracks", "Refinery"),
        ),
        request_id="request-scripted-suffix",
    )
    assert [step.canonical_action for step in suffix.proposal.steps] == [
        "BUILD BARRACKSTECHLAB"
    ]


def test_scripted_macro_client_rejects_unknown_race_action() -> None:
    with pytest.raises(ValueError, match="pinned race vocabulary"):
        ScriptedMacroPolicyClient(
            race="terran",
            actions=["Pylon"],
            objective="Reject cross-race actions.",
        )


def test_scripted_macro_client_rejects_ambiguous_duplicate_actions() -> None:
    with pytest.raises(ValueError, match="must be unique"):
        ScriptedMacroPolicyClient(
            race="terran",
            actions=["Marine", "Marine"],
            objective="Reject ambiguous confirmation cursors.",
        )
