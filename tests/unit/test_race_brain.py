from __future__ import annotations

import asyncio

from rtscortex.contracts import (
    ActionArgumentType,
    AvailableAction,
    EconomyState,
    ObservationEnvelope,
    SC2State,
)
from rtscortex.cortex import (
    DeterministicSituationAnalyzer,
    HIMAEnsemblePolicyClient,
    RaceBrainProposalResponse,
    RaceBrainStrategicContext,
)
from rtscortex.policy.hima import (
    HIMA_ADAPTER_VERSION,
    HIMA_PARSER_VERSION,
    HIMA_PINNED_REVISIONS,
    HIMA_VOCABULARY_VERSION,
    HIMAInputContext,
    HIMALiveHealth,
    HIMALiveProposalResponse,
    HIMAObservationAdapter,
    HIMAProposalParser,
)


class _Client:
    def __init__(self, cluster: str, output: str) -> None:
        self.cluster = cluster
        self.output = output
        self.calls = 0

    async def health(self) -> HIMALiveHealth:
        model_id = f"SNUMPR/Protoss-{self.cluster}"
        return HIMALiveHealth(
            model_id=model_id,
            model_revision=HIMA_PINNED_REVISIONS[model_id],
            adapter_version=HIMA_ADAPTER_VERSION,
            parser_version=HIMA_PARSER_VERSION,
            vocabulary_version=HIMA_VOCABULARY_VERSION,
        )

    async def propose(
        self,
        context: HIMAInputContext,
        *,
        request_id: str | None = None,
    ) -> HIMALiveProposalResponse:
        self.calls += 1
        snapshot = HIMAObservationAdapter().adapt_context(context)
        return HIMALiveProposalResponse(
            request_id=request_id or self.cluster,
            run_id=context.observation.run_id,
            episode_id=context.observation.episode_id,
            step_id=context.observation.step_id,
            game_loop=context.observation.game_loop,
            projection_hash=snapshot.projection_hash,
            proposal=HIMAProposalParser().parse(self.output),
        )

    async def close(self) -> None:
        return None


class _ScheduledClient(_Client):
    def __init__(self, cluster: str, events: list[str]) -> None:
        super().__init__(cluster, "Actions: ['Pylon']")
        self.events = events

    async def propose(
        self,
        context: HIMAInputContext,
        *,
        request_id: str | None = None,
    ) -> HIMALiveProposalResponse:
        self.events.append(f"start:{self.cluster}")
        await asyncio.sleep(0.01)
        response = await super().propose(context, request_id=request_id)
        self.events.append(f"end:{self.cluster}")
        return response


def _observation() -> ObservationEnvelope:
    return ObservationEnvelope(
        run_id="run",
        episode_id="episode",
        step_id=0,
        game_loop=0,
        state=SC2State(
            economy=EconomyState(
                minerals=200,
                supply_used=12,
                supply_cap=15,
                workers=12,
            )
        ),
        available_actions=[
            AvailableAction(
                name="Build_Pylon_Screen",
                argument_names=["screen"],
                argument_types=[ActionArgumentType.POSITION],
                actor_scopes=["Builder/Probe-1"],
                argument_candidates=[[[65, 90]]],
            )
        ],
    )


def test_race_brain_queries_all_three_and_selects_legal_frontier() -> None:
    clients = {
        "a": _Client("a", "Actions: ['Pylon']"),
        "b": _Client("b", "Actions: ['Gateway']"),
        "c": _Client("c", "Actions: ['Stargate']"),
    }
    client = HIMAEnsemblePolicyClient(clients, race="protoss")
    observation = _observation()

    response = asyncio.run(
        client.propose(
            HIMAInputContext(observation=observation),
            request_id="ensemble-request",
            strategic_context=RaceBrainStrategicContext(
                situation=DeterministicSituationAnalyzer().assess(observation)
            ),
        )
    )

    assert isinstance(response, RaceBrainProposalResponse)
    assert response.selected_member_id == "hima-protoss-a"
    assert [member.member_id for member in response.members] == [
        "hima-protoss-a",
        "hima-protoss-b",
        "hima-protoss-c",
    ]
    assert all(member.calls == 1 for member in clients.values())
    assert response.selected.proposal.steps[0].canonical_action == "BUILD PYLON"
    assert response.valid_member_count == 3
    assert response.degraded_member_ids == ()


def test_race_brain_isolates_invalid_member_output() -> None:
    clients = {
        "a": _Client("a", "Actions: ['Pylon']"),
        "b": _Client("b", "Actions: ['supply:12', 'warpGateResearch']"),
        "c": _Client("c", "Actions: ['Pylon', 'Gateway']"),
    }
    client = HIMAEnsemblePolicyClient(clients, race="protoss")

    response = asyncio.run(
        client.propose(
            HIMAInputContext(observation=_observation()),
            request_id="degraded-member-request",
        )
    )

    assert response.selected_member_id == "hima-protoss-c"
    assert response.valid_member_count == 2
    assert response.degraded_member_ids == ("hima-protoss-b",)
    invalid = next(member for member in response.members if member.cluster == "b")
    assert invalid.frontier is not None
    assert invalid.frontier.classification.value == "parse_error"
    assert invalid.score < 0


def test_unsupported_frontier_is_a_runtime_gap_not_a_degraded_member() -> None:
    clients = {
        "a": _Client("a", "Actions: ['RoboticsFacility']"),
        "b": _Client("b", "Actions: ['Pylon']"),
        "c": _Client("c", "Actions: ['Gateway']"),
    }
    client = HIMAEnsemblePolicyClient(clients, race="protoss")

    response = asyncio.run(
        client.propose(
            HIMAInputContext(observation=_observation()),
            request_id="runtime-gap-request",
        )
    )

    unsupported = next(member for member in response.members if member.cluster == "a")
    assert unsupported.frontier is not None
    assert unsupported.frontier.classification.value == "unsupported_by_runtime"
    assert response.valid_member_count == 3
    assert response.degraded_member_ids == ()


def test_race_brain_health_contains_all_checkpoint_provenance() -> None:
    clients = {cluster: _Client(cluster, "Actions: ['Pylon']") for cluster in ("a", "b", "c")}
    client = HIMAEnsemblePolicyClient(clients, race="protoss")

    health = asyncio.run(client.health())

    assert health.race == "protoss"
    assert len(health.members) == 3
    assert [member.health.model_id for member in health.members] == [
        "SNUMPR/Protoss-a",
        "SNUMPR/Protoss-b",
        "SNUMPR/Protoss-c",
    ]


def test_race_brain_runs_device_groups_in_parallel_and_members_in_order() -> None:
    events: list[str] = []
    client = HIMAEnsemblePolicyClient(
        {cluster: _ScheduledClient(cluster, events) for cluster in ("a", "b", "c")},
        race="protoss",
        execution_groups=(("a",), ("b", "c")),
    )

    response = asyncio.run(client.propose(HIMAInputContext(observation=_observation())))

    assert response.execution_groups == (("a",), ("b", "c"))
    assert events.index("start:a") < events.index("end:b")
    assert events.index("start:b") < events.index("end:a")
    assert events.index("end:b") < events.index("start:c")
