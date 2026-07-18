"""Three-specialist HIMA race brain with deterministic strategic coordination."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Mapping, Sequence
from typing import Literal, Protocol

from pydantic import Field

from rtscortex.contracts.models import ContractModel
from rtscortex.cortex.macro import runtime_frontier
from rtscortex.cortex.models import SituationAssessment, ThreatLevel
from rtscortex.playbook.models import PlaybookRuleKind, PlaybookSelection
from rtscortex.policy.hima import HIMAInputContext, HIMALiveHealth, HIMALiveProposalResponse
from rtscortex.policy.models import PolicyActionAssessment, PolicyActionClassification
from rtscortex.races import race_profile

RACE_BRAIN_COORDINATOR_VERSION = "deterministic-race-brain-v1"
HIMACluster = Literal["a", "b", "c"]
_HIMA_CLUSTERS: tuple[HIMACluster, ...] = ("a", "b", "c")


class RaceBrainMemberHealth(ContractModel):
    member_id: str = Field(min_length=1)
    cluster: HIMACluster
    health: HIMALiveHealth


class RaceBrainHealth(ContractModel):
    status: Literal["ready"] = "ready"
    model_id: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    race: Literal["protoss", "terran", "zerg"]
    coordinator_version: str = RACE_BRAIN_COORDINATOR_VERSION
    execution_groups: tuple[tuple[HIMACluster, ...], ...] = ((_HIMA_CLUSTERS),)
    members: tuple[RaceBrainMemberHealth, ...] = Field(min_length=3, max_length=3)


class RaceBrainStrategicContext(ContractModel):
    situation: SituationAssessment
    playbook: PlaybookSelection | None = None


class RaceBrainMemberProposal(ContractModel):
    member_id: str = Field(min_length=1)
    cluster: HIMACluster
    response: HIMALiveProposalResponse
    frontier: PolicyActionAssessment | None = None
    score: float
    score_reasons: tuple[str, ...]


class RaceBrainProposalResponse(ContractModel):
    protocol_version: Literal["1.0"] = "1.0"
    request_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    step_id: int = Field(ge=0)
    game_loop: int = Field(ge=0)
    race: Literal["protoss", "terran", "zerg"]
    coordinator_version: str = RACE_BRAIN_COORDINATOR_VERSION
    execution_groups: tuple[tuple[HIMACluster, ...], ...] = ((_HIMA_CLUSTERS),)
    selected_member_id: str = Field(min_length=1)
    selected: HIMALiveProposalResponse
    members: tuple[RaceBrainMemberProposal, ...] = Field(min_length=3, max_length=3)
    valid_member_count: int = Field(default=3, ge=0, le=3)
    degraded_member_ids: tuple[str, ...] = ()
    playbook_lesson_ids: tuple[str, ...] = ()
    rationale: str = Field(min_length=1)


MacroPolicyResponse = HIMALiveProposalResponse | RaceBrainProposalResponse
MacroPolicyHealth = HIMALiveHealth | RaceBrainHealth


class HIMAEnsembleMemberClient(Protocol):
    async def health(self) -> HIMALiveHealth: ...

    async def propose(
        self,
        context: HIMAInputContext,
        *,
        request_id: str | None = None,
    ) -> HIMALiveProposalResponse: ...

    async def close(self) -> None: ...


class HIMAEnsembleMemberSidecar(Protocol):
    async def start(self) -> HIMALiveHealth: ...

    async def restart(self) -> HIMALiveHealth: ...

    async def close(self) -> None: ...


class HIMAEnsemblePolicyClient:
    """Query all three specialists for one race and coordinate their proposals."""

    def __init__(
        self,
        clients: Mapping[str, HIMAEnsembleMemberClient],
        *,
        race: Literal["protoss", "terran", "zerg"],
        execution_groups: Sequence[Sequence[HIMACluster]] | None = None,
    ) -> None:
        if tuple(sorted(clients)) != ("a", "b", "c"):
            raise ValueError("race brain requires exactly the a/b/c specialist clients")
        self._clients = dict(clients)
        self.race = race
        groups = execution_groups or (_HIMA_CLUSTERS,)
        normalized: tuple[tuple[HIMACluster, ...], ...] = tuple(tuple(group) for group in groups)
        flattened = tuple(cluster for group in normalized for cluster in group)
        if tuple(sorted(flattened)) != _HIMA_CLUSTERS or len(flattened) != 3:
            raise ValueError("execution groups must contain each a/b/c specialist once")
        self.execution_groups = normalized

    async def health(self) -> RaceBrainHealth:
        members: list[RaceBrainMemberHealth] = []
        for cluster in _HIMA_CLUSTERS:
            client = self._clients[cluster]
            health = await client.health()
            expected_model_id = f"SNUMPR/{self.race.title()}-{cluster}"
            if health.model_id != expected_model_id:
                raise ValueError(
                    "race brain member identity mismatch: "
                    f"{health.model_id!r} != {expected_model_id!r}"
                )
            members.append(
                RaceBrainMemberHealth(
                    member_id=f"hima-{self.race}-{cluster}",
                    cluster=cluster,
                    health=health,
                )
            )
        revision = hashlib.sha256(
            "|".join(member.health.model_revision for member in members).encode()
        ).hexdigest()
        return RaceBrainHealth(
            model_id=f"SNUMPR/HIMA-{self.race.title()}-Ensemble",
            model_revision=revision,
            race=self.race,
            execution_groups=self.execution_groups,
            members=tuple(members),
        )

    async def propose(
        self,
        context: HIMAInputContext,
        *,
        request_id: str | None = None,
        strategic_context: RaceBrainStrategicContext | None = None,
    ) -> RaceBrainProposalResponse:
        outer_request_id = (
            request_id
            or hashlib.sha256(
                (
                    f"{context.observation.run_id}|{context.observation.episode_id}|"
                    f"{context.observation.step_id}|ensemble"
                ).encode()
            ).hexdigest()
        )

        async def propose_group(
            group: Sequence[HIMACluster],
        ) -> list[tuple[HIMACluster, HIMALiveProposalResponse]]:
            group_responses: list[tuple[HIMACluster, HIMALiveProposalResponse]] = []
            for cluster in group:
                response = await self._clients[cluster].propose(
                    context,
                    request_id=f"{outer_request_id}:{cluster}",
                )
                group_responses.append((cluster, response))
            return group_responses

        grouped_responses = await asyncio.gather(
            *(propose_group(group) for group in self.execution_groups)
        )
        response_by_cluster = {
            cluster: response for group in grouped_responses for cluster, response in group
        }
        responses = [(cluster, response_by_cluster[cluster]) for cluster in _HIMA_CLUSTERS]
        members = _coordinate(responses, context, strategic_context, self.race)
        selected = max(members, key=lambda item: item.score)
        degraded_member_ids = tuple(
            member.member_id for member in members if not _member_proposal_is_valid(member)
        )
        lesson_ids = (
            ()
            if strategic_context is None or strategic_context.playbook is None
            else strategic_context.playbook.lesson_ids
        )
        return RaceBrainProposalResponse(
            request_id=outer_request_id,
            run_id=context.observation.run_id,
            episode_id=context.observation.episode_id,
            step_id=context.observation.step_id,
            game_loop=context.observation.game_loop,
            race=self.race,
            execution_groups=self.execution_groups,
            selected_member_id=selected.member_id,
            selected=selected.response,
            members=tuple(members),
            valid_member_count=len(members) - len(degraded_member_ids),
            degraded_member_ids=degraded_member_ids,
            playbook_lesson_ids=lesson_ids,
            rationale=(
                f"Selected {selected.member_id} with score {selected.score:.1f}: "
                + ", ".join(selected.score_reasons)
            ),
        )

    async def close(self) -> None:
        await asyncio.gather(
            *(client.close() for client in self._clients.values()),
            return_exceptions=True,
        )


class HIMAEnsembleSidecar:
    """Own the three process-isolated workers behind one race brain."""

    def __init__(
        self,
        sidecars: Sequence[HIMAEnsembleMemberSidecar],
        client: HIMAEnsemblePolicyClient,
    ) -> None:
        if len(sidecars) != 3:
            raise ValueError("race brain sidecar group requires exactly three processes")
        self._sidecars = tuple(sidecars)
        self._client = client

    async def start(self) -> RaceBrainHealth:
        started: list[HIMAEnsembleMemberSidecar] = []
        try:
            for sidecar in self._sidecars:
                await sidecar.start()
                started.append(sidecar)
        except BaseException:
            await asyncio.gather(
                *(sidecar.close() for sidecar in started),
                return_exceptions=True,
            )
            raise
        return await self._client.health()

    async def restart(self) -> RaceBrainHealth:
        for sidecar in self._sidecars:
            await sidecar.restart()
        return await self._client.health()

    async def close(self) -> None:
        await asyncio.gather(
            *(sidecar.close() for sidecar in reversed(self._sidecars)),
            return_exceptions=True,
        )


def selected_hima_response(response: MacroPolicyResponse) -> HIMALiveProposalResponse:
    return response.selected if isinstance(response, RaceBrainProposalResponse) else response


def _coordinate(
    responses: Sequence[tuple[HIMACluster, HIMALiveProposalResponse]],
    context: HIMAInputContext,
    strategic_context: RaceBrainStrategicContext | None,
    race: str,
) -> list[RaceBrainMemberProposal]:
    recommendations = _playbook_recommendations(strategic_context)
    avoid_actions = _playbook_avoid_actions(strategic_context)
    members: list[RaceBrainMemberProposal] = []
    for cluster, response in responses:
        frontier = runtime_frontier(
            response.proposal,
            context.observation,
            context.previous_actions,
            race_profile(race).data,
        )
        score, reasons = _proposal_score(
            response,
            frontier,
            strategic_context,
            recommendations,
            avoid_actions,
        )
        members.append(
            RaceBrainMemberProposal(
                member_id=f"hima-{race}-{cluster}",
                cluster=cluster,
                response=response,
                frontier=frontier,
                score=score,
                score_reasons=tuple(reasons),
            )
        )
    return members


def _proposal_score(
    response: HIMALiveProposalResponse,
    frontier: PolicyActionAssessment | None,
    strategic_context: RaceBrainStrategicContext | None,
    recommendations: set[str],
    avoid_actions: set[str],
) -> tuple[float, list[str]]:
    if frontier is None:
        score = -200.0
        reasons = ["no runtime frontier"]
    else:
        weights = {
            PolicyActionClassification.MAPPED_LEGAL_NOW: 100.0,
            PolicyActionClassification.MAPPED_DEFERRED: 70.0,
            PolicyActionClassification.MAPPED_FUTURE: 35.0,
            PolicyActionClassification.OBSOLETE: 10.0,
            PolicyActionClassification.UNSUPPORTED_BY_RUNTIME: -100.0,
            PolicyActionClassification.ILLEGAL_ACTION: -150.0,
            PolicyActionClassification.PARSE_ERROR: -200.0,
        }
        score = weights[frontier.classification]
        reasons = [frontier.classification.value]
        action = frontier.source_action
        if action in recommendations:
            score += 20.0
            reasons.append("promoted playbook support")
        if action in avoid_actions:
            score -= 25.0
            reasons.append("promoted playbook warning")
        if strategic_context is not None:
            situation = strategic_context.situation
            combat_action = any(
                token in action for token in ("ZEALOT", "STALKER", "ADEPT", "VOIDRAY")
            )
            if situation.threat_level in {ThreatLevel.HIGH, ThreatLevel.CRITICAL}:
                score += 8.0 if combat_action else -8.0
                reasons.append("threat-aware combat bias" if combat_action else "threat penalty")
    if response.proposal.steps:
        score += min(len(response.proposal.steps), 10) * 0.1
        reasons.append("ordered plan depth")
    return score, reasons


def _playbook_recommendations(
    strategic_context: RaceBrainStrategicContext | None,
) -> set[str]:
    if strategic_context is None or strategic_context.playbook is None:
        return set()
    return {
        action
        for hit in strategic_context.playbook.hits
        if hit.lesson.rule_kind is PlaybookRuleKind.STRATEGY
        if (action := hit.lesson.recommended_action) is not None
    }


def _playbook_avoid_actions(
    strategic_context: RaceBrainStrategicContext | None,
) -> set[str]:
    if strategic_context is None or strategic_context.playbook is None:
        return set()
    return {
        action
        for hit in strategic_context.playbook.hits
        if hit.lesson.rule_kind is PlaybookRuleKind.STRATEGY
        if (action := hit.lesson.avoid_action) is not None
    }


def _member_proposal_is_valid(member: RaceBrainMemberProposal) -> bool:
    if member.frontier is None:
        return False
    return member.frontier.classification not in {
        PolicyActionClassification.PARSE_ERROR,
        PolicyActionClassification.ILLEGAL_ACTION,
    }
