"""Deterministic macro-policy client for live engineering canaries."""

from __future__ import annotations

import json

from rtscortex.policy.hima import (
    HIMA_ADAPTER_VERSION,
    HIMAInputContext,
    HIMALiveHealth,
    HIMALiveProposalResponse,
    HIMAObservationAdapter,
    HIMAProposalParser,
)
from rtscortex.policy.hima.race_vocabulary import (
    HIMA_PARSER_VERSIONS,
    HIMA_VOCABULARY_VERSIONS,
)
from rtscortex.policy.models import MacroPolicyProposal

SCRIPTED_MACRO_REVISION = "scripted-macro-v1"


class ScriptedMacroPolicyClient:
    """Return one fixed, race-validated macro sequence on every planning cycle."""

    def __init__(
        self,
        *,
        race: str,
        actions: list[str],
        objective: str,
    ) -> None:
        self.race = race.casefold()
        self._adapter = HIMAObservationAdapter(race=self.race)
        self._parser = HIMAProposalParser(race=self.race)
        normalized_actions = tuple(action.strip() for action in actions)
        if len(set(normalized_actions)) != len(normalized_actions):
            raise ValueError("scripted macro actions must be unique")
        proposal = self._parse_actions(normalized_actions, objective)
        if proposal.diagnostics or len(proposal.steps) != len(actions):
            diagnostics = ", ".join(item.code for item in proposal.diagnostics)
            raise ValueError(
                "scripted macro actions must use the pinned race vocabulary"
                + (f": {diagnostics}" if diagnostics else "")
            )
        self._actions = normalized_actions
        self._objective = objective
        self._empty_proposal = proposal.model_copy(
            update={
                "strategic_objective": objective,
                "steps": [],
                "raw_output": "Actions: []",
            }
        )
        self._completed_prefix = 0
        self._closed = False

    def _parse_actions(
        self,
        actions: tuple[str, ...],
        objective: str,
    ) -> MacroPolicyProposal:
        raw_output = f"Reason: {objective} Actions: {json.dumps(actions)}"
        return self._parser.parse(raw_output).model_copy(update={"strategic_objective": objective})

    async def health(self) -> HIMALiveHealth:
        if self._closed:
            raise RuntimeError("scripted macro client is closed")
        return HIMALiveHealth(
            model_id=f"RTSCortex/Scripted-{self.race.title()}",
            model_revision=SCRIPTED_MACRO_REVISION,
            adapter_version=HIMA_ADAPTER_VERSION,
            parser_version=HIMA_PARSER_VERSIONS[self.race],
            vocabulary_version=HIMA_VOCABULARY_VERSIONS[self.race],
        )

    async def propose(
        self,
        context: HIMAInputContext,
        *,
        request_id: str | None = None,
    ) -> HIMALiveProposalResponse:
        if self._closed:
            raise RuntimeError("scripted macro client is closed")
        observed = {
            _canonical_evidence(value)
            for value in (
                *context.previous_actions,
                *(unit.unit_type for unit in context.observation.state.own_units),
                *(unit.unit_type for unit in context.observation.state.own_structures),
                *context.observation.state.upgrades,
            )
        }
        while self._completed_prefix < len(self._actions):
            expected = _canonical_evidence(self._actions[self._completed_prefix])
            if expected not in observed:
                break
            self._completed_prefix += 1
        remaining = self._actions[self._completed_prefix :]
        proposal = (
            self._parse_actions(remaining, self._objective) if remaining else self._empty_proposal
        )
        snapshot = self._adapter.adapt_context(context)
        observation = context.observation
        return HIMALiveProposalResponse(
            request_id=request_id or "scripted-request",
            run_id=observation.run_id,
            episode_id=observation.episode_id,
            step_id=observation.step_id,
            game_loop=observation.game_loop,
            projection_hash=snapshot.projection_hash,
            proposal=proposal,
        )

    async def close(self) -> None:
        self._closed = True


def _canonical_evidence(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())
