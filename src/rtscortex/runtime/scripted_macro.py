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
        raw_output = f"Reason: {objective} Actions: {json.dumps(actions)}"
        proposal = self._parser.parse(raw_output)
        if proposal.diagnostics or len(proposal.steps) != len(actions):
            diagnostics = ", ".join(item.code for item in proposal.diagnostics)
            raise ValueError(
                "scripted macro actions must use the pinned race vocabulary"
                + (f": {diagnostics}" if diagnostics else "")
            )
        self._proposal = proposal.model_copy(update={"strategic_objective": objective})
        self._closed = False

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
        snapshot = self._adapter.adapt_context(context)
        observation = context.observation
        return HIMALiveProposalResponse(
            request_id=request_id or "scripted-request",
            run_id=observation.run_id,
            episode_id=observation.episode_id,
            step_id=observation.step_id,
            game_loop=observation.game_loop,
            projection_hash=snapshot.projection_hash,
            proposal=self._proposal,
        )

    async def close(self) -> None:
        self._closed = True
