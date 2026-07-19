"""Typed, model-independent data used by the HIMA policy adapter."""

from __future__ import annotations

import json
from hashlib import sha256
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from rtscortex.contracts import ObservationEnvelope

HIMA_ADAPTER_VERSION: Literal["hima-github-json-v2"] = "hima-github-json-v2"
HIMA_PARSER_VERSION = "hima-protoss-parser-v5"
HIMA_VOCABULARY_VERSION = "hima-protoss-60-v2"
HIMA_UPSTREAM_REVISION = "6b2a4084f9d6c0d7739aedb860685cf9dfd90d35"
HIMA_LIVE_PROTOCOL_VERSION: Literal["1.0"] = "1.0"


class HIMAModel(BaseModel):
    """Base model for immutable HIMA adapter data."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class HIMAInputContext(HIMAModel):
    """Runtime-neutral input from which the exact HIMA payload is projected."""

    observation: ObservationEnvelope
    previous_actions: tuple[str, ...] = ()


class HIMAObservationSnapshot(HIMAModel):
    """The exact five-field JSON projection visible to the HIMA checkpoint."""

    adapter_version: Literal["hima-github-json-v2"] = HIMA_ADAPTER_VERSION
    supply_used: int = Field(ge=0)
    supply_capacity: int = Field(ge=0)
    unit: dict[str, int]
    research: tuple[str, ...] = ()
    previous_action: tuple[str, ...] = ()

    def upstream_payload(self) -> dict[str, object]:
        """Return fields in the order used by upstream ``generate_input``."""

        return {
            "supply_used": self.supply_used,
            "supply_capacity": self.supply_capacity,
            "unit": dict(self.unit),
            "research": list(self.research),
            "previous_action": list(self.previous_action),
        }

    @property
    def projection_hash(self) -> str:
        """Return a stable fingerprint of exactly the information shown to HIMA."""

        payload = json.dumps(
            self.upstream_payload(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return sha256(payload).hexdigest()


class HIMALiveProposalRequest(HIMAModel):
    """One correlated request sent to the isolated HIMA policy process."""

    protocol_version: Literal["1.0"] = HIMA_LIVE_PROTOCOL_VERSION
    request_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    step_id: int = Field(ge=0)
    game_loop: int = Field(ge=0)
    snapshot: HIMAObservationSnapshot


class HIMALiveHealth(HIMAModel):
    """Readiness and immutable model identity reported by the HIMA process."""

    status: Literal["ready"] = "ready"
    protocol_version: Literal["1.0"] = HIMA_LIVE_PROTOCOL_VERSION
    model_id: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    adapter_version: str = Field(min_length=1)
    parser_version: str = Field(min_length=1)
    vocabulary_version: str = Field(min_length=1)


class HIMAMacroAction(HIMAModel):
    """One member of a pinned race-specific HIMA macro-action vocabulary."""

    upstream_action_id: int = Field(ge=100, le=330)
    upstream_name: str = Field(min_length=1)
    canonical_action: str
    category: Literal["train", "build", "research"]
    aliases: tuple[str, ...] = ()

    @property
    def upstream_index(self) -> int:
        """Compatibility alias; HIMA uses sparse action IDs, not list indices."""

        return self.upstream_action_id
