"""Typed, model-independent data used by the HIMA policy adapter."""

from __future__ import annotations

import json
from hashlib import sha256
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

HIMA_ADAPTER_VERSION: Literal["hima-github-json-v2"] = "hima-github-json-v2"
HIMA_PARSER_VERSION = "hima-protoss-parser-v2"
HIMA_VOCABULARY_VERSION = "hima-protoss-60-v2"
HIMA_UPSTREAM_REVISION = "6b2a4084f9d6c0d7739aedb860685cf9dfd90d35"


class HIMAModel(BaseModel):
    """Base model for immutable HIMA adapter data."""

    model_config = ConfigDict(extra="forbid", frozen=True)


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


class HIMAMacroAction(HIMAModel):
    """One member of the pinned Protoss macro-action vocabulary."""

    upstream_action_id: int = Field(ge=100, le=325)
    upstream_name: str = Field(min_length=1)
    canonical_action: str
    category: Literal["train", "build", "research"]
    aliases: tuple[str, ...] = ()

    @property
    def upstream_index(self) -> int:
        """Compatibility alias; HIMA uses sparse action IDs, not list indices."""

        return self.upstream_action_id
