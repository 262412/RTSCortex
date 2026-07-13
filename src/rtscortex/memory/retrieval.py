"""Optional semantic-memory boundary reserved for future vector retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class MemoryHit:
    content: str
    source: str
    score: float


class MemoryRetriever(Protocol):
    async def search(self, query: str, *, limit: int = 5) -> list[MemoryHit]: ...


class DisabledMemoryRetriever:
    """Explicit no-op used until semantic retrieval is enabled by an experiment."""

    async def search(self, query: str, *, limit: int = 5) -> list[MemoryHit]:
        del query, limit
        return []
