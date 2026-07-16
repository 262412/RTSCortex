"""Shared Worker-side gameplay effect result types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class EffectVerdict:
    """One final gameplay-effect verdict for a deferred command."""

    command_id: str
    success: bool
    failure_reason: Optional[str] = None
    status: str = "failed"
    failure_code: Optional[str] = None
    evidence: Optional[dict[str, Any]] = None

