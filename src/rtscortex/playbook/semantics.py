"""Shared semantic classification for executable Playbook rules."""

from __future__ import annotations

from rtscortex.playbook.models import PlaybookRuleCategory, PlaybookRuleKind


def evaluation_kind(category: PlaybookRuleCategory) -> PlaybookRuleKind:
    """Return the single evaluation contract used across the Playbook stack."""

    if category in {
        PlaybookRuleCategory.ENGINE_INVARIANT,
        PlaybookRuleCategory.EXECUTION_GUARD,
        PlaybookRuleCategory.TACTICAL_RESPONSE,
    }:
        return PlaybookRuleKind.EXECUTION_GUARD
    return PlaybookRuleKind.STRATEGY
