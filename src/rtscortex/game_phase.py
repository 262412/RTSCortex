"""Shared game-phase vocabulary without importing the Cortex package."""

from enum import StrEnum


class GamePhase(StrEnum):
    EARLY = "early"
    TECHNOLOGY = "technology"
    PRODUCTION = "production"
    COMBAT = "combat"
