"""Shared timing contract for RAW build effects and placement ownership."""

from __future__ import annotations

DEFAULT_ACTION_EFFECT_TIMEOUT_GAME_LOOPS = 112
ACTIVE_BUILD_ORDER_TIMEOUT_MULTIPLIER = 4
NEXUS_ACTIVE_BUILD_ORDER_TIMEOUT_MULTIPLIER = 12
POST_ORDER_EFFECT_GRACE_GAME_LOOPS = 32


def build_effect_max_lifetime(base_timeout_game_loops: int, action_name: str) -> int:
    """Return the longest verifier lifetime for one accepted build command."""

    multiplier = (
        NEXUS_ACTIVE_BUILD_ORDER_TIMEOUT_MULTIPLIER
        if action_name
        in {
            "Build_Nexus_Near",
            "Build_CommandCenter_Near",
            "Build_Hatchery_Near",
        }
        else ACTIVE_BUILD_ORDER_TIMEOUT_MULTIPLIER
    )
    return int(base_timeout_game_loops) * multiplier


def build_reservation_expiry(
    game_loop: int,
    action_name: str,
    *,
    base_timeout_game_loops: int = DEFAULT_ACTION_EFFECT_TIMEOUT_GAME_LOOPS,
) -> int:
    """Keep placement ownership through hard timeout and post-order evidence grace."""

    return (
        int(game_loop)
        + build_effect_max_lifetime(base_timeout_game_loops, action_name)
        + POST_ORDER_EFFECT_GRACE_GAME_LOOPS
    )
