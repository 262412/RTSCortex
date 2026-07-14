"""Deterministic wall-clock pacing for manually stepped SC2 games."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Optional

SC2_GAME_LOOPS_PER_SECOND = 22.4
SC2_STEP_MUL = 1


@dataclass(frozen=True)
class StepTiming:
    """Timing information for one permitted SC2 game-loop step."""

    deadline: float
    slept_seconds: float
    late_by_seconds: float
    skipped_deadlines: int


class FixedRateGameClock:
    """Pace one-game-loop steps against absolute monotonic deadlines.

    Missed deadlines are skipped instead of replayed, so a delayed planner never
    triggers a burst of catch-up steps. Keeping deadlines on the original cadence
    also prevents ordinary sleep overhead from accumulating as drift.
    """

    def __init__(
        self,
        speed_multiplier: float = 1.0,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if not math.isfinite(speed_multiplier) or speed_multiplier <= 0:
            raise ValueError("speed_multiplier must be a finite positive number")
        self._speed_multiplier = speed_multiplier
        self._monotonic = monotonic
        self._sleeper = sleeper
        self._next_deadline: Optional[float] = None

    @property
    def step_mul(self) -> int:
        """Return the fixed SC2 step size; pacing never skips game loops."""

        return SC2_STEP_MUL

    @property
    def loops_per_second(self) -> float:
        """Return the target number of SC2 game loops per wall-clock second."""

        return SC2_GAME_LOOPS_PER_SECOND * self._speed_multiplier

    @property
    def seconds_per_step(self) -> float:
        """Return the wall-clock interval between one-loop SC2 steps."""

        return 1.0 / self.loops_per_second

    @property
    def next_deadline(self) -> Optional[float]:
        """Return the next scheduled monotonic deadline, if pacing has started."""

        return self._next_deadline

    def reset(self) -> None:
        """Make the next step establish a fresh monotonic schedule."""

        self._next_deadline = None

    def wait_for_step(self) -> StepTiming:
        """Wait until one SC2 loop may advance and schedule the following loop."""

        now = self._monotonic()
        deadline = self._next_deadline
        if deadline is None:
            deadline = now + self.seconds_per_step

        sleep_seconds = max(0.0, deadline - now)
        if sleep_seconds:
            self._sleeper(sleep_seconds)
        after_wait = self._monotonic()

        next_deadline = deadline + self.seconds_per_step
        skipped_deadlines = 0
        if next_deadline <= after_wait:
            skipped_deadlines = (
                math.floor((after_wait - next_deadline) / self.seconds_per_step) + 1
            )
            next_deadline += skipped_deadlines * self.seconds_per_step
        self._next_deadline = next_deadline

        return StepTiming(
            deadline=deadline,
            slept_seconds=sleep_seconds,
            late_by_seconds=max(0.0, after_wait - deadline),
            skipped_deadlines=skipped_deadlines,
        )


class InitialPlanningBarrierState(Enum):
    """Lifecycle states for the one-time initial planning barrier."""

    WAITING = "waiting"
    RELEASED = "released"


class InitialPlanningBarrier:
    """Track whether SC2 stepping must wait for the initial plan."""

    def __init__(self) -> None:
        self._state = InitialPlanningBarrierState.WAITING

    @property
    def state(self) -> InitialPlanningBarrierState:
        return self._state

    @property
    def blocks_steps(self) -> bool:
        return self._state is InitialPlanningBarrierState.WAITING

    def release(self) -> None:
        """Permanently allow steps for the current episode."""

        self._state = InitialPlanningBarrierState.RELEASED

    def reset(self) -> None:
        """Block steps for the next episode's initial plan."""

        self._state = InitialPlanningBarrierState.WAITING
