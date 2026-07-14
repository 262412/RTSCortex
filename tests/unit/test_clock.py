from __future__ import annotations

import math

import pytest
from rtscortex_llm_pysc2.clock import (
    FixedRateGameClock,
    InitialPlanningBarrier,
    InitialPlanningBarrierState,
)


class FakeTime:
    def __init__(self, *, now: float = 0.0, oversleep: float = 0.0) -> None:
        self.now = now
        self.oversleep = oversleep
        self.sleep_calls: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleep_calls.append(seconds)
        self.now += seconds + self.oversleep

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_fixed_rate_clock_converts_speed_multiplier_to_single_loop_rate() -> None:
    fake_time = FakeTime(now=10.0)
    clock = FixedRateGameClock(
        speed_multiplier=0.25,
        monotonic=fake_time.monotonic,
        sleeper=fake_time.sleep,
    )

    timing = clock.wait_for_step()

    assert clock.step_mul == 1
    assert clock.loops_per_second == pytest.approx(5.6)
    assert clock.seconds_per_step == pytest.approx(1 / 5.6)
    assert fake_time.sleep_calls == [pytest.approx(1 / 5.6)]
    assert timing.deadline == pytest.approx(10.0 + 1 / 5.6)
    assert timing.slept_seconds == pytest.approx(1 / 5.6)
    assert timing.late_by_seconds == 0.0
    assert timing.skipped_deadlines == 0


def test_fixed_rate_clock_uses_absolute_deadlines_without_cumulative_drift() -> None:
    fake_time = FakeTime(oversleep=0.01)
    clock = FixedRateGameClock(
        speed_multiplier=1.0,
        monotonic=fake_time.monotonic,
        sleeper=fake_time.sleep,
    )
    period = 1 / 22.4

    first = clock.wait_for_step()
    second = clock.wait_for_step()

    assert first.deadline == pytest.approx(period)
    assert second.deadline == pytest.approx(2 * period)
    assert fake_time.sleep_calls == [
        pytest.approx(period),
        pytest.approx(period - 0.01),
    ]
    assert clock.next_deadline == pytest.approx(3 * period)


def test_fixed_rate_clock_skips_missed_deadlines_instead_of_catching_up() -> None:
    fake_time = FakeTime()
    clock = FixedRateGameClock(
        speed_multiplier=1.0,
        monotonic=fake_time.monotonic,
        sleeper=fake_time.sleep,
    )
    period = clock.seconds_per_step

    clock.wait_for_step()
    fake_time.advance(10 * period)
    late_timing = clock.wait_for_step()

    assert late_timing.slept_seconds == 0.0
    assert late_timing.late_by_seconds == pytest.approx(9 * period)
    assert late_timing.skipped_deadlines == 9
    assert clock.next_deadline == pytest.approx(12 * period)

    clock.wait_for_step()

    assert fake_time.sleep_calls[-1] == pytest.approx(period)


def test_fixed_rate_clock_reset_reanchors_the_monotonic_schedule() -> None:
    fake_time = FakeTime()
    clock = FixedRateGameClock(
        speed_multiplier=0.5,
        monotonic=fake_time.monotonic,
        sleeper=fake_time.sleep,
    )

    clock.wait_for_step()
    clock.reset()
    fake_time.advance(3.0)
    timing = clock.wait_for_step()

    assert timing.deadline == pytest.approx(3.0 + 2 / 22.4 + 2 / 22.4)


@pytest.mark.parametrize("speed_multiplier", [0.0, -0.1, math.nan, math.inf])
def test_fixed_rate_clock_rejects_invalid_speed_multiplier(speed_multiplier: float) -> None:
    with pytest.raises(ValueError, match="speed_multiplier"):
        FixedRateGameClock(speed_multiplier=speed_multiplier)


def test_initial_planning_barrier_blocks_until_released_and_can_reset() -> None:
    barrier = InitialPlanningBarrier()

    assert barrier.state.value == InitialPlanningBarrierState.WAITING.value
    assert barrier.blocks_steps is True

    barrier.release()
    barrier.release()

    assert barrier.state.value == InitialPlanningBarrierState.RELEASED.value
    assert barrier.blocks_steps is False

    barrier.reset()

    assert barrier.state.value == InitialPlanningBarrierState.WAITING.value
    assert barrier.blocks_steps is True
