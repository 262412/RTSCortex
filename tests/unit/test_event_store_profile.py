from __future__ import annotations

from scripts.profile_event_store import profile


def test_event_store_profile_reports_required_acceptance_metrics() -> None:
    result = profile(loops=20, events_per_loop=3)

    assert result["loops_per_second"] > 0
    assert result["events_per_loop"] == 3
    assert result["bytes_per_loop"] > 0
    assert result["queue_peak"] <= result["queue_capacity"]
    assert result["writer_lag_ms_max"] >= 0
