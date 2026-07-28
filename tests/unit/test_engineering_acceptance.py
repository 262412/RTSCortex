from __future__ import annotations

from pathlib import Path

from rtscortex.evaluation.engineering import (
    REQUIRED_ENGINEERING_GATES,
    build_engineering_gate_report,
)
from rtscortex.memory import StoredEvent


def _event(
    event_id: int,
    event_type: str,
    payload: dict[str, object],
) -> StoredEvent:
    return StoredEvent(
        event_id=event_id,
        run_id="run",
        episode_id="episode",
        step_id=event_id,
        event_type=event_type,
        created_at=f"2026-07-28T00:00:{event_id:02d}+00:00",
        payload=payload,
    )


def test_missing_required_engineering_evidence_fails_closed(tmp_path: Path) -> None:
    report = build_engineering_gate_report(
        [],
        run_dir=tmp_path,
        natural_run_baseline_bytes_per_loop=None,
    )

    assert report["accepted"] is False
    assert set(report["gates"]) == set(REQUIRED_ENGINEERING_GATES)
    assert "effective_loops_per_second" in report["missing_required_metrics"]
    assert "natural_run_disk_reduction_ratio" in report["missing_required_metrics"]


def test_build_gates_use_effect_evidence_not_api_acceptance(tmp_path: Path) -> None:
    events = [
        _event(
            1,
            "execution",
            {
                "protocol_version": "1.1",
                "run_id": "run",
                "episode_id": "episode",
                "step_id": 1,
                "command_id": "build-1",
                "success": False,
                "action_name": "Build_Pylon_Screen",
                "actor": "Builder/Empty",
                "status": "failed",
                "execution_stage": "effect_verification",
                "failure_code": "no_build_start_evidence",
                "primitive_trace": [
                    {
                        "function": "Build_Pylon_pt",
                        "origin": "translator",
                        "ordinal": 0,
                        "total": 1,
                        "accepted": True,
                    }
                ],
                "effect_evidence": {
                    "effect_kind": "build",
                    "target_position": [30.0, 40.0],
                    "validated_target_position": [30.0, 40.0],
                    "emitted_target_position": [30.0, 40.0],
                    "builder_tag": "0x1",
                    "build_started": False,
                },
            },
        )
    ]

    report = build_engineering_gate_report(
        events,
        run_dir=tmp_path,
        natural_run_baseline_bytes_per_loop=100.0,
    )

    assert report["metrics"]["build_start_coverage"] == 0.0
    assert report["metrics"]["build_confirmation_rate"] == 0.0
    assert report["metrics"]["build_failure_rate"] == 1.0
    assert report["gates"]["build_start_coverage"]["passed"] is False
    assert report["gates"]["build_confirmation_rate"]["passed"] is False
    assert report["gates"]["build_failure_rate"]["passed"] is False


def test_performance_and_disk_metrics_are_thresholded(tmp_path: Path) -> None:
    (tmp_path / "events.jsonl").write_bytes(b"x" * 400)
    events = [
        _event(1, "observation", {"game_loop": 1, "state": {}}),
        _event(
            2,
            "event_store_performance",
            {
                "subscriber_callbacks_under_durable_lock": 0,
            },
        ),
        _event(3, "observation", {"game_loop": 2, "state": {}}),
    ]

    report = build_engineering_gate_report(
        events,
        run_dir=tmp_path,
        natural_run_baseline_bytes_per_loop=400.0,
    )

    assert report["metrics"]["effective_loops_per_second"] == 1.0
    assert report["metrics"]["natural_run_disk_reduction_ratio"] == 2.0
    assert report["gates"]["effective_loops_per_second"]["passed"] is False
    assert report["gates"]["natural_run_disk_reduction_ratio"]["passed"] is False
