from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rtscortex.console import ConsoleSession, FrameMetadata, LiveConsoleHub, create_console_app
from rtscortex.memory import EventStore


def _console(
    tmp_path: Path, *, queue_size: int = 256
) -> tuple[EventStore, LiveConsoleHub, TestClient]:
    store = EventStore(tmp_path / "events.sqlite3", tmp_path / "events.jsonl")
    hub = LiveConsoleHub(
        ConsoleSession(
            run_id="run",
            episode_id="episode",
            status="running",
            scenario="Simple64",
            seed=0,
            model="qwen3-8b",
        ),
        subscriber_queue_size=queue_size,
    )
    app = create_console_app(
        store,
        hub.session(),
        hub,
        frontend_event_limit=100,
        heartbeat_seconds=0.05,
    )
    return store, hub, TestClient(app)


def _frame(sequence: int, *, run_id: str = "run", episode_id: str = "episode") -> FrameMetadata:
    return FrameMetadata(
        kind="screen",
        run_id=run_id,
        episode_id=episode_id,
        step_id=sequence,
        game_loop=sequence * 8,
        frame_sequence=sequence,
        captured_at=datetime.now(UTC),
        width=256,
        height=256,
    )


def test_console_app_exposes_only_read_only_observability_routes(tmp_path: Path) -> None:
    store, _, client = _console(tmp_path)
    store.append_event(
        run_id="run",
        episode_id="episode",
        step_id=0,
        event_type="observation",
        payload={"game_loop": 8},
    )

    with client:
        health = client.get("/console/api/v1/health")
        session = client.get("/console/api/v1/session")
        events = client.get("/console/api/v1/events", params={"after_event_id": 0})
        control = client.post("/v1/tick", json={})

    assert health.json()["read_only"] is True
    assert health.json()["protocol_version"] == "1.1"
    assert session.json()["session"]["run_id"] == "run"
    assert session.json()["latest_event_id"] == 1
    assert events.json()["events"][0]["event_type"] == "observation"
    assert events.json()["next_after_event_id"] == 1
    assert control.status_code == 404
    store.close()


def test_console_app_can_serve_built_static_frontend(tmp_path: Path) -> None:
    store, hub, _ = _console(tmp_path)
    static_dir = tmp_path / "dist"
    static_dir.mkdir()
    (static_dir / "index.html").write_text("<main>RTSCortex</main>", encoding="utf-8")
    app = create_console_app(store, hub.session(), hub, static_dir)

    with TestClient(app) as client:
        page = client.get("/")
        health = client.get("/console/api/v1/health")

    assert page.text == "<main>RTSCortex</main>"
    assert health.status_code == 200
    store.close()


def test_console_hub_keeps_only_latest_valid_frame(tmp_path: Path) -> None:
    store, hub, client = _console(tmp_path)
    jpeg = b"\xff\xd8frame\xff\xd9"

    assert hub.put_frame(_frame(2), jpeg) is True
    assert hub.put_frame(_frame(1), jpeg) is False
    with pytest.raises(ValueError, match="run_id"):
        hub.put_frame(_frame(3, run_id="other"), jpeg)
    with pytest.raises(ValueError, match="episode_id"):
        hub.put_frame(_frame(3, episode_id="other"), jpeg)
    with pytest.raises(ValueError, match="JPEG"):
        hub.put_frame(_frame(3), b"not-an-image")

    with client:
        response = client.get("/console/api/v1/frames/screen")
        missing = client.get("/console/api/v1/frames/minimap")

    assert response.content == jpeg
    assert response.headers["etag"] == '"screen-2"'
    assert response.headers["x-frame-game-loop"] == "16"
    assert missing.status_code == 404
    store.close()


def test_console_websocket_backfills_and_streams_persisted_events(tmp_path: Path) -> None:
    store, _, client = _console(tmp_path)
    first = store.append_event(
        run_id="run",
        episode_id="episode",
        step_id=0,
        event_type="planner_started",
        payload={},
    )

    with (
        client,
        client.websocket_connect(
            "/console/api/v1/stream", params={"after_event_id": 0}
        ) as websocket,
    ):
        backfill = websocket.receive_json()
        second = store.append_event(
            run_id="run",
            episode_id="episode",
            step_id=1,
            event_type="decision",
            payload={"commands": []},
        )
        live = websocket.receive_json()

    assert backfill["type"] == "stored_event"
    assert backfill["event"]["event_id"] == first.event_id
    assert live["type"] == "stored_event"
    assert live["event"]["event_id"] == second.event_id
    store.close()


def test_console_hub_requests_resync_for_slow_subscriber(tmp_path: Path) -> None:
    store, hub, _ = _console(tmp_path, queue_size=1)
    first = store.append_event(
        run_id="run",
        episode_id="episode",
        step_id=0,
        event_type="one",
        payload={},
    )
    second = store.append_event(
        run_id="run",
        episode_id="episode",
        step_id=1,
        event_type="two",
        payload={},
    )

    async def receive_overflow() -> dict[str, object]:
        subscription = hub.subscribe()
        hub.publish_event(first)
        hub.publish_event(second)
        await asyncio.sleep(0)
        message = await subscription.receive()
        subscription.close()
        return message.model_dump(mode="json")

    message = asyncio.run(receive_overflow())

    assert message["type"] == "resync_required"
    assert message["after_event_id"] == 0
    store.close()
