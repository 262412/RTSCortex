from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import httpx

from rtscortex.api import create_app
from rtscortex.console import ConsoleSession, LiveConsoleHub
from rtscortex.contracts import (
    ActionSource,
    EpisodeOutcome,
    EpisodeResult,
    ExecutionReport,
    ExecutionStage,
    ExecutionStatus,
)
from rtscortex.memory import EventStore
from rtscortex.providers import FakeProvider
from rtscortex.runtime import RuntimeEngine
from tests.helpers import make_config, make_observation


def test_versioned_api_health_and_tick(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    runtime = RuntimeEngine(
        config=config,
        store=EventStore(tmp_path / "events.sqlite3", tmp_path / "events.jsonl"),
        provider=FakeProvider(),
    )

    async def execute() -> None:
        transport = httpx.ASGITransport(app=create_app(runtime))
        try:
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://runtime.test",
            ) as client:
                health = await client.get("/healthz")
                assert health.json()["protocol_version"] == "1.1"
                response = await client.post(
                    "/v1/tick",
                    json=make_observation().model_dump(mode="json"),
                )
                assert response.status_code == 200
                assert response.json()["commands"][0]["name"] == "Attack_Unit"
                command_id = response.json()["commands"][0]["command_id"]
                execution = await client.post(
                    "/v1/execution",
                    json=ExecutionReport(
                        run_id="run-1",
                        episode_id="episode-1",
                        step_id=0,
                        command_id=command_id,
                        success=True,
                        action_name="Attack_Unit",
                        actor="army",
                        source=ActionSource.PLANNER,
                        requested_arguments=["0x1"],
                        resolved_arguments=["0x1"],
                        status=ExecutionStatus.SUCCEEDED,
                        execution_stage=ExecutionStage.PYSC2_ACCEPTANCE,
                    ).model_dump(mode="json"),
                )
                assert execution.json() == {"status": "recorded"}
                episode = await client.post(
                    "/v1/episode/end",
                    json=EpisodeResult(
                        run_id="run-1",
                        episode_id="episode-1",
                        scenario="mock",
                        seed=0,
                        outcome=EpisodeOutcome.VICTORY,
                        steps=1,
                    ).model_dump(mode="json"),
                )
                assert episode.json() == {"status": "recorded"}
        finally:
            await runtime.close()

    asyncio.run(execute())


def test_live_api_rejects_legacy_protocol(tmp_path: Path) -> None:
    runtime = RuntimeEngine(
        config=make_config(tmp_path),
        store=EventStore(tmp_path / "events.sqlite3", tmp_path / "events.jsonl"),
        provider=FakeProvider(),
    )

    async def execute() -> None:
        transport = httpx.ASGITransport(app=create_app(runtime))
        try:
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://runtime.test",
            ) as client:
                payload = make_observation().model_dump(mode="json")
                payload["protocol_version"] = "1.0"
                response = await client.post("/v1/tick", json=payload)
                assert response.status_code == 409
                assert "expected 1.1" in response.json()["detail"]
        finally:
            await runtime.close()

    asyncio.run(execute())


def test_runtime_api_accepts_console_frames_only_when_hub_is_enabled(tmp_path: Path) -> None:
    runtime = RuntimeEngine(
        config=make_config(tmp_path),
        store=EventStore(tmp_path / "events.sqlite3", tmp_path / "events.jsonl"),
        provider=FakeProvider(),
    )
    hub = LiveConsoleHub(ConsoleSession(run_id="run-1", episode_id="episode-1", status="running"))

    async def execute() -> None:
        headers = {
            "content-type": "image/jpeg",
            "x-rtscortex-protocol-version": "1.1",
            "x-rtscortex-run-id": "run-1",
            "x-rtscortex-episode-id": "episode-1",
            "x-rtscortex-step-id": "3",
            "x-rtscortex-game-loop": "24",
            "x-rtscortex-frame-sequence": "7",
            "x-rtscortex-captured-at": datetime.now(UTC).isoformat(),
            "x-rtscortex-width": "256",
            "x-rtscortex-height": "256",
        }
        jpeg = b"\xff\xd8screen\xff\xd9"
        try:
            enabled = httpx.ASGITransport(app=create_app(runtime, console_hub=hub))
            async with httpx.AsyncClient(
                transport=enabled,
                base_url="http://runtime.test",
            ) as client:
                response = await client.post(
                    "/internal/console/v1/frame/screen",
                    headers=headers,
                    content=jpeg,
                )
                assert response.status_code == 204
                frame = hub.latest_frame("screen")
                assert frame is not None
                assert frame.metadata.game_loop == 24
                assert frame.content == jpeg

            disabled = httpx.ASGITransport(app=create_app(runtime))
            async with httpx.AsyncClient(
                transport=disabled,
                base_url="http://runtime.test",
            ) as client:
                response = await client.post(
                    "/internal/console/v1/frame/screen",
                    headers=headers,
                    content=jpeg,
                )
                assert response.status_code == 404
        finally:
            await runtime.close()

    asyncio.run(execute())


def test_managed_api_lifespan_starts_and_closes_runtime(tmp_path: Path) -> None:
    class TrackingRuntime(RuntimeEngine):
        started = False
        closed = False

        async def start(self) -> None:
            self.started = True

        async def close(self) -> None:
            self.closed = True
            await super().close()

    runtime = TrackingRuntime(
        config=make_config(tmp_path),
        store=EventStore(tmp_path / "events.sqlite3", tmp_path / "events.jsonl"),
        provider=FakeProvider(),
    )
    app = create_app(runtime, manage_runtime_lifecycle=True)

    async def exercise() -> None:
        async with app.router.lifespan_context(app):
            assert runtime.started is True
            assert runtime.closed is False
        assert runtime.closed is True

    asyncio.run(exercise())
