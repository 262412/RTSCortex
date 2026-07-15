from __future__ import annotations

import asyncio
from pathlib import Path

import httpx

from rtscortex.api import create_app
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
