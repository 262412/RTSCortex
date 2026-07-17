from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from rtscortex.policy.hima import (
    HIMAInputContext,
    HIMALiveBusyError,
    HIMALivePolicyClient,
    HIMALivePolicyService,
    HIMALiveProposalRequest,
    HIMALiveProtocolError,
    HIMALiveTimeoutError,
    HIMAObservationAdapter,
    create_hima_live_app,
)
from rtscortex.policy.models import PolicyObservationFixture
from rtscortex.policy.subagents import HIMA_PROTOSS_SPECS
from tests.helpers import make_observation


class FakePersistentGenerator:
    def __init__(self, response: str = "Actions: ['Probe', 'Pylon']") -> None:
        self.response = response
        self.load_calls = 0
        self.user_messages: list[str] = []
        self.active = 0
        self.max_active = 0

    async def load(self) -> None:
        self.load_calls += 1

    async def generate(self, *, user_message: str) -> str:
        self.user_messages.append(user_message)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0)
        self.active -= 1
        return self.response


class BlockingPersistentGenerator(FakePersistentGenerator):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def generate(self, *, user_message: str) -> str:
        self.user_messages.append(user_message)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.started.set()
        try:
            await self.release.wait()
            return self.response
        finally:
            self.active -= 1


def _context() -> HIMAInputContext:
    observation = make_observation(game_loop=224).model_copy(
        update={"text_observation": "SECRET ENEMY STATE"}
    )
    return HIMAInputContext(
        observation=observation,
        previous_actions=("Probe",),
    )


def _request(request_id: str) -> HIMALiveProposalRequest:
    context = _context()
    return HIMALiveProposalRequest(
        request_id=request_id,
        run_id=context.observation.run_id,
        episode_id=context.observation.episode_id,
        step_id=context.observation.step_id,
        game_loop=context.observation.game_loop,
        snapshot=HIMAObservationAdapter().adapt_context(context),
    )


def test_live_context_and_shadow_fixture_use_the_same_projection() -> None:
    context = _context()
    fixture = PolicyObservationFixture(
        fixture_id="same-input",
        observation=context.observation,
        previous_actions=list(context.previous_actions),
    )
    adapter = HIMAObservationAdapter()

    live_snapshot, live_message = adapter.prepare_context(context)
    shadow_snapshot, shadow_message = adapter.prepare(fixture)

    assert live_snapshot == shadow_snapshot
    assert live_message == shadow_message
    assert "SECRET" not in live_message


def test_live_service_loads_once_and_rejects_overlapping_generation() -> None:
    generator = BlockingPersistentGenerator()
    service = HIMALivePolicyService(HIMA_PROTOSS_SPECS[0], generator)

    async def execute() -> None:
        await service.start()
        await service.start()
        first = asyncio.create_task(service.propose(_request("first")))
        await generator.started.wait()
        with pytest.raises(HIMALiveBusyError, match="already in progress"):
            await service.propose(_request("second"))
        generator.release.set()
        await first

    asyncio.run(execute())

    assert generator.load_calls == 1
    assert generator.max_active == 1
    assert len(generator.user_messages) == 1
    payload = json.loads(generator.user_messages[0])
    assert list(payload) == [
        "supply_used",
        "supply_capacity",
        "unit",
        "research",
        "previous_action",
    ]
    assert payload["previous_action"] == ["Probe"]


def test_cancelled_caller_keeps_inference_slot_until_generation_finishes() -> None:
    generator = BlockingPersistentGenerator()
    service = HIMALivePolicyService(HIMA_PROTOSS_SPECS[0], generator)

    async def execute() -> None:
        await service.start()
        caller = asyncio.create_task(service.propose(_request("cancelled")))
        await generator.started.wait()
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller

        with pytest.raises(HIMALiveBusyError, match="already in progress"):
            await service.propose(_request("while-thread-still-running"))

        generator.release.set()
        while generator.active:
            await asyncio.sleep(0)
        result = await service.propose(_request("after-finish"))
        assert result.request_id == "after-finish"

    asyncio.run(execute())

    assert generator.max_active == 1
    assert len(generator.user_messages) == 2


def test_live_app_and_client_round_trip_without_control_routes() -> None:
    generator = FakePersistentGenerator()
    service = HIMALivePolicyService(HIMA_PROTOSS_SPECS[0], generator)
    app = create_hima_live_app(service)

    async def execute() -> None:
        await service.start()
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://hima.test",
        ) as http_client:
            client = HIMALivePolicyClient(
                http_client,
                expected_model_id="SNUMPR/Protoss-a",
            )
            health = await client.health()
            result = await client.propose(_context(), request_id="live-1")
            control = await http_client.post("/v1/tick", json={})

        assert health.status == "ready"
        assert health.model_id == "SNUMPR/Protoss-a"
        assert result.request_id == "live-1"
        assert result.run_id == _context().observation.run_id
        assert result.game_loop == 224
        assert result.proposal.steps[1].canonical_action == "BUILD PYLON"
        assert control.status_code == 404

    asyncio.run(execute())


def test_live_app_is_unavailable_before_checkpoint_start() -> None:
    app = create_hima_live_app(
        HIMALivePolicyService(HIMA_PROTOSS_SPECS[0], FakePersistentGenerator())
    )

    async def execute() -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://hima.test",
        ) as client:
            health = await client.get("/healthz")
            proposal = await client.post(
                "/internal/policy/v1/propose",
                json={
                    "protocol_version": "1.0",
                    "request_id": "early",
                    "run_id": _context().observation.run_id,
                    "episode_id": _context().observation.episode_id,
                    "step_id": _context().observation.step_id,
                    "game_loop": _context().observation.game_loop,
                    "snapshot": HIMAObservationAdapter()
                    .adapt_context(_context())
                    .model_dump(mode="json"),
                },
            )

        assert health.status_code == 503
        assert proposal.status_code == 503

    asyncio.run(execute())


def test_live_client_rejects_wrong_checkpoint_identity() -> None:
    service = HIMALivePolicyService(HIMA_PROTOSS_SPECS[0], FakePersistentGenerator())
    app = create_hima_live_app(service)

    async def execute() -> None:
        await service.start()
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://hima.test",
        ) as http_client:
            client = HIMALivePolicyClient(
                http_client,
                expected_model_id="SNUMPR/Protoss-b",
            )
            with pytest.raises(HIMALiveProtocolError, match="identity"):
                await client.health()

    asyncio.run(execute())


def test_live_client_rejects_stale_adapter_parser_or_vocabulary() -> None:
    app = FastAPI()

    @app.get("/healthz")
    async def stale_health() -> dict[str, str]:
        return {
            "status": "ready",
            "protocol_version": "1.0",
            "model_id": "SNUMPR/Protoss-a",
            "model_revision": "revision",
            "adapter_version": "stale-adapter",
            "parser_version": "stale-parser",
            "vocabulary_version": "stale-vocabulary",
        }

    async def execute() -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://hima.test",
        ) as http_client:
            client = HIMALivePolicyClient(http_client)
            with pytest.raises(HIMALiveProtocolError, match="versions"):
                await client.health()

    asyncio.run(execute())


def test_live_client_rejects_uncorrelated_response() -> None:
    context = _context()
    snapshot = HIMAObservationAdapter().adapt_context(context)
    proposal = {
        "strategic_objective": "opening",
        "steps": [],
        "vocabulary_version": "test",
        "parser_version": "test",
    }
    app = FastAPI()

    @app.post("/internal/policy/v1/propose")
    async def mismatched_response() -> dict[str, Any]:
        return {
            "protocol_version": "1.0",
            "request_id": "another-request",
            "run_id": context.observation.run_id,
            "episode_id": context.observation.episode_id,
            "step_id": context.observation.step_id,
            "game_loop": context.observation.game_loop,
            "projection_hash": snapshot.projection_hash,
            "proposal": proposal,
        }

    async def execute() -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://hima.test",
        ) as http_client:
            client = HIMALivePolicyClient(http_client)
            with pytest.raises(HIMALiveProtocolError, match="does not match"):
                await client.propose(context, request_id="expected-request")

    asyncio.run(execute())


def test_live_client_wraps_invalid_payloads_as_protocol_errors() -> None:
    async def invalid_payload(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not-json", request=request)

    async def execute() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(invalid_payload),
            base_url="http://hima.test",
        ) as http_client:
            client = HIMALivePolicyClient(http_client)
            with pytest.raises(HIMALiveProtocolError, match="health response"):
                await client.health()
            with pytest.raises(HIMALiveProtocolError, match="proposal response"):
                await client.propose(_context())

    asyncio.run(execute())


def test_live_client_classifies_httpx_timeouts() -> None:
    async def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("deadline exceeded", request=request)

    async def execute() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(timeout),
            base_url="http://hima.test",
        ) as http_client:
            client = HIMALivePolicyClient(http_client)
            with pytest.raises(HIMALiveTimeoutError, match="health"):
                await client.health()
            with pytest.raises(HIMALiveTimeoutError, match="proposal"):
                await client.propose(_context())

    asyncio.run(execute())


def test_live_app_returns_conflict_while_inference_is_running() -> None:
    generator = BlockingPersistentGenerator()
    service = HIMALivePolicyService(HIMA_PROTOSS_SPECS[0], generator)
    app = create_hima_live_app(service)

    async def execute() -> None:
        await service.start()
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://hima.test",
        ) as client:
            first = asyncio.create_task(
                client.post(
                    "/internal/policy/v1/propose",
                    content=_request("first-http").model_dump_json(),
                    headers={"content-type": "application/json"},
                )
            )
            await generator.started.wait()
            conflict = await client.post(
                "/internal/policy/v1/propose",
                content=_request("second-http").model_dump_json(),
                headers={"content-type": "application/json"},
            )
            generator.release.set()
            completed = await first

        assert conflict.status_code == 409
        assert completed.status_code == 200

    asyncio.run(execute())


def test_uds_client_requires_absolute_path_and_positive_timeout() -> None:
    with pytest.raises(ValueError, match="absolute"):
        HIMALivePolicyClient.for_unix_socket("relative.sock")
    with pytest.raises(ValueError, match="positive"):
        HIMALivePolicyClient.for_unix_socket("/tmp/hima.sock", timeout_seconds=0)
