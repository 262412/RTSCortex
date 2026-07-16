"""Typed, UDS-compatible transport for a persistent local HIMA policy."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Self
from uuid import uuid4

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import Field, ValidationError

from rtscortex.policy.hima.models import (
    HIMA_ADAPTER_VERSION,
    HIMA_LIVE_PROTOCOL_VERSION,
    HIMA_PARSER_VERSION,
    HIMA_VOCABULARY_VERSION,
    HIMAInputContext,
    HIMALiveHealth,
    HIMALiveProposalRequest,
    HIMAModel,
)
from rtscortex.policy.hima.observation import HIMAObservationAdapter
from rtscortex.policy.hima.parser import HIMAProposalParser
from rtscortex.policy.hima.subagent import (
    HIMA_PINNED_REVISIONS,
    HIMAPersistentTextGenerator,
    HIMAPolicySubagent,
)
from rtscortex.policy.models import MacroPolicyProposal, PolicySubagentSpec

__all__ = [
    "HIMALiveHealth",
    "HIMALiveBusyError",
    "HIMALivePolicyClient",
    "HIMALivePolicyService",
    "HIMALiveProposalRequest",
    "HIMALiveProposalResponse",
    "HIMALiveProtocolError",
    "HIMALiveTimeoutError",
    "create_hima_live_app",
]


class HIMALiveProposalResponse(HIMAModel):
    """One correlated proposal returned by the persistent HIMA process."""

    protocol_version: Literal["1.0"] = HIMA_LIVE_PROTOCOL_VERSION
    request_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    step_id: int = Field(ge=0)
    game_loop: int = Field(ge=0)
    projection_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    proposal: MacroPolicyProposal


class HIMALiveProtocolError(RuntimeError):
    """Raised when a HIMA sidecar response cannot match its request."""


class HIMALiveBusyError(RuntimeError):
    """Raised when the single HIMA inference slot is already occupied."""


class HIMALiveTimeoutError(TimeoutError):
    """Raised when transport I/O to the HIMA sidecar exceeds its deadline."""


class HIMALivePolicyService:
    """Own one loaded HIMA generator and serialize all model invocations."""

    def __init__(
        self,
        spec: PolicySubagentSpec,
        generator: HIMAPersistentTextGenerator,
        *,
        adapter: HIMAObservationAdapter | None = None,
        parser: HIMAProposalParser | None = None,
    ) -> None:
        self._generator = generator
        self._adapter = adapter or HIMAObservationAdapter()
        self._subagent = HIMAPolicySubagent(
            spec,
            generator,
            self._adapter,
            parser or HIMAProposalParser(),
        )
        self._started = False
        self._start_lock = asyncio.Lock()
        self._proposal_state_lock = asyncio.Lock()
        self._proposal_task: asyncio.Task[MacroPolicyProposal] | None = None

    @property
    def model_id(self) -> str:
        return self._subagent.spec.model_id

    @property
    def model_revision(self) -> str:
        return self._subagent.model_revision

    async def start(self) -> None:
        """Load the local checkpoint exactly once before reporting readiness."""

        async with self._start_lock:
            if self._started:
                return
            await self._generator.load()
            self._started = True

    def health(self) -> HIMALiveHealth:
        """Return readiness only after the local checkpoint has loaded."""

        if not self._started:
            raise RuntimeError("HIMA policy service has not loaded its checkpoint")
        return HIMALiveHealth(
            model_id=self.model_id,
            model_revision=self.model_revision,
            adapter_version=HIMA_ADAPTER_VERSION,
            parser_version=HIMA_PARSER_VERSION,
            vocabulary_version=HIMA_VOCABULARY_VERSION,
        )

    async def propose(
        self,
        request: HIMALiveProposalRequest,
    ) -> HIMALiveProposalResponse:
        """Generate and parse one projected request without concurrent GPU calls."""

        if not self._started:
            raise RuntimeError("HIMA policy service has not loaded its checkpoint")
        async with self._proposal_state_lock:
            inflight = self._proposal_task
            if inflight is not None and not inflight.done():
                raise HIMALiveBusyError("HIMA policy inference is already in progress")
            task = asyncio.create_task(
                self._subagent.propose_snapshot(request.snapshot),
                name=f"hima-proposal-{request.request_id}",
            )
            self._proposal_task = task
            task.add_done_callback(self._proposal_finished)

        # A disconnected or timed-out HTTP caller must not cancel the underlying
        # to_thread-backed GPU inference. The task remains the occupied inference
        # slot until it actually finishes, so a second request is rejected.
        proposal = await asyncio.shield(task)
        return HIMALiveProposalResponse(
            request_id=request.request_id,
            run_id=request.run_id,
            episode_id=request.episode_id,
            step_id=request.step_id,
            game_loop=request.game_loop,
            projection_hash=request.snapshot.projection_hash,
            proposal=proposal,
        )

    def _proposal_finished(
        self,
        task: asyncio.Task[MacroPolicyProposal],
    ) -> None:
        if self._proposal_task is task:
            self._proposal_task = None
        if not task.cancelled():
            # Consume a detached task exception when its HTTP caller has already
            # gone away. Awaiters can still retrieve the same exception.
            task.exception()


def create_hima_live_app(service: HIMALivePolicyService) -> FastAPI:
    """Create the private policy app; callers decide which UDS runs it."""

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await service.start()
        yield

    app = FastAPI(
        title="RTSCortex HIMA Policy",
        version=HIMA_LIVE_PROTOCOL_VERSION,
        lifespan=lifespan,
    )

    @app.get("/healthz", response_model=HIMALiveHealth)
    async def healthz() -> HIMALiveHealth:
        try:
            return service.health()
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    @app.post(
        "/internal/policy/v1/propose",
        response_model=HIMALiveProposalResponse,
    )
    async def propose(
        request: HIMALiveProposalRequest,
    ) -> HIMALiveProposalResponse:
        try:
            return await service.propose(request)
        except HIMALiveBusyError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    return app


class HIMALivePolicyClient:
    """Validated async client for the private HIMA policy protocol."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        expected_model_id: str | None = None,
        owns_client: bool = False,
    ) -> None:
        if expected_model_id is not None and expected_model_id not in HIMA_PINNED_REVISIONS:
            raise ValueError(f"unrecognized pinned HIMA model: {expected_model_id}")
        self._client = client
        self._expected_model_id = expected_model_id
        self._owns_client = owns_client
        self._adapter = HIMAObservationAdapter()

    @classmethod
    def for_unix_socket(
        cls,
        socket_path: str | Path,
        *,
        timeout_seconds: float = 12.0,
        expected_model_id: str | None = None,
    ) -> Self:
        """Build a local-only client; no TCP transport is exposed here."""

        path = Path(socket_path).expanduser()
        if not path.is_absolute():
            raise ValueError("HIMA socket_path must be absolute")
        if timeout_seconds <= 0:
            raise ValueError("HIMA timeout_seconds must be positive")
        transport = httpx.AsyncHTTPTransport(uds=str(path))
        client = httpx.AsyncClient(
            transport=transport,
            base_url="http://hima.local",
            timeout=timeout_seconds,
        )
        return cls(
            client,
            expected_model_id=expected_model_id,
            owns_client=True,
        )

    async def health(self) -> HIMALiveHealth:
        try:
            response = await self._client.get("/healthz")
        except httpx.TimeoutException as error:
            raise HIMALiveTimeoutError("HIMA health request timed out") from error
        response.raise_for_status()
        try:
            health = HIMALiveHealth.model_validate_json(response.content)
        except ValidationError as error:
            raise HIMALiveProtocolError(
                "HIMA health response does not match the live protocol"
            ) from error
        if (
            health.adapter_version != HIMA_ADAPTER_VERSION
            or health.parser_version != HIMA_PARSER_VERSION
            or health.vocabulary_version != HIMA_VOCABULARY_VERSION
        ):
            raise HIMALiveProtocolError(
                "HIMA health adapter/parser/vocabulary versions do not match "
                "this Runtime"
            )
        if self._expected_model_id is not None:
            expected_revision = HIMA_PINNED_REVISIONS[self._expected_model_id]
            if (
                health.model_id != self._expected_model_id
                or health.model_revision != expected_revision
            ):
                raise HIMALiveProtocolError(
                    "HIMA health identity does not match the configured checkpoint"
                )
        return health

    async def propose(
        self,
        context: HIMAInputContext,
        *,
        request_id: str | None = None,
    ) -> HIMALiveProposalResponse:
        request = HIMALiveProposalRequest(
            request_id=request_id or uuid4().hex,
            run_id=context.observation.run_id,
            episode_id=context.observation.episode_id,
            step_id=context.observation.step_id,
            game_loop=context.observation.game_loop,
            snapshot=self._adapter.adapt_context(context),
        )
        try:
            response = await self._client.post(
                "/internal/policy/v1/propose",
                content=request.model_dump_json(),
                headers={"content-type": "application/json"},
            )
        except httpx.TimeoutException as error:
            raise HIMALiveTimeoutError("HIMA proposal request timed out") from error
        if response.status_code == 409:
            raise HIMALiveBusyError("HIMA policy inference is already in progress")
        response.raise_for_status()
        try:
            result = HIMALiveProposalResponse.model_validate_json(response.content)
        except ValidationError as error:
            raise HIMALiveProtocolError(
                "HIMA proposal response does not match the live protocol"
            ) from error
        self._validate_response(request, result)
        return result

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def _validate_response(
        self,
        request: HIMALiveProposalRequest,
        response: HIMALiveProposalResponse,
    ) -> None:
        if (
            response.request_id,
            response.run_id,
            response.episode_id,
            response.step_id,
            response.game_loop,
            response.projection_hash,
        ) != (
            request.request_id,
            request.run_id,
            request.episode_id,
            request.step_id,
            request.game_loop,
            request.snapshot.projection_hash,
        ):
            raise HIMALiveProtocolError("HIMA proposal response does not match its request")
