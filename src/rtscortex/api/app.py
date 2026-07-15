"""FastAPI transport for environment workers."""

from typing import Annotated

from fastapi import FastAPI, Header, HTTPException, Request, Response
from pydantic import ValidationError

from rtscortex import __version__
from rtscortex.console import FrameMetadata, LiveConsoleHub
from rtscortex.console.models import FrameKind
from rtscortex.contracts import (
    CURRENT_PROTOCOL_VERSION,
    ActionBatch,
    EpisodeResult,
    ExecutionReport,
    ObservationEnvelope,
)
from rtscortex.runtime import RuntimeEngine


def create_app(
    engine: RuntimeEngine,
    *,
    console_hub: LiveConsoleHub | None = None,
) -> FastAPI:
    app = FastAPI(title="RTSCortex Runtime", version=__version__)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {
            "status": "ok",
            "protocol_version": CURRENT_PROTOCOL_VERSION,
            "runtime_version": __version__,
        }

    @app.post("/v1/tick", response_model=ActionBatch)
    async def tick(observation: ObservationEnvelope) -> ActionBatch:
        _require_current_protocol(observation.protocol_version)
        return await engine.tick(observation)

    @app.post("/v1/execution")
    async def execution(report: ExecutionReport) -> dict[str, str]:
        _require_current_protocol(report.protocol_version)
        engine.record_execution(report)
        return {"status": "recorded"}

    @app.post("/v1/episode/end")
    async def end_episode(result: EpisodeResult) -> dict[str, str]:
        _require_current_protocol(result.protocol_version)
        engine.end_episode(result)
        return {"status": "recorded"}

    if console_hub is not None:

        @app.post("/internal/console/v1/frame/{kind}", status_code=204)
        async def console_frame(
            kind: FrameKind,
            request: Request,
            protocol_version: Annotated[str, Header(alias="X-RTSCortex-Protocol-Version")],
            run_id: Annotated[str, Header(alias="X-RTSCortex-Run-Id")],
            episode_id: Annotated[str, Header(alias="X-RTSCortex-Episode-Id")],
            step_id: Annotated[int, Header(alias="X-RTSCortex-Step-Id")],
            game_loop: Annotated[int, Header(alias="X-RTSCortex-Game-Loop")],
            frame_sequence: Annotated[int, Header(alias="X-RTSCortex-Frame-Sequence")],
            captured_at: Annotated[str, Header(alias="X-RTSCortex-Captured-At")],
            width: Annotated[int, Header(alias="X-RTSCortex-Width")],
            height: Annotated[int, Header(alias="X-RTSCortex-Height")],
        ) -> Response:
            _require_current_protocol(protocol_version)
            if request.headers.get("content-type") != "image/jpeg":
                raise HTTPException(status_code=415, detail="console frames must be image/jpeg")
            content = await request.body()
            if len(content) > 1_000_000:
                raise HTTPException(status_code=413, detail="console frame exceeds 1 MB")
            try:
                metadata = FrameMetadata.model_validate(
                    {
                        "kind": kind,
                        "run_id": run_id,
                        "episode_id": episode_id,
                        "step_id": step_id,
                        "game_loop": game_loop,
                        "frame_sequence": frame_sequence,
                        "captured_at": captured_at,
                        "width": width,
                        "height": height,
                        "protocol_version": protocol_version,
                    }
                )
                console_hub.put_frame(metadata, content)
            except ValidationError as error:
                raise HTTPException(status_code=422, detail=error.errors()) from error
            except ValueError as error:
                raise HTTPException(status_code=409, detail=str(error)) from error
            return Response(status_code=204)

    return app


def _require_current_protocol(protocol_version: str) -> None:
    if protocol_version != CURRENT_PROTOCOL_VERSION:
        raise HTTPException(
            status_code=409,
            detail=(
                f"live protocol version {protocol_version} is not supported; "
                f"expected {CURRENT_PROTOCOL_VERSION}"
            ),
        )
