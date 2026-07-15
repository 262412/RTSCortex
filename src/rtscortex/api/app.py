"""FastAPI transport for environment workers."""

from fastapi import FastAPI, HTTPException

from rtscortex import __version__
from rtscortex.contracts import (
    CURRENT_PROTOCOL_VERSION,
    ActionBatch,
    EpisodeResult,
    ExecutionReport,
    ObservationEnvelope,
)
from rtscortex.runtime import RuntimeEngine


def create_app(engine: RuntimeEngine) -> FastAPI:
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
