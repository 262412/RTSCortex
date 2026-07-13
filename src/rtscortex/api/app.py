"""FastAPI transport for environment workers."""

from fastapi import FastAPI

from rtscortex import __version__
from rtscortex.contracts import ActionBatch, EpisodeResult, ExecutionReport, ObservationEnvelope
from rtscortex.runtime import RuntimeEngine


def create_app(engine: RuntimeEngine) -> FastAPI:
    app = FastAPI(title="RTSCortex Runtime", version=__version__)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok", "protocol_version": "1.0", "runtime_version": __version__}

    @app.post("/v1/tick", response_model=ActionBatch)
    async def tick(observation: ObservationEnvelope) -> ActionBatch:
        return await engine.tick(observation)

    @app.post("/v1/execution")
    async def execution(report: ExecutionReport) -> dict[str, str]:
        engine.record_execution(report)
        return {"status": "recorded"}

    @app.post("/v1/episode/end")
    async def end_episode(result: EpisodeResult) -> dict[str, str]:
        engine.end_episode(result)
        return {"status": "recorded"}

    return app
