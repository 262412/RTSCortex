"""Read-only HTTP and WebSocket application for the RTSCortex live console."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

from rtscortex import __version__
from rtscortex.console.hub import LiveConsoleHub
from rtscortex.console.models import (
    ConsoleEvent,
    ConsoleEventPage,
    ConsoleHealth,
    ConsoleSession,
    ConsoleSessionSnapshot,
    FrameKind,
    HeartbeatMessage,
    ResyncRequiredMessage,
    StoredEventMessage,
)
from rtscortex.memory import EventStore

FRAME_KINDS: tuple[FrameKind, ...] = ("screen", "minimap")


def create_console_app(
    store: EventStore,
    session: ConsoleSession,
    hub: LiveConsoleHub,
    static_dir: Path | None = None,
    *,
    frontend_event_limit: int = 5_000,
    heartbeat_seconds: float = 10.0,
) -> FastAPI:
    """Create an app that exposes observability data and no control routes."""

    if frontend_event_limit < 1:
        raise ValueError("frontend_event_limit must be positive")
    if heartbeat_seconds <= 0:
        raise ValueError("heartbeat_seconds must be positive")
    if session.run_id != hub.run_id:
        raise ValueError("console session and hub run_id must match")

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        unsubscribe = store.subscribe(hub.publish_event)
        try:
            yield
        finally:
            unsubscribe()

    app = FastAPI(
        title="RTSCortex Live Console",
        version=__version__,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )

    @app.get("/console/api/v1/health", response_model=ConsoleHealth)
    async def health() -> ConsoleHealth:
        return ConsoleHealth()

    @app.get("/console/api/v1/session", response_model=ConsoleSessionSnapshot)
    async def session_snapshot() -> ConsoleSessionSnapshot:
        current = hub.session()
        return ConsoleSessionSnapshot(
            session=current,
            latest_event_id=store.latest_event_id(current.run_id),
            frames={
                kind: None if (frame := hub.latest_frame(kind)) is None else frame.metadata
                for kind in FRAME_KINDS
            },
        )

    @app.get("/console/api/v1/events", response_model=ConsoleEventPage)
    async def events(
        after_event_id: int = Query(default=0, ge=0),
        limit: int = Query(default=500, ge=1),
    ) -> ConsoleEventPage:
        page_limit = min(limit, frontend_event_limit)
        stored = store.events_after(hub.run_id, after_event_id, page_limit + 1)
        has_more = len(stored) > page_limit
        visible = stored[:page_limit]
        next_id = visible[-1].event_id if visible else after_event_id
        return ConsoleEventPage(
            events=[ConsoleEvent.from_stored(event) for event in visible],
            next_after_event_id=next_id,
            has_more=has_more,
        )

    @app.get("/console/api/v1/frames/{kind}")
    async def latest_frame(kind: FrameKind) -> Response:
        frame = hub.latest_frame(kind)
        if frame is None:
            raise HTTPException(status_code=404, detail=f"{kind} frame is not available")
        metadata = frame.metadata
        return Response(
            content=frame.content,
            media_type=metadata.content_type,
            headers={
                "Cache-Control": "no-store",
                "ETag": f'"{kind}-{metadata.frame_sequence}"',
                "X-Frame-Sequence": str(metadata.frame_sequence),
                "X-Frame-Game-Loop": str(metadata.game_loop),
                "X-Frame-Captured-At": metadata.captured_at.isoformat(),
            },
        )

    @app.websocket("/console/api/v1/stream")
    async def stream(websocket: WebSocket, after_event_id: int = Query(default=0, ge=0)) -> None:
        await websocket.accept()
        subscription = hub.subscribe()
        try:
            backlog = store.events_after(
                hub.run_id,
                after_event_id,
                frontend_event_limit + 1,
            )
            if len(backlog) > frontend_event_limit:
                await websocket.send_json(
                    ResyncRequiredMessage(
                        reason="backlog_exceeded",
                        after_event_id=after_event_id,
                    ).model_dump(mode="json")
                )
            else:
                for event in backlog:
                    await websocket.send_json(
                        StoredEventMessage(event=ConsoleEvent.from_stored(event)).model_dump(
                            mode="json"
                        )
                    )
            while True:
                try:
                    message = await asyncio.wait_for(
                        subscription.receive(), timeout=heartbeat_seconds
                    )
                except TimeoutError:
                    message = HeartbeatMessage(latest_event_id=store.latest_event_id(hub.run_id))
                await websocket.send_json(message.model_dump(mode="json"))
        except (WebSocketDisconnect, asyncio.CancelledError, RuntimeError):
            pass
        finally:
            subscription.close()

    if static_dir is not None:
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="console-static")

    return app
