"""Bounded, best-effort fan-out for live console events and frames."""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass

from rtscortex.console.models import (
    ConsoleEvent,
    ConsoleMessage,
    ConsoleRunStatus,
    ConsoleSession,
    FrameAvailableMessage,
    FrameKind,
    FrameMetadata,
    ResyncRequiredMessage,
    RunStatusMessage,
    StoredEventMessage,
)
from rtscortex.memory import StoredEvent


@dataclass(frozen=True)
class LatestFrame:
    metadata: FrameMetadata
    content: bytes


@dataclass
class _Subscriber:
    loop: asyncio.AbstractEventLoop
    queue: asyncio.Queue[ConsoleMessage]


class ConsoleSubscription:
    def __init__(self, hub: LiveConsoleHub, subscriber_id: int, subscriber: _Subscriber) -> None:
        self._hub = hub
        self._subscriber_id = subscriber_id
        self._subscriber = subscriber

    async def receive(self) -> ConsoleMessage:
        return await self._subscriber.queue.get()

    def close(self) -> None:
        self._hub.unsubscribe(self._subscriber_id)


class LiveConsoleHub:
    """Keep only current transient state and fan it out without blocking producers."""

    def __init__(self, session: ConsoleSession, *, subscriber_queue_size: int = 256) -> None:
        if subscriber_queue_size < 1:
            raise ValueError("subscriber_queue_size must be positive")
        self._lock = threading.RLock()
        self._session = session
        self._subscriber_queue_size = subscriber_queue_size
        self._subscribers: dict[int, _Subscriber] = {}
        self._next_subscriber_id = 0
        self._frames: dict[FrameKind, LatestFrame] = {}

    @property
    def run_id(self) -> str:
        return self._session.run_id

    def session(self) -> ConsoleSession:
        with self._lock:
            return self._session.model_copy(deep=True)

    def set_status(self, status: ConsoleRunStatus, *, episode_id: str | None = None) -> None:
        with self._lock:
            updated = self._session.model_copy(
                update={
                    "status": status,
                    "episode_id": (
                        episode_id if episode_id is not None else self._session.episode_id
                    ),
                }
            )
        validated = ConsoleSession.model_validate(updated.model_dump())
        with self._lock:
            self._session = validated
        self._broadcast(RunStatusMessage(session=validated))

    def publish_event(self, event: StoredEvent) -> None:
        if event.run_id != self.run_id:
            return
        self._broadcast(StoredEventMessage(event=ConsoleEvent.from_stored(event)))

    def put_frame(self, metadata: FrameMetadata, content: bytes) -> bool:
        """Store a newer JPEG frame and emit a small availability notification."""

        if metadata.run_id != self.run_id:
            raise ValueError("frame run_id does not match console session")
        session_episode_id = self.session().episode_id
        if session_episode_id is not None and metadata.episode_id != session_episode_id:
            raise ValueError("frame episode_id does not match console session")
        if not content or not content.startswith(b"\xff\xd8") or not content.endswith(b"\xff\xd9"):
            raise ValueError("frame content must be a JPEG image")
        with self._lock:
            current = self._frames.get(metadata.kind)
            if current is not None and metadata.frame_sequence <= current.metadata.frame_sequence:
                return False
            self._frames[metadata.kind] = LatestFrame(metadata=metadata, content=bytes(content))
        self._broadcast(FrameAvailableMessage(frame=metadata))
        return True

    def latest_frame(self, kind: FrameKind) -> LatestFrame | None:
        with self._lock:
            return self._frames.get(kind)

    def subscribe(self) -> ConsoleSubscription:
        loop = asyncio.get_running_loop()
        subscriber = _Subscriber(
            loop=loop,
            queue=asyncio.Queue(maxsize=self._subscriber_queue_size),
        )
        with self._lock:
            subscriber_id = self._next_subscriber_id
            self._next_subscriber_id += 1
            self._subscribers[subscriber_id] = subscriber
        return ConsoleSubscription(self, subscriber_id, subscriber)

    def unsubscribe(self, subscriber_id: int) -> None:
        with self._lock:
            self._subscribers.pop(subscriber_id, None)

    def _broadcast(self, message: ConsoleMessage) -> None:
        with self._lock:
            subscribers = tuple(self._subscribers.items())
        for subscriber_id, subscriber in subscribers:
            try:
                subscriber.loop.call_soon_threadsafe(self._offer, subscriber, message)
            except RuntimeError:
                self.unsubscribe(subscriber_id)

    @staticmethod
    def _offer(subscriber: _Subscriber, message: ConsoleMessage) -> None:
        if subscriber.queue.full():
            while not subscriber.queue.empty():
                subscriber.queue.get_nowait()
            subscriber.queue.put_nowait(
                ResyncRequiredMessage(
                    reason="subscriber_overflow",
                    after_event_id=0,
                )
            )
            return
        subscriber.queue.put_nowait(message)
