"""Strongly typed payloads exposed by the read-only console API."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

from rtscortex import __version__
from rtscortex.contracts import CURRENT_PROTOCOL_VERSION
from rtscortex.memory import StoredEvent

ConsoleRunStatus: TypeAlias = Literal["starting", "running", "completed", "failed", "historical"]
FrameKind: TypeAlias = Literal["screen", "minimap"]


class ConsoleModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ConsoleSession(ConsoleModel):
    run_id: str = Field(min_length=1)
    episode_id: str | None = None
    status: ConsoleRunStatus = "starting"
    scenario: str | None = None
    seed: int | None = None
    model: str | None = None
    stale_after_seconds: float = Field(default=2.0, gt=0.0)
    frontend_event_limit: int = Field(default=5_000, ge=100, le=100_000)
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    protocol_version: Literal["1.1"] = CURRENT_PROTOCOL_VERSION
    runtime_version: str = __version__


class FrameMetadata(ConsoleModel):
    kind: FrameKind
    run_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    step_id: int = Field(ge=0)
    game_loop: int = Field(ge=0)
    frame_sequence: int = Field(ge=0)
    captured_at: datetime
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    protocol_version: Literal["1.1"] = CURRENT_PROTOCOL_VERSION
    content_type: Literal["image/jpeg"] = "image/jpeg"


class ConsoleEvent(ConsoleModel):
    event_id: int
    run_id: str
    episode_id: str
    step_id: int
    event_type: str
    created_at: str
    payload: dict[str, Any]

    @classmethod
    def from_stored(cls, event: StoredEvent) -> ConsoleEvent:
        return cls.model_validate(event.__dict__)


class ConsoleHealth(ConsoleModel):
    status: Literal["ok"] = "ok"
    read_only: Literal[True] = True
    protocol_version: Literal["1.1"] = CURRENT_PROTOCOL_VERSION
    runtime_version: str = __version__


class ConsoleSessionSnapshot(ConsoleModel):
    session: ConsoleSession
    latest_event_id: int = Field(ge=0)
    frames: dict[FrameKind, FrameMetadata | None]


class ConsoleEventPage(ConsoleModel):
    events: list[ConsoleEvent]
    next_after_event_id: int = Field(ge=0)
    has_more: bool


class StoredEventMessage(ConsoleModel):
    type: Literal["stored_event"] = "stored_event"
    event: ConsoleEvent


class FrameAvailableMessage(ConsoleModel):
    type: Literal["frame_available"] = "frame_available"
    frame: FrameMetadata


class RunStatusMessage(ConsoleModel):
    type: Literal["run_status"] = "run_status"
    session: ConsoleSession


class HeartbeatMessage(ConsoleModel):
    type: Literal["heartbeat"] = "heartbeat"
    sent_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    latest_event_id: int = Field(ge=0)


class ResyncRequiredMessage(ConsoleModel):
    type: Literal["resync_required"] = "resync_required"
    reason: Literal["subscriber_overflow", "backlog_exceeded"]
    after_event_id: int = Field(ge=0)


ConsoleMessage: TypeAlias = (
    StoredEventMessage
    | FrameAvailableMessage
    | RunStatusMessage
    | HeartbeatMessage
    | ResyncRequiredMessage
)
