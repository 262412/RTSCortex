"""Best-effort, in-memory RGB frame publishing for the Live Console."""

from __future__ import annotations

import importlib
import queue
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from typing import Any, Optional, Protocol

import httpx


@dataclass(frozen=True)
class EncodedImage:
    body: bytes
    width: int
    height: int


@dataclass(frozen=True)
class FrameSample:
    step_id: int
    game_loop: int
    sequence: int
    captured_at: str
    screen: Optional[Any]
    minimap: Optional[Any]


class FrameUploader(Protocol):
    def upload(self, kind: str, sample: FrameSample, image: EncodedImage) -> None: ...

    def close(self) -> None: ...


FrameEncoder = Callable[[Any, int], EncodedImage]


class RuntimeFrameUploader:
    """Upload JPEG frames to the Runtime's private UDS-only endpoints."""

    def __init__(
        self,
        *,
        run_id: str,
        episode_id: str,
        base_url: str,
        unix_socket: Optional[str],
        timeout_seconds: float = 1.0,
        client: Optional[httpx.Client] = None,
    ) -> None:
        self._run_id = run_id
        self._episode_id = episode_id
        if client is None:
            transport = httpx.HTTPTransport(uds=unix_socket) if unix_socket else None
            client = httpx.Client(
                base_url=base_url,
                transport=transport,
                timeout=timeout_seconds,
            )
        self._client = client

    def upload(self, kind: str, sample: FrameSample, image: EncodedImage) -> None:
        response = self._client.post(
            f"/internal/console/v1/frame/{kind}",
            content=image.body,
            headers={
                "Content-Type": "image/jpeg",
                "X-RTSCortex-Protocol-Version": "1.1",
                "X-RTSCortex-Run-Id": self._run_id,
                "X-RTSCortex-Episode-Id": self._episode_id,
                "X-RTSCortex-Step-Id": str(sample.step_id),
                "X-RTSCortex-Game-Loop": str(sample.game_loop),
                "X-RTSCortex-Frame-Sequence": str(sample.sequence),
                "X-RTSCortex-Captured-At": sample.captured_at,
                "X-RTSCortex-Width": str(image.width),
                "X-RTSCortex-Height": str(image.height),
            },
        )
        response.raise_for_status()

    def close(self) -> None:
        self._client.close()


class RGBFramePublisher:
    """Sample frames by wall clock and publish only the newest pending sample."""

    def __init__(
        self,
        *,
        uploader: FrameUploader,
        frame_fps: float,
        jpeg_quality: int,
        encoder: Optional[FrameEncoder] = None,
        monotonic: Callable[[], float] = time.monotonic,
        captured_at: Optional[Callable[[], str]] = None,
    ) -> None:
        if frame_fps <= 0:
            raise ValueError("frame_fps must be positive")
        if not 1 <= jpeg_quality <= 95:
            raise ValueError("jpeg_quality must be between 1 and 95")
        self._uploader = uploader
        self._frame_interval_seconds = 1.0 / frame_fps
        self._jpeg_quality = jpeg_quality
        self._encoder = encoder or _encode_jpeg
        self._monotonic = monotonic
        self._captured_at = captured_at or _utc_now
        self._queue: queue.Queue[FrameSample] = queue.Queue(maxsize=1)
        self._state_lock = threading.Lock()
        self._stop = threading.Event()
        self._next_capture_at = 0.0
        self._sequence = 0
        self._closed = False
        self._thread = threading.Thread(
            target=self._run,
            name="rtscortex-rgb-publisher",
            daemon=True,
        )
        self._thread.start()

    def submit(self, observation: Any, *, step_id: int, game_loop: int) -> bool:
        """Queue the current RGB pair without waiting for encoding or network I/O."""

        try:
            screen = _observation_value(observation, "rgb_screen")
            minimap = _observation_value(observation, "rgb_minimap")
            if screen is None and minimap is None:
                return False
            now = self._monotonic()
            with self._state_lock:
                if self._closed or now < self._next_capture_at:
                    return False
                self._next_capture_at = now + self._frame_interval_seconds
                self._sequence += 1
                sequence = self._sequence
            sample = FrameSample(
                step_id=step_id,
                game_loop=game_loop,
                sequence=sequence,
                captured_at=self._captured_at(),
                screen=screen,
                minimap=minimap,
            )
            self._replace_pending(sample)
        except Exception:
            return False
        return True

    def close(self, timeout_seconds: float = 2.0) -> None:
        """Drain at most the newest pending sample and stop the uploader thread."""

        with self._state_lock:
            if self._closed:
                return
            self._closed = True
        self._stop.set()
        self._thread.join(timeout=max(0.0, timeout_seconds))

    def _replace_pending(self, sample: FrameSample) -> None:
        try:
            self._queue.put_nowait(sample)
            return
        except queue.Full:
            pass
        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._queue.put_nowait(sample)
        except queue.Full:
            # The worker won the race and another submitter filled the single slot.
            pass

    def _run(self) -> None:
        try:
            while not self._stop.is_set() or not self._queue.empty():
                try:
                    sample = self._queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                self._publish(sample)
        finally:
            self._uploader.close()

    def _publish(self, sample: FrameSample) -> None:
        for kind, pixels in (("screen", sample.screen), ("minimap", sample.minimap)):
            if pixels is None:
                continue
            try:
                image = self._encoder(pixels, self._jpeg_quality)
                self._uploader.upload(kind, sample, image)
            except Exception:
                # Live Console is observational. Its failures must never stop SC2.
                continue


def _encode_jpeg(pixels: Any, quality: int) -> EncodedImage:
    """Encode an RGB numpy-like array with Pillow without touching the filesystem."""

    image_module: Any = importlib.import_module("PIL.Image")
    # PySC2 4.10 exposes render planes as int32 even though every channel is 8-bit.
    # Pillow only accepts three-channel arrays after they are normalized to uint8.
    normalized = pixels.astype("uint8", copy=False)
    image = image_module.fromarray(normalized).convert("RGB")
    output = BytesIO()
    image.save(output, format="JPEG", quality=quality, optimize=False)
    width, height = image.size
    return EncodedImage(body=output.getvalue(), width=int(width), height=int(height))


def _observation_value(observation: Any, name: str) -> Optional[Any]:
    if isinstance(observation, Mapping):
        return observation.get(name)
    return getattr(observation, name, None)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
