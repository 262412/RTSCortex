"""Focused Python 3.9 tests for the Live Console RGB publisher."""

from __future__ import annotations

import threading
import unittest
from types import SimpleNamespace
from typing import Any, Optional
from unittest import mock

import httpx
import numpy as np
from rtscortex_llm_pysc2 import entrypoint
from rtscortex_llm_pysc2.frame_stream import (
    EncodedImage,
    FrameSample,
    RGBFramePublisher,
    RuntimeFrameUploader,
    _encode_jpeg,
)
from rtscortex_llm_pysc2.worker import RTSCortexMainAgent, WorkerSettings


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


class _RecordingUploader:
    def __init__(self, *, block_first: bool = False) -> None:
        self.calls: list[tuple[str, FrameSample, EncodedImage]] = []
        self.started = threading.Event()
        self.release = threading.Event()
        self.closed = False
        self.block_first = block_first

    def upload(self, kind: str, sample: FrameSample, image: EncodedImage) -> None:
        self.calls.append((kind, sample, image))
        if self.block_first and len(self.calls) == 1:
            self.started.set()
            if not self.release.wait(timeout=2):
                raise TimeoutError("test did not release uploader")

    def close(self) -> None:
        self.closed = True


def _fake_encoder(pixels: Any, quality: int) -> EncodedImage:
    return EncodedImage(body=f"{pixels}:{quality}".encode(), width=2, height=1)


class RGBFramePublisherTests(unittest.TestCase):
    def test_wall_clock_limiter_and_latest_frame_replacement(self) -> None:
        clock = _Clock()
        uploader = _RecordingUploader(block_first=True)
        publisher = RGBFramePublisher(
            uploader=uploader,
            frame_fps=2,
            jpeg_quality=75,
            encoder=_fake_encoder,
            monotonic=clock,
            captured_at=lambda: "2026-07-15T00:00:00Z",
        )

        self.assertTrue(
            publisher.submit(
                {"rgb_screen": "first"},
                step_id=1,
                game_loop=10,
            )
        )
        self.assertTrue(uploader.started.wait(timeout=1))
        clock.value = 0.2
        self.assertFalse(
            publisher.submit(
                {"rgb_screen": "rate-limited"},
                step_id=2,
                game_loop=20,
            )
        )
        clock.value = 0.5
        self.assertTrue(
            publisher.submit(
                {"rgb_screen": "superseded"},
                step_id=3,
                game_loop=30,
            )
        )
        clock.value = 1.0
        self.assertTrue(
            publisher.submit(
                {"rgb_screen": "latest"},
                step_id=4,
                game_loop=40,
            )
        )

        uploader.release.set()
        publisher.close()

        self.assertEqual([call[1].step_id for call in uploader.calls], [1, 4])
        self.assertEqual(uploader.calls[-1][2].body, b"latest:75")
        self.assertTrue(uploader.closed)

    def test_missing_rgb_and_encoder_failures_never_escape(self) -> None:
        clock = _Clock()
        failed = threading.Event()
        uploader = _RecordingUploader()

        def encoder(pixels: Any, quality: int) -> EncodedImage:
            if pixels == "bad":
                failed.set()
                raise ValueError("not an RGB array")
            return _fake_encoder(pixels, quality)

        publisher = RGBFramePublisher(
            uploader=uploader,
            frame_fps=2,
            jpeg_quality=75,
            encoder=encoder,
            monotonic=clock,
        )
        self.assertFalse(publisher.submit({}, step_id=0, game_loop=0))
        self.assertTrue(publisher.submit({"rgb_screen": "bad"}, step_id=1, game_loop=1))
        self.assertTrue(failed.wait(timeout=1))
        clock.value = 0.5
        self.assertTrue(publisher.submit({"rgb_screen": "good"}, step_id=2, game_loop=2))

        publisher.close()

        self.assertEqual([call[2].body for call in uploader.calls], [b"good:75"])
        self.assertTrue(uploader.closed)

    def test_pillow_encoder_keeps_image_in_memory(self) -> None:
        pixels = np.zeros((3, 5, 3), dtype=np.int32)
        pixels[1, 2] = [255, 128, 64]

        encoded = _encode_jpeg(pixels, 80)

        self.assertEqual((encoded.width, encoded.height), (5, 3))
        self.assertTrue(encoded.body.startswith(b"\xff\xd8"))
        self.assertTrue(encoded.body.endswith(b"\xff\xd9"))


class RuntimeFrameUploaderTests(unittest.TestCase):
    def test_upload_uses_raw_jpeg_and_provenance_headers(self) -> None:
        captured: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return httpx.Response(204, request=request)

        client = httpx.Client(
            base_url="http://rtscortex",
            transport=httpx.MockTransport(handler),
        )
        uploader = RuntimeFrameUploader(
            run_id="run-1",
            episode_id="episode-1",
            base_url="http://unused",
            unix_socket=None,
            client=client,
        )
        sample = FrameSample(
            step_id=7,
            game_loop=112,
            sequence=3,
            captured_at="2026-07-15T00:00:00Z",
            screen=None,
            minimap=None,
        )

        uploader.upload(
            "screen",
            sample,
            EncodedImage(body=b"jpeg", width=256, height=256),
        )
        uploader.close()

        request: Optional[httpx.Request] = captured[0] if captured else None
        self.assertIsNotNone(request)
        assert request is not None
        self.assertEqual(request.url.path, "/internal/console/v1/frame/screen")
        self.assertEqual(request.content, b"jpeg")
        self.assertEqual(request.headers["content-type"], "image/jpeg")
        self.assertEqual(request.headers["x-rtscortex-protocol-version"], "1.1")
        self.assertEqual(request.headers["x-rtscortex-run-id"], "run-1")
        self.assertEqual(request.headers["x-rtscortex-episode-id"], "episode-1")
        self.assertEqual(request.headers["x-rtscortex-step-id"], "7")
        self.assertEqual(request.headers["x-rtscortex-game-loop"], "112")
        self.assertEqual(request.headers["x-rtscortex-frame-sequence"], "3")
        self.assertEqual(request.headers["x-rtscortex-width"], "256")
        self.assertEqual(request.headers["x-rtscortex-height"], "256")


class WorkerFrameIntegrationTests(unittest.TestCase):
    def test_worker_settings_read_console_environment(self) -> None:
        environment = {
            "RTSCORTEX_RUN_ID": "run-frame",
            "RTSCORTEX_EPISODE_ID": "episode-frame",
            "RTSCORTEX_CONSOLE_ENABLED": "true",
            "RTSCORTEX_CONSOLE_FRAME_FPS": "2.5",
            "RTSCORTEX_CONSOLE_JPEG_QUALITY": "81",
        }
        with mock.patch.dict("os.environ", environment, clear=True):
            settings = WorkerSettings.from_environment()

        self.assertTrue(settings.console_enabled)
        self.assertEqual(settings.console_frame_fps, 2.5)
        self.assertEqual(settings.console_jpeg_quality, 81)

    def test_agent_submission_and_close_are_best_effort(self) -> None:
        class FailingPublisher:
            def __init__(self) -> None:
                self.closed = False

            def submit(self, observation: Any, *, step_id: int, game_loop: int) -> bool:
                raise RuntimeError("console unavailable")

            def close(self) -> None:
                self.closed = True

        publisher = FailingPublisher()
        agent = object.__new__(RTSCortexMainAgent)
        agent._frame_publisher = publisher
        agent.steps = 9
        observation = SimpleNamespace(
            observation=SimpleNamespace(rgb_screen="pixels", game_loop=[112])
        )

        agent._submit_console_frame(observation)
        agent._close_frame_publisher()
        agent._close_frame_publisher()

        self.assertTrue(publisher.closed)
        self.assertIsNone(agent._frame_publisher)

    def test_standalone_entrypoint_requests_rgb_without_rendering(self) -> None:
        environment = {
            "RTSCORTEX_CONSOLE_ENABLED": "true",
            "RTSCORTEX_CONSOLE_RGB_SCREEN_SIZE": "320",
            "RTSCORTEX_CONSOLE_RGB_MINIMAP_SIZE": "160",
        }
        with (
            mock.patch.dict("os.environ", environment, clear=True),
            mock.patch.object(entrypoint.subprocess, "run") as run,
        ):
            entrypoint.main()

        command = run.call_args.args[0]
        self.assertEqual(command[command.index("--rgb_screen_size") + 1], "320")
        self.assertEqual(command[command.index("--rgb_minimap_size") + 1], "160")
        self.assertEqual(command[command.index("--action_space") + 1], "FEATURES")
        self.assertIn("--render=false", command)
        self.assertIn("--save_replay=false", command)


if __name__ == "__main__":
    unittest.main()
