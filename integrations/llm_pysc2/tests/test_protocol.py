"""Python 3.9 coverage for durable placement-transition delivery."""

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from typing import Any

import httpx
from rtscortex_llm_pysc2.protocol import (
    PlacementTransitionConflictError,
    PlacementTransitionDeliveryError,
    RuntimeClient,
)


def _event(transition_id: str = "placement-transition:test") -> dict[str, Any]:
    return {
        "protocol_version": "1.1",
        "run_id": "run-1",
        "episode_id": "episode-1",
        "step_id": 1,
        "command_id": "command-1",
        "action_name": "Build_Pylon_Screen",
        "transition_id": transition_id,
        "builder_tag": "0xb1",
        "builder_lease_state": "acquired",
        "transition": {
            "reservation_id": "reservation-1",
            "structure_type": "Pylon",
            "footprint_cells": [[1, 1], [1, 2], [2, 1], [2, 2]],
            "previous_state": "unreserved",
            "next_state": "reserved",
            "game_loop": 1,
        },
    }


class PlacementTransitionOutboxTests(unittest.TestCase):
    def test_lost_ack_retries_the_exact_payload_once(self) -> None:
        requests: list[bytes] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request.content)
            if len(requests) == 1:
                raise httpx.ReadTimeout("ack lost", request=request)
            return httpx.Response(
                200,
                request=request,
                json={"status": "already_recorded"},
            )

        with tempfile.TemporaryDirectory() as directory:
            client = RuntimeClient(
                placement_outbox_path=str(Path(directory) / "outbox.sqlite3"),
                client=httpx.Client(
                    base_url="http://rtscortex",
                    transport=httpx.MockTransport(handler),
                ),
            )
            client.placement_transition(_event())
            client.close()

            self.assertEqual(len(requests), 2)
            self.assertEqual(requests[0], requests[1])
            with sqlite3.connect(Path(directory) / "outbox.sqlite3") as connection:
                count = connection.execute(
                    "SELECT COUNT(*) FROM placement_transition_outbox"
                ).fetchone()[0]
            self.assertEqual(count, 0)

    def test_restart_recovers_the_same_durable_event(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            outbox = Path(directory) / "outbox.sqlite3"

            def unavailable(request: httpx.Request) -> httpx.Response:
                raise httpx.ConnectError("runtime unavailable", request=request)

            first = RuntimeClient(
                placement_outbox_path=str(outbox),
                placement_retry_attempts=1,
                client=httpx.Client(
                    base_url="http://rtscortex",
                    transport=httpx.MockTransport(unavailable),
                ),
            )
            with self.assertRaises(PlacementTransitionDeliveryError):
                first.placement_transition(_event())
            first.close()

            delivered: list[dict[str, Any]] = []
            paths: list[str] = []

            def recovered(request: httpx.Request) -> httpx.Response:
                paths.append(request.url.path)
                if request.url.path == "/v1/tick":
                    return httpx.Response(200, request=request, json={"commands": []})
                delivered.append(json.loads(request.content))
                return httpx.Response(200, request=request, json={"status": "recorded"})

            second = RuntimeClient(
                placement_outbox_path=str(outbox),
                client=httpx.Client(
                    base_url="http://rtscortex",
                    transport=httpx.MockTransport(recovered),
                ),
            )
            second.tick({})
            second.close()

            self.assertEqual(paths, ["/v1/tick", "/v1/placement/transition"])
            self.assertEqual(delivered, [_event()])
            with sqlite3.connect(outbox) as connection:
                count = connection.execute(
                    "SELECT COUNT(*) FROM placement_transition_outbox"
                ).fetchone()[0]
            self.assertEqual(count, 0)

    def test_same_transition_id_with_different_payload_is_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            outbox = Path(directory) / "outbox.sqlite3"

            def unavailable(request: httpx.Request) -> httpx.Response:
                raise httpx.ConnectError("runtime unavailable", request=request)

            client = RuntimeClient(
                placement_outbox_path=str(outbox),
                placement_retry_attempts=1,
                client=httpx.Client(
                    base_url="http://rtscortex",
                    transport=httpx.MockTransport(unavailable),
                ),
            )
            with self.assertRaises(PlacementTransitionDeliveryError):
                client.placement_transition(_event())
            conflicting = _event()
            conflicting["builder_tag"] = "0xb2"
            with self.assertRaises(PlacementTransitionConflictError):
                client.placement_transition(conflicting)
            client.close()

    def test_http_409_is_a_fatal_identity_conflict(self) -> None:
        def conflict(request: httpx.Request) -> httpx.Response:
            return httpx.Response(409, request=request, json={"detail": "different payload"})

        with tempfile.TemporaryDirectory() as directory:
            client = RuntimeClient(
                placement_outbox_path=str(Path(directory) / "outbox.sqlite3"),
                client=httpx.Client(
                    base_url="http://rtscortex",
                    transport=httpx.MockTransport(conflict),
                ),
            )
            with self.assertRaises(PlacementTransitionConflictError):
                client.placement_transition(_event())
            client.close()


if __name__ == "__main__":
    unittest.main()
