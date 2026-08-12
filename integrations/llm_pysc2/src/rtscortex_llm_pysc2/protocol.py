"""HTTP transport for the versioned RTSCortex worker API."""

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Optional, cast

import httpx

from rtscortex_llm_pysc2.performance import PhaseProfiler


class PlacementTransitionDeliveryError(RuntimeError):
    """The exact transition remains durable and may be retried after restart."""


class PlacementTransitionConflictError(RuntimeError):
    """The server or local outbox rejected a conflicting retry identity."""


class _PlacementTransitionOutbox:
    def __init__(self, path: Path, *, limit: int) -> None:
        if limit < 1:
            raise ValueError("placement outbox limit must be positive")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(str(path), check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._limit = limit
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS placement_transition_outbox (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                transition_id TEXT NOT NULL UNIQUE,
                payload_hash TEXT NOT NULL,
                payload_json TEXT NOT NULL
            )
            """
        )
        self._connection.commit()

    def enqueue(self, event: dict[str, Any]) -> None:
        transition_id = event.get("transition_id")
        if not isinstance(transition_id, str) or not transition_id:
            raise ValueError("placement transition requires a non-empty transition_id")
        payload_json = json.dumps(event, sort_keys=True, separators=(",", ":"))
        payload_hash = hashlib.sha256(payload_json.encode()).hexdigest()
        with self._lock:
            existing = self._connection.execute(
                """
                SELECT payload_hash FROM placement_transition_outbox
                WHERE transition_id = ?
                """,
                (transition_id,),
            ).fetchone()
            if existing is not None:
                if str(existing["payload_hash"]) != payload_hash:
                    raise PlacementTransitionConflictError(
                        "placement transition retry used the same ID with a different payload"
                    )
                return
            count = int(
                self._connection.execute(
                    "SELECT COUNT(*) FROM placement_transition_outbox"
                ).fetchone()[0]
            )
            if count >= self._limit:
                raise PlacementTransitionDeliveryError(
                    f"placement transition outbox reached its {self._limit}-event limit"
                )
            self._connection.execute(
                """
                INSERT INTO placement_transition_outbox (
                    transition_id, payload_hash, payload_json
                ) VALUES (?, ?, ?)
                """,
                (transition_id, payload_hash, payload_json),
            )
            self._connection.commit()

    def oldest(self) -> Optional[tuple[str, dict[str, Any]]]:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT transition_id, payload_json
                FROM placement_transition_outbox
                ORDER BY sequence
                LIMIT 1
                """
            ).fetchone()
        if row is None:
            return None
        return str(row["transition_id"]), cast(dict[str, Any], json.loads(row["payload_json"]))

    def acknowledge(self, transition_id: str) -> None:
        with self._lock:
            self._connection.execute(
                "DELETE FROM placement_transition_outbox WHERE transition_id = ?",
                (transition_id,),
            )
            self._connection.commit()

    def close(self) -> None:
        self._connection.close()


class RuntimeClient:
    def __init__(
        self,
        base_url: str = "http://rtscortex",
        unix_socket: Optional[str] = None,
        timeout_seconds: float = 1.0,
        placement_outbox_path: Optional[str] = None,
        placement_retry_attempts: int = 3,
        placement_outbox_limit: int = 64,
        placement_timeout_seconds: float = 2.0,
        client: Optional[httpx.Client] = None,
        profiler: Optional[PhaseProfiler] = None,
    ) -> None:
        if placement_retry_attempts < 1:
            raise ValueError("placement retry attempts must be positive")
        if placement_timeout_seconds <= 0:
            raise ValueError("placement timeout must be positive")
        transport = httpx.HTTPTransport(uds=unix_socket) if unix_socket else None
        self.client = client or httpx.Client(
            base_url=base_url,
            transport=transport,
            timeout=timeout_seconds,
        )
        self.profiler = profiler or PhaseProfiler()
        self._tick_count = 0
        self._placement_retry_attempts = placement_retry_attempts
        self._placement_timeout_seconds = placement_timeout_seconds
        self._placement_outbox = (
            None
            if placement_outbox_path is None
            else _PlacementTransitionOutbox(
                Path(placement_outbox_path),
                limit=placement_outbox_limit,
            )
        )

    def health(self) -> dict[str, Any]:
        response = self.client.get("/healthz")
        response.raise_for_status()
        payload = cast(dict[str, Any], response.json())
        protocol_version = payload.get("protocol_version")
        if protocol_version != "1.1":
            raise RuntimeError(
                "RTSCortex live protocol mismatch: "
                f"worker requires 1.1, runtime reported {protocol_version!r}"
            )
        return payload

    def tick(self, observation: dict[str, Any]) -> dict[str, Any]:
        with self.profiler.measure("json_encode"):
            encoded = json.dumps(observation, ensure_ascii=False, separators=(",", ":"))
        with self.profiler.measure("request_transport"):
            response = self.client.post(
                "/v1/tick",
                content=encoded.encode(),
                headers={"content-type": "application/json"},
            )
        response.raise_for_status()
        runtime_tick_ms = response.headers.get("x-rtscortex-runtime-tick-ms")
        if runtime_tick_ms is not None:
            self.profiler.observe_milliseconds("runtime_tick", float(runtime_tick_ms))
        with self.profiler.measure("response_decode"):
            payload = cast(dict[str, Any], json.loads(response.content))
        # A restarted Runtime reconstructs episode and command lifecycle state
        # during tick. Only then is it safe to replay a transition whose ACK
        # was lost by the previous Worker process.
        self.retry_placement_transitions()
        self._tick_count += 1
        if self._tick_count % 64 == 0:
            self.performance_profile(
                run_id=str(observation["run_id"]),
                episode_id=str(observation["episode_id"]),
                step_id=int(observation["step_id"]),
                game_loop=int(observation["game_loop"]),
            )
        return payload

    def performance_profile(
        self,
        *,
        run_id: str,
        episode_id: str,
        step_id: int,
        game_loop: int,
    ) -> None:
        response = self.client.post(
            "/v1/performance",
            json={
                "protocol_version": "1.1",
                "run_id": run_id,
                "episode_id": episode_id,
                "step_id": step_id,
                "game_loop": game_loop,
                "phases": self.profiler.snapshot(),
            },
        )
        response.raise_for_status()

    def execution(self, report: dict[str, Any]) -> None:
        response = self.client.post("/v1/execution", json=report)
        response.raise_for_status()

    def authoritative_build_preflight(self, result: dict[str, Any]) -> None:
        response = self.client.post(
            "/v1/build/preflight",
            json=result,
        )
        response.raise_for_status()

    def placement_transition(self, event: dict[str, Any]) -> None:
        if self._placement_outbox is None:
            self._deliver_placement_transition(event)
            return
        self._placement_outbox.enqueue(event)
        self.retry_placement_transitions()

    def retry_placement_transitions(self) -> None:
        if self._placement_outbox is None:
            return
        while True:
            pending = self._placement_outbox.oldest()
            if pending is None:
                return
            transition_id, event = pending
            self._deliver_placement_transition(event)
            self._placement_outbox.acknowledge(transition_id)

    def _deliver_placement_transition(self, event: dict[str, Any]) -> None:
        last_error: Optional[BaseException] = None
        for _attempt in range(self._placement_retry_attempts):
            try:
                response = self.client.post(
                    "/v1/placement/transition",
                    json=event,
                    timeout=self._placement_timeout_seconds,
                )
                if response.status_code == 409:
                    raise PlacementTransitionConflictError(
                        "runtime rejected placement transition identity: " + response.text
                    )
                response.raise_for_status()
                payload = cast(dict[str, Any], response.json())
                if payload.get("status") not in {"recorded", "already_recorded"}:
                    raise PlacementTransitionDeliveryError(
                        "runtime returned an invalid placement transition acknowledgement"
                    )
                return
            except PlacementTransitionConflictError:
                raise
            except httpx.HTTPStatusError as error:
                if 400 <= error.response.status_code < 500:
                    raise PlacementTransitionDeliveryError(
                        "runtime rejected placement transition: " + error.response.text
                    ) from error
                last_error = error
            except (httpx.TransportError, PlacementTransitionDeliveryError) as error:
                last_error = error
        raise PlacementTransitionDeliveryError(
            "placement transition delivery failed; exact payload remains in the outbox"
        ) from last_error

    def end_episode(self, result: dict[str, Any]) -> None:
        self.performance_profile(
            run_id=str(result["run_id"]),
            episode_id=str(result["episode_id"]),
            step_id=int(result["steps"]),
            game_loop=int(result["steps"]),
        )
        response = self.client.post("/v1/episode/end", json=result)
        response.raise_for_status()

    def close(self) -> None:
        try:
            self.client.close()
        finally:
            if self._placement_outbox is not None:
                self._placement_outbox.close()
