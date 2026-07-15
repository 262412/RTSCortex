"""Run a disposable historical Console backed by the real Python event store."""

from __future__ import annotations

import argparse
import threading
from pathlib import Path
from tempfile import TemporaryDirectory

import uvicorn
from fastapi import Request
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response

from rtscortex.console import ConsoleSession, LiveConsoleHub, create_console_app
from rtscortex.memory import EventStore

RUN_ID = "e2e-console-run"
EPISODE_ID = "e2e-episode-0"
COMMAND_ID = "cmd-build-pylon-001"


def _append_initial_events(store: EventStore) -> None:
    events: list[tuple[str, dict[str, object]]] = [
        (
            "observation",
            {
                "game_loop": 64,
                "economy": {"minerals": 400, "vespene": 0, "workers": 12},
            },
        ),
        ("module_started", {"game_loop": 64, "module": "Reflection"}),
        (
            "context_prepared",
            {
                "game_loop": 64,
                "module": "Planning",
                "original_chars": 8200,
                "final_chars": 3900,
                "compression_ratio": 0.476,
                "retained_observations": 4,
                "retained_lessons": 2,
                "model": "Qwen/Qwen3-8B",
                "prompt_version": "planning-v1",
            },
        ),
        (
            "module_result",
            {
                "game_loop": 64,
                "module": "Reflection",
                "output": {"lesson": "Keep the first Pylon near the mineral line."},
                "model_call": True,
                "latency_ms": 842,
                "usage": {"prompt_tokens": 900, "completion_tokens": 52, "total_tokens": 952},
            },
        ),
        (
            "module_result",
            {
                "game_loop": 64,
                "module": "Planning",
                "output": {
                    "goal": "Establish Protoss production",
                    "steps": ["Build a Pylon", "Build a Gateway"],
                },
                "model_call": True,
                "latency_ms": 1260,
                "usage": {
                    "prompt_tokens": 1024,
                    "completion_tokens": 86,
                    "total_tokens": 1110,
                },
            },
        ),
        (
            "module_result",
            {
                "game_loop": 64,
                "module": "Action",
                "output": {
                    "commands": [
                        {
                            "command_id": COMMAND_ID,
                            "action_name": "Build_Pylon_Screen",
                            "arguments": [[66, 88]],
                        }
                    ]
                },
            },
        ),
        (
            "decision",
            {
                "game_loop": 64,
                "commands": [
                    {
                        "command_id": COMMAND_ID,
                        "action_name": "Build_Pylon_Screen",
                        "actor": "Builder",
                        "source": "planner",
                    }
                ],
            },
        ),
        (
            "execution",
            {
                "game_loop": 80,
                "command_id": COMMAND_ID,
                "action_name": "Build_Pylon_Screen",
                "status": "succeeded",
                "execution_stage": "effect_verification",
                "effect_evidence": {"new_structure_tag": "0xabc", "confirmed_loop": 80},
            },
        ),
        (
            "validation_failed",
            {
                "game_loop": 96,
                "command_id": "cmd-invalid-attack-001",
                "action_name": "Attack_Unit",
                "status": "failed",
                "failure_code": "friendly_target",
            },
        ),
    ]
    for step_id, (event_type, payload) in enumerate(events):
        store.append_event(
            run_id=RUN_ID,
            episode_id=EPISODE_ID,
            step_id=step_id,
            event_type=event_type,
            payload=payload,
        )


def _append_reconnect_events(
    store: EventStore,
    client_ready: threading.Event,
    stopping: threading.Event,
) -> None:
    if not client_ready.wait(timeout=30.0) or stopping.wait(timeout=2.0):
        return
    for offset, game_loop in enumerate((112, 128, 144), start=1):
        if stopping.is_set():
            return
        store.append_event(
            run_id=RUN_ID,
            episode_id=EPISODE_ID,
            step_id=9 + offset,
            event_type="console_fixture_tick",
            payload={"game_loop": game_loop, "status": "healthy", "sequence": offset},
        )
        if stopping.wait(timeout=0.25):
            return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--static-dir", type=Path, required=True)
    arguments = parser.parse_args()
    static_dir = arguments.static_dir.resolve()
    if not (static_dir / "index.html").is_file():
        raise SystemExit(f"Console frontend is not built: {static_dir}")

    client_ready = threading.Event()
    stopping = threading.Event()
    with TemporaryDirectory(prefix="rtscortex-console-e2e-") as temporary_directory:
        root = Path(temporary_directory)
        store = EventStore(root / "events.sqlite3", root / "events.jsonl")
        _append_initial_events(store)
        session = ConsoleSession(
            run_id=RUN_ID,
            episode_id=EPISODE_ID,
            status="historical",
            scenario="Simple64",
            seed=0,
            model="Qwen/Qwen3-8B",
        )
        hub = LiveConsoleHub(session)
        app = create_console_app(store, session, hub, static_dir)

        @app.middleware("http")
        async def mark_client_ready(
            request: Request,
            call_next: RequestResponseEndpoint,
        ) -> Response:
            response = await call_next(request)
            if request.url.path == "/console/api/v1/session":
                client_ready.set()
            return response

        producer = threading.Thread(
            target=_append_reconnect_events,
            args=(store, client_ready, stopping),
            name="console-e2e-event-producer",
            daemon=True,
        )
        producer.start()
        try:
            uvicorn.run(app, host="127.0.0.1", port=arguments.port, log_level="warning")
        finally:
            stopping.set()
            producer.join(timeout=2.0)
            store.close()


if __name__ == "__main__":
    main()
