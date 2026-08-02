from __future__ import annotations

import json

import httpx
import pytest
from rtscortex_llm_pysc2.performance import PhaseProfiler
from rtscortex_llm_pysc2.protocol import RuntimeClient


def test_runtime_client_profiles_encode_transport_decode_and_runtime_tick() -> None:
    profiles: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/tick":
            return httpx.Response(
                200,
                content=json.dumps({"commands": []}).encode(),
                headers={"X-RTSCortex-Runtime-Tick-Ms": "2.5"},
            )
        if request.url.path == "/v1/performance":
            profiles.append(json.loads(request.content))
            return httpx.Response(200, json={"status": "recorded"})
        raise AssertionError(request.url.path)

    profiler = PhaseProfiler()
    http = httpx.Client(
        base_url="http://rtscortex",
        transport=httpx.MockTransport(handler),
    )
    client = RuntimeClient(client=http, profiler=profiler)
    observation = {
        "run_id": "run",
        "episode_id": "episode",
        "step_id": 1,
        "game_loop": 16,
    }

    assert client.tick(observation) == {"commands": []}
    client.performance_profile(
        run_id="run",
        episode_id="episode",
        step_id=1,
        game_loop=16,
    )

    phases = profiles[0]["phases"]
    assert isinstance(phases, dict)
    assert set(phases) >= {"json_encode", "request_transport", "response_decode", "runtime_tick"}
    assert phases["runtime_tick"]["total_ms"] == pytest.approx(2.5)
    client.close()
