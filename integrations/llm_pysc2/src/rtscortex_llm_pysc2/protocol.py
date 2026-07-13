"""JSON transport and LLM-PySC2 text-action rendering."""

import json
from typing import Any, Optional, cast

import httpx


class RuntimeClient:
    def __init__(
        self,
        base_url: str = "http://rtscortex",
        unix_socket: Optional[str] = None,
        timeout_seconds: float = 1.0,
    ) -> None:
        transport = httpx.HTTPTransport(uds=unix_socket) if unix_socket else None
        self.client = httpx.Client(
            base_url=base_url,
            transport=transport,
            timeout=timeout_seconds,
        )

    def health(self) -> dict[str, Any]:
        response = self.client.get("/healthz")
        response.raise_for_status()
        return cast(dict[str, Any], response.json())

    def tick(self, observation: dict[str, Any]) -> dict[str, Any]:
        response = self.client.post("/v1/tick", json=observation)
        response.raise_for_status()
        return cast(dict[str, Any], response.json())

    def execution(self, report: dict[str, Any]) -> None:
        response = self.client.post("/v1/execution", json=report)
        response.raise_for_status()

    def end_episode(self, result: dict[str, Any]) -> None:
        response = self.client.post("/v1/episode/end", json=result)
        response.raise_for_status()

    def close(self) -> None:
        self.client.close()


def render_action_batch(batch: dict[str, Any]) -> str:
    """Render a validated ActionBatch in LLM-PySC2's Team/action text format."""

    grouped: dict[str, list[dict[str, Any]]] = {}
    for command in batch.get("commands", []):
        grouped.setdefault(command["actor"], []).append(command)
    lines = ["Actions:"]
    for actor, commands in grouped.items():
        lines.append(f"    Team {actor}:")
        for command in commands:
            arguments = ", ".join(_format_argument(value) for value in command["arguments"])
            lines.append(f"        <{command['name']}({arguments})>")
    return "\n".join(lines)


def _format_argument(value: Any) -> str:
    if isinstance(value, (list, dict)):
        return json.dumps(value, separators=(",", ":"))
    return str(value)
