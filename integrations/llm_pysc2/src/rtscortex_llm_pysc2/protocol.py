"""HTTP transport for the versioned RTSCortex worker API."""

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
        payload = cast(dict[str, Any], response.json())
        protocol_version = payload.get("protocol_version")
        if protocol_version != "1.1":
            raise RuntimeError(
                "RTSCortex live protocol mismatch: "
                f"worker requires 1.1, runtime reported {protocol_version!r}"
            )
        return payload

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
