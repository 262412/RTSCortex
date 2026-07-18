"""Serve one pinned HIMA checkpoint over a private Unix-domain socket."""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

import uvicorn

from rtscortex.policy.hima.live import HIMALivePolicyService, create_hima_live_app
from rtscortex.policy.hima.subagent import HIMA_PINNED_REVISIONS, TransformersHIMAGenerator
from rtscortex.policy.subagents import HIMA_ALL_SPECS


def run_worker(
    *,
    socket_path: Path,
    model_id: str,
    model_path: Path,
    device: str,
    max_new_tokens: int,
    allow_unlicensed_weights: bool,
) -> None:
    """Load once during app startup, then serve serialized proposal requests."""

    if not socket_path.is_absolute():
        raise ValueError("HIMA socket path must be absolute")
    if not socket_path.parent.is_dir():
        raise FileNotFoundError(f"HIMA socket parent does not exist: {socket_path.parent}")
    spec = next(
        (candidate for candidate in HIMA_ALL_SPECS if candidate.model_id == model_id),
        None,
    )
    if spec is None:
        raise ValueError(f"no HIMA policy spec exists for {model_id}")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    generator = TransformersHIMAGenerator(
        model_path,
        model_id=model_id,
        allow_unlicensed_weights=allow_unlicensed_weights,
        device=device,
        max_new_tokens=max_new_tokens,
    )
    app = create_hima_live_app(HIMALivePolicyService(spec, generator))
    config = uvicorn.Config(
        app,
        uds=str(socket_path),
        log_level="info",
        access_log=False,
    )
    asyncio.run(uvicorn.Server(config).serve())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--model-id", choices=tuple(HIMA_PINNED_REVISIONS), required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--allow-unlicensed-weights", action="store_true")
    arguments = parser.parse_args()
    run_worker(
        socket_path=arguments.socket,
        model_id=arguments.model_id,
        model_path=arguments.model_path,
        device=arguments.device,
        max_new_tokens=arguments.max_new_tokens,
        allow_unlicensed_weights=arguments.allow_unlicensed_weights,
    )


if __name__ == "__main__":
    main()
