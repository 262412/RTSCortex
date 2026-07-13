"""Process entrypoint for the fixed v0.1 PvZ scenario."""

from __future__ import annotations

import os
import subprocess
import sys


def main() -> None:
    """Start PySC2 only when the worker command is explicitly invoked."""

    command = [
        sys.executable,
        "-m",
        "pysc2.bin.agent",
        "--map",
        "pvz_task1_level1",
        "--agent_race",
        "protoss",
        "--parallel",
        "1",
        "--agent",
        "rtscortex_llm_pysc2.worker.RTSCortexMainAgent",
        "--random_seed",
        os.environ.get("RTSCORTEX_SEED", "0"),
    ]
    subprocess.run(command, check=True)
