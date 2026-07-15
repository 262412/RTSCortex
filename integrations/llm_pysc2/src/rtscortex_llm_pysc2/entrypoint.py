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
        os.environ.get("RTSCORTEX_SCENARIO", "pvz_task1_level1"),
        "--agent_race",
        os.environ.get("RTSCORTEX_AGENT_RACE", "protoss"),
        "--agent2",
        "Bot",
        "--agent2_race",
        os.environ.get("RTSCORTEX_OPPONENT_RACE", "random"),
        "--difficulty",
        os.environ.get("RTSCORTEX_OPPONENT_DIFFICULTY", "very_hard"),
        "--bot_build",
        os.environ.get("RTSCORTEX_OPPONENT_BUILD", "random"),
        "--step_mul",
        os.environ.get("RTSCORTEX_STEP_MUL", "1"),
        "--parallel",
        "1",
        "--render=false",
        "--save_replay=false",
        "--agent",
        "rtscortex_llm_pysc2.worker.RTSCortexMainAgent",
        "--random_seed",
        os.environ.get("RTSCORTEX_SEED", "0"),
    ]
    game_steps_per_episode = os.environ.get("RTSCORTEX_GAME_STEPS_PER_EPISODE")
    if game_steps_per_episode is not None:
        command.extend(["--game_steps_per_episode", game_steps_per_episode])
    if os.environ.get("RTSCORTEX_CONSOLE_ENABLED", "false").strip().lower() == "true":
        command.extend(
            [
                "--rgb_screen_size",
                os.environ.get("RTSCORTEX_CONSOLE_RGB_SCREEN_SIZE", "256"),
                "--rgb_minimap_size",
                os.environ.get("RTSCORTEX_CONSOLE_RGB_MINIMAP_SIZE", "128"),
                "--action_space",
                "FEATURES",
            ]
        )
    subprocess.run(command, check=True)
