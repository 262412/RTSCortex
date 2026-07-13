from __future__ import annotations

import stat
from pathlib import Path

import pytest

from rtscortex.config import EnvironmentSettings
from rtscortex.runtime.live import LiveEnvironmentError, prepare_live_worker
from tests.helpers import make_config


def test_prepare_live_worker_builds_fixed_pysc2_command(tmp_path: Path) -> None:
    base_python = tmp_path / "python3.9"
    base_python.write_text("#!/bin/sh\necho 'Python 3.9.99'\n", encoding="utf-8")
    base_python.chmod(base_python.stat().st_mode | stat.S_IXUSR)
    worker_python = tmp_path / "worker-venv/bin/python"
    worker_python.parent.mkdir(parents=True)
    worker_python.symlink_to(base_python)

    sc2_path = tmp_path / "StarCraftII"
    executable = sc2_path / "Versions/Base75689/SC2_x64"
    executable.parent.mkdir(parents=True)
    executable.touch()
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
    scenario_map = sc2_path / "Maps/llm_pysc2/pvz_task1_level1.SC2Map"
    scenario_map.parent.mkdir(parents=True)
    scenario_map.touch()

    upstream_source = tmp_path / "third_party/LLM-PySC2/llm_pysc2/agents/llm_pysc2_agent_main.py"
    upstream_source.parent.mkdir(parents=True)
    upstream_source.write_text(
        "elif not self._all_agent_waiting_response_finished():\n"
        "    func_id, func_call = (0, actions.FUNCTIONS.no_op())\n"
        "    return func_call\n"
        "elif not self._all_agent_executing_finished():\n"
        "    pass\n",
        encoding="utf-8",
    )
    runner_source = tmp_path / "third_party/LLM-PySC2/pysc2/bin/agent.py"
    runner_source.parent.mkdir(parents=True)
    runner_source.write_text(
        'flags.DEFINE_integer("random_seed", None, "Random seed")\nrandom_seed=FLAGS.random_seed\n',
        encoding="utf-8",
    )

    config = make_config(tmp_path).model_copy(
        update={
            "environment": EnvironmentSettings(
                adapter="llm_pysc2",
                sc2_path=sc2_path,
                worker_python=worker_python,
                max_steps=42,
            )
        }
    )

    spec = prepare_live_worker(config, tmp_path, environment={})

    assert spec.sc2_path == sc2_path
    assert spec.command == (
        str(worker_python),
        "-m",
        "pysc2.bin.agent",
        "--map",
        "pvz_task1_level1",
        "--agent",
        "rtscortex_llm_pysc2.worker.RTSCortexMainAgent",
        "--agent_race",
        "protoss",
        "--parallel",
        "1",
        "--render=false",
        "--save_replay=false",
        "--max_agent_steps",
        "42",
        "--random_seed",
        "0",
    )


def test_prepare_live_worker_reports_all_missing_prerequisites(tmp_path: Path) -> None:
    config = make_config(tmp_path).model_copy(
        update={
            "environment": EnvironmentSettings(
                adapter="llm_pysc2",
                worker_python=tmp_path / "missing-python",
            )
        }
    )

    with pytest.raises(LiveEnvironmentError) as captured:
        prepare_live_worker(config, tmp_path, environment={})

    message = str(captured.value)
    assert "worker Python is missing" in message
    assert "SC2PATH is unset" in message
    assert "waiting-response patch is not applied" in message
    assert "random-seed patch is not applied" in message
