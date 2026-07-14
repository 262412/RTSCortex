from __future__ import annotations

import stat
from pathlib import Path

import pytest
from rtscortex_llm_pysc2 import entrypoint as worker_entrypoint

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
    executable = sc2_path / "Versions/Base92440/SC2_x64"
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
    _write_build_coordinate_patch_source(tmp_path)

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
        "--agent2",
        "Bot",
        "--agent2_race",
        "random",
        "--difficulty",
        "very_hard",
        "--bot_build",
        "random",
        "--step_mul",
        "1",
        "--parallel",
        "1",
        "--render=false",
        "--save_replay=false",
        "--max_agent_steps",
        "42",
        "--random_seed",
        "0",
    )

    executable.unlink()
    old_executable = sc2_path / "Versions/Base75689/SC2_x64"
    old_executable.parent.mkdir(parents=True)
    old_executable.touch()
    old_executable.chmod(old_executable.stat().st_mode | stat.S_IXUSR)
    with pytest.raises(LiveEnvironmentError, match="older than required build 92440"):
        prepare_live_worker(config, tmp_path, environment={})


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
    assert "build-coordinate patch is not applied" in message


def test_prepare_live_worker_accepts_2s3z_on_sc2_410(tmp_path: Path) -> None:
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
    scenario_map = sc2_path / "Maps/llm_smac/2s3z.SC2Map"
    scenario_map.parent.mkdir(parents=True)
    scenario_map.touch()
    _write_worker_patch_sources(tmp_path)

    config = make_config(tmp_path).model_copy(
        update={
            "environment": EnvironmentSettings(
                adapter="llm_pysc2",
                scenario="2s3z",
                sc2_path=sc2_path,
                worker_python=worker_python,
                max_steps=128,
            )
        }
    )

    spec = prepare_live_worker(config, tmp_path, environment={})

    assert spec.sc2_path == sc2_path
    assert spec.command[spec.command.index("--map") + 1] == "2s3z"
    assert spec.command[spec.command.index("--max_agent_steps") + 1] == "128"


def test_prepare_live_worker_builds_official_melee_bot_command(tmp_path: Path) -> None:
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
    scenario_map = sc2_path / "Maps/Melee/Simple64.SC2Map"
    scenario_map.parent.mkdir(parents=True)
    scenario_map.touch()
    _write_worker_patch_sources(tmp_path)

    config = make_config(tmp_path).model_copy(
        update={
            "environment": EnvironmentSettings(
                adapter="llm_pysc2",
                scenario="Simple64",
                sc2_path=sc2_path,
                worker_python=worker_python,
                max_steps=40_000,
                agent_race="protoss",
                opponent_race="zerg",
                opponent_difficulty="easy",
                opponent_build="macro",
                step_mul=1,
                game_steps_per_episode=28_800,
            )
        }
    )

    spec = prepare_live_worker(config, tmp_path, environment={})

    assert spec.command == (
        str(worker_python),
        "-m",
        "pysc2.bin.agent",
        "--map",
        "Simple64",
        "--agent",
        "rtscortex_llm_pysc2.worker.RTSCortexMainAgent",
        "--agent_race",
        "protoss",
        "--agent2",
        "Bot",
        "--agent2_race",
        "zerg",
        "--difficulty",
        "easy",
        "--bot_build",
        "macro",
        "--step_mul",
        "1",
        "--game_steps_per_episode",
        "28800",
        "--parallel",
        "1",
        "--render=false",
        "--save_replay=false",
        "--max_agent_steps",
        "40000",
        "--random_seed",
        "0",
    )


def test_worker_entrypoint_forwards_melee_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[list[str]] = []

    def run(command: list[str], *, check: bool) -> None:
        assert check is True
        captured.append(command)

    monkeypatch.setenv("RTSCORTEX_SCENARIO", "Simple64")
    monkeypatch.setenv("RTSCORTEX_AGENT_RACE", "protoss")
    monkeypatch.setenv("RTSCORTEX_OPPONENT_RACE", "zerg")
    monkeypatch.setenv("RTSCORTEX_OPPONENT_DIFFICULTY", "easy")
    monkeypatch.setenv("RTSCORTEX_OPPONENT_BUILD", "macro")
    monkeypatch.setenv("RTSCORTEX_STEP_MUL", "1")
    monkeypatch.setenv("RTSCORTEX_GAME_STEPS_PER_EPISODE", "28800")
    monkeypatch.setattr("rtscortex_llm_pysc2.entrypoint.subprocess.run", run)

    worker_entrypoint.main()

    command = captured[0]
    assert command[command.index("--map") + 1] == "Simple64"
    assert command[command.index("--agent2") + 1] == "Bot"
    assert command[command.index("--agent2_race") + 1] == "zerg"
    assert command[command.index("--difficulty") + 1] == "easy"
    assert command[command.index("--bot_build") + 1] == "macro"
    assert command[command.index("--step_mul") + 1] == "1"
    assert command[command.index("--game_steps_per_episode") + 1] == "28800"
    assert "--save_replay=false" in command


def test_prepare_live_worker_rejects_unsupported_scenario(tmp_path: Path) -> None:
    config = make_config(tmp_path).model_copy(
        update={
            "environment": EnvironmentSettings(
                adapter="llm_pysc2",
                scenario="unknown",
                worker_python=tmp_path / "python",
            )
        }
    )

    with pytest.raises(LiveEnvironmentError, match="unsupported live scenario"):
        prepare_live_worker(config, tmp_path, environment={})


def _write_worker_patch_sources(project_root: Path) -> None:
    upstream_source = (
        project_root / "third_party/LLM-PySC2/llm_pysc2/agents/llm_pysc2_agent_main.py"
    )
    upstream_source.parent.mkdir(parents=True)
    upstream_source.write_text(
        "elif not self._all_agent_waiting_response_finished():\n"
        "    func_id, func_call = (0, actions.FUNCTIONS.no_op())\n"
        "    return func_call\n"
        "elif not self._all_agent_executing_finished():\n"
        "    pass\n",
        encoding="utf-8",
    )
    runner_source = project_root / "third_party/LLM-PySC2/pysc2/bin/agent.py"
    runner_source.parent.mkdir(parents=True)
    runner_source.write_text(
        'flags.DEFINE_integer("random_seed", None, "Random seed")\nrandom_seed=FLAGS.random_seed\n',
        encoding="utf-8",
    )
    _write_build_coordinate_patch_source(project_root)


def _write_build_coordinate_patch_source(project_root: Path) -> None:
    source = project_root / "third_party/LLM-PySC2/llm_pysc2/lib/llm_action.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        "feature_screen.power[y0][x0]\n"
        "feature_screen.creep[y0][x0]\n"
        "feature_screen.buildable[y][x]\n"
        "feature_screen.pathable[y][x]\n"
        "feature_screen.player_relative[y][x]\n",
        encoding="utf-8",
    )
