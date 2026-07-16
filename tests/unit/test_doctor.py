from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import httpx
import pytest

import rtscortex.cli.doctor as doctor_module
from rtscortex.cli.doctor import (
    _provider_check,
    _sc2_checks,
    _worker_packages_check,
    _worker_patch_check,
    _worker_python_check,
    run_doctor,
)
from rtscortex.config import ExperimentConfig, ProviderSettings


def core_venv_status(project_root: Path) -> str:
    checks = run_doctor(project_root)
    return next(check.status for check in checks if check.name == "core_venv")


def test_doctor_accepts_uv_virtualenv_directory(tmp_path: Path) -> None:
    venv = tmp_path / ".venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
    (venv / "bin" / "python").touch()

    assert core_venv_status(tmp_path) == "ok"


def test_doctor_rejects_incomplete_virtualenv_directory(tmp_path: Path) -> None:
    (tmp_path / ".venv").mkdir()

    assert core_venv_status(tmp_path) == "error"


def test_provider_check_requires_configured_model(monkeypatch: pytest.MonkeyPatch) -> None:
    config = ExperimentConfig(
        provider=ProviderSettings(
            kind="openai_compatible",
            model="Qwen/Qwen3-8B",
            base_url="http://model.test/v1",
        )
    )

    def get(*args: Any, **kwargs: Any) -> httpx.Response:
        del args, kwargs
        return httpx.Response(
            200,
            request=httpx.Request("GET", "http://model.test/v1/models"),
            json={"data": [{"id": "Qwen/Qwen3-8B"}]},
        )

    monkeypatch.setattr(httpx, "get", get)

    check = _provider_check(config)

    assert check.status == "ok"
    assert "Qwen/Qwen3-8B" in check.detail


def test_worker_python_requires_python_39(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    python = tmp_path / "python"
    python.touch()

    def completed(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        del args, kwargs
        return subprocess.CompletedProcess([], 0, stdout="Python 3.9.25\n", stderr="")

    monkeypatch.setattr(subprocess, "run", completed)

    assert _worker_python_check(python, required=True).status == "ok"


def test_worker_package_probe_surfaces_import_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    python = tmp_path / "python"
    python.touch()

    def failed_probe(path: str) -> tuple[int, str]:
        assert path == str(python)
        return 1, "ModuleNotFoundError: missing_package"

    monkeypatch.setattr(doctor_module, "_probe_worker_packages", failed_probe)

    check = _worker_packages_check(python, required=False)

    assert check.status == "error"
    assert check.detail == "ModuleNotFoundError: missing_package"


def test_worker_patch_is_required_only_for_live_runs(tmp_path: Path) -> None:
    assert _worker_patch_check(tmp_path, required=False).status == "optional"
    assert _worker_patch_check(tmp_path, required=True).status == "error"

    source = tmp_path / "third_party/LLM-PySC2/llm_pysc2/agents/llm_pysc2_agent_main.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "elif not self._all_agent_waiting_response_finished():\n"
        "    func_id, func_call = (0, actions.FUNCTIONS.no_op())\n"
        "    return func_call\n"
        "elif not self._all_agent_executing_finished():\n"
        "    pass\n"
        "return translator settlement no_op\n"
        "if tag not in self.unit_uid_disappear:\n"
        "agent.curr_action_name != 'No_Operation'\n"
        "wait for confirmed disappearance\n"
        "keep the action pending\n"
        "agent.last_execution_abort = {\n"
        "'failure_code': 'actor_not_available'\n"
        "team head unit is unavailable before action translation\n"
        "except FileExistsError:\n"
        "llm_pysc2_global_log_id = max(llm_pysc2_global_log_id, self.log_id)\n",
        encoding="utf-8",
    )

    runner_source = tmp_path / "third_party/LLM-PySC2/pysc2/bin/agent.py"
    runner_source.parent.mkdir(parents=True)
    runner_source.write_text(
        'flags.DEFINE_integer("random_seed", None, "Random seed")\nrandom_seed=FLAGS.random_seed\n',
        encoding="utf-8",
    )
    run_loop_source = tmp_path / "third_party/LLM-PySC2/pysc2/env/run_loop.py"
    run_loop_source.parent.mkdir(parents=True, exist_ok=True)
    run_loop_source.write_text(
        'on_episode_truncated = getattr(agent, "on_episode_truncated", None)\n'
        "on_episode_truncated(total_frames)\n",
        encoding="utf-8",
    )

    action_source = tmp_path / "third_party/LLM-PySC2/llm_pysc2/lib/llm_action.py"
    action_source.parent.mkdir(parents=True)
    action_source.write_text(
        "feature_screen.power[y0][x0]\n"
        "feature_screen.creep[y0][x0]\n"
        "feature_screen.buildable[y][x]\n"
        "feature_screen.pathable[y][x]\n"
        "feature_screen.player_relative[y][x]\n"
        "if unit.tag == tag:\n"
        "(0, F.no_op, ())\n"
        "if not unit.is_on_screen or not (0 < unit.x < size_screen:\n"
        "is not a neutral resource anchor\n"
        "def full_footprint_valid(center_x, center_y):\n"
        "feature_screen.player_relative[y][x] != 0\n"
        "not buildable for a complete footprint\n"
        "def resource_clearance_score(center_x, center_y, resources):\n"
        "has no complete footprint with valid resource clearance\n"
        "candidates.append((clearance_score, centroid_distance, candidate_x, candidate_y))\n"
        "pixel_scale = size_screen / SCREEN_WORLD_GRID\n"
        "sample_stride = max(1, int(pixel_scale))\n"
        "minimum, ideal, maximum = 7 * pixel_scale, 8.5 * pixel_scale, 10 * pixel_scale\n"
        "minimum, ideal, maximum = 6 * pixel_scale, 7.5 * pixel_scale, 9 * pixel_scale\n"
        "feature_screen.visibility_map[y][x] != features.Visibility.VISIBLE\n"
        "unit.display_type != 1 or not unit.is_on_screen\n",
        encoding="utf-8",
    )
    translation_source = tmp_path / "third_party/LLM-PySC2/llm_pysc2/agents/llm_pysc2_agent.py"
    translation_source.write_text(
        "self.last_translation_result\n"
        "'requested_function_id': requested_function_id\n"
        "'emitted_function_id': func_id\n"
        "'ordinal': translation_ordinal\n"
        "'total': self._rtscortex_translation_total\n"
        "all_args_valid = all_args_valid and func_valid\n"
        "MainAgent confirms unit death across observations\n"
        "team['unit_tags'] = list(dict.fromkeys(team['unit_tags']))\n",
        encoding="utf-8",
    )

    funcs_source = tmp_path / "third_party/LLM-PySC2/llm_pysc2/agents/main_agent_funcs.py"
    funcs_source.write_text(
        "Advance confirmed-death state on every observation\n"
        "tag for tag, steps in self.unit_disappear_steps.items() if steps >= 40\n"
        "tag for tag in agent.unit_tag_list if tag not in self.unit_uid_disappear\n"
        "tag for tag in team['unit_tags'] if tag not in self.unit_uid_disappear\n"
        "self.config.ENABLE_AUTO_WORKER_MANAGE and self.is_all_nexus_full is False\n"
        "_rtscortex_reserved_worker_tags\n"
        "HoldPosition_quick('now')\n"
        "Reserved worker\n",
        encoding="utf-8",
    )

    assert _worker_patch_check(tmp_path, required=True).status == "ok"

    complete_main_source = source.read_text(encoding="utf-8")
    source.write_text(
        complete_main_source.replace("except FileExistsError:\n", ""),
        encoding="utf-8",
    )
    concurrent_log_check = _worker_patch_check(tmp_path, required=True)
    assert concurrent_log_check.status == "error"
    assert "0011-allocate-log-directories-atomically.patch" in concurrent_log_check.detail
    source.write_text(complete_main_source, encoding="utf-8")

    run_loop_source.write_text("", encoding="utf-8")
    truncation_check = _worker_patch_check(tmp_path, required=True)
    assert truncation_check.status == "error"
    assert "0010-report-max-frame-truncation.patch" in truncation_check.detail
    run_loop_source.write_text(
        'on_episode_truncated = getattr(agent, "on_episode_truncated", None)\n'
        "on_episode_truncated(total_frames)\n",
        encoding="utf-8",
    )

    complete_action_source = action_source.read_text(encoding="utf-8")
    action_source.write_text(
        complete_action_source.replace("sample_stride = max(1, int(pixel_scale))\n", ""),
        encoding="utf-8",
    )
    exact_scale_check = _worker_patch_check(tmp_path, required=True)
    assert exact_scale_check.status == "error"
    assert "0009-use-exact-nexus-screen-scale.patch" in exact_scale_check.detail
    action_source.write_text(complete_action_source, encoding="utf-8")

    complete_funcs_source = funcs_source.read_text(encoding="utf-8")
    funcs_source.write_text(
        complete_funcs_source.replace("Advance confirmed-death state on every observation\n", ""),
        encoding="utf-8",
    )
    locked_death_check = _worker_patch_check(tmp_path, required=True)
    assert locked_death_check.status == "error"
    assert "0007-preserve-transient-team-units.patch" in locked_death_check.detail
    funcs_source.write_text(complete_funcs_source, encoding="utf-8")

    funcs_source.write_text(
        complete_funcs_source.replace("_rtscortex_reserved_worker_tags\n", ""),
        encoding="utf-8",
    )
    reserved_worker_check = _worker_patch_check(tmp_path, required=True)
    assert reserved_worker_check.status == "error"
    assert "0013-preserve-reserved-builder-worker.patch" in reserved_worker_check.detail
    funcs_source.write_text(complete_funcs_source, encoding="utf-8")

    complete_main_source = source.read_text(encoding="utf-8")
    source.write_text(
        complete_main_source.replace("agent.curr_action_name != 'No_Operation'\n", ""),
        encoding="utf-8",
    )
    control_noop_check = _worker_patch_check(tmp_path, required=True)
    assert control_noop_check.status == "error"
    assert "0007-preserve-transient-team-units.patch" in control_noop_check.detail
    source.write_text(complete_main_source, encoding="utf-8")

    translation_source.write_text(
        translation_source.read_text(encoding="utf-8").replace(
            "team['unit_tags'] = list(dict.fromkeys(team['unit_tags']))\n", ""
        ),
        encoding="utf-8",
    )
    check = _worker_patch_check(tmp_path, required=True)
    assert check.status == "error"
    assert "0007-preserve-transient-team-units.patch" in check.detail


def test_sc2_checks_executable_and_installed_scenario_map(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_map = tmp_path / "third_party/LLM-PySC2/llm_pysc2/maps/llm_pysc2/pvz_task1_level1.SC2Map"
    source_map.parent.mkdir(parents=True)
    source_map.touch()
    sc2_path = tmp_path / "StarCraftII"
    executable = sc2_path / "Versions/Base92440/SC2_x64"
    executable.parent.mkdir(parents=True)
    executable.touch()
    installed_map = sc2_path / "Maps/llm_pysc2/pvz_task1_level1.SC2Map"
    installed_map.parent.mkdir(parents=True)
    installed_map.touch()
    monkeypatch.setenv("SC2PATH", str(sc2_path))

    checks = _sc2_checks(tmp_path, required=True)

    assert {check.name: check.status for check in checks} == {
        "scenario_map_source": "ok",
        "starcraft_ii": "ok",
        "scenario_map_installed": "ok",
    }


def test_sc2_checks_rejects_build_older_than_fixed_pvz_map(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_map = tmp_path / "third_party/LLM-PySC2/llm_pysc2/maps/llm_pysc2/pvz_task1_level1.SC2Map"
    source_map.parent.mkdir(parents=True)
    source_map.touch()
    sc2_path = tmp_path / "StarCraftII"
    executable = sc2_path / "Versions/Base75689/SC2_x64"
    executable.parent.mkdir(parents=True)
    executable.touch()
    installed_map = sc2_path / "Maps/llm_pysc2/pvz_task1_level1.SC2Map"
    installed_map.parent.mkdir(parents=True)
    installed_map.touch()
    monkeypatch.setenv("SC2PATH", str(sc2_path))

    checks = _sc2_checks(tmp_path, required=True)
    check = next(check for check in checks if check.name == "starcraft_ii")

    assert check.status == "error"
    assert "requires Base92440" in check.detail


def test_sc2_checks_accepts_2s3z_on_base75689(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_map = tmp_path / "third_party/LLM-PySC2/llm_pysc2/maps/llm_smac/2s3z.SC2Map"
    source_map.parent.mkdir(parents=True)
    source_map.touch()
    sc2_path = tmp_path / "StarCraftII"
    executable = sc2_path / "Versions/Base75689/SC2_x64"
    executable.parent.mkdir(parents=True)
    executable.touch()
    installed_map = sc2_path / "Maps/llm_smac/2s3z.SC2Map"
    installed_map.parent.mkdir(parents=True)
    installed_map.touch()
    monkeypatch.setenv("SC2PATH", str(sc2_path))

    checks = _sc2_checks(tmp_path, required=True, scenario="2s3z")

    assert {check.name: check.status for check in checks} == {
        "scenario_map_source": "ok",
        "starcraft_ii": "ok",
        "scenario_map_installed": "ok",
    }


def test_sc2_checks_accept_official_melee_map_without_submodule_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sc2_path = tmp_path / "StarCraftII"
    executable = sc2_path / "Versions/Base75689/SC2_x64"
    executable.parent.mkdir(parents=True)
    executable.touch()
    installed_map = sc2_path / "Maps/Melee/Simple64.SC2Map"
    installed_map.parent.mkdir(parents=True)
    installed_map.touch()
    monkeypatch.setenv("SC2PATH", str(sc2_path))

    checks = _sc2_checks(tmp_path, required=True, scenario="Simple64")

    assert {check.name: check.status for check in checks} == {
        "scenario_map_source": "optional",
        "starcraft_ii": "ok",
        "scenario_map_installed": "ok",
    }
    source_check = next(check for check in checks if check.name == "scenario_map_source")
    assert "official map pack" in source_check.detail


def test_sc2_checks_rejects_incomplete_configured_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SC2PATH", str(tmp_path / "incomplete"))

    checks = _sc2_checks(tmp_path, required=False)

    assert next(check for check in checks if check.name == "starcraft_ii").status == "error"
