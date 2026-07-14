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
        "    pass\n",
        encoding="utf-8",
    )

    runner_source = tmp_path / "third_party/LLM-PySC2/pysc2/bin/agent.py"
    runner_source.parent.mkdir(parents=True)
    runner_source.write_text(
        'flags.DEFINE_integer("random_seed", None, "Random seed")\nrandom_seed=FLAGS.random_seed\n',
        encoding="utf-8",
    )

    assert _worker_patch_check(tmp_path, required=True).status == "ok"


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
