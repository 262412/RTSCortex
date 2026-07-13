"""Environment diagnostics that do not mutate the host."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from rtscortex.runtime.live import (
    random_seed_patch_is_applied,
    waiting_response_patch_is_applied,
)

EXPECTED_LLM_PYSC2_COMMIT = "551c863475c0c4a96a181080974d24b59589e9f3"
SCENARIO_MAP = "pvz_task1_level1.SC2Map"
SCENARIO_MAP_RELATIVE = Path("llm_pysc2") / SCENARIO_MAP


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str


def run_doctor(project_root: Path, *, require_sc2: bool = False) -> list[Check]:
    checks = [
        Check(
            "python",
            "ok" if sys.version_info >= (3, 11) else "error",
            sys.version.split()[0],
        ),
        Check("uv", "ok" if shutil.which("uv") else "error", shutil.which("uv") or "missing"),
    ]
    checks.append(_core_venv_check(project_root / ".venv"))
    submodule = project_root / "third_party" / "LLM-PySC2"
    commit = _git_commit(submodule)
    checks.append(
        Check(
            "llm_pysc2",
            "ok" if commit == EXPECTED_LLM_PYSC2_COMMIT else "error",
            commit or "submodule missing",
        )
    )
    worker_python_value = os.environ.get("RTSCORTEX_LLM_PYSC2_PYTHON")
    worker_python = (
        Path(worker_python_value).expanduser()
        if worker_python_value
        else Path.home() / "fastscratch/envs/rtscortex-llm-pysc2/bin/python"
    )
    checks.append(_worker_python_check(worker_python, required=require_sc2))
    checks.append(_worker_packages_check(worker_python, required=require_sc2))
    checks.append(_worker_patch_check(project_root, required=require_sc2))
    checks.extend(_sc2_checks(project_root, required=require_sc2))
    checks.append(_socket_parent_check())
    return checks


def _core_venv_check(venv: Path) -> Check:
    python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    valid = (venv / "pyvenv.cfg").is_file() and python.is_file()
    if valid:
        layout = "symlink" if venv.is_symlink() else "directory"
        detail = f"{venv.resolve()} ({layout})"
    else:
        detail = (
            f"{venv.resolve()} (invalid virtual environment)"
            if venv.exists()
            else "missing virtual environment"
        )
    return Check("core_venv", "ok" if valid else "error", detail)


def _git_commit(path: Path) -> str | None:
    if not path.exists():
        return None
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _worker_python_check(python: Path, *, required: bool) -> Check:
    if not python.is_file():
        status = "error" if required else "optional"
        return Check("worker_python", status, f"missing: {python}")
    completed = subprocess.run(
        [str(python), "--version"],
        check=False,
        capture_output=True,
        text=True,
    )
    version = (completed.stdout or completed.stderr).strip()
    valid = completed.returncode == 0 and version.startswith("Python 3.9.")
    return Check(
        "worker_python",
        "ok" if valid else "error",
        f"{python} ({version or 'version check failed'})",
    )


def _worker_packages_check(python: Path, *, required: bool) -> Check:
    if not python.is_file():
        status = "error" if required else "optional"
        return Check("worker_packages", status, "worker Python is missing")
    return_code, output = _probe_worker_packages(str(python))
    if return_code == 0:
        return Check("worker_packages", "ok", output)
    return Check(
        "worker_packages",
        "error",
        output or "failed to import bridge and LLM-PySC2 packages",
    )


def _worker_patch_check(project_root: Path, *, required: bool) -> Check:
    missing = []
    if not waiting_response_patch_is_applied(project_root):
        missing.append("0001-return-noop-while-awaiting-runtime.patch")
    if not random_seed_patch_is_applied(project_root):
        missing.append("0002-pass-random-seed-to-sc2env.patch")
    status = "ok" if not missing else ("error" if required else "optional")
    detail = "all worker patches applied" if not missing else "apply " + ", ".join(missing)
    return Check("worker_patch", status, detail)


@lru_cache(maxsize=4)
def _probe_worker_packages(python: str) -> tuple[int, str]:
    probe = (
        "from importlib.metadata import version; "
        "import llm_pysc2.agents, pysc2, rtscortex_llm_pysc2; "
        "print('bridge=' + version('rtscortex-llm-pysc2') + "
        "' llm-pysc2=' + version('llm-pysc2') + "
        "' protobuf=' + version('protobuf'))"
    )
    environment = dict(os.environ)
    environment["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
    completed = subprocess.run(
        [python, "-c", probe],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    output = completed.stdout.strip() if completed.returncode == 0 else completed.stderr.strip()
    return completed.returncode, output.splitlines()[-1] if output else ""


def _sc2_checks(project_root: Path, *, required: bool) -> list[Check]:
    source_map = project_root / "third_party/LLM-PySC2/llm_pysc2/maps" / SCENARIO_MAP_RELATIVE
    source_map_status = "ok" if source_map.is_file() else "error"
    checks = [Check("scenario_map_source", source_map_status, str(source_map))]

    sc2_path_value = os.environ.get("SC2PATH")
    if not sc2_path_value:
        status = "error" if required else "optional"
        checks.extend(
            [
                Check("starcraft_ii", status, "SC2PATH is unset"),
                Check("scenario_map_installed", status, "SC2PATH is unset"),
            ]
        )
        return checks

    sc2_path = Path(sc2_path_value).expanduser()
    executables = sorted((sc2_path / "Versions").glob("Base*/SC2_x64"))
    if (sc2_path / "SC2_x64").is_file():
        executables.append(sc2_path / "SC2_x64")
    if executables:
        executable = executables[-1]
        version = executable.parent.name if executable.parent.name.startswith("Base") else "unknown"
        checks.append(Check("starcraft_ii", "ok", f"{executable} ({version})"))
    else:
        checks.append(Check("starcraft_ii", "error", f"no SC2_x64 below {sc2_path}"))

    installed_map = sc2_path / "Maps" / SCENARIO_MAP_RELATIVE
    checks.append(
        Check(
            "scenario_map_installed",
            "ok" if installed_map.is_file() else "error",
            str(installed_map),
        )
    )
    return checks


def _socket_parent_check() -> Check:
    if os.name == "nt":
        return Check("runtime_socket", "optional", "Unix sockets are not used on Windows")
    runtime_value = os.environ.get("RTSCORTEX_RUNTIME_ROOT")
    runtime_root = (
        Path(runtime_value).expanduser()
        if runtime_value
        else Path.home() / "fastscratch/rtscortex_runtime"
    )
    existing_parent = runtime_root
    while not existing_parent.exists() and existing_parent != existing_parent.parent:
        existing_parent = existing_parent.parent
    writable = existing_parent.is_dir() and os.access(existing_parent, os.W_OK | os.X_OK)
    return Check(
        "runtime_socket",
        "ok" if writable else "error",
        f"{runtime_root} (parent: {existing_parent})",
    )
