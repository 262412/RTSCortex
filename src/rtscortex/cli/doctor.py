"""Environment diagnostics that do not mutate the host."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

EXPECTED_LLM_PYSC2_COMMIT = "551c863475c0c4a96a181080974d24b59589e9f3"


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
    venv = project_root / ".venv"
    checks.append(
        Check(
            "core_venv",
            "ok" if venv.is_symlink() else "error",
            str(venv.resolve()) if venv.exists() else "missing symlink",
        )
    )
    submodule = project_root / "third_party" / "LLM-PySC2"
    commit = _git_commit(submodule)
    checks.append(
        Check(
            "llm_pysc2",
            "ok" if commit == EXPECTED_LLM_PYSC2_COMMIT else "error",
            commit or "submodule missing",
        )
    )
    sc2_path_value = os.environ.get("SC2PATH")
    sc2_path = None if not sc2_path_value else Path(sc2_path_value).expanduser()
    sc2_present = sc2_path is not None and sc2_path.exists()
    checks.append(
        Check(
            "starcraft_ii",
            "ok" if sc2_present else ("error" if require_sc2 else "optional"),
            str(sc2_path) if sc2_present else "SC2PATH is unset or missing",
        )
    )
    return checks


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
