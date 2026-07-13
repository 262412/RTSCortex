from __future__ import annotations

from pathlib import Path

from rtscortex.cli.doctor import run_doctor


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
