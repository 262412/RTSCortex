from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts.prepare_reviewed_llm_pysc2_runtime import prepare_reviewed_source


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_reviewed_runtime_uses_a_clean_gitlink_without_mutating_it(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init", "--quiet", str(source)], check=True)
    _git(source, "config", "user.email", "test@example.com")
    _git(source, "config", "user.name", "RTSCortex Test")
    upstream_file = source / "agent.py"
    upstream_file.write_text("VALUE = 'upstream'\n", encoding="utf-8")
    _git(source, "add", "agent.py")
    _git(source, "commit", "--quiet", "-m", "upstream")
    gitlink = _git(source, "rev-parse", "HEAD")

    upstream_file.write_text("VALUE = 'reviewed'\n", encoding="utf-8")
    patch = subprocess.run(
        ["git", "-C", str(source), "diff", "--binary"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    _git(source, "checkout", "--", "agent.py")
    patch_directory = tmp_path / "patches"
    patch_directory.mkdir()
    (patch_directory / "0001-reviewed.patch").write_text(patch, encoding="utf-8")

    output_root = tmp_path / "reviewed"
    manifest = prepare_reviewed_source(
        source,
        output_root,
        patch_directory,
        expected_gitlink=gitlink,
    )

    target = output_root / "third_party" / "LLM-PySC2"
    assert _git(source, "status", "--porcelain") == ""
    assert _git(target, "rev-parse", "HEAD") == gitlink
    assert (target / "agent.py").read_text(encoding="utf-8") == "VALUE = 'reviewed'\n"
    assert _git(target, "status", "--porcelain") == "M agent.py"
    assert manifest["source_gitlink"] == gitlink
    assert manifest["patch_count"] == 1
    assert (output_root / "reviewed-source.json").is_file()

    with pytest.raises(ValueError, match="already exists"):
        prepare_reviewed_source(
            source,
            output_root,
            patch_directory,
            expected_gitlink=gitlink,
        )


def test_reviewed_runtime_rejects_a_dirty_source(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init", "--quiet", str(source)], check=True)
    _git(source, "config", "user.email", "test@example.com")
    _git(source, "config", "user.name", "RTSCortex Test")
    (source / "agent.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(source, "add", "agent.py")
    _git(source, "commit", "--quiet", "-m", "upstream")
    gitlink = _git(source, "rev-parse", "HEAD")
    (source / "agent.py").write_text("VALUE = 2\n", encoding="utf-8")
    patch_directory = tmp_path / "patches"
    patch_directory.mkdir()
    (patch_directory / "0001-reviewed.patch").write_text("not used\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must be clean"):
        prepare_reviewed_source(
            source,
            tmp_path / "reviewed",
            patch_directory,
            expected_gitlink=gitlink,
        )
