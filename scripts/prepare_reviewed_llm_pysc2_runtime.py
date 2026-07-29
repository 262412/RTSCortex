"""Prepare an isolated LLM-PySC2 source tree with the reviewed patch chain."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


def prepare_reviewed_source(
    source: Path,
    output_root: Path,
    patch_directory: Path,
    *,
    expected_gitlink: str,
) -> dict[str, object]:
    source = source.resolve()
    output_root = output_root.resolve()
    patch_directory = patch_directory.resolve()
    target = output_root / "third_party" / "LLM-PySC2"
    if output_root.exists():
        raise ValueError(f"reviewed source root already exists: {output_root}")
    if _git(source, "rev-parse", "HEAD") != expected_gitlink:
        raise ValueError("LLM-PySC2 source HEAD does not match the expected gitlink")
    if _git(source, "status", "--porcelain"):
        raise ValueError("LLM-PySC2 source must be clean before preparing the reviewed copy")
    patches = tuple(sorted(patch_directory.glob("*.patch")))
    if not patches:
        raise ValueError(f"no reviewed patches found below {patch_directory}")

    target.parent.mkdir(parents=True)
    subprocess.run(
        ["git", "clone", "--quiet", "--shared", "--no-checkout", str(source), str(target)],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(target), "checkout", "--quiet", "--detach", expected_gitlink],
        check=True,
    )
    for patch in patches:
        subprocess.run(
            ["git", "-C", str(target), "apply", "--check", str(patch)],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(target), "apply", str(patch)],
            check=True,
        )

    patch_records = [{"name": patch.name, "sha256": _sha256_file(patch)} for patch in patches]
    patch_set_sha256 = _sha256_json(patch_records)
    reviewed_diff = subprocess.run(
        ["git", "-C", str(target), "diff", "--binary"],
        check=True,
        capture_output=True,
    ).stdout
    reviewed_tree = reviewed_source_tree_manifest(target)
    manifest = {
        "schema_version": "1.1",
        "artifact_kind": "reviewed-llm-pysc2-runtime",
        "source_gitlink": expected_gitlink,
        "patch_count": len(patches),
        "patch_set_sha256": patch_set_sha256,
        "reviewed_diff_sha256": hashlib.sha256(reviewed_diff).hexdigest(),
        **reviewed_tree,
        "patches": patch_records,
    }
    (output_root / "reviewed-source.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_json(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def reviewed_source_tree_manifest(repository: Path) -> dict[str, object]:
    """Hash every runtime-visible file, including ignored and untracked files."""

    repository = repository.resolve()
    entries: list[dict[str, object]] = []
    for current_root, directory_names, file_names in os.walk(
        repository,
        followlinks=False,
    ):
        root = Path(current_root)
        directory_names[:] = sorted(name for name in directory_names if name != ".git")
        for name in tuple(directory_names):
            path = root / name
            if not path.is_symlink():
                continue
            entries.append(_tree_entry(repository, path))
            directory_names.remove(name)
        for name in sorted(file_names):
            path = root / name
            if path.name == ".git":
                continue
            entries.append(_tree_entry(repository, path))
    entries.sort(key=lambda entry: str(entry["path"]))
    untracked = subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        check=True,
        capture_output=True,
    ).stdout.split(b"\0")
    untracked_paths = sorted(
        path.decode("utf-8", errors="surrogateescape") for path in untracked if path
    )
    return {
        "reviewed_tree_sha256": _sha256_json(entries),
        "reviewed_tree_file_count": len(entries),
        "reviewed_untracked_file_count": len(untracked_paths),
        "reviewed_untracked_paths": untracked_paths,
    }


def _tree_entry(repository: Path, path: Path) -> dict[str, object]:
    relative = path.relative_to(repository).as_posix()
    if path.is_symlink():
        target = os.readlink(path)
        return {
            "path": relative,
            "kind": "symlink",
            "size": len(target.encode()),
            "sha256": hashlib.sha256(target.encode()).hexdigest(),
        }
    content = path.read_bytes()
    return {
        "path": relative,
        "kind": "file",
        "size": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--patch-directory", type=Path, required=True)
    parser.add_argument("--expected-gitlink", required=True)
    arguments = parser.parse_args()
    manifest = prepare_reviewed_source(
        arguments.source,
        arguments.output_root,
        arguments.patch_directory,
        expected_gitlink=arguments.expected_gitlink,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
