"""Run source-bound deterministic recovery canaries for formal experiments."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from qualification_attempt import (
    attempt_fields,
    create_only_json,
    load_attempt_manifest,
    validate_attempt_provenance,
)

_TESTS = (
    "tests/integration/test_runtime.py::test_runtime_restart_recovers_plan_execution_and_episode_summary",
    "tests/integration/test_runtime.py::test_runtime_recovers_dispatched_transition_without_redispatch",
    "tests/integration/test_cortex_runtime.py::test_recovery_replays_expansion_reopen_after_checkpoint",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-git-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attempt-manifest", type=Path)
    parser.add_argument("--seed-ids", default="0,1,2")
    arguments = parser.parse_args()
    seed_ids = tuple(int(value) for value in arguments.seed_ids.split(",") if value != "")
    if len(seed_ids) != len(set(seed_ids)):
        raise SystemExit("seed IDs must be unique")
    git_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip()
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *_TESTS],
        check=False,
    )
    passed = result.returncode == 0 and git_sha == arguments.expected_git_sha
    artifact = {
        "format_version": "1.1",
        "created_at": datetime.now(UTC).isoformat(),
        "git_sha": git_sha,
        "expected_git_sha": arguments.expected_git_sha,
        "seed_ids": list(seed_ids),
        "tests": list(_TESTS),
        "pytest_exit_code": result.returncode,
        "recovery_evidence_present": passed,
        "checkpoint_tail_recovery_bounded": passed,
        "passed": passed,
    }
    if arguments.attempt_manifest is not None:
        attempt = load_attempt_manifest(arguments.attempt_manifest.parent)
        artifact.update(attempt_fields(attempt))
        artifact["attempt"] = attempt_fields(attempt)
        validate_attempt_provenance(artifact, attempt, artifact_name="recovery evidence")
    create_only_json(arguments.output, artifact)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
