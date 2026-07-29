"""Run source-bound deterministic recovery canaries for formal experiments."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

_TESTS = (
    "tests/integration/test_runtime.py::test_runtime_restart_recovers_plan_execution_and_episode_summary",
    "tests/integration/test_runtime.py::test_runtime_recovers_dispatched_transition_without_redispatch",
    "tests/integration/test_cortex_runtime.py::test_recovery_replays_expansion_reopen_after_checkpoint",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-git-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
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
        "format_version": "1.0",
        "created_at": datetime.now(UTC).isoformat(),
        "git_sha": git_sha,
        "expected_git_sha": arguments.expected_git_sha,
        "tests": list(_TESTS),
        "pytest_exit_code": result.returncode,
        "passed": passed,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
