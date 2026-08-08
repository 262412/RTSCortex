"""Build a revision-bound soft baseline and isolated hard-shadow probe database."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

from rtscortex.playbook import (
    PlaybookRule,
    PlaybookRuleCategory,
    PlaybookRuleEffect,
    PlaybookRuleKind,
    PlaybookRuleStatus,
    PlaybookRuleStrength,
    PlaybookRunLearner,
    PlaybookStore,
    evaluation_kind,
)

_SC2_VERSION_PATTERN = re.compile(r"^Version: (B[0-9]+) \(SC2\.([^)]+)\)$", re.MULTILINE)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sc2_version(worker_stderr: Path) -> tuple[str, str]:
    matches = set(_SC2_VERSION_PATTERN.findall(worker_stderr.read_text(encoding="utf-8")))
    if len(matches) != 1:
        raise ValueError(f"run must attest exactly one SC2 build and patch: {worker_stderr}")
    build, patch = next(iter(matches))
    return str(build), str(patch)


def _load_source_rows(
    status_path: Path,
    run_directories: tuple[Path, ...],
    *,
    sc2_patch: str,
) -> tuple[list[dict[str, Any]], str, str, str]:
    rows = list(csv.DictReader(status_path.open(encoding="utf-8"), delimiter="\t"))
    expected_paths = {path.resolve() for path in run_directories}
    observed_paths = {
        Path(row.get("run_dir", "")).expanduser().resolve() for row in rows if row.get("run_dir")
    }
    if len(rows) != 3 or observed_paths != expected_paths:
        raise ValueError("source status must contain exactly the three requested run directories")

    source_entries: list[dict[str, Any]] = []
    builds: set[str] = set()
    source_git_shas: set[str] = set()
    attestation_payloads: set[str] = set()
    for row in rows:
        run_directory = Path(row["run_dir"]).expanduser().resolve()
        gates_path = run_directory / "engineering-gates.json"
        events_path = run_directory / "events.jsonl"
        worker_stderr = run_directory / "worker.stderr.log"
        if not all(path.is_file() for path in (gates_path, events_path, worker_stderr)):
            raise ValueError(f"source run is missing required artifacts: {run_directory}")
        gates = json.loads(gates_path.read_text(encoding="utf-8"))
        evidence = gates.get("evidence", {})
        seed = int(row["seed"])
        source_git_sha = row.get("git_head_before", "")
        checks = {
            "exit_code": row.get("exit_code") == "0",
            "accepted": row.get("accepted") == "true" and gates.get("accepted") is True,
            "diagnostic_only": evidence.get("diagnostic_only") is False,
            "git_sha": (
                row.get("git_head_before")
                == row.get("git_head_after")
                == evidence.get("expected_git_sha")
                and len(source_git_sha) == 40
                and all(character in "0123456789abcdef" for character in source_git_sha)
            ),
            "submodule": (
                bool(row.get("submodule_commit_before"))
                and row.get("submodule_commit_before") == row.get("submodule_commit_after")
            ),
            "reviewed_commit": (
                row.get("reviewed_commit_before")
                == row.get("reviewed_commit_after")
                == row.get("submodule_commit_before")
            ),
            "reviewed_tree": (
                bool(row.get("reviewed_tree_before"))
                and row.get("reviewed_tree_before") == row.get("reviewed_tree_after")
            ),
        }
        failures = [name for name, accepted in checks.items() if not accepted]
        if failures:
            raise ValueError(
                f"source run {run_directory.name} failed attestation: {', '.join(failures)}"
            )
        sc2_build, observed_patch = _sc2_version(worker_stderr)
        if observed_patch != sc2_patch:
            raise ValueError(
                f"source run {run_directory.name} used SC2 {observed_patch}, expected {sc2_patch}"
            )
        builds.add(sc2_build)
        source_git_shas.add(source_git_sha)
        attestation = json.dumps(
            {
                "git_sha": source_git_sha,
                "submodule_commit": row["submodule_commit_before"],
                "reviewed_commit": row["reviewed_commit_before"],
                "reviewed_tree": row["reviewed_tree_before"],
                "sc2_build": sc2_build,
                "sc2_patch": observed_patch,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        attestation_payloads.add(attestation)
        source_entries.append(
            {
                "seed_id": seed,
                "run_id": run_directory.name,
                "run_directory": str(run_directory),
                "events_sha256": _sha256_file(events_path),
                "engineering_gates_sha256": _sha256_file(gates_path),
                "worker_stderr_sha256": _sha256_file(worker_stderr),
            }
        )
    if len(builds) != 1 or len(source_git_shas) != 1 or len(attestation_payloads) != 1:
        raise ValueError("source runs do not share one source and SC2 attestation")
    fingerprint = hashlib.sha256(next(iter(attestation_payloads)).encode()).hexdigest()
    return source_entries, next(iter(builds)), fingerprint, next(iter(source_git_shas))


def _eligible_parent(
    rule: PlaybookRule,
    *,
    source_run_ids: set[str],
    source_seeds: set[int],
) -> bool:
    uncensored_runs = set(rule.source_run_ids) - set(rule.censored_source_run_ids)
    uncensored_seeds = set(rule.source_seeds) - set(rule.censored_source_seeds)
    return (
        rule.status is PlaybookRuleStatus.ACTIVE
        and rule.strength is PlaybookRuleStrength.SOFT
        and rule.effect is PlaybookRuleEffect.AVOID
        and evaluation_kind(rule.category) is PlaybookRuleKind.EXECUTION_GUARD
        and (
            rule.category is not PlaybookRuleCategory.EXECUTION_GUARD
            or rule.retry_guard is not None
        )
        and uncensored_runs == source_run_ids
        and uncensored_seeds == source_seeds
        and len(uncensored_runs) >= 3
        and len(uncensored_seeds) >= 3
        and rule.confidence >= 0.9
        and rule.contradiction_count == 0
        and rule.shadow_state_count >= 48
        and rule.false_block_rate <= 0.01
        and bool(rule.action_names or rule.role_ids)
    )


def prepare_hard_qualification(
    *,
    run_directories: tuple[Path, ...],
    source_status: Path,
    baseline_output: Path,
    probe_output: Path,
    plan_output: Path,
    expected_git_sha: str,
    sc2_patch: str,
    max_probes: int,
) -> dict[str, Any]:
    """Create fresh qualification artifacts without mutating any source run."""

    output_paths = (baseline_output, probe_output, plan_output)
    existing = [str(path) for path in output_paths if path.exists()]
    if existing:
        raise ValueError("hard qualification outputs already exist: " + ", ".join(existing))
    source_entries, sc2_build, source_fingerprint, source_git_sha = _load_source_rows(
        source_status,
        run_directories,
        sc2_patch=sc2_patch,
    )
    baseline_output.parent.mkdir(parents=True, exist_ok=True)
    store = PlaybookStore(baseline_output)
    try:
        learning = PlaybookRunLearner(store).learn(
            run_directories,
            agent_race="protoss",
            opponent_race="zerg",
        )
        learned_run_ids = {episode.run_id for episode in learning.learned_episodes}
        learned_seeds = {episode.seed for episode in learning.learned_episodes}
        parent_candidates = [
            rule
            for rule in store.rules()
            if _eligible_parent(
                rule,
                source_run_ids=learned_run_ids,
                source_seeds=learned_seeds,
            )
        ]
        parent_candidates.sort(key=lambda rule: (-rule.shadow_state_count, rule.rule_id))
        if not parent_candidates:
            raise ValueError(
                "no three-seed active soft execution rule is eligible for qualification"
            )
        if len(parent_candidates) > max_probes:
            raise ValueError(
                f"eligible hard qualification probes exceed max_probes={max_probes}: "
                f"{len(parent_candidates)}"
            )
        source_binding = {
            "kind": "hard_qualification_source_binding",
            "source_git_sha": source_git_sha,
            "qualification_git_sha": expected_git_sha,
            "sc2_build": sc2_build,
            "sc2_patch": sc2_patch,
            "source_attestation_fingerprint": source_fingerprint,
            "source_status_sha256": _sha256_file(source_status),
            "source_runs": source_entries,
        }
        bound_parents = [
            store.upsert_rule(
                parent.model_copy(
                    update={
                        "code_revision": expected_git_sha,
                        "sc2_patch": sc2_patch,
                        "evidence": {
                            **parent.evidence,
                            "hard_qualification_source_binding": source_binding,
                        },
                    }
                )
            )
            for parent in parent_candidates
        ]
    finally:
        store.close()

    baseline_sha256 = _sha256_file(baseline_output)
    shutil.copy2(baseline_output, probe_output)
    probe_store = PlaybookStore(probe_output)
    try:
        for parent in bound_parents:
            probe_store.upsert_rule(
                parent.model_copy(
                    update={
                        "effect": PlaybookRuleEffect.FORBID,
                        "strength": PlaybookRuleStrength.HARD,
                        "evidence": {
                            **parent.evidence,
                            "hard_qualification_probe": {
                                "parent_rule_id": parent.rule_id,
                                "parent_canonical_key": parent.canonical_key,
                                "baseline_sha256": baseline_sha256,
                                "expected_git_sha": expected_git_sha,
                                "sc2_build": sc2_build,
                                "sc2_patch": sc2_patch,
                            },
                        },
                    }
                )
            )
    finally:
        probe_store.close()
    probe_sha256 = _sha256_file(probe_output)
    plan = {
        "schema_version": "1.0",
        "artifact_kind": "playbook-hard-qualification-plan",
        "expected_git_sha": expected_git_sha,
        "source_git_sha": source_git_sha,
        "sc2_build": sc2_build,
        "sc2_patch": sc2_patch,
        "baseline_path": str(baseline_output.resolve()),
        "baseline_sha256": baseline_sha256,
        "probe_baseline_path": str(probe_output.resolve()),
        "probe_baseline_sha256": probe_sha256,
        "parent_rule_ids": [parent.rule_id for parent in bound_parents],
        "source_attestation_fingerprint": source_fingerprint,
        "source_status_path": str(source_status.resolve()),
        "source_status_sha256": _sha256_file(source_status),
        "source_runs": source_entries,
    }
    plan_output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run-dir", type=Path, action="append", required=True)
    parser.add_argument("--source-status", type=Path, required=True)
    parser.add_argument("--baseline-output", type=Path, required=True)
    parser.add_argument("--probe-output", type=Path, required=True)
    parser.add_argument("--plan-output", type=Path, required=True)
    parser.add_argument("--expected-git-sha", required=True)
    parser.add_argument("--sc2-patch", required=True)
    parser.add_argument("--max-probes", type=int, default=8)
    arguments = parser.parse_args()
    plan = prepare_hard_qualification(
        run_directories=tuple(path.resolve() for path in arguments.source_run_dir),
        source_status=arguments.source_status.resolve(),
        baseline_output=arguments.baseline_output.resolve(),
        probe_output=arguments.probe_output.resolve(),
        plan_output=arguments.plan_output.resolve(),
        expected_git_sha=arguments.expected_git_sha,
        sc2_patch=arguments.sc2_patch,
        max_probes=arguments.max_probes,
    )
    print(json.dumps(plan, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
