"""Analyze natural-terminal hard-shadow probes and emit one typed manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import yaml

from rtscortex.memory import read_event_log
from rtscortex.playbook import (
    PlaybookHardQualificationRunEvidence,
    PlaybookStore,
    analyze_hard_qualification_evaluations,
    build_hard_qualification_manifest,
)
from scripts.analyze_playbook_experiment import _run_metrics

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
        return "unknown", "unknown"
    return next(iter(matches))


def _shadow_config_is_valid(config_path: Path, *, expected_database: Path) -> bool:
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        playbook = payload["cortex"]["playbook"]
        observed_database = Path(playbook["database_path"]).expanduser().resolve()
    except (OSError, KeyError, TypeError, ValueError):
        return False
    return (
        playbook.get("enabled") is True
        and playbook.get("learning_mode") == "frozen"
        and playbook.get("rule_mode") == "shadow"
        and playbook.get("hard_readiness_required") is False
        and observed_database == expected_database.resolve()
    )


def analyze_hard_qualification(
    *,
    plan_path: Path,
    baseline_path: Path,
    probe_path: Path,
    status_path: Path,
    engineering_baseline_path: Path,
    recovery_evidence_path: Path,
    expected_git_sha: str,
    sc2_patch: str,
    working_database: Path,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan_checks = {
        "schema": plan.get("schema_version") == "1.0",
        "kind": plan.get("artifact_kind") == "playbook-hard-qualification-plan",
        "git_sha": plan.get("expected_git_sha") == expected_git_sha,
        "sc2_patch": plan.get("sc2_patch") == sc2_patch,
        "baseline_path": Path(str(plan.get("baseline_path", ""))).resolve()
        == baseline_path.resolve(),
        "baseline_sha256": plan.get("baseline_sha256") == _sha256_file(baseline_path),
        "probe_path": Path(str(plan.get("probe_baseline_path", ""))).resolve()
        == probe_path.resolve(),
        "probe_sha256": plan.get("probe_baseline_sha256") == _sha256_file(probe_path),
    }
    failed_plan_checks = [name for name, accepted in plan_checks.items() if not accepted]
    if failed_plan_checks:
        raise ValueError("hard qualification plan mismatch: " + ", ".join(failed_plan_checks))

    engineering_baseline = json.loads(engineering_baseline_path.read_text(encoding="utf-8"))
    recovery_evidence = json.loads(recovery_evidence_path.read_text(encoding="utf-8"))
    rows = list(csv.DictReader(status_path.open(encoding="utf-8"), delimiter="\t"))
    expected_seeds = {int(entry["seed_id"]) for entry in plan["source_runs"]}
    observed_seeds = {int(row["seed"]) for row in rows}
    if len(rows) != 3 or observed_seeds != expected_seeds:
        raise ValueError("qualification status must contain one row for every source seed")

    common_run_evidence: dict[int, dict[str, Any]] = {}
    events_by_seed: dict[int, tuple[Any, ...]] = {}
    run_reports: list[dict[str, Any]] = []
    for row in rows:
        seed = int(row["seed"])
        run_directory = Path(row["run_dir"]).expanduser().resolve()
        events_path = run_directory / "events.jsonl"
        gates_path = run_directory / "engineering-gates.json"
        worker_stderr = run_directory / "worker.stderr.log"
        required_paths = (events_path, gates_path, worker_stderr, run_directory / "config.yaml")
        if not all(path.is_file() for path in required_paths):
            raise ValueError(f"qualification run is missing required artifacts: {run_directory}")
        run_metrics = _run_metrics(
            row,
            natural_run_baseline_bytes_per_loop=float(
                engineering_baseline["natural_run_bytes_per_game_loop"]
            ),
            expected_git_sha=expected_git_sha,
            recovery_evidence=recovery_evidence,
        )
        sc2_build, observed_patch = _sc2_version(worker_stderr)
        config_valid = _shadow_config_is_valid(
            run_directory / "config.yaml",
            expected_database=working_database,
        )
        source_valid = (
            run_metrics.source_tree_clean
            and run_metrics.source_commit_matches_expected_sha
            and run_metrics.source_attestation_consistent
            and run_metrics.source_attestation_fingerprint is not None
        )
        immutable_probe = (
            row.get("playbook_before_sha256") == plan["probe_baseline_sha256"]
            and row.get("playbook_after_sha256") == plan["probe_baseline_sha256"]
        )
        run_accepted = (
            int(row["exit_code"]) == 0
            and run_metrics.natural_terminal
            and run_metrics.engineering_accepted
            and source_valid
            and config_valid
            and immutable_probe
            and sc2_build == plan["sc2_build"]
            and observed_patch == sc2_patch
        )
        events = tuple(read_event_log(events_path))
        events_by_seed[seed] = events
        common_run_evidence[seed] = {
            "seed_id": seed,
            "run_id": run_directory.name,
            "run_directory": str(run_directory),
            "events_sha256": _sha256_file(events_path),
            "engineering_gates_sha256": _sha256_file(gates_path),
            "worker_stderr_sha256": _sha256_file(worker_stderr),
            "playbook_before_sha256": row.get("playbook_before_sha256", ""),
            "playbook_after_sha256": row.get("playbook_after_sha256", ""),
            "git_sha": expected_git_sha,
            "source_attestation_fingerprint": (
                run_metrics.source_attestation_fingerprint or "0" * 64
            ),
            "sc2_build": sc2_build,
            "sc2_patch": observed_patch,
            "natural_terminal": run_metrics.natural_terminal,
            "engineering_accepted": run_accepted,
            "analysis_evidence_overflow_count": run_metrics.analysis_evidence_overflow_count,
        }
        run_reports.append(
            {
                "seed_id": seed,
                "run_id": run_directory.name,
                "run_directory": str(run_directory),
                "accepted": run_accepted,
                "natural_terminal": run_metrics.natural_terminal,
                "engineering_accepted": run_metrics.engineering_accepted,
                "source_attestation_consistent": source_valid,
                "shadow_config_valid": config_valid,
                "probe_baseline_immutable": immutable_probe,
                "sc2_build": sc2_build,
                "sc2_patch": observed_patch,
                "analysis_evidence_overflow_count": (run_metrics.analysis_evidence_overflow_count),
            }
        )

    store = PlaybookStore(baseline_path, read_only=True)
    try:
        rules_by_id = {rule.rule_id: rule for rule in store.rules()}
    finally:
        store.close()
    accepted_manifests: list[Any] = []
    parent_reports: list[dict[str, Any]] = []
    for parent_rule_id in plan["parent_rule_ids"]:
        parent = rules_by_id.get(parent_rule_id)
        if parent is None:
            parent_reports.append(
                {"parent_rule_id": parent_rule_id, "accepted": False, "reason": "missing_parent"}
            )
            continue
        run_evidence: list[PlaybookHardQualificationRunEvidence] = []
        counterfactual_by_seed: dict[str, dict[str, Any]] = {}
        for seed in sorted(events_by_seed):
            observed = analyze_hard_qualification_evaluations(
                events_by_seed[seed],
                rule_id=parent_rule_id,
            )
            counterfactual_by_seed[str(seed)] = {
                "shadow_would_block_application_count": (
                    observed.shadow_would_block_application_count
                ),
                "resolved_counterfactual_count": observed.resolved_counterfactual_count,
                "unresolved_counterfactual_count": observed.unresolved_counterfactual_count,
                "execution_false_block_count": observed.execution_false_block_count,
                "execution_false_block_rate": observed.execution_false_block_rate,
                "invalid_counterfactual_evidence_count": (
                    observed.invalid_counterfactual_evidence_count
                ),
            }
            run_evidence.append(
                PlaybookHardQualificationRunEvidence(
                    **common_run_evidence[seed],
                    invalid_counterfactual_evidence_count=(
                        observed.invalid_counterfactual_evidence_count
                    ),
                    shadow_would_block_application_count=(
                        observed.shadow_would_block_application_count
                    ),
                    resolved_counterfactual_count=observed.resolved_counterfactual_count,
                    unresolved_counterfactual_count=observed.unresolved_counterfactual_count,
                    execution_false_block_count=observed.execution_false_block_count,
                )
            )
        try:
            manifest = build_hard_qualification_manifest(
                parent=parent,
                baseline_sha256=plan["baseline_sha256"],
                probe_baseline_sha256=plan["probe_baseline_sha256"],
                git_sha=expected_git_sha,
                sc2_patch=sc2_patch,
                runs=run_evidence,
            )
        except ValueError as error:
            parent_reports.append(
                {
                    "parent_rule_id": parent_rule_id,
                    "accepted": False,
                    "reason": str(error),
                    "counterfactual_by_seed": counterfactual_by_seed,
                }
            )
            continue
        accepted_manifests.append(manifest)
        parent_reports.append(
            {
                "parent_rule_id": parent_rule_id,
                "accepted": True,
                "counterfactual_resolved_count": manifest.counterfactual_resolved_count,
                "counterfactual_false_block_rate": manifest.counterfactual_false_block_rate,
                "counterfactual_by_seed": counterfactual_by_seed,
            }
        )

    accepted_manifests.sort(
        key=lambda manifest: (
            manifest.counterfactual_false_block_rate,
            -manifest.counterfactual_resolved_count,
            manifest.parent_rule_id,
        )
    )
    selected = accepted_manifests[0] if accepted_manifests else None
    report = {
        "schema_version": "1.0",
        "artifact_kind": "playbook-hard-qualification-report",
        "expected_git_sha": expected_git_sha,
        "sc2_patch": sc2_patch,
        "baseline_sha256": plan["baseline_sha256"],
        "probe_baseline_sha256": plan["probe_baseline_sha256"],
        "runs": sorted(run_reports, key=lambda item: item["seed_id"]),
        "parents": parent_reports,
        "accepted_parent_rule_ids": [manifest.parent_rule_id for manifest in accepted_manifests],
        "selected_parent_rule_id": None if selected is None else selected.parent_rule_id,
        "accepted": selected is not None,
    }
    return report, None if selected is None else selected.model_dump(mode="json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--engineering-baseline", type=Path, required=True)
    parser.add_argument("--recovery-evidence", type=Path, required=True)
    parser.add_argument("--expected-git-sha", required=True)
    parser.add_argument("--sc2-patch", required=True)
    parser.add_argument("--working-database", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    arguments = parser.parse_args()
    report, manifest = analyze_hard_qualification(
        plan_path=arguments.plan.resolve(),
        baseline_path=arguments.baseline.resolve(),
        probe_path=arguments.probe.resolve(),
        status_path=arguments.status.resolve(),
        engineering_baseline_path=arguments.engineering_baseline.resolve(),
        recovery_evidence_path=arguments.recovery_evidence.resolve(),
        expected_git_sha=arguments.expected_git_sha,
        sc2_patch=arguments.sc2_patch,
        working_database=arguments.working_database.resolve(),
    )
    arguments.report_output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if manifest is None:
        raise SystemExit(1)
    arguments.manifest_output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
