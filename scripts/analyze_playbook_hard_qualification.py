"""Analyze natural-terminal hard-shadow probes and emit one typed manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TypeVar

import yaml

from rtscortex.memory import read_event_log
from rtscortex.playbook import (
    PlaybookHardQualificationRunEvidence,
    PlaybookRule,
    PlaybookRuleEffect,
    PlaybookRuleStatus,
    PlaybookRuleStrength,
    PlaybookStore,
    analyze_hard_qualification_evaluations,
    analyze_typed_retry_coverage,
    build_hard_qualification_manifest,
    playbook_predicate_fingerprint,
    playbook_rule_fingerprint,
)
from rtscortex.playbook.selection import runtime_hard_rule_candidates
from scripts.analyze_playbook_experiment import _run_metrics

_SC2_VERSION_PATTERN = re.compile(r"^Version: (B[0-9]+) \(SC2\.([^)]+)\)$", re.MULTILINE)
_BatchValue = TypeVar("_BatchValue")


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
    build, patch = next(iter(matches))
    return str(build), str(patch)


def _shadow_config_is_valid(config_path: Path, *, expected_database: Path) -> bool:
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        playbook = payload["cortex"]["playbook"]
        observed_database = Path(playbook["database_path"]).expanduser().resolve()
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError):
        return False
    return (
        playbook.get("enabled") is True
        and playbook.get("learning_mode") == "frozen"
        and playbook.get("rule_mode") == "shadow"
        and playbook.get("hard_readiness_required") is False
        and observed_database == expected_database.resolve()
    )


def _runtime_context_from_config(config_path: Path) -> dict[str, object]:
    """Return only immutable runtime context values needed by guard replay."""

    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        environment = payload["environment"]
        if not isinstance(environment, dict):
            return {}
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError):
        return {}
    context: dict[str, object] = {}
    for field in ("agent_race", "opponent_race", "scenario"):
        value = environment.get(field)
        if isinstance(value, str) and value:
            context["map_name" if field == "scenario" else field] = value
    return context


def _qualification_config_fingerprint(config_path: Path) -> str | None:
    """Hash the generated qualification config independent of its seed."""

    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, yaml.YAMLError):
        return None
    if not isinstance(payload, dict):
        return None
    run = payload.get("run")
    if not isinstance(run, dict) or not isinstance(run.get("seed"), int):
        return None
    normalized = dict(payload)
    normalized_run = dict(run)
    normalized_run.pop("seed", None)
    normalized["run"] = normalized_run
    try:
        serialized = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return None
    return hashlib.sha256(serialized.encode()).hexdigest()


def _index_batch_status_rows(
    plan: Mapping[str, Any],
    rows: Sequence[Mapping[str, str]],
) -> dict[tuple[str, int], dict[str, str]]:
    """Validate and index the exact planned batch-by-seed run matrix."""

    batches = plan.get("batches")
    source_runs = plan.get("source_runs")
    if not isinstance(batches, list) or not batches:
        raise ValueError("hard qualification plan has no probe batches")
    if not isinstance(source_runs, list) or not source_runs:
        raise ValueError("hard qualification plan has no source seeds")
    batch_by_id = {str(batch["batch_id"]): batch for batch in batches if isinstance(batch, dict)}
    if len(batch_by_id) != len(batches):
        raise ValueError("hard qualification plan contains duplicate batch IDs")
    expected_seeds = tuple(sorted(int(source["seed_id"]) for source in source_runs))
    if expected_seeds != (0, 1, 2):
        raise ValueError("hard qualification plan must cover source seeds 0/1/2 exactly once")
    expected_keys = [
        (str(batch["batch_id"]), seed)
        for batch in batches
        if isinstance(batch, dict)
        for seed in expected_seeds
    ]
    expected_git_sha = str(plan.get("expected_git_sha", ""))
    source_attestation = plan.get("source_attestation")
    if not isinstance(source_attestation, dict):
        raise ValueError("hard qualification plan is missing source attestation")
    indexed: dict[tuple[str, int], dict[str, str]] = {}
    for row in rows:
        batch_id = row.get("batch_id", "")
        batch = batch_by_id.get(batch_id)
        if batch is None:
            raise ValueError(f"unknown batch ID in status: {batch_id or '<missing>'}")
        try:
            seed = int(row.get("seed", ""))
            batch_index = int(row.get("batch_index", ""))
        except ValueError as error:
            raise ValueError("invalid batch index or seed in status") from error
        if batch_index != int(batch["batch_index"]):
            raise ValueError(f"batch index mismatch for {batch_id}")
        key = (batch_id, seed)
        if key in indexed:
            raise ValueError(f"duplicate batch-seed status: {batch_id}:{seed}")
        if key not in expected_keys:
            raise ValueError(f"unexpected batch-seed status: {batch_id}:{seed}")
        if (
            Path(row.get("probe_path", "")).expanduser().resolve()
            != Path(str(batch["probe_baseline_path"])).resolve()
        ):
            raise ValueError(f"probe path mismatch for {batch_id}:{seed}")
        if row.get("probe_sha256") != batch.get("probe_baseline_sha256"):
            raise ValueError(f"probe SHA mismatch for {batch_id}:{seed}")
        if row.get("exit_code") != "0":
            raise ValueError(f"nonzero batch-seed status: {batch_id}:{seed}")
        if not (
            row.get("git_head_before") == row.get("git_head_after") == expected_git_sha
            and row.get("superproject_dirty_before") == "false"
            and row.get("superproject_dirty_after") == "false"
            and row.get("submodule_dirty_before") == "false"
            and row.get("submodule_dirty_after") == "false"
        ):
            raise ValueError(f"Git attestation mismatch for {batch_id}:{seed}")
        expected_submodule = str(source_attestation.get("submodule_commit", ""))
        expected_reviewed_commit = str(
            source_attestation.get("reviewed_commit", expected_submodule)
        )
        expected_reviewed_tree = str(
            source_attestation.get(
                "reviewed_tree_sha256",
                source_attestation.get("reviewed_tree", ""),
            )
        )
        submodule_values = {
            row.get("submodule_commit_before"),
            row.get("submodule_commit_after"),
            row.get("submodule_gitlink_before"),
            row.get("submodule_gitlink_after"),
        }
        if submodule_values != {expected_submodule}:
            raise ValueError(f"submodule attestation mismatch for {batch_id}:{seed}")
        if (
            len(
                {
                    row.get("submodule_diff_sha256_before"),
                    row.get("submodule_diff_sha256_after"),
                }
            )
            != 1
        ):
            raise ValueError(f"submodule diff attestation mismatch for {batch_id}:{seed}")
        if {
            row.get("reviewed_source_commit_before"),
            row.get("reviewed_source_commit_after"),
        } != {expected_reviewed_commit}:
            raise ValueError(f"reviewed commit attestation mismatch for {batch_id}:{seed}")
        if {
            row.get("reviewed_source_tree_sha256_before"),
            row.get("reviewed_source_tree_sha256_after"),
        } != {expected_reviewed_tree}:
            raise ValueError(f"reviewed tree attestation mismatch for {batch_id}:{seed}")
        if (
            len(
                {
                    row.get("reviewed_source_diff_sha256_before"),
                    row.get("reviewed_source_diff_sha256_after"),
                }
            )
            != 1
        ):
            raise ValueError(f"reviewed diff attestation mismatch for {batch_id}:{seed}")
        indexed[key] = dict(row)
    missing = [key for key in expected_keys if key not in indexed]
    if missing:
        labels = [f"{batch_id}:{seed}" for batch_id, seed in missing]
        raise ValueError("missing batch-seed status: " + ", ".join(labels))
    if len(indexed) != len(expected_keys):
        raise ValueError("extra batch-seed status rows")
    return indexed


def _values_for_parent_batch(
    parent_rule_id: str,
    *,
    batch_id_by_parent: Mapping[str, str],
    values_by_batch_seed: Mapping[tuple[str, int], _BatchValue],
    expected_seeds: Sequence[int],
) -> dict[int, _BatchValue]:
    batch_id = batch_id_by_parent.get(parent_rule_id)
    if batch_id is None:
        raise ValueError(f"eligible parent has no probe batch: {parent_rule_id}")
    selected: dict[int, _BatchValue] = {}
    for seed in expected_seeds:
        key = (batch_id, int(seed))
        if key not in values_by_batch_seed:
            raise ValueError(f"missing parent batch evidence: {parent_rule_id}:{batch_id}:{seed}")
        selected[int(seed)] = values_by_batch_seed[key]
    return selected


def _require_complete_batch_run_audit(run_reports: Sequence[Mapping[str, Any]]) -> None:
    """Reject the whole qualification when any planned run fails common gates."""

    rejected = [
        f"{report.get('batch_id', '<missing>')}:{report.get('seed_id', '<missing>')}"
        for report in run_reports
        if report.get("accepted") is not True
    ]
    if rejected:
        raise ValueError("hard qualification batch/run integrity failed: " + ", ".join(rejected))


def _select_global_manifest(candidates: Sequence[Any]) -> tuple[Any | None, list[Any]]:
    ordered = sorted(
        candidates,
        key=lambda manifest: (
            manifest.counterfactual_false_block_rate,
            -manifest.counterfactual_resolved_count,
            manifest.parent_rule_id,
        ),
    )
    return (ordered[0] if ordered else None), ordered


def build_incomplete_hard_qualification_report(
    plan: Mapping[str, Any],
    rows: Sequence[Mapping[str, str]],
    *,
    reason: str,
    runner_exit_code: int,
) -> dict[str, Any]:
    """Describe a terminal runner failure without producing acceptance evidence."""

    batches = [batch for batch in plan.get("batches", []) if isinstance(batch, dict)]
    expected_seeds = tuple(sorted(int(source["seed_id"]) for source in plan.get("source_runs", [])))
    expected_keys = [(str(batch["batch_id"]), seed) for batch in batches for seed in expected_seeds]
    observed: dict[tuple[str, int], Mapping[str, str]] = {}
    unexpected: list[str] = []
    for row in rows:
        try:
            key = (row.get("batch_id", ""), int(row.get("seed", "")))
        except ValueError:
            unexpected.append(f"{row.get('batch_id', '<missing>')}:<invalid>")
            continue
        if key in observed or key not in expected_keys:
            unexpected.append(f"{key[0]}:{key[1]}")
            continue
        observed[key] = row
    completed: list[str] = []
    failed: list[str] = []
    for batch_id, seed in expected_keys:
        observed_row = observed.get((batch_id, seed))
        if observed_row is None:
            continue
        label = f"{batch_id}:{seed}"
        if observed_row.get("exit_code") == "0":
            completed.append(label)
        else:
            failed.append(label)
    unrun = [
        f"{batch_id}:{seed}" for batch_id, seed in expected_keys if (batch_id, seed) not in observed
    ]
    batch_id_by_parent = {
        str(parent_rule_id): str(batch["batch_id"])
        for batch in batches
        for parent_rule_id in batch.get("parent_rule_ids", [])
    }
    eligible_parent_ids = [str(value) for value in plan.get("eligible_parent_ids", [])]
    parents = [
        {
            "parent_rule_id": parent_rule_id,
            "batch_id": batch_id_by_parent.get(parent_rule_id),
            "accepted": False,
            "selection_status": "unrun",
            "reason": "hard_qualification_batch_audit_incomplete",
        }
        for parent_rule_id in eligible_parent_ids
    ]
    return {
        "schema_version": "2.0",
        "artifact_kind": "playbook-hard-qualification-report",
        "expected_git_sha": plan.get("expected_git_sha"),
        "source_git_sha": plan.get("source_git_sha"),
        "accepted": False,
        "batch_audit_complete": False,
        "runner_failure_reason": reason,
        "runner_exit_code": runner_exit_code,
        "eligible_parent_count": len(eligible_parent_ids),
        "batches": batches,
        "completed_batch_seed_runs": completed,
        "failed_batch_seed_runs": failed,
        "unrun_batch_seed_runs": unrun,
        "unexpected_batch_seed_runs": unexpected,
        "runs": [dict(row) for row in rows],
        "parents": parents,
        "accepted_parent_rule_ids": [],
        "selected_parent_rule_id": None,
        "selected_batch_id": None,
        "selected_probe_baseline_sha256": None,
    }


def _validate_batch_plan(
    plan: Mapping[str, Any],
    *,
    plan_path: Path,
    baseline_path: Path,
    expected_git_sha: str,
    sc2_patch: str,
) -> tuple[
    list[dict[str, Any]],
    dict[str, str],
    dict[str, dict[str, PlaybookRule]],
]:
    """Validate plan identity, partition integrity, and every probe database."""

    checks = {
        "schema": plan.get("schema_version") == "2.0",
        "kind": plan.get("artifact_kind") == "playbook-hard-qualification-plan",
        "git_sha": plan.get("expected_git_sha") == expected_git_sha,
        "sc2_patch": plan.get("sc2_patch") == sc2_patch,
        "baseline_path": Path(str(plan.get("baseline_path", ""))).resolve()
        == baseline_path.resolve(),
        "baseline_sha256": baseline_path.is_file()
        and plan.get("baseline_sha256") == _sha256_file(baseline_path),
        "source_status": Path(str(plan.get("source_status_path", ""))).is_file(),
    }
    source_status_path = Path(str(plan.get("source_status_path", "")))
    if source_status_path.is_file():
        checks["source_status_sha256"] = plan.get("source_status_sha256") == _sha256_file(
            source_status_path
        )
    failed = [name for name, accepted in checks.items() if not accepted]
    if failed:
        raise ValueError("hard qualification plan mismatch: " + ", ".join(failed))
    if plan_path.resolve() == baseline_path.resolve():
        raise ValueError("hard qualification plan and baseline paths must differ")

    eligible_parent_ids_raw = plan.get("eligible_parent_ids")
    eligible_parent_records = plan.get("eligible_parents")
    batches_raw = plan.get("batches")
    if not isinstance(eligible_parent_ids_raw, list) or not eligible_parent_ids_raw:
        raise ValueError("hard qualification plan has no eligible parents")
    eligible_parent_ids = [str(parent_rule_id) for parent_rule_id in eligible_parent_ids_raw]
    if len(eligible_parent_ids) != len(set(eligible_parent_ids)):
        raise ValueError("hard qualification plan repeats eligible parent IDs")
    if not isinstance(eligible_parent_records, list) or len(eligible_parent_records) != len(
        eligible_parent_ids
    ):
        raise ValueError("hard qualification plan has incomplete eligible parent records")
    recorded_ids = [
        str(record.get("parent_rule_id", ""))
        for record in eligible_parent_records
        if isinstance(record, dict)
    ]
    recorded_ranks = [
        record.get("global_rank") for record in eligible_parent_records if isinstance(record, dict)
    ]
    if recorded_ids != eligible_parent_ids or recorded_ranks != list(
        range(1, len(eligible_parent_ids) + 1)
    ):
        raise ValueError("eligible parent records do not preserve global rank order")
    if not isinstance(batches_raw, list) or not batches_raw:
        raise ValueError("hard qualification plan has no batches")
    if not all(isinstance(batch, dict) for batch in batches_raw):
        raise ValueError("hard qualification plan contains malformed batches")
    batches = [dict(batch) for batch in batches_raw]
    max_probes_per_batch = int(plan.get("max_probes_per_batch", 0))
    max_probe_batches = int(plan.get("max_probe_batches", 0))
    if not 1 <= max_probes_per_batch <= 8 or not 1 <= max_probe_batches <= 2:
        raise ValueError("hard qualification plan exceeds bounded batch limits")
    if len(batches) > max_probe_batches:
        raise ValueError("hard qualification plan exceeds max_probe_batches")

    flattened: list[str] = []
    probe_paths: set[Path] = set()
    batch_id_by_parent: dict[str, str] = {}
    probe_rules_by_batch: dict[str, dict[str, PlaybookRule]] = {}
    baseline_store = PlaybookStore(baseline_path, read_only=True)
    try:
        baseline_rules = {rule.rule_id: rule for rule in baseline_store.rules()}
    finally:
        baseline_store.close()
    for parent_rule_id in eligible_parent_ids:
        parent = baseline_rules.get(parent_rule_id)
        if parent is None:
            raise ValueError(f"soft baseline is missing eligible parent: {parent_rule_id}")
        if not (
            parent.status is PlaybookRuleStatus.ACTIVE
            and parent.strength is PlaybookRuleStrength.SOFT
            and parent.effect is PlaybookRuleEffect.AVOID
        ):
            raise ValueError(f"soft baseline parent semantics drifted: {parent_rule_id}")

    for expected_index, batch in enumerate(batches):
        batch_id = str(batch.get("batch_id", ""))
        if batch_id != f"batch-{expected_index:03d}" or batch.get("batch_index") != expected_index:
            raise ValueError("hard qualification batches are not canonically indexed")
        members_raw = batch.get("parent_rule_ids")
        if not isinstance(members_raw, list):
            raise ValueError(f"{batch_id} has malformed parent_rule_ids")
        members = [str(parent_rule_id) for parent_rule_id in members_raw]
        if not 1 <= len(members) <= max_probes_per_batch:
            raise ValueError(f"{batch_id} exceeds max_probes_per_batch")
        if batch.get("batch_size") != len(members):
            raise ValueError(f"{batch_id} batch size does not match its members")
        if len(members) != len(set(members)):
            raise ValueError(f"{batch_id} repeats a parent")
        flattened.extend(members)
        for parent_rule_id in members:
            if parent_rule_id in batch_id_by_parent:
                raise ValueError(f"eligible parent appears in multiple batches: {parent_rule_id}")
            batch_id_by_parent[parent_rule_id] = batch_id

        probe_path = Path(str(batch.get("probe_baseline_path", ""))).resolve()
        if probe_path in probe_paths or not probe_path.is_file():
            raise ValueError(f"{batch_id} probe path is missing or duplicated")
        probe_paths.add(probe_path)
        if batch.get("probe_baseline_sha256") != _sha256_file(probe_path):
            raise ValueError(f"{batch_id} probe SHA mismatch")
        probe_store = PlaybookStore(probe_path, read_only=True)
        try:
            probe_rules = probe_store.rules()
        finally:
            probe_store.close()
        probe_rule_map = {rule.rule_id: rule for rule in probe_rules}
        active_hard_ids = {
            rule.rule_id
            for rule in runtime_hard_rule_candidates(
                probe_rules,
                agent_race=None,
                opponent_race=None,
                map_name=None,
            )
        }
        if active_hard_ids != set(members):
            raise ValueError(f"{batch_id} runtime hard candidate set differs from plan")
        for parent_rule_id in eligible_parent_ids:
            rule = probe_rule_map.get(parent_rule_id)
            if rule is None:
                raise ValueError(f"{batch_id} is missing eligible parent {parent_rule_id}")
            if parent_rule_id in active_hard_ids:
                valid = (
                    rule.status is PlaybookRuleStatus.ACTIVE
                    and rule.strength is PlaybookRuleStrength.HARD
                    and rule.effect is PlaybookRuleEffect.FORBID
                )
            else:
                valid = (
                    rule.status is PlaybookRuleStatus.SUSPENDED
                    and rule.strength is PlaybookRuleStrength.SOFT
                    and rule.effect is PlaybookRuleEffect.AVOID
                )
            if not valid:
                raise ValueError(f"{batch_id} has invalid isolation for {parent_rule_id}")
        validation = batch.get("isolation_validation")
        if not isinstance(validation, dict) or validation.get("verified") is not True:
            raise ValueError(f"{batch_id} lacks successful isolation validation")
        probe_rules_by_batch[batch_id] = probe_rule_map

    if flattened != eligible_parent_ids or len(flattened) != len(set(flattened)):
        raise ValueError("hard qualification batch union is incomplete or overlapping")
    partition_integrity = plan.get("partition_integrity")
    if not isinstance(partition_integrity, dict) or partition_integrity.get("verified") is not True:
        raise ValueError("hard qualification plan lacks verified partition integrity")
    if int(plan.get("eligible_parent_count", -1)) != len(eligible_parent_ids):
        raise ValueError("hard qualification eligible parent count mismatch")
    if int(plan.get("required_batch_count", -1)) != len(batches):
        raise ValueError("hard qualification required batch count mismatch")
    return batches, batch_id_by_parent, probe_rules_by_batch


def analyze_hard_qualification(
    *,
    plan_path: Path,
    baseline_path: Path,
    status_path: Path,
    engineering_baseline_path: Path,
    recovery_evidence_path: Path,
    expected_git_sha: str,
    sc2_patch: str,
    working_database: Path,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    batches, batch_id_by_parent, probe_rules_by_batch = _validate_batch_plan(
        plan,
        plan_path=plan_path,
        baseline_path=baseline_path,
        expected_git_sha=expected_git_sha,
        sc2_patch=sc2_patch,
    )

    engineering_baseline = json.loads(engineering_baseline_path.read_text(encoding="utf-8"))
    recovery_evidence = json.loads(recovery_evidence_path.read_text(encoding="utf-8"))
    rows = list(csv.DictReader(status_path.open(encoding="utf-8"), delimiter="\t"))
    indexed_rows = _index_batch_status_rows(plan, rows)
    expected_seeds = tuple(sorted(int(entry["seed_id"]) for entry in plan["source_runs"]))
    planned_source_by_seed = {int(entry["seed_id"]): entry for entry in plan["source_runs"]}

    common_run_evidence: dict[tuple[str, int], dict[str, Any]] = {}
    events_by_batch_seed: dict[tuple[str, int], tuple[Any, ...]] = {}
    context_values_by_batch_seed: dict[tuple[str, int], dict[str, object]] = {}
    run_reports: list[dict[str, Any]] = []
    qualification_config_fingerprint: str | None = None
    observed_run_directories: set[Path] = set()
    for batch in batches:
        batch_id = str(batch["batch_id"])
        batch_index = int(batch["batch_index"])
        probe_sha256 = str(batch["probe_baseline_sha256"])
        for seed in expected_seeds:
            row = indexed_rows[(batch_id, seed)]
            run_directory = Path(row.get("run_dir", "")).expanduser().resolve()
            if not row.get("run_dir") or run_directory in observed_run_directories:
                raise ValueError(
                    f"qualification run directory is missing or reused: {batch_id}:{seed}"
                )
            observed_run_directories.add(run_directory)
            events_path = run_directory / "events.jsonl"
            gates_path = run_directory / "engineering-gates.json"
            worker_stderr = run_directory / "worker.stderr.log"
            config_path = run_directory / "config.yaml"
            required_paths = (events_path, gates_path, worker_stderr, config_path)
            if not all(path.is_file() for path in required_paths):
                raise ValueError(
                    f"qualification run is missing required artifacts: {run_directory}"
                )
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
                config_path,
                expected_database=working_database,
            )
            config_sha256 = _sha256_file(config_path)
            generated_config_fingerprint = _qualification_config_fingerprint(config_path)
            if qualification_config_fingerprint is None:
                qualification_config_fingerprint = generated_config_fingerprint
            try:
                run_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            except (OSError, TypeError, ValueError, yaml.YAMLError):
                run_config = {}
            run_seed = (
                run_config.get("run", {}).get("seed")
                if isinstance(run_config, dict) and isinstance(run_config.get("run"), dict)
                else None
            )
            planned_source = planned_source_by_seed.get(seed, {})
            config_bound = (
                config_valid
                and generated_config_fingerprint is not None
                and generated_config_fingerprint == qualification_config_fingerprint
                and run_seed == seed
            )
            source_valid = (
                run_metrics.source_tree_clean
                and run_metrics.source_commit_matches_expected_sha
                and run_metrics.source_attestation_consistent
                and run_metrics.source_attestation_fingerprint is not None
            )
            immutable_probe = (
                row.get("playbook_before_sha256") == probe_sha256
                and row.get("playbook_after_sha256") == probe_sha256
            )
            run_accepted = (
                run_metrics.natural_terminal
                and run_metrics.engineering_accepted
                and source_valid
                and config_valid
                and config_bound
                and immutable_probe
                and sc2_build == plan["sc2_build"]
                and observed_patch == sc2_patch
            )
            key = (batch_id, seed)
            events_by_batch_seed[key] = tuple(read_event_log(events_path))
            context_values_by_batch_seed[key] = _runtime_context_from_config(config_path)
            common_run_evidence[key] = {
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
                "analysis_evidence_overflow_count": (run_metrics.analysis_evidence_overflow_count),
                "config_sha256": config_sha256,
            }
            run_reports.append(
                {
                    "batch_id": batch_id,
                    "batch_index": batch_index,
                    "seed_id": seed,
                    "run_id": run_directory.name,
                    "run_directory": str(run_directory),
                    "accepted": run_accepted,
                    "natural_terminal": run_metrics.natural_terminal,
                    "engineering_accepted": run_metrics.engineering_accepted,
                    "source_attestation_consistent": source_valid,
                    "shadow_config_valid": config_valid,
                    "qualification_config_bound": config_bound,
                    "source_config_sha256": planned_source.get("config_sha256"),
                    "qualification_config_fingerprint": generated_config_fingerprint,
                    "probe_baseline_path": str(batch["probe_baseline_path"]),
                    "probe_baseline_sha256": probe_sha256,
                    "probe_baseline_immutable": immutable_probe,
                    "sc2_build": sc2_build,
                    "sc2_patch": observed_patch,
                    "analysis_evidence_overflow_count": (
                        run_metrics.analysis_evidence_overflow_count
                    ),
                }
            )

    _require_complete_batch_run_audit(run_reports)

    store = PlaybookStore(baseline_path, read_only=True)
    try:
        rules_by_id = {rule.rule_id: rule for rule in store.rules()}
    finally:
        store.close()
    accepted_manifests: list[Any] = []
    parent_reports: list[dict[str, Any]] = []
    eligible_records_by_id = {
        str(record["parent_rule_id"]): record for record in plan["eligible_parents"]
    }
    batch_by_id = {str(batch["batch_id"]): batch for batch in batches}
    for parent_rule_id in plan["eligible_parent_ids"]:
        parent_rule_id = str(parent_rule_id)
        batch_id = batch_id_by_parent[parent_rule_id]
        batch = batch_by_id[batch_id]
        parent = rules_by_id.get(parent_rule_id)
        probe_rule = probe_rules_by_batch[batch_id].get(parent_rule_id)
        if parent is None or probe_rule is None:
            raise ValueError(f"validated parent disappeared during analysis: {parent_rule_id}")
        expected_rule_fingerprint = playbook_rule_fingerprint(probe_rule)
        expected_predicate_fingerprint = playbook_predicate_fingerprint(probe_rule)
        parent_events = _values_for_parent_batch(
            parent_rule_id,
            batch_id_by_parent=batch_id_by_parent,
            values_by_batch_seed=events_by_batch_seed,
            expected_seeds=expected_seeds,
        )
        parent_contexts = _values_for_parent_batch(
            parent_rule_id,
            batch_id_by_parent=batch_id_by_parent,
            values_by_batch_seed=context_values_by_batch_seed,
            expected_seeds=expected_seeds,
        )
        parent_common_evidence = _values_for_parent_batch(
            parent_rule_id,
            batch_id_by_parent=batch_id_by_parent,
            values_by_batch_seed=common_run_evidence,
            expected_seeds=expected_seeds,
        )
        run_evidence: list[PlaybookHardQualificationRunEvidence] = []
        counterfactual_by_seed: dict[str, dict[str, Any]] = {}
        for seed in expected_seeds:
            observed = analyze_hard_qualification_evaluations(
                parent_events[seed],
                rule_id=parent_rule_id,
                expected_rule_fingerprint=expected_rule_fingerprint,
                expected_predicate_fingerprint=expected_predicate_fingerprint,
            )
            typed_retry = analyze_typed_retry_coverage(
                parent_events[seed],
                rule=probe_rule,
                expected_rule_fingerprint=expected_rule_fingerprint,
                expected_predicate_fingerprint=expected_predicate_fingerprint,
                context_values=parent_contexts[seed],
            )
            counterfactual_by_seed[str(seed)] = {
                "shadow_would_block_application_count": (
                    observed.shadow_would_block_application_count
                ),
                "resolved_counterfactual_count": observed.resolved_counterfactual_count,
                "unresolved_counterfactual_count": observed.unresolved_counterfactual_count,
                "structurally_unobservable_counterfactual_count": (
                    observed.structurally_unobservable_counterfactual_count
                ),
                "execution_false_block_count": observed.execution_false_block_count,
                "invalid_application_count": observed.invalid_application_count,
                "application_without_evaluation_count": (
                    observed.application_without_evaluation_count
                ),
                "evaluation_without_application_count": (
                    observed.evaluation_without_application_count
                ),
                "execution_false_block_rate": observed.execution_false_block_rate,
                "invalid_counterfactual_evidence_count": (
                    observed.invalid_counterfactual_evidence_count
                ),
                "typed_retry_opportunity_count": typed_retry.typed_retry_opportunity_count,
                "typed_retry_application_count": typed_retry.typed_retry_application_count,
                "typed_retry_coverage_unavailable": (typed_retry.typed_retry_coverage_unavailable),
                "typed_retry_coverage_reasons": list(typed_retry.unavailable_reasons),
                "typed_retry_opportunity_ids": list(typed_retry.opportunity_ids),
                "typed_retry_application_ids": list(typed_retry.application_ids),
                "typed_retry_opportunity_bindings": [
                    binding.as_dict() for binding in typed_retry.opportunity_bindings
                ],
            }
            run_evidence.append(
                PlaybookHardQualificationRunEvidence(
                    **parent_common_evidence[seed],
                    invalid_counterfactual_evidence_count=(
                        observed.invalid_counterfactual_evidence_count
                    ),
                    shadow_would_block_application_count=(
                        observed.shadow_would_block_application_count
                    ),
                    resolved_counterfactual_count=observed.resolved_counterfactual_count,
                    unresolved_counterfactual_count=observed.unresolved_counterfactual_count,
                    execution_false_block_count=observed.execution_false_block_count,
                    structurally_unobservable_counterfactual_count=(
                        observed.structurally_unobservable_counterfactual_count
                    ),
                    invalid_application_count=observed.invalid_application_count,
                    application_without_evaluation_count=(
                        observed.application_without_evaluation_count
                    ),
                    evaluation_without_application_count=(
                        observed.evaluation_without_application_count
                    ),
                    rule_fingerprint=expected_rule_fingerprint,
                    predicate_fingerprint=expected_predicate_fingerprint,
                    typed_retry_opportunity_count=typed_retry.typed_retry_opportunity_count,
                    typed_retry_application_count=typed_retry.typed_retry_application_count,
                    typed_retry_coverage_unavailable=(typed_retry.typed_retry_coverage_unavailable),
                    typed_retry_coverage_reasons=typed_retry.unavailable_reasons,
                )
            )
        try:
            manifest = build_hard_qualification_manifest(
                parent=parent,
                baseline_sha256=plan["baseline_sha256"],
                probe_baseline_sha256=batch["probe_baseline_sha256"],
                git_sha=expected_git_sha,
                sc2_patch=sc2_patch,
                runs=run_evidence,
            )
        except ValueError as error:
            parent_reports.append(
                {
                    "parent_rule_id": parent_rule_id,
                    "global_rank": eligible_records_by_id[parent_rule_id]["global_rank"],
                    "batch_id": batch_id,
                    "batch_index": batch["batch_index"],
                    "probe_baseline_sha256": batch["probe_baseline_sha256"],
                    "accepted": False,
                    "selection_status": "rejected",
                    "reason": str(error),
                    "counterfactual_by_seed": counterfactual_by_seed,
                }
            )
            continue
        accepted_manifests.append(manifest)
        parent_reports.append(
            {
                "parent_rule_id": parent_rule_id,
                "global_rank": eligible_records_by_id[parent_rule_id]["global_rank"],
                "batch_id": batch_id,
                "batch_index": batch["batch_index"],
                "probe_baseline_sha256": batch["probe_baseline_sha256"],
                "accepted": True,
                "selection_status": "accepted_not_selected",
                "counterfactual_resolved_count": manifest.counterfactual_resolved_count,
                "counterfactual_unresolved_count": manifest.counterfactual_unresolved_count,
                "counterfactual_structurally_unobservable_count": (
                    manifest.counterfactual_structurally_unobservable_count
                ),
                "invalid_counterfactual_evidence_count": sum(
                    item.invalid_counterfactual_evidence_count
                    for item in manifest.qualification_runs
                ),
                "counterfactual_false_block_rate": manifest.counterfactual_false_block_rate,
                "counterfactual_by_seed": counterfactual_by_seed,
                "typed_retry_opportunity_count_by_seed": (
                    manifest.typed_retry_opportunity_count_by_seed
                ),
                "typed_retry_application_count_by_seed": (
                    manifest.typed_retry_application_count_by_seed
                ),
                "typed_retry_coverage_unavailable_by_seed": (
                    manifest.typed_retry_coverage_unavailable_by_seed
                ),
            }
        )

    selected, accepted_manifests = _select_global_manifest(accepted_manifests)
    if selected is not None:
        for parent_report in parent_reports:
            if parent_report["parent_rule_id"] == selected.parent_rule_id:
                parent_report["selection_status"] = "selected"
                break
    selected_batch_id = None if selected is None else batch_id_by_parent[selected.parent_rule_id]
    selected_probe_sha256 = (
        None
        if selected_batch_id is None
        else batch_by_id[selected_batch_id]["probe_baseline_sha256"]
    )
    batch_reports = [
        {
            **batch,
            "runs": [run for run in run_reports if run["batch_id"] == batch["batch_id"]],
        }
        for batch in batches
    ]
    report = {
        "schema_version": "2.0",
        "artifact_kind": "playbook-hard-qualification-report",
        "expected_git_sha": expected_git_sha,
        "source_git_sha": plan["source_git_sha"],
        "sc2_patch": sc2_patch,
        "baseline_sha256": plan["baseline_sha256"],
        "eligible_parent_count": len(plan["eligible_parent_ids"]),
        "batch_audit_complete": True,
        "batches": batch_reports,
        "completed_batch_seed_runs": [
            f"{batch['batch_id']}:{seed}" for batch in batches for seed in expected_seeds
        ],
        "failed_batch_seed_runs": [],
        "unrun_batch_seed_runs": [],
        "unexpected_batch_seed_runs": [],
        "runs": sorted(
            run_reports,
            key=lambda item: (item["batch_index"], item["seed_id"]),
        ),
        "parents": parent_reports,
        "accepted_parent_rule_ids": [manifest.parent_rule_id for manifest in accepted_manifests],
        "selected_parent_rule_id": None if selected is None else selected.parent_rule_id,
        "selected_batch_id": selected_batch_id,
        "selected_probe_baseline_sha256": selected_probe_sha256,
        "accepted": selected is not None,
    }
    return report, None if selected is None else selected.model_dump(mode="json")


def write_hard_qualification_result(
    report: dict[str, Any],
    manifest: dict[str, Any] | None,
    *,
    report_output: Path,
    manifest_output: Path,
) -> tuple[int, str]:
    """Write one terminal result without allowing a rejected manifest downstream."""

    resolved_report = report_output.expanduser().resolve()
    resolved_manifest = manifest_output.expanduser().resolve()
    exit_code = 0 if manifest is not None and report.get("accepted") is True else 1
    status = "accepted" if exit_code == 0 else "rejected"
    terminal_report = {
        **report,
        "status": status,
        "exit_code": exit_code,
        "report_path": str(resolved_report),
    }
    resolved_report.write_text(
        json.dumps(terminal_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if exit_code:
        if resolved_manifest.exists():
            raise ValueError(
                f"hard qualification rejected but a manifest already exists: {resolved_manifest}"
            )
        return (
            exit_code,
            f"hard_qualification rejected exit_code=1 report={resolved_report}",
        )
    assert manifest is not None
    resolved_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return (
        exit_code,
        "hard_qualification accepted exit_code=0 "
        f"report={resolved_report} manifest={resolved_manifest}",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--engineering-baseline", type=Path, required=True)
    parser.add_argument("--recovery-evidence", type=Path, required=True)
    parser.add_argument("--expected-git-sha", required=True)
    parser.add_argument("--sc2-patch", required=True)
    parser.add_argument("--working-database", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--terminal-rejection-reason")
    parser.add_argument("--runner-exit-code", type=int, default=1)
    arguments = parser.parse_args()
    plan_path = arguments.plan.resolve()
    status_path = arguments.status.resolve()
    try:
        plan_payload = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        plan_payload = {
            "schema_version": "2.0",
            "artifact_kind": "playbook-hard-qualification-plan",
            "expected_git_sha": arguments.expected_git_sha,
            "eligible_parent_ids": [],
            "batches": [],
            "source_runs": [],
        }
    try:
        status_rows = list(csv.DictReader(status_path.open(encoding="utf-8"), delimiter="\t"))
    except (OSError, csv.Error):
        status_rows = []
    if arguments.terminal_rejection_reason:
        report = build_incomplete_hard_qualification_report(
            plan_payload,
            status_rows,
            reason=arguments.terminal_rejection_reason,
            runner_exit_code=arguments.runner_exit_code,
        )
        manifest = None
    else:
        try:
            report, manifest = analyze_hard_qualification(
                plan_path=plan_path,
                baseline_path=arguments.baseline.resolve(),
                status_path=status_path,
                engineering_baseline_path=arguments.engineering_baseline.resolve(),
                recovery_evidence_path=arguments.recovery_evidence.resolve(),
                expected_git_sha=arguments.expected_git_sha,
                sc2_patch=arguments.sc2_patch,
                working_database=arguments.working_database.resolve(),
            )
        except Exception as error:
            report = build_incomplete_hard_qualification_report(
                plan_payload,
                status_rows,
                reason=f"analysis_failed:{type(error).__name__}:{error}",
                runner_exit_code=1,
            )
            manifest = None
    exit_code, message = write_hard_qualification_result(
        report,
        manifest,
        report_output=arguments.report_output,
        manifest_output=arguments.manifest_output,
    )
    print(message, file=sys.stderr if exit_code else sys.stdout)
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
