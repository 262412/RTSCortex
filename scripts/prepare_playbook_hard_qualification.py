"""Build a revision-bound soft baseline and isolated hard-shadow probe database."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from rtscortex.memory import read_event_log
from rtscortex.playbook import (
    PlaybookRule,
    PlaybookRuleCategory,
    PlaybookRuleEffect,
    PlaybookRuleKind,
    PlaybookRuleStatus,
    PlaybookRuleStrength,
    PlaybookRunLearner,
    PlaybookStore,
    TypedRetryCoverageBySeed,
    analyze_typed_retry_coverage_by_seed,
    evaluation_kind,
)
from rtscortex.playbook.selection import runtime_hard_rule_candidates

_SC2_VERSION_PATTERN = re.compile(r"^Version: (B[0-9]+) \(SC2\.([^)]+)\)$", re.MULTILINE)
_RUNTIME_MAX_HARD_RULES = 8
_MAX_PROBE_BATCHES = 2


class ProbeBatchCapacityError(ValueError):
    """Fail-closed capacity error that retains every ordered eligible parent."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        super().__init__(json.dumps(payload, sort_keys=True))


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


def _runtime_context_from_config(config_path: Path) -> dict[str, object]:
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


def _load_source_rows(
    status_path: Path,
    run_directories: tuple[Path, ...],
    *,
    sc2_patch: str,
) -> tuple[list[dict[str, Any]], str, str, str, dict[str, Any]]:
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
    config_hashes_by_seed: dict[int, str] = {}
    for row in rows:
        run_directory = Path(row["run_dir"]).expanduser().resolve()
        gates_path = run_directory / "engineering-gates.json"
        events_path = run_directory / "events.jsonl"
        worker_stderr = run_directory / "worker.stderr.log"
        config_path = run_directory / "config.yaml"
        if not all(
            path.is_file() for path in (gates_path, events_path, worker_stderr, config_path)
        ):
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
                bool(row.get("reviewed_tree_sha256_before"))
                and row.get("reviewed_tree_sha256_before") == row.get("reviewed_tree_sha256_after")
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
        config_sha256 = _sha256_file(config_path)
        if seed in config_hashes_by_seed:
            raise ValueError(f"source status contains duplicate seed {seed}")
        config_hashes_by_seed[seed] = config_sha256
        attestation = json.dumps(
            {
                "git_sha": source_git_sha,
                "submodule_commit": row["submodule_commit_before"],
                "reviewed_commit": row["reviewed_commit_before"],
                "reviewed_tree": row["reviewed_tree_sha256_before"],
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
                "config_path": str(config_path),
                "config_sha256": config_sha256,
            }
        )
    if (
        len(builds) != 1
        or len(source_git_shas) != 1
        or len(attestation_payloads) != 1
        or set(config_hashes_by_seed) != {0, 1, 2}
    ):
        raise ValueError("source runs must cover seeds 0/1/2 with one source and SC2 attestation")
    fingerprint_payload = {
        "common_attestation": next(iter(attestation_payloads)),
        "config_sha256_by_seed": {
            str(seed): config_hashes_by_seed[seed] for seed in sorted(config_hashes_by_seed)
        },
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    common_attestation = json.loads(next(iter(attestation_payloads)))
    return (
        source_entries,
        next(iter(builds)),
        fingerprint,
        next(iter(source_git_shas)),
        common_attestation,
    )


def _eligible_parent(
    rule: PlaybookRule,
    *,
    source_run_ids: set[str],
    source_seeds: set[int],
    typed_retry_coverage: TypedRetryCoverageBySeed,
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
        and typed_retry_coverage.has_real_opportunity_every_seed
    )


def _partition_probe_batches(
    parents: Sequence[PlaybookRule],
    *,
    max_probes_per_batch: int,
    max_probe_batches: int,
) -> tuple[tuple[PlaybookRule, ...], ...]:
    """Return a deterministic, bounded, complete partition of eligible parents."""

    if not 1 <= max_probes_per_batch <= _RUNTIME_MAX_HARD_RULES:
        raise ValueError(f"max_probes_per_batch must be within 1-{_RUNTIME_MAX_HARD_RULES}")
    if not 1 <= max_probe_batches <= _MAX_PROBE_BATCHES:
        raise ValueError(f"max_probe_batches must be within 1-{_MAX_PROBE_BATCHES}")
    ordered = tuple(sorted(parents, key=lambda rule: (-rule.shadow_state_count, rule.rule_id)))
    parent_ids = [rule.rule_id for rule in ordered]
    if len(parent_ids) != len(set(parent_ids)):
        raise ValueError("eligible hard qualification parents contain duplicate rule IDs")
    required_batch_count = (
        0 if not ordered else (len(ordered) + max_probes_per_batch - 1) // max_probes_per_batch
    )
    if required_batch_count > max_probe_batches:
        raise ProbeBatchCapacityError(
            {
                "reason": "eligible_probe_capacity_exceeded",
                "eligible_parent_count": len(ordered),
                "eligible_parent_ids": parent_ids,
                "max_probes_per_batch": max_probes_per_batch,
                "max_probe_batches": max_probe_batches,
                "total_probe_capacity": max_probes_per_batch * max_probe_batches,
                "required_batch_count": required_batch_count,
            }
        )
    return tuple(
        ordered[offset : offset + max_probes_per_batch]
        for offset in range(0, len(ordered), max_probes_per_batch)
    )


def _probe_batch_path(probe_output: Path, batch_index: int) -> Path:
    return probe_output.with_name(
        f"{probe_output.stem}.batch-{batch_index:03d}{probe_output.suffix}"
    )


def _partition_integrity(
    eligible_parent_ids: Sequence[str],
    batches: Sequence[dict[str, Any]],
    *,
    max_probes_per_batch: int,
    max_probe_batches: int,
) -> dict[str, Any]:
    flattened = [
        str(parent_rule_id)
        for batch in batches
        for parent_rule_id in batch.get("parent_rule_ids", [])
    ]
    eligible = list(eligible_parent_ids)
    duplicate_ids = sorted(
        parent_rule_id for parent_rule_id in set(flattened) if flattened.count(parent_rule_id) > 1
    )
    missing_ids = [parent_rule_id for parent_rule_id in eligible if parent_rule_id not in flattened]
    extra_ids = [parent_rule_id for parent_rule_id in flattened if parent_rule_id not in eligible]
    batch_sizes = [len(batch.get("parent_rule_ids", [])) for batch in batches]
    ordered_union_complete = flattened == eligible
    intersections_empty = not duplicate_ids
    sizes_within_limit = all(1 <= size <= max_probes_per_batch for size in batch_sizes)
    batch_count_within_limit = 1 <= len(batches) <= max_probe_batches
    verified = all(
        (
            ordered_union_complete,
            intersections_empty,
            not missing_ids,
            not extra_ids,
            sizes_within_limit,
            batch_count_within_limit,
        )
    )
    return {
        "verified": verified,
        "eligible_parent_count": len(eligible),
        "partitioned_parent_count": len(flattened),
        "ordered_union_complete": ordered_union_complete,
        "batch_intersections_empty": intersections_empty,
        "batch_sizes_within_limit": sizes_within_limit,
        "batch_count_within_limit": batch_count_within_limit,
        "batch_sizes": batch_sizes,
        "duplicate_parent_ids": duplicate_ids,
        "missing_parent_ids": missing_ids,
        "extra_parent_ids": extra_ids,
    }


def _create_probe_batch(
    *,
    baseline_path: Path,
    probe_path: Path,
    eligible_parents: Sequence[PlaybookRule],
    batch_members: Sequence[PlaybookRule],
    batch_id: str,
    batch_index: int,
    baseline_sha256: str,
    expected_git_sha: str,
    sc2_build: str,
    sc2_patch: str,
    max_probes_per_batch: int,
) -> dict[str, Any]:
    """Create and read back one isolated hard-shadow probe database."""

    eligible_ids = [parent.rule_id for parent in eligible_parents]
    member_ids = [parent.rule_id for parent in batch_members]
    member_id_set = set(member_ids)
    if not member_ids or len(member_ids) > max_probes_per_batch:
        raise ValueError(f"{batch_id} must contain 1-{max_probes_per_batch} parents")
    if len(member_ids) != len(member_id_set) or not member_id_set.issubset(eligible_ids):
        raise ValueError(f"{batch_id} contains duplicate or ineligible parents")
    if probe_path.exists():
        raise ValueError(f"hard qualification probe already exists: {probe_path}")
    if _sha256_file(baseline_path) != baseline_sha256:
        raise ValueError("soft baseline changed before probe batch creation")

    shutil.copy2(baseline_path, probe_path)
    probe_store = PlaybookStore(probe_path)
    try:
        for parent in eligible_parents:
            if parent.rule_id not in member_id_set:
                probe_store.upsert_rule(
                    parent.model_copy(update={"status": PlaybookRuleStatus.SUSPENDED})
                )
                continue
            probe_store.upsert_rule(
                parent.model_copy(
                    update={
                        "status": PlaybookRuleStatus.ACTIVE,
                        "effect": PlaybookRuleEffect.FORBID,
                        "strength": PlaybookRuleStrength.HARD,
                        "evidence": {
                            **parent.evidence,
                            "hard_qualification_probe": {
                                "batch_id": batch_id,
                                "batch_index": batch_index,
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

    if _sha256_file(baseline_path) != baseline_sha256:
        raise ValueError("soft baseline changed while creating probe batches")
    readback = PlaybookStore(probe_path, read_only=True)
    try:
        rules = readback.rules()
    finally:
        readback.close()
    rules_by_id = {rule.rule_id: rule for rule in rules}
    active_hard_ids = [
        rule.rule_id
        for rule in runtime_hard_rule_candidates(
            rules,
            agent_race=None,
            opponent_race=None,
            map_name=None,
        )
    ]
    suspended_ids = [
        parent_rule_id
        for parent_rule_id in eligible_ids
        if parent_rule_id not in member_id_set
        and parent_rule_id in rules_by_id
        and rules_by_id[parent_rule_id].status is PlaybookRuleStatus.SUSPENDED
        and rules_by_id[parent_rule_id].strength is PlaybookRuleStrength.SOFT
        and rules_by_id[parent_rule_id].effect is PlaybookRuleEffect.AVOID
    ]
    member_semantics_valid = all(
        parent_rule_id in rules_by_id
        and rules_by_id[parent_rule_id].status is PlaybookRuleStatus.ACTIVE
        and rules_by_id[parent_rule_id].strength is PlaybookRuleStrength.HARD
        and rules_by_id[parent_rule_id].effect is PlaybookRuleEffect.FORBID
        for parent_rule_id in member_ids
    )
    expected_suspended_ids = [
        parent_rule_id for parent_rule_id in eligible_ids if parent_rule_id not in member_id_set
    ]
    active_hard_set_matches = set(active_hard_ids) == member_id_set
    suspended_set_matches = suspended_ids == expected_suspended_ids
    verified = (
        active_hard_set_matches
        and len(active_hard_ids) <= max_probes_per_batch
        and suspended_set_matches
        and member_semantics_valid
    )
    validation = {
        "verified": verified,
        "active_hard_parent_ids": [
            parent_rule_id for parent_rule_id in member_ids if parent_rule_id in active_hard_ids
        ],
        "runtime_active_hard_rule_ids": active_hard_ids,
        "suspended_eligible_parent_ids": suspended_ids,
        "active_hard_set_matches_batch": active_hard_set_matches,
        "nonmember_isolation_complete": suspended_set_matches,
        "member_semantics_valid": member_semantics_valid,
        "hard_candidate_count_within_limit": len(active_hard_ids) <= max_probes_per_batch,
    }
    if not verified:
        raise ValueError(
            f"probe batch isolation validation failed for {batch_id}: "
            + json.dumps(validation, sort_keys=True)
        )
    return validation


def prepare_hard_qualification(
    *,
    run_directories: tuple[Path, ...],
    source_status: Path,
    baseline_output: Path,
    probe_output: Path,
    plan_output: Path,
    expected_git_sha: str,
    sc2_patch: str,
    max_probes_per_batch: int,
    max_probe_batches: int,
) -> dict[str, Any]:
    """Create one soft baseline and bounded isolated hard-shadow probe batches."""

    output_paths = (baseline_output, probe_output, plan_output)
    existing = [str(path) for path in output_paths if path.exists()]
    existing.extend(
        str(path)
        for path in sorted(
            probe_output.parent.glob(f"{probe_output.stem}.batch-*{probe_output.suffix}")
        )
    )
    if existing:
        raise ValueError("hard qualification outputs already exist: " + ", ".join(existing))
    (
        source_entries,
        sc2_build,
        source_fingerprint,
        source_git_sha,
        raw_source_attestation,
    ) = _load_source_rows(source_status, run_directories, sc2_patch=sc2_patch)
    source_attestation = {
        "git_sha": raw_source_attestation["git_sha"],
        "submodule_commit": raw_source_attestation["submodule_commit"],
        "submodule_gitlink": raw_source_attestation["submodule_commit"],
        "reviewed_commit": raw_source_attestation["reviewed_commit"],
        "reviewed_tree_sha256": raw_source_attestation["reviewed_tree"],
        "sc2_build": raw_source_attestation["sc2_build"],
        "sc2_patch": raw_source_attestation["sc2_patch"],
    }
    events_by_seed = {
        int(entry["seed_id"]): tuple(
            read_event_log(Path(str(entry["run_directory"])) / "events.jsonl")
        )
        for entry in source_entries
    }
    context_values_by_seed = {
        int(entry["seed_id"]): _runtime_context_from_config(
            Path(str(entry["run_directory"])) / "config.yaml"
        )
        for entry in source_entries
    }
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
        typed_coverage_by_rule: dict[str, TypedRetryCoverageBySeed] = {}
        typed_diagnostics: dict[str, dict[str, object]] = {}
        candidate_pool = [
            rule
            for rule in store.rules()
            if (
                rule.status is PlaybookRuleStatus.ACTIVE
                and rule.strength is PlaybookRuleStrength.SOFT
                and evaluation_kind(rule.category) is PlaybookRuleKind.EXECUTION_GUARD
            )
        ]
        for rule in candidate_pool:
            coverage = analyze_typed_retry_coverage_by_seed(
                events_by_seed,
                rule=rule,
                context_values_by_seed=context_values_by_seed,
            )
            typed_coverage_by_rule[rule.rule_id] = coverage
            typed_diagnostics[rule.rule_id] = coverage.as_dict()
        parent_candidates = [
            rule
            for rule in candidate_pool
            if _eligible_parent(
                rule,
                source_run_ids=learned_run_ids,
                source_seeds=learned_seeds,
                typed_retry_coverage=typed_coverage_by_rule[rule.rule_id],
            )
        ]
        if not parent_candidates:
            raise ValueError(
                "no three-seed active soft execution rule is eligible for qualification; "
                "typed_retry_coverage_unavailable diagnostics per rule/seed: "
                + json.dumps(typed_diagnostics, sort_keys=True)
            )
        batches = _partition_probe_batches(
            parent_candidates,
            max_probes_per_batch=max_probes_per_batch,
            max_probe_batches=max_probe_batches,
        )
        parent_candidates = [parent for batch in batches for parent in batch]
        source_binding = {
            "kind": "hard_qualification_source_binding",
            "source_git_sha": source_git_sha,
            "qualification_git_sha": expected_git_sha,
            "sc2_build": sc2_build,
            "sc2_patch": sc2_patch,
            "source_attestation_fingerprint": source_fingerprint,
            "source_attestation": source_attestation,
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
                            "typed_retry_coverage": typed_coverage_by_rule[
                                parent.rule_id
                            ].as_dict(),
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
    bound_by_id = {parent.rule_id: parent for parent in bound_parents}
    batch_records: list[dict[str, Any]] = []
    for batch_index, batch in enumerate(batches):
        batch_id = f"batch-{batch_index:03d}"
        batch_members = tuple(bound_by_id[parent.rule_id] for parent in batch)
        probe_path = _probe_batch_path(probe_output, batch_index)
        validation = _create_probe_batch(
            baseline_path=baseline_output,
            probe_path=probe_path,
            eligible_parents=bound_parents,
            batch_members=batch_members,
            batch_id=batch_id,
            batch_index=batch_index,
            baseline_sha256=baseline_sha256,
            expected_git_sha=expected_git_sha,
            sc2_build=sc2_build,
            sc2_patch=sc2_patch,
            max_probes_per_batch=max_probes_per_batch,
        )
        batch_records.append(
            {
                "batch_id": batch_id,
                "batch_index": batch_index,
                "batch_size": len(batch_members),
                "parent_rule_ids": [parent.rule_id for parent in batch_members],
                "suspended_eligible_parent_ids": [
                    parent.rule_id
                    for parent in bound_parents
                    if parent.rule_id not in {member.rule_id for member in batch_members}
                ],
                "probe_baseline_path": str(probe_path.resolve()),
                "probe_baseline_sha256": _sha256_file(probe_path),
                "isolation_validation": validation,
            }
        )
    eligible_parent_ids = [parent.rule_id for parent in bound_parents]
    partition_integrity = _partition_integrity(
        eligible_parent_ids,
        batch_records,
        max_probes_per_batch=max_probes_per_batch,
        max_probe_batches=max_probe_batches,
    )
    if not partition_integrity["verified"]:
        raise ValueError(
            "hard qualification probe partition validation failed: "
            + json.dumps(partition_integrity, sort_keys=True)
        )
    if _sha256_file(baseline_output) != baseline_sha256:
        raise ValueError("soft baseline changed after probe batch creation")

    eligible_parent_records = []
    for global_rank, parent in enumerate(bound_parents, start=1):
        coverage = typed_coverage_by_rule[parent.rule_id]
        eligible_parent_records.append(
            {
                "global_rank": global_rank,
                "parent_rule_id": parent.rule_id,
                "canonical_key": parent.canonical_key,
                "action_names": list(parent.action_names),
                "failure_code": (
                    None if parent.retry_guard is None else parent.retry_guard.failure_code
                ),
                "shadow_state_count": parent.shadow_state_count,
                "typed_retry_opportunity_count_by_seed": {
                    str(seed): count
                    for seed, count in coverage.typed_retry_opportunity_count_by_seed.items()
                },
                "typed_retry_application_count_by_seed": {
                    str(seed): count
                    for seed, count in coverage.typed_retry_application_count_by_seed.items()
                },
                "typed_retry_coverage": coverage.as_dict(),
                "batch_id": next(
                    record["batch_id"]
                    for record in batch_records
                    if parent.rule_id in record["parent_rule_ids"]
                ),
            }
        )
    plan = {
        "schema_version": "2.0",
        "artifact_kind": "playbook-hard-qualification-plan",
        "expected_git_sha": expected_git_sha,
        "source_git_sha": source_git_sha,
        "sc2_build": sc2_build,
        "sc2_patch": sc2_patch,
        "baseline_path": str(baseline_output.resolve()),
        "baseline_sha256": baseline_sha256,
        "max_probes_per_batch": max_probes_per_batch,
        "max_probe_batches": max_probe_batches,
        "total_probe_capacity": max_probes_per_batch * max_probe_batches,
        "required_batch_count": len(batch_records),
        "eligible_parent_count": len(bound_parents),
        "eligible_parent_ids": eligible_parent_ids,
        "eligible_parents": eligible_parent_records,
        "batches": batch_records,
        "partition_integrity": partition_integrity,
        "source_attestation_fingerprint": source_fingerprint,
        "source_attestation": source_attestation,
        "source_status_path": str(source_status.resolve()),
        "source_status_sha256": _sha256_file(source_status),
        "source_runs": source_entries,
        "typed_retry_opportunity_count_by_parent_seed": {
            rule.rule_id: {
                str(seed): count
                for seed, count in typed_coverage_by_rule[
                    rule.rule_id
                ].typed_retry_opportunity_count_by_seed.items()
            }
            for rule in bound_parents
        },
        "typed_retry_application_count_by_parent_seed": {
            rule.rule_id: {
                str(seed): count
                for seed, count in typed_coverage_by_rule[
                    rule.rule_id
                ].typed_retry_application_count_by_seed.items()
            }
            for rule in bound_parents
        },
        "typed_retry_coverage_unavailable_by_parent_seed": {
            rule_id: {
                str(seed): list(reasons)
                for seed, reasons in coverage.unavailable_reasons_by_seed.items()
            }
            for rule_id, coverage in typed_coverage_by_rule.items()
            if coverage.typed_retry_coverage_unavailable
        },
        "typed_retry_coverage_diagnostics": typed_diagnostics,
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
    parser.add_argument("--max-probes-per-batch", type=int, default=8)
    parser.add_argument("--max-probe-batches", type=int, default=2)
    arguments = parser.parse_args()
    plan = prepare_hard_qualification(
        run_directories=tuple(path.resolve() for path in arguments.source_run_dir),
        source_status=arguments.source_status.resolve(),
        baseline_output=arguments.baseline_output.resolve(),
        probe_output=arguments.probe_output.resolve(),
        plan_output=arguments.plan_output.resolve(),
        expected_git_sha=arguments.expected_git_sha,
        sc2_patch=arguments.sc2_patch,
        max_probes_per_batch=arguments.max_probes_per_batch,
        max_probe_batches=arguments.max_probe_batches,
    )
    print(json.dumps(plan, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
