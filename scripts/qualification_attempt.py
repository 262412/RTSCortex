"""Atomic attempt claims and provenance checks for qualification run roots."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ATTEMPT_MANIFEST_NAME = "attempt-manifest.json"
ATTEMPT_FIELDS = (
    "run_set_id",
    "expected_git_sha",
    "attempt_id",
    "slurm_job_id",
    "slurm_restart_count",
    "started_at",
)
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_ATTEMPT_ID_RE = re.compile(r"^attempt:[0-9a-f]{32}$")


class AttemptClaimError(RuntimeError):
    """The run root cannot be claimed for a new, non-restarted attempt."""


def _restart_count(value: str | int | None) -> int:
    if value is None or value == "":
        return 0
    if isinstance(value, bool):
        raise AttemptClaimError("SLURM_RESTART_COUNT must be a non-negative integer")
    text = str(value)
    if not text.isdigit():
        raise AttemptClaimError("SLURM_RESTART_COUNT must be a non-negative integer")
    return int(text)


def _started_at(value: str | None) -> str:
    timestamp = datetime.now(UTC).isoformat() if value is None else value
    try:
        parsed = datetime.fromisoformat(timestamp)
    except ValueError as error:
        raise AttemptClaimError("started timestamp must be ISO-8601") from error
    if parsed.tzinfo is None:
        raise AttemptClaimError("started timestamp must include a timezone")
    return timestamp


def _manifest_fields(
    *,
    run_set_id: str,
    expected_git_sha: str,
    attempt_id: str,
    slurm_job_id: str,
    slurm_restart_count: int,
    started_at: str,
) -> dict[str, Any]:
    return {
        "run_set_id": run_set_id,
        "expected_git_sha": expected_git_sha,
        "attempt_id": attempt_id,
        "slurm_job_id": slurm_job_id,
        "slurm_restart_count": slurm_restart_count,
        "started_at": started_at,
    }


def claim_attempt_root(
    run_set_dir: str | Path,
    *,
    expected_git_sha: str,
    run_set_id: str | None = None,
    slurm_job_id: str | None = None,
    slurm_restart_count: str | int | None = None,
    started_at: str | None = None,
) -> dict[str, Any]:
    """Atomically claim a previously absent run root and create its manifest.

    Restart refusal and all input validation happen before creating the parent or
    run root.  The root itself is claimed with one ``mkdir`` operation, and the
    manifest is created with exclusive ``O_EXCL`` semantics.  A claimed root is
    intentionally never removed on a later failure: it is evidence that the
    attempt started and must not be replayed into.
    """

    if not _SHA_RE.fullmatch(expected_git_sha):
        raise AttemptClaimError("expected Git revision must be a full lowercase SHA")
    restart = _restart_count(slurm_restart_count)
    if restart > 0:
        raise AttemptClaimError("refusing restart count > 0: SLURM_RESTART_COUNT must be 0")
    root = Path(run_set_dir).expanduser().resolve()
    if not root.name or root.name in {".", ".."}:
        raise AttemptClaimError("run-set directory must have a non-empty identity")
    identity = root.name if run_set_id is None else str(run_set_id)
    if identity != root.name:
        raise AttemptClaimError("run_set_id must match the run-set directory name")
    job_id = "unknown" if slurm_job_id is None or slurm_job_id == "" else str(slurm_job_id)
    timestamp = _started_at(started_at)
    attempt_id = f"attempt:{uuid.uuid4().hex}"
    manifest = {
        "format_version": "1.0",
        "artifact_kind": "qualification-attempt-manifest",
        **_manifest_fields(
            run_set_id=identity,
            expected_git_sha=expected_git_sha,
            attempt_id=attempt_id,
            slurm_job_id=job_id,
            slurm_restart_count=restart,
            started_at=timestamp,
        ),
    }

    # Do not even create a parent for a rejected restart.  The parent creation is
    # the only setup write needed before the atomic run-root claim.
    root.parent.mkdir(parents=True, exist_ok=True)
    try:
        root.mkdir()
    except FileExistsError as error:
        raise AttemptClaimError(f"run-set root already exists: {root}") from error
    manifest_path = root / ATTEMPT_MANIFEST_NAME
    try:
        with manifest_path.open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as error:
        raise AttemptClaimError(f"attempt manifest already exists: {manifest_path}") from error
    return manifest


def load_attempt_manifest(run_set_dir: str | Path) -> dict[str, Any]:
    root = Path(run_set_dir).expanduser().resolve()
    path = root / ATTEMPT_MANIFEST_NAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as error:
        raise ValueError(f"could not read attempt manifest: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError("attempt manifest must be a JSON object")
    if (
        payload.get("format_version") != "1.0"
        or payload.get("artifact_kind") != "qualification-attempt-manifest"
    ):
        raise ValueError("attempt manifest format or artifact kind is invalid")
    missing = [field for field in ATTEMPT_FIELDS if field not in payload]
    if missing:
        raise ValueError(f"attempt manifest is missing fields: {', '.join(missing)}")
    if not _SHA_RE.fullmatch(str(payload["expected_git_sha"])):
        raise ValueError("attempt manifest expected_git_sha is invalid")
    if payload["run_set_id"] != root.name:
        raise ValueError("attempt manifest run_set_id does not match its run root")
    if not isinstance(payload["attempt_id"], str) or not _ATTEMPT_ID_RE.fullmatch(
        payload["attempt_id"]
    ):
        raise ValueError("attempt manifest attempt_id is invalid")
    if not isinstance(payload["slurm_job_id"], str) or not payload["slurm_job_id"].strip():
        raise ValueError("attempt manifest slurm_job_id is invalid")
    if not isinstance(payload["slurm_restart_count"], int) or isinstance(
        payload["slurm_restart_count"], bool
    ):
        raise ValueError("attempt manifest restart count is invalid")
    if payload["slurm_restart_count"] != 0:
        raise ValueError("attempt manifest has a non-zero restart count")
    if not isinstance(payload["started_at"], str):
        raise ValueError("attempt manifest started_at is invalid")
    try:
        parsed_started_at = datetime.fromisoformat(payload["started_at"])
    except ValueError as error:
        raise ValueError("attempt manifest started_at is invalid") from error
    if parsed_started_at.tzinfo is None:
        raise ValueError("attempt manifest started_at is invalid")
    return payload


def attempt_fields(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return the flat provenance fields copied into every qualification artifact."""

    missing = [field for field in ATTEMPT_FIELDS if field not in manifest]
    if missing:
        raise ValueError(f"attempt provenance is missing fields: {', '.join(missing)}")
    return {field: manifest[field] for field in ATTEMPT_FIELDS}


def validate_attempt_provenance(
    payload: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    artifact_name: str,
) -> None:
    """Fail closed unless an artifact carries the exact claimed attempt fields."""

    expected = attempt_fields(manifest)
    mismatched = [field for field, value in expected.items() if payload.get(field) != value]
    nested = payload.get("attempt")
    if nested is not None:
        if not isinstance(nested, Mapping):
            mismatched.append("attempt")
        else:
            mismatched.extend(
                f"attempt.{field}"
                for field, value in expected.items()
                if nested.get(field) != value
            )
    if mismatched:
        raise ValueError(f"{artifact_name} attempt provenance mismatch: {', '.join(mismatched)}")


def validate_recovery_reference(
    evidence: Mapping[str, Any],
    manifest: Mapping[str, Any],
    recovery_path: str | Path,
) -> None:
    """Validate recovery path/hash plus the qualification seed provenance."""

    validate_attempt_provenance(evidence, manifest, artifact_name="qualification evidence")
    reference = evidence.get("recovery_evidence")
    if not isinstance(reference, Mapping):
        raise ValueError("qualification evidence is missing recovery evidence reference")
    expected_path = Path(str(reference.get("path", ""))).expanduser().resolve()
    actual_path = Path(recovery_path).expanduser().resolve()
    if expected_path != actual_path:
        raise ValueError("recovery evidence path does not match the claimed artifact")
    expected_hash = reference.get("sha256")
    actual_hash = hashlib.sha256(actual_path.read_bytes()).hexdigest()
    if not isinstance(expected_hash, str) or actual_hash != expected_hash:
        raise ValueError("recovery evidence hash does not match the claimed artifact")
    expected_seeds = evidence.get("seed_ids", evidence.get("qualification_seed_ids"))
    recovery_payload = json.loads(actual_path.read_text(encoding="utf-8"))
    if not isinstance(recovery_payload, Mapping):
        raise ValueError("recovery evidence must be a JSON object")
    validate_attempt_provenance(recovery_payload, manifest, artifact_name="recovery evidence")
    observed_seeds = recovery_payload.get(
        "seed_ids", recovery_payload.get("qualification_seed_ids")
    )
    if expected_seeds != observed_seeds:
        raise ValueError("recovery evidence seed provenance does not match qualification evidence")
    if recovery_payload.get("passed") is not True:
        raise ValueError("recovery evidence did not pass")


def _reference_identity(value: Any) -> tuple[Path, str] | None:
    if not isinstance(value, Mapping):
        return None
    raw_path = value.get("path")
    sha256 = value.get("sha256")
    if not isinstance(raw_path, str) or not raw_path or not isinstance(sha256, str):
        return None
    return Path(raw_path).expanduser().resolve(), sha256


def validate_source_qualification(
    *,
    run_set_dir: str | Path,
    manifest: Mapping[str, Any],
    source_attestation_path: str | Path,
    recovery_path: str | Path,
    qualification_evidence_path: str | Path,
    status_path: str | Path,
    expected_seeds: tuple[int, ...] = (0, 1, 2),
) -> None:
    """Join source attestation, recovery, evidence, status rows, and gates."""

    root = Path(run_set_dir).expanduser().resolve()
    if manifest.get("run_set_id") != root.name:
        raise ValueError("attempt manifest does not belong to the qualification run root")
    source_path = Path(source_attestation_path).expanduser().resolve()
    recovery = Path(recovery_path).expanduser().resolve()
    qualification_path = Path(qualification_evidence_path).expanduser().resolve()
    rows_path = Path(status_path).expanduser().resolve()
    expected_artifacts = {
        root / "source-attestation.json": source_path,
        root / "recovery-canary.json": recovery,
        root / "qualification-evidence.json": qualification_path,
        root / "qualification-status.tsv": rows_path,
    }
    if any(expected.resolve() != observed for expected, observed in expected_artifacts.items()):
        raise ValueError("qualification artifact is outside the claimed run root")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    evidence = json.loads(qualification_path.read_text(encoding="utf-8"))
    if not isinstance(source, Mapping) or not isinstance(evidence, Mapping):
        raise ValueError("source attestation and qualification evidence must be objects")
    validate_attempt_provenance(source, manifest, artifact_name="source attestation")
    validate_attempt_provenance(evidence, manifest, artifact_name="qualification evidence")
    if (
        source.get("format_version") != "1.0"
        or source.get("artifact_kind") != "qualification-source-attestation"
        or source.get("git_sha") != manifest.get("expected_git_sha")
    ):
        raise ValueError("source attestation identity is invalid")
    if evidence.get("seed_ids") != list(expected_seeds):
        raise ValueError("qualification evidence does not bind the exact seed set")
    validate_recovery_reference(evidence, manifest, recovery)
    source_reference = evidence.get("source_attestation")
    if not isinstance(source_reference, Mapping):
        raise ValueError("qualification evidence is missing source attestation reference")
    if Path(str(source_reference.get("path", ""))).expanduser().resolve() != source_path:
        raise ValueError("source attestation path does not match qualification evidence")
    if source_reference.get("sha256") != hashlib.sha256(source_path.read_bytes()).hexdigest():
        raise ValueError("source attestation hash does not match qualification evidence")
    baseline_reference = evidence.get("natural_run_baseline")
    if not isinstance(baseline_reference, Mapping):
        raise ValueError("qualification evidence is missing the disk baseline reference")
    baseline_path = Path(str(baseline_reference.get("path", ""))).expanduser().resolve()
    baseline_sha256 = baseline_reference.get("sha256")
    try:
        observed_baseline_sha256 = hashlib.sha256(baseline_path.read_bytes()).hexdigest()
    except OSError as error:
        raise ValueError("qualification disk baseline is missing") from error
    if baseline_sha256 != observed_baseline_sha256:
        raise ValueError("qualification disk baseline hash does not match")
    expected_source = {
        "git_sha": source.get("git_sha"),
        "submodule_gitlink": source.get("submodule_gitlink"),
        "submodule_commit": source.get("submodule_commit"),
        "submodule_diff_sha256": source.get("submodule_diff_sha256"),
        "reviewed_commit": source.get("reviewed_commit"),
        "reviewed_diff_sha256": source.get("reviewed_diff_sha256"),
        "reviewed_tree_sha256": source.get("reviewed_tree_sha256"),
    }
    if not all(isinstance(value, str) and value for value in expected_source.values()):
        raise ValueError("source attestation is missing immutable source identities")
    if source.get("superproject_dirty") is not False or source.get("submodule_dirty") is not False:
        raise ValueError("source attestation is dirty")
    reviewed_manifest_path = (
        Path(str(source.get("reviewed_source_manifest", ""))).expanduser().resolve()
    )
    if reviewed_manifest_path != root / "reviewed-source" / "reviewed-source.json":
        raise ValueError("reviewed-source manifest is outside the claimed run root")
    reviewed_manifest_sha256 = source.get("reviewed_source_manifest_sha256")
    try:
        observed_reviewed_manifest_sha256 = hashlib.sha256(
            reviewed_manifest_path.read_bytes()
        ).hexdigest()
    except OSError as error:
        raise ValueError("reviewed-source manifest is missing") from error
    if reviewed_manifest_sha256 != observed_reviewed_manifest_sha256:
        raise ValueError("reviewed-source manifest hash does not match source attestation")
    try:
        rows = list(csv.DictReader(rows_path.open(encoding="utf-8"), delimiter="\t"))
    except (OSError, csv.Error) as error:
        raise ValueError("could not read qualification status rows") from error
    try:
        observed_seeds = {int(row.get("seed", "-1")) for row in rows}
    except (TypeError, ValueError) as error:
        raise ValueError("qualification status has an invalid seed") from error
    if len(rows) != len(expected_seeds) or observed_seeds != set(expected_seeds):
        raise ValueError("qualification status rows do not cover the exact seed set")
    run_directories = {Path(row.get("run_dir", "")).expanduser().resolve() for row in rows}
    if len(run_directories) != len(rows):
        raise ValueError("qualification status rows reuse a run directory")
    for row in rows:
        seed = int(row["seed"])
        if row.get("exit_code") != "0" or row.get("accepted") != "true":
            raise ValueError(f"seed {seed} did not finish with accepted outer status")
        row_provenance = {
            "run_set_id": row.get("run_set_id"),
            "expected_git_sha": row.get("expected_git_sha"),
            "attempt_id": row.get("attempt_id"),
            "slurm_job_id": row.get("slurm_job_id"),
            "slurm_restart_count": int(row.get("slurm_restart_count", "-1")),
            "started_at": row.get("started_at"),
        }
        validate_attempt_provenance(row_provenance, manifest, artifact_name=f"seed {seed} status")
        if (
            not row.get("git_head_before")
            or not row.get("git_head_after")
            or row.get("git_head_before") != row.get("git_head_after")
        ):
            raise ValueError(f"seed {seed} source SHA is not stable")
        if row.get("git_head_before") != expected_source["git_sha"]:
            raise ValueError(f"seed {seed} source SHA does not match source attestation")
        if (
            row.get("superproject_dirty_before") != "false"
            or row.get("superproject_dirty_after") != "false"
            or row.get("submodule_dirty_before") != "false"
            or row.get("submodule_dirty_after") != "false"
        ):
            raise ValueError(f"seed {seed} source attestation is dirty")
        for field in (
            "submodule_commit",
            "submodule_gitlink",
            "submodule_diff_sha256",
        ):
            before = row.get(f"{field}_before")
            after = row.get(f"{field}_after")
            expected = expected_source[field]
            if before != after or before != expected:
                raise ValueError(f"seed {seed} {field} provenance mismatch")
        for field, row_field in (
            ("reviewed_commit", "reviewed_commit"),
            ("reviewed_diff_sha256", "reviewed_source_diff_sha256"),
            ("reviewed_tree_sha256", "reviewed_tree_sha256"),
        ):
            before = row.get(f"{row_field}_before")
            after = row.get(f"{row_field}_after")
            expected = expected_source[field]
            if before != after or before != expected:
                raise ValueError(f"seed {seed} {field} provenance mismatch")
        run_dir = Path(row.get("run_dir", "")).expanduser().resolve()
        if not run_dir.is_dir():
            raise ValueError(f"seed {seed} run directory is missing")
        gates_path = run_dir / "engineering-gates.json"
        if not gates_path.is_file():
            raise ValueError(f"seed {seed} engineering gates are missing")
        events_path = run_dir / "events.jsonl"
        for artifact_name, artifact_path, status_field in (
            ("event journal", events_path, "events_sha256"),
            ("engineering gates", gates_path, "engineering_gates_sha256"),
        ):
            expected_sha256 = row.get(status_field)
            try:
                observed_sha256 = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
            except OSError as error:
                raise ValueError(f"seed {seed} {artifact_name} is missing") from error
            if expected_sha256 != observed_sha256:
                raise ValueError(f"seed {seed} {artifact_name} hash mismatch")
        gates = json.loads(gates_path.read_text(encoding="utf-8"))
        if not isinstance(gates, Mapping):
            raise ValueError(f"seed {seed} engineering gates are not an object")
        gate_evidence = gates.get("evidence")
        if not isinstance(gate_evidence, Mapping):
            raise ValueError(f"seed {seed} engineering gates have no evidence")
        validate_attempt_provenance(
            gate_evidence,
            manifest,
            artifact_name=f"seed {seed} engineering gates",
        )
        if _reference_identity(gate_evidence.get("source_attestation")) != _reference_identity(
            source_reference
        ):
            raise ValueError(f"seed {seed} source attestation provenance mismatch")
        if _reference_identity(gate_evidence.get("recovery_evidence")) != _reference_identity(
            evidence.get("recovery_evidence")
        ):
            raise ValueError(f"seed {seed} recovery evidence provenance mismatch")
        if _reference_identity(gate_evidence.get("natural_run_baseline")) != _reference_identity(
            evidence.get("natural_run_baseline")
        ):
            raise ValueError(f"seed {seed} disk baseline provenance mismatch")
        if (
            gate_evidence.get("evidence_kind") != "three-seed-qualification"
            or gate_evidence.get("diagnostic_only") is not False
            or Path(str(gate_evidence.get("manifest_path", ""))).expanduser().resolve()
            != qualification_path
        ):
            raise ValueError(f"seed {seed} qualification manifest provenance mismatch")
        if gate_evidence.get("seed_ids") != list(expected_seeds):
            raise ValueError(f"seed {seed} engineering gates have wrong seed provenance")
        gate_results = gates.get("gates")
        if (
            gates.get("accepted") is not True
            or not isinstance(gate_results, Mapping)
            or len(gate_results) != 37
            or any(
                not isinstance(result, Mapping) or result.get("passed") is not True
                for result in gate_results.values()
            )
        ):
            raise ValueError(f"seed {seed} engineering gates are not 37/37 accepted")
        try:
            events = [
                json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()
            ]
        except (OSError, ValueError, TypeError) as error:
            raise ValueError(f"seed {seed} event journal is invalid") from error
        episode_results = [
            event
            for event in events
            if isinstance(event, Mapping) and event.get("event_type") == "episode_result"
        ]
        if len(episode_results) != 1:
            raise ValueError(f"seed {seed} does not have exactly one terminal episode result")
        if not events or any(event.get("run_id") != run_dir.name for event in events):
            raise ValueError(f"seed {seed} event journal has inconsistent run identity")
        episode_ids = {event.get("episode_id") for event in events}
        if len(episode_ids) != 1 or None in episode_ids:
            raise ValueError(f"seed {seed} event journal has inconsistent episode identity")
        result_payload = episode_results[0].get("payload")
        if (
            not isinstance(result_payload, Mapping)
            or result_payload.get("seed") != seed
            or result_payload.get("run_id") != run_dir.name
        ):
            raise ValueError(f"seed {seed} run identity does not match its status row")


def create_only_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Write JSON through exclusive-create semantics for terminal artifacts."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as error:
        raise AttemptClaimError(f"artifact already exists: {target}") from error


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    claim = subparsers.add_parser("claim")
    claim.add_argument("run_set_dir", type=Path)
    claim.add_argument("--expected-git-sha", required=True)
    claim.add_argument("--run-set-id")
    claim.add_argument("--slurm-job-id")
    claim.add_argument("--slurm-restart-count")
    arguments = parser.parse_args()
    if arguments.command == "claim":
        claim_attempt_root(
            arguments.run_set_dir,
            expected_git_sha=arguments.expected_git_sha,
            run_set_id=arguments.run_set_id,
            slurm_job_id=arguments.slurm_job_id,
            slurm_restart_count=arguments.slurm_restart_count,
        )


if __name__ == "__main__":
    main()
