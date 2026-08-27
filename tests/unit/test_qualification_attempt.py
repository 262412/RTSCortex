from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from rtscortex.evaluation.report import ReportError, _load_qualification_evidence
from scripts.qualification_attempt import (
    AttemptClaimError,
    claim_attempt_root,
    create_only_json,
    load_attempt_manifest,
    validate_attempt_provenance,
    validate_recovery_reference,
    validate_source_qualification,
)

GIT_SHA = "a" * 40


def _write_valid_source_qualification(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    run_set = tmp_path / "qualification-001"
    manifest = claim_attempt_root(
        run_set,
        expected_git_sha=GIT_SHA,
        slurm_job_id="12345",
    )
    attempt = {
        field: manifest[field]
        for field in (
            "run_set_id",
            "expected_git_sha",
            "attempt_id",
            "slurm_job_id",
            "slurm_restart_count",
            "started_at",
        )
    }
    source = run_set / "source-attestation.json"
    reviewed_source = run_set / "reviewed-source"
    reviewed_source.mkdir()
    reviewed_manifest = reviewed_source / "reviewed-source.json"
    reviewed_manifest.write_text('{"reviewed":true}\n', encoding="utf-8")
    source_payload = {
        "format_version": "1.0",
        "artifact_kind": "qualification-source-attestation",
        **attempt,
        "attempt": attempt,
        "git_sha": GIT_SHA,
        "superproject_dirty": False,
        "submodule_gitlink": "b" * 40,
        "submodule_commit": "b" * 40,
        "submodule_dirty": False,
        "submodule_diff_sha256": "c" * 64,
        "reviewed_commit": "b" * 40,
        "reviewed_diff_sha256": "d" * 64,
        "reviewed_tree_sha256": "e" * 64,
        "reviewed_source_manifest": str(reviewed_manifest.resolve()),
        "reviewed_source_manifest_sha256": hashlib.sha256(
            reviewed_manifest.read_bytes()
        ).hexdigest(),
    }
    source.write_text(json.dumps(source_payload), encoding="utf-8")
    recovery = run_set / "recovery-canary.json"
    recovery.write_text(
        json.dumps(
            {
                **attempt,
                "attempt": attempt,
                "passed": True,
                "seed_ids": [0, 1, 2],
            }
        ),
        encoding="utf-8",
    )
    source_reference = {
        "path": str(source.resolve()),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    baseline = run_set / "baseline.json"
    baseline.write_text('{"natural_run_bytes_per_game_loop":1}\n', encoding="utf-8")
    recovery_reference = {
        "path": str(recovery.resolve()),
        "sha256": hashlib.sha256(recovery.read_bytes()).hexdigest(),
    }
    baseline_reference = {
        "path": str(baseline.resolve()),
        "sha256": hashlib.sha256(baseline.read_bytes()).hexdigest(),
    }
    evidence = run_set / "qualification-evidence.json"
    evidence_payload = {
        **attempt,
        "attempt": attempt,
        "evidence_kind": "three-seed-qualification",
        "diagnostic_only": False,
        "seed_ids": [0, 1, 2],
        "recovery_evidence": recovery_reference,
        "source_attestation": source_reference,
        "natural_run_baseline": baseline_reference,
    }
    evidence.write_text(
        json.dumps(evidence_payload),
        encoding="utf-8",
    )
    header = (
        "seed",
        "exit_code",
        "run_dir",
        "events_sha256",
        "engineering_gates_sha256",
        "accepted",
        "run_set_id",
        "expected_git_sha",
        "attempt_id",
        "slurm_job_id",
        "slurm_restart_count",
        "started_at",
        "git_head_before",
        "git_head_after",
        "superproject_dirty_before",
        "superproject_dirty_after",
        "submodule_commit_before",
        "submodule_commit_after",
        "submodule_dirty_before",
        "submodule_dirty_after",
        "submodule_gitlink_before",
        "submodule_gitlink_after",
        "submodule_diff_sha256_before",
        "submodule_diff_sha256_after",
        "reviewed_commit_before",
        "reviewed_commit_after",
        "reviewed_source_diff_sha256_before",
        "reviewed_source_diff_sha256_after",
        "reviewed_tree_sha256_before",
        "reviewed_tree_sha256_after",
    )
    rows = []
    for seed in (0, 1, 2):
        run_dir = tmp_path / f"cortex-seed-{seed}"
        run_dir.mkdir()
        events_path = run_dir / "events.jsonl"
        events_path.write_text(
            json.dumps(
                {
                    "run_id": run_dir.name,
                    "episode_id": "episode-0",
                    "event_type": "episode_result",
                    "payload": {"seed": seed, "run_id": run_dir.name},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        gates_path = run_dir / "engineering-gates.json"
        gates_path.write_text(
            json.dumps(
                {
                    "accepted": True,
                    "gates": {f"gate-{index}": {"passed": True} for index in range(37)},
                    "evidence": {
                        **evidence_payload,
                        "manifest_path": str(evidence.resolve()),
                        "natural_run_baseline": {
                            **baseline_reference,
                            "source_issue": "SCX-PT-034",
                            "bytes_per_game_loop": 1.0,
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        rows.append(
            (
                str(seed),
                "0",
                str(run_dir),
                hashlib.sha256(events_path.read_bytes()).hexdigest(),
                hashlib.sha256(gates_path.read_bytes()).hexdigest(),
                "true",
                *(
                    str(attempt[field])
                    for field in (
                        "run_set_id",
                        "expected_git_sha",
                        "attempt_id",
                        "slurm_job_id",
                        "slurm_restart_count",
                        "started_at",
                    )
                ),
                GIT_SHA,
                GIT_SHA,
                "false",
                "false",
                "b" * 40,
                "b" * 40,
                "false",
                "false",
                "b" * 40,
                "b" * 40,
                "c" * 64,
                "c" * 64,
                "b" * 40,
                "b" * 40,
                "d" * 64,
                "d" * 64,
                "e" * 64,
                "e" * 64,
            )
        )
    status = run_set / "qualification-status.tsv"
    status.write_text(
        "\t".join(header) + "\n" + "".join("\t".join(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return run_set, manifest


def test_fresh_root_is_atomically_claimed_with_attempt_manifest(tmp_path: Path) -> None:
    run_set = tmp_path / "qualification-001"

    manifest = claim_attempt_root(
        run_set,
        expected_git_sha=GIT_SHA,
        slurm_job_id="12345",
        slurm_restart_count="0",
    )

    assert run_set.is_dir()
    payload = json.loads((run_set / "attempt-manifest.json").read_text(encoding="utf-8"))
    assert payload == manifest
    assert payload["run_set_id"] == run_set.name
    assert payload["expected_git_sha"] == GIT_SHA
    assert payload["attempt_id"]
    assert payload["slurm_job_id"] == "12345"
    assert payload["slurm_restart_count"] == 0
    assert payload["started_at"]


def test_fresh_claim_allows_create_only_preparation_artifacts(tmp_path: Path) -> None:
    run_set = tmp_path / "qualification-001"
    manifest = claim_attempt_root(run_set, expected_git_sha=GIT_SHA)

    create_only_json(run_set / "recovery-canary.json", {**manifest, "passed": True})
    create_only_json(run_set / "qualification-evidence.json", manifest)
    (run_set / "source-status.txt").open("x", encoding="utf-8").close()
    reviewed_source = run_set / "reviewed-source"
    reviewed_source.mkdir()

    assert (run_set / "attempt-manifest.json").is_file()
    assert (run_set / "recovery-canary.json").is_file()
    assert (run_set / "qualification-evidence.json").is_file()
    assert (run_set / "source-status.txt").is_file()
    assert reviewed_source.is_dir()


def test_source_qualification_joins_one_attempt_across_all_seed_artifacts(
    tmp_path: Path,
) -> None:
    run_set, manifest = _write_valid_source_qualification(tmp_path)

    validate_source_qualification(
        run_set_dir=run_set,
        manifest=manifest,
        source_attestation_path=run_set / "source-attestation.json",
        recovery_path=run_set / "recovery-canary.json",
        qualification_evidence_path=run_set / "qualification-evidence.json",
        status_path=run_set / "qualification-status.tsv",
    )


def test_source_qualification_rejects_seed_run_identity_mismatch(tmp_path: Path) -> None:
    run_set, manifest = _write_valid_source_qualification(tmp_path)
    events = tmp_path / "cortex-seed-1" / "events.jsonl"
    payload = json.loads(events.read_text(encoding="utf-8"))
    payload["payload"]["seed"] = 0
    events.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    status = run_set / "qualification-status.tsv"
    lines = status.read_text(encoding="utf-8").splitlines()
    header = lines[0].split("\t")
    seed_index = header.index("seed")
    events_hash_index = header.index("events_sha256")
    for index, line in enumerate(lines[1:], start=1):
        fields = line.split("\t")
        if fields[seed_index] == "1":
            fields[events_hash_index] = hashlib.sha256(events.read_bytes()).hexdigest()
            lines[index] = "\t".join(fields)
            break
    status.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="run identity"):
        validate_source_qualification(
            run_set_dir=run_set,
            manifest=manifest,
            source_attestation_path=run_set / "source-attestation.json",
            recovery_path=run_set / "recovery-canary.json",
            qualification_evidence_path=run_set / "qualification-evidence.json",
            status_path=run_set / "qualification-status.tsv",
        )


def test_source_qualification_rejects_changed_seed_gate_hash(tmp_path: Path) -> None:
    run_set, manifest = _write_valid_source_qualification(tmp_path)
    gates = tmp_path / "cortex-seed-2" / "engineering-gates.json"
    payload = json.loads(gates.read_text(encoding="utf-8"))
    payload["accepted"] = False
    gates.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="engineering gates hash mismatch"):
        validate_source_qualification(
            run_set_dir=run_set,
            manifest=manifest,
            source_attestation_path=run_set / "source-attestation.json",
            recovery_path=run_set / "recovery-canary.json",
            qualification_evidence_path=run_set / "qualification-evidence.json",
            status_path=run_set / "qualification-status.tsv",
        )


def test_source_qualification_rejects_wrong_source_sha(tmp_path: Path) -> None:
    run_set, manifest = _write_valid_source_qualification(tmp_path)
    source = run_set / "source-attestation.json"
    source_payload = json.loads(source.read_text(encoding="utf-8"))
    source_payload["git_sha"] = "f" * 40
    source.write_text(json.dumps(source_payload), encoding="utf-8")
    evidence = run_set / "qualification-evidence.json"
    evidence_payload = json.loads(evidence.read_text(encoding="utf-8"))
    evidence_payload["source_attestation"]["sha256"] = hashlib.sha256(
        source.read_bytes()
    ).hexdigest()
    evidence.write_text(json.dumps(evidence_payload), encoding="utf-8")

    with pytest.raises(ValueError, match="source attestation identity"):
        validate_source_qualification(
            run_set_dir=run_set,
            manifest=manifest,
            source_attestation_path=source,
            recovery_path=run_set / "recovery-canary.json",
            qualification_evidence_path=evidence,
            status_path=run_set / "qualification-status.tsv",
        )


def test_second_invocation_refuses_without_changing_hashes_or_mtime(tmp_path: Path) -> None:
    run_set = tmp_path / "qualification-001"
    claim_attempt_root(run_set, expected_git_sha=GIT_SHA)
    marker = run_set / "qualification-status.tsv"
    marker.write_text("immutable\n", encoding="utf-8")
    before = {
        path: (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
        for path in run_set.rglob("*")
        if path.is_file()
    }

    with pytest.raises(AttemptClaimError, match="already exists"):
        claim_attempt_root(run_set, expected_git_sha=GIT_SHA)

    after = {
        path: (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
        for path in run_set.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_restart_count_one_refuses_before_creating_parent_or_root(tmp_path: Path) -> None:
    parent = tmp_path / "not-created"
    run_set = parent / "qualification-001"

    with pytest.raises(AttemptClaimError, match="restart count"):
        claim_attempt_root(
            run_set,
            expected_git_sha=GIT_SHA,
            slurm_job_id="12345",
            slurm_restart_count="1",
        )

    assert not parent.exists()


def test_attempt_manifest_rejects_wrong_run_set_identity(tmp_path: Path) -> None:
    run_set = tmp_path / "qualification-001"
    claim_attempt_root(run_set, expected_git_sha=GIT_SHA)
    manifest_path = run_set / "attempt-manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["run_set_id"] = "another-run-set"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="run_set_id"):
        load_attempt_manifest(run_set)


def test_post_preemption_replay_refuses_and_leaves_artifacts_unchanged(tmp_path: Path) -> None:
    run_set = tmp_path / "qualification-001"
    claim_attempt_root(run_set, expected_git_sha=GIT_SHA)
    recovery = run_set / "recovery-canary.json"
    recovery.write_text('{"passed": false}\n', encoding="utf-8")
    before = recovery.stat().st_mtime_ns, hashlib.sha256(recovery.read_bytes()).hexdigest()

    with pytest.raises(AttemptClaimError):
        claim_attempt_root(run_set, expected_git_sha=GIT_SHA)

    assert (
        recovery.stat().st_mtime_ns,
        hashlib.sha256(recovery.read_bytes()).hexdigest(),
    ) == before


def test_recovery_reference_requires_matching_hash_and_attempt_provenance(tmp_path: Path) -> None:
    run_set = tmp_path / "qualification-001"
    manifest = claim_attempt_root(run_set, expected_git_sha=GIT_SHA)
    recovery = run_set / "recovery-canary.json"
    recovery_payload = {**manifest, "passed": True, "seed_ids": [0, 1, 2]}
    recovery.write_text(json.dumps(recovery_payload, sort_keys=True) + "\n", encoding="utf-8")
    evidence = {
        **manifest,
        "recovery_evidence": {
            "path": str(recovery.resolve()),
            "sha256": hashlib.sha256(recovery.read_bytes()).hexdigest(),
        },
        "seed_ids": [0, 1, 2],
    }

    validate_attempt_provenance(recovery_payload, manifest, artifact_name="recovery")
    validate_recovery_reference(evidence, manifest, recovery)

    tampered_hash = {
        **evidence,
        "recovery_evidence": {
            **evidence["recovery_evidence"],
            "sha256": "0" * 64,
        },
    }
    with pytest.raises(ValueError, match="recovery evidence hash"):
        validate_recovery_reference(tampered_hash, manifest, recovery)

    tampered_seed = {**evidence, "seed_ids": [0, 1, 9]}
    with pytest.raises(ValueError, match="seed provenance"):
        validate_recovery_reference(tampered_seed, manifest, recovery)


def test_attempt_provenance_rejects_mismatched_gate_or_seed_row() -> None:
    manifest = {
        "run_set_id": "qualification-001",
        "expected_git_sha": GIT_SHA,
        "attempt_id": "attempt-001",
        "slurm_job_id": "12345",
        "slurm_restart_count": 0,
        "started_at": "2026-08-13T00:00:00+00:00",
    }

    with pytest.raises(ValueError, match="attempt provenance"):
        validate_attempt_provenance(
            {**manifest, "attempt_id": "attempt-other"},
            manifest,
            artifact_name="engineering gates",
        )


def test_report_loader_flattens_and_checks_attempt_provenance(tmp_path: Path) -> None:
    run_set = tmp_path / "qualification-001"
    manifest = claim_attempt_root(run_set, expected_git_sha=GIT_SHA)
    attempt = {
        field: manifest[field]
        for field in (
            "run_set_id",
            "expected_git_sha",
            "attempt_id",
            "slurm_job_id",
            "slurm_restart_count",
            "started_at",
        )
    }
    recovery = run_set / "recovery-canary.json"
    recovery.write_text(
        json.dumps(
            {
                "format_version": "1.1",
                "git_sha": GIT_SHA,
                **attempt,
                "attempt": attempt,
                "passed": True,
                "seed_ids": [0, 1, 2],
            }
        ),
        encoding="utf-8",
    )
    source = run_set / "source-attestation.json"
    source.write_text(
        json.dumps({"git_sha": GIT_SHA, **attempt, "attempt": attempt}),
        encoding="utf-8",
    )
    baseline = run_set / "baseline.json"
    baseline.write_text(
        json.dumps({"source_issue": "SCX-PT-034", "natural_run_bytes_per_game_loop": 1.0}),
        encoding="utf-8",
    )
    evidence = {
        "format_version": "1.0",
        "evidence_kind": "three-seed-qualification",
        "diagnostic_only": False,
        **attempt,
        "attempt": attempt,
        "seed_ids": [0, 1, 2],
        "recovery_evidence": {
            "path": str(recovery),
            "sha256": hashlib.sha256(recovery.read_bytes()).hexdigest(),
        },
        "source_attestation": {
            "path": str(source),
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        },
        "natural_run_baseline": {
            "path": str(baseline),
            "sha256": hashlib.sha256(baseline.read_bytes()).hexdigest(),
        },
    }
    evidence_path = run_set / "qualification-evidence.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")

    _baseline, _recovery, _sha, loaded = _load_qualification_evidence(evidence_path)
    assert loaded["attempt_id"] == manifest["attempt_id"]
    assert loaded["attempt"]["run_set_id"] == manifest["run_set_id"]
    recovery_payload = json.loads(recovery.read_text(encoding="utf-8"))
    recovery_payload["attempt_id"] = "attempt:wrong"
    recovery.write_text(json.dumps(recovery_payload), encoding="utf-8")
    evidence["recovery_evidence"]["sha256"] = hashlib.sha256(recovery.read_bytes()).hexdigest()
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    with pytest.raises(ReportError, match="attempt provenance"):
        _load_qualification_evidence(evidence_path)


def test_report_loader_rejects_nested_attempt_mismatch(tmp_path: Path) -> None:
    run_set = tmp_path / "qualification-001"
    manifest = claim_attempt_root(run_set, expected_git_sha=GIT_SHA)
    attempt = {
        field: manifest[field]
        for field in (
            "run_set_id",
            "expected_git_sha",
            "attempt_id",
            "slurm_job_id",
            "slurm_restart_count",
            "started_at",
        )
    }
    recovery = run_set / "recovery-canary.json"
    recovery.write_text(
        json.dumps(
            {
                "format_version": "1.1",
                **attempt,
                "attempt": attempt,
                "git_sha": GIT_SHA,
                "seed_ids": [0, 1, 2],
            }
        ),
        encoding="utf-8",
    )
    baseline = run_set / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "source_issue": "SCX-PT-034",
                "natural_run_bytes_per_game_loop": 1.0,
            }
        ),
        encoding="utf-8",
    )
    evidence = {
        "format_version": "1.0",
        "evidence_kind": "three-seed-qualification",
        "diagnostic_only": False,
        **attempt,
        "attempt": {**attempt, "attempt_id": "attempt:wrong"},
        "seed_ids": [0, 1, 2],
        "recovery_evidence": {
            "path": str(recovery),
            "sha256": hashlib.sha256(recovery.read_bytes()).hexdigest(),
        },
        "natural_run_baseline": {
            "path": str(baseline),
            "sha256": hashlib.sha256(baseline.read_bytes()).hexdigest(),
        },
    }
    evidence_path = run_set / "qualification-evidence.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")

    with pytest.raises(ReportError, match="attempt provenance"):
        _load_qualification_evidence(evidence_path)


def test_three_seed_wrapper_refuses_replay_before_recovery_and_preserves_artifacts(
    tmp_path: Path,
) -> None:
    root = tmp_path / "qualification-001"
    root.mkdir()
    for name in (
        "source-status.txt",
        "qualification-metadata.txt",
        "qualification-status.tsv",
        "recovery-canary.json",
        "qualification-evidence.json",
        "seed-0.log",
    ):
        (root / name).write_text(f"old:{name}\n", encoding="utf-8")
    reviewed_source = root / "reviewed-source"
    reviewed_source.mkdir()
    (reviewed_source / "reviewed-source.json").write_text(
        '{"immutable":true}\n',
        encoding="utf-8",
    )
    before = {
        path: (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
    }
    script = Path(__file__).parents[2] / "scripts" / "run_protoss_three_seed_qualification.sh"
    result = subprocess.run(
        ["bash", str(script), str(root), "--expected-git-sha", GIT_SHA],
        env={
            **os.environ,
            "RTSCORTEX_CORE_PYTHON": "/bin/true",
            "RTSCORTEX_CORE_CLI": "/bin/true",
            "RTSCORTEX_CLAIM_PYTHON": "python3",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "already exists" in result.stderr
    assert "recovery" not in result.stderr.lower()
    after = {
        path: (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_three_seed_wrapper_restart_one_does_not_create_root_or_recovery(tmp_path: Path) -> None:
    root = tmp_path / "qualification-001"
    script = Path(__file__).parents[2] / "scripts" / "run_protoss_three_seed_qualification.sh"
    result = subprocess.run(
        ["bash", str(script), str(root), "--expected-git-sha", GIT_SHA],
        env={
            **os.environ,
            "SLURM_RESTART_COUNT": "1",
            "RTSCORTEX_CORE_PYTHON": "/bin/true",
            "RTSCORTEX_CORE_CLI": "/bin/true",
            "RTSCORTEX_CLAIM_PYTHON": "python3",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "restart count" in result.stderr
    assert not root.exists()
