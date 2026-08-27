#!/usr/bin/env bash
set -euo pipefail
set -o noclobber
export PYTHONDONTWRITEBYTECODE=1

if [[ $# -ne 3 || "$2" != "--expected-git-sha" ]]; then
  echo "usage: $0 <run-set-dir> --expected-git-sha <sha>" >&2
  exit 2
fi

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_set_dir="$1"
expected_git_sha="$3"
core_python="${RTSCORTEX_CORE_PYTHON:-/mnt/fastscratch/users/tbczhang/envs/rtscortex-core/bin/python}"
core_cli="${RTSCORTEX_CORE_CLI:-/mnt/fastscratch/users/tbczhang/envs/rtscortex-core/bin/rtscortex}"
config="${repo_dir}/configs/experiments/live_simple64_hima_protoss_ensemble_cortex_v0_5_qualification.yaml"
engineering_baseline="${repo_dir}/configs/acceptance/protoss_natural_terminal_v1.json"
working_playbook="/mnt/scratch/users/tbczhang/outputs/RTSCortex/cortex-playbook-qualification-working.sqlite3"
seeds=(0 1 2)

if [[ ! "${expected_git_sha}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "expected Git revision must be a full lowercase SHA" >&2
  exit 2
fi
if [[ ! -x "${core_python}" || ! -x "${core_cli}" ]]; then
  echo "RTSCortex core Python/CLI is missing or not executable" >&2
  exit 2
fi
export PYTHONPATH="${repo_dir}/src:${repo_dir}/integrations/llm_pysc2/src${PYTHONPATH:+:${PYTHONPATH}}"
claim_python="${RTSCORTEX_CLAIM_PYTHON:-${core_python}}"
"${claim_python}" "${repo_dir}/scripts/qualification_attempt.py" claim \
  "${run_set_dir}" \
  --expected-git-sha "${expected_git_sha}" \
  --slurm-job-id "${SLURM_JOB_ID:-unknown}" \
  --slurm-restart-count "${SLURM_RESTART_COUNT:-0}"
run_set_dir="$(readlink -f "${run_set_dir}")"
recovery_evidence="${run_set_dir}/recovery-canary.json"
qualification_evidence="${run_set_dir}/qualification-evidence.json"
reviewed_source_root="${run_set_dir}/reviewed-source"
status_file="${run_set_dir}/qualification-status.tsv"
attempt_manifest="${run_set_dir}/attempt-manifest.json"
attempt_manifest_sha256="$(sha256sum "${attempt_manifest}" | awk '{print $1}')"
IFS=$'\t' read -r attempt_run_set_id attempt_expected_git_sha attempt_id attempt_job_id \
  attempt_restart_count attempt_started_at < <(
  "${claim_python}" - "${attempt_manifest}" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
print("\t".join(str(payload[field]) for field in (
    "run_set_id", "expected_git_sha", "attempt_id", "slurm_job_id",
    "slurm_restart_count", "started_at",
)))
PY
)

cd "${repo_dir}"
git_head="$(git rev-parse HEAD)"
superproject_dirty="$(test -n "$(git status --porcelain --ignore-submodules=dirty)" && echo true || echo false)"
submodule_gitlink="$(git ls-tree HEAD third_party/LLM-PySC2 | awk '{print $3}')"
submodule_commit="$(git -C third_party/LLM-PySC2 rev-parse HEAD)"
submodule_dirty="$(test -n "$(git -C third_party/LLM-PySC2 status --porcelain)" && echo true || echo false)"
submodule_diff_sha256="$(git -C third_party/LLM-PySC2 diff --binary | sha256sum | awk '{print $1}')"
if [[ "${git_head}" != "${expected_git_sha}" \
  || "${superproject_dirty}" != "false" \
  || "${submodule_dirty}" != "false" \
  || "${submodule_commit}" != "${submodule_gitlink}" ]]; then
  echo "qualification requires clean ${expected_git_sha} and exact clean submodule ${submodule_gitlink}" >&2
  exit 2
fi

git status --short > "${run_set_dir}/source-status.txt"
"${core_python}" scripts/run_recovery_acceptance_canary.py \
  --expected-git-sha "${expected_git_sha}" \
  --attempt-manifest "${attempt_manifest}" \
  --seed-ids "0,1,2" \
  --output "${recovery_evidence}"

"${core_python}" scripts/prepare_reviewed_llm_pysc2_runtime.py \
  --source third_party/LLM-PySC2 \
  --output-root "${reviewed_source_root}" \
  --patch-directory integrations/llm_pysc2/patches \
  --expected-gitlink "${submodule_gitlink}"
export RTSCORTEX_REVIEWED_SOURCE_ROOT="${reviewed_source_root}"
reviewed_source_manifest="${reviewed_source_root}/reviewed-source.json"
reviewed_source_manifest_sha256="$(sha256sum "${reviewed_source_manifest}" | awk '{print $1}')"
reviewed_llm_pysc2="${reviewed_source_root}/third_party/LLM-PySC2"
reviewed_source_commit="$(git -C "${reviewed_llm_pysc2}" rev-parse HEAD)"
reviewed_source_diff_sha256="$(git -C "${reviewed_llm_pysc2}" diff --binary | sha256sum | awk '{print $1}')"
reviewed_source_tree_sha256="$(
  "${core_python}" -m scripts.hash_reviewed_source_tree \
    "${reviewed_llm_pysc2}" --field reviewed_tree_sha256
)"

source_attestation="${run_set_dir}/source-attestation.json"
"${core_python}" - "${attempt_manifest}" "${source_attestation}" \
  "${expected_git_sha}" "${submodule_gitlink}" "${submodule_commit}" \
  "${submodule_diff_sha256}" "${reviewed_source_manifest}" \
  "${reviewed_source_manifest_sha256}" "${reviewed_source_commit}" \
  "${reviewed_source_diff_sha256}" \
  "${reviewed_source_tree_sha256}" <<'PY'
import sys
from pathlib import Path

from scripts.qualification_attempt import attempt_fields, create_only_json, load_attempt_manifest

(manifest_raw, output_raw, expected_git_sha, submodule_gitlink, submodule_commit,
 submodule_diff_sha256, reviewed_manifest, reviewed_manifest_sha256,
 reviewed_commit, reviewed_diff_sha256, reviewed_tree_sha256) = sys.argv[1:]
manifest = load_attempt_manifest(Path(manifest_raw).parent)
payload = {
    "format_version": "1.0",
    "artifact_kind": "qualification-source-attestation",
    **attempt_fields(manifest),
    "attempt": attempt_fields(manifest),
    "git_sha": expected_git_sha,
    "superproject_dirty": False,
    "submodule_gitlink": submodule_gitlink,
    "submodule_commit": submodule_commit,
    "submodule_dirty": False,
    "submodule_diff_sha256": submodule_diff_sha256,
    "reviewed_commit": reviewed_commit,
    "reviewed_diff_sha256": reviewed_diff_sha256,
    "reviewed_tree_sha256": reviewed_tree_sha256,
    "reviewed_source_manifest": str(Path(reviewed_manifest).resolve()),
    "reviewed_source_manifest_sha256": reviewed_manifest_sha256,
}
create_only_json(output_raw, payload)
PY

"${core_python}" - "${attempt_manifest}" "${qualification_evidence}" \
  "${expected_git_sha}" "${recovery_evidence}" "${engineering_baseline}" \
  "${source_attestation}" <<'PY'
import hashlib
import sys
from pathlib import Path

from scripts.qualification_attempt import attempt_fields, create_only_json, load_attempt_manifest

(manifest_raw, output_raw, expected_git_sha, recovery_raw, baseline_raw,
 source_attestation_raw) = sys.argv[1:]
manifest = load_attempt_manifest(Path(manifest_raw).parent)
recovery = Path(recovery_raw).resolve()
baseline = Path(baseline_raw).resolve()
source_attestation = Path(source_attestation_raw).resolve()
payload = {
    "format_version": "1.0",
    "evidence_kind": "three-seed-qualification",
    "diagnostic_only": False,
    **attempt_fields(manifest),
    "attempt": attempt_fields(manifest),
    "expected_git_sha": expected_git_sha,
    "seed_ids": [0, 1, 2],
    "recovery_evidence": {
        "path": str(recovery),
        "sha256": hashlib.sha256(recovery.read_bytes()).hexdigest(),
    },
    "source_attestation": {
        "path": str(source_attestation),
        "sha256": hashlib.sha256(source_attestation.read_bytes()).hexdigest(),
    },
    "natural_run_baseline": {
        "path": str(baseline),
        "sha256": hashlib.sha256(baseline.read_bytes()).hexdigest(),
    },
}
create_only_json(output_raw, payload)
PY

capture_attestation() {
  source_git_head="$(git rev-parse HEAD)"
  source_superproject_dirty="$(test -n "$(git status --porcelain --ignore-submodules=dirty)" && echo true || echo false)"
  source_submodule_commit="$(git -C third_party/LLM-PySC2 rev-parse HEAD)"
  source_submodule_dirty="$(test -n "$(git -C third_party/LLM-PySC2 status --porcelain)" && echo true || echo false)"
  source_submodule_gitlink="$(git ls-tree HEAD third_party/LLM-PySC2 | awk '{print $3}')"
  source_submodule_diff_sha256="$(git -C third_party/LLM-PySC2 diff --binary | sha256sum | awk '{print $1}')"
  source_reviewed_commit="$(git -C "${reviewed_llm_pysc2}" rev-parse HEAD)"
  source_reviewed_diff_sha256="$(git -C "${reviewed_llm_pysc2}" diff --binary | sha256sum | awk '{print $1}')"
  source_reviewed_tree_sha256="$(
    "${core_python}" -m scripts.hash_reviewed_source_tree \
      "${reviewed_llm_pysc2}" --field reviewed_tree_sha256
  )"
}

attestation_matches() {
  [[ "${source_git_head}" == "${expected_git_sha}" ]] \
    && [[ "${source_superproject_dirty}" == "false" ]] \
    && [[ "${source_submodule_commit}" == "${submodule_commit}" ]] \
    && [[ "${source_submodule_dirty}" == "false" ]] \
    && [[ "${source_submodule_gitlink}" == "${submodule_gitlink}" ]] \
    && [[ "${source_submodule_diff_sha256}" == "${submodule_diff_sha256}" ]] \
    && [[ "${source_reviewed_commit}" == "${reviewed_source_commit}" ]] \
    && [[ "${source_reviewed_diff_sha256}" == "${reviewed_source_diff_sha256}" ]] \
    && [[ "${source_reviewed_tree_sha256}" == "${reviewed_source_tree_sha256}" ]]
}

{
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "evidence_scope=three-seed-qualification"
  echo "diagnostic_only=false"
  echo "formal_24_run=false"
  echo "run_set_id=${attempt_run_set_id}"
  echo "attempt_id=${attempt_id}"
  echo "slurm_job_id=${attempt_job_id}"
  echo "slurm_restart_count=${attempt_restart_count}"
  echo "attempt_started_at=${attempt_started_at}"
  echo "attempt_manifest=${attempt_manifest}"
  echo "attempt_manifest_sha256=${attempt_manifest_sha256}"
  echo "expected_git_sha=${expected_git_sha}"
  echo "submodule_gitlink=${submodule_gitlink}"
  echo "submodule_diff_sha256=${submodule_diff_sha256}"
  echo "reviewed_source_manifest=${reviewed_source_manifest}"
  echo "reviewed_source_manifest_sha256=${reviewed_source_manifest_sha256}"
  echo "reviewed_source_tree_sha256=${reviewed_source_tree_sha256}"
  echo "qualification_evidence=${qualification_evidence}"
} > "${run_set_dir}/qualification-metadata.txt"

printf "seed\texit_code\trun_dir\tevents_sha256\tengineering_gates_sha256\taccepted\trun_set_id\texpected_git_sha\tattempt_id\tslurm_job_id\tslurm_restart_count\tstarted_at\tgit_head_before\tgit_head_after\tsuperproject_dirty_before\tsuperproject_dirty_after\tsubmodule_commit_before\tsubmodule_commit_after\tsubmodule_dirty_before\tsubmodule_dirty_after\tsubmodule_gitlink_before\tsubmodule_gitlink_after\tsubmodule_diff_sha256_before\tsubmodule_diff_sha256_after\treviewed_commit_before\treviewed_commit_after\treviewed_source_diff_sha256_before\treviewed_source_diff_sha256_after\treviewed_tree_sha256_before\treviewed_tree_sha256_after\n" > "${status_file}"
overall_status=0
for seed in "${seeds[@]}"; do
  rm -f "${working_playbook}" "${working_playbook}-shm" "${working_playbook}-wal"
  capture_attestation
  if ! attestation_matches; then
    echo "source attestation changed before seed ${seed}" >&2
    exit 2
  fi
  git_head_before="${source_git_head}"
  superproject_dirty_before="${source_superproject_dirty}"
  submodule_commit_before="${source_submodule_commit}"
  submodule_dirty_before="${source_submodule_dirty}"
  reviewed_commit_before="${source_reviewed_commit}"
  submodule_gitlink_before="${source_submodule_gitlink}"
  submodule_diff_sha256_before="${source_submodule_diff_sha256}"
  reviewed_source_diff_sha256_before="${source_reviewed_diff_sha256}"
  reviewed_tree_sha256_before="${source_reviewed_tree_sha256}"
  log_path="${run_set_dir}/seed-${seed}.log"
  : > "${log_path}"
  set +e
  SC2PATH="/mnt/scratch/users/tbczhang/StarCraftII" \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    "${core_cli}" run \
      --config "${config}" \
      --seed "${seed}" \
      --qualification-evidence "${qualification_evidence}" \
    2>&1 | tee -a "${log_path}"
  run_status=${PIPESTATUS[0]}
  set -e
  run_dir="$(sed -n 's/^Artifacts: //p' "${log_path}" | tail -n 1)"
  events_sha256=""
  engineering_gates_sha256=""
  capture_attestation
  if ! attestation_matches; then
    echo "source attestation changed during seed ${seed}" >&2
    run_status=86
  fi
  accepted=false
  if [[ ${run_status} -eq 0 && -n "${run_dir}" && -f "${run_dir}/engineering-gates.json" ]]; then
    if [[ -f "${run_dir}/events.jsonl" ]]; then
      events_sha256="$(sha256sum "${run_dir}/events.jsonl" | awk '{print $1}')"
    fi
    engineering_gates_sha256="$(sha256sum "${run_dir}/engineering-gates.json" | awk '{print $1}')"
    set +e
    "${core_python}" - "${run_dir}" "${expected_git_sha}" "${attempt_manifest}" "${source_attestation}" <<'PY'
import json
import sys
import hashlib
from pathlib import Path
from scripts.qualification_attempt import load_attempt_manifest, validate_attempt_provenance

run_dir = Path(sys.argv[1])
expected_git_sha = sys.argv[2]
attempt_manifest = Path(sys.argv[3])
source_attestation = Path(sys.argv[4]).resolve()
attempt = load_attempt_manifest(attempt_manifest.parent)
gates = json.loads((run_dir / "engineering-gates.json").read_text(encoding="utf-8"))
metrics = gates.get("metrics", {})
diagnostics = gates.get("diagnostics", {})
evidence = gates.get("evidence", {})
gate_results = gates.get("gates", {})

def exact_zero_integer(value):
    return isinstance(value, int) and not isinstance(value, bool) and value == 0

required = {
    "build_start_coverage": lambda value: value == 1.0,
    "build_confirmation_rate": lambda value: value is not None and value >= 0.9,
    "build_failure_rate": lambda value: value is not None and value <= 0.1,
    "terminal_collapse_non_recovery_macro_dispatch_count": exact_zero_integer,
    "semantic_build_failure_streak_bounded": lambda value: value is True,
    "authoritative_build_pre_dispatch_circuit_bounded": lambda value: value is True,
    "production_confirmation_complete": lambda value: value == 1.0,
    "recovery_evidence_present": lambda value: value is True,
    "checkpoint_tail_recovery_bounded": lambda value: value is True,
    "postgame_semantic_event_coverage": lambda value: value == 1.0,
    "natural_run_disk_reduction_ratio": lambda value: value is not None and value >= 4.0,
    "defense_inventory_within_cap": lambda value: value is True,
    "unchanged_failed_target_redispatch_zero": lambda value: value is True,
}
valid = gates.get("format_version") == "1.2"
valid = valid and gates.get("accepted") is True
valid = valid and all(check(metrics.get(name)) for name, check in required.items())
valid = valid and gate_results.get(
    "terminal_collapse_non_recovery_macro_dispatch_count", {}
).get("passed") is True
valid = valid and diagnostics.get("unchanged_failed_target_redispatch_count") == 0
valid = valid and exact_zero_integer(
    diagnostics.get("terminal_collapse_non_recovery_macro_dispatch_count")
)
valid = valid and exact_zero_integer(
    diagnostics.get("terminal_collapse_macro_lineage_unknown_count")
)
valid = valid and exact_zero_integer(
    diagnostics.get("authoritative_build_pre_dispatch_post_open_command_count")
)
valid = valid and exact_zero_integer(
    diagnostics.get("authoritative_build_pre_dispatch_post_open_dispatch_count")
)
valid = valid and exact_zero_integer(
    diagnostics.get("authoritative_build_pre_dispatch_post_open_rejection_count")
)
valid = valid and exact_zero_integer(
    diagnostics.get("authoritative_build_pre_dispatch_missing_identity_count")
)
for diagnostic_name in (
    "authoritative_build_pre_dispatch_post_open_primitive_count",
    "authoritative_build_pre_dispatch_post_open_approach_primitive_count",
    "authoritative_build_pre_dispatch_raw_circuit_open_violation_count",
    "authoritative_build_pre_dispatch_identity_inconsistency_count",
    "authoritative_build_pre_dispatch_invalid_transition_count",
    "authoritative_build_pre_dispatch_missing_open_count",
    "authoritative_build_pre_dispatch_producer_inconsistency_count",
):
    valid = valid and exact_zero_integer(diagnostics.get(diagnostic_name))
valid = valid and evidence.get("expected_git_sha") == expected_git_sha
valid = valid and evidence.get("diagnostic_only") is False
validate_attempt_provenance(evidence, attempt, artifact_name="engineering gates")
source_reference = evidence.get("source_attestation")
valid = valid and isinstance(source_reference, dict)
valid = valid and Path(str(source_reference.get("path", ""))).expanduser().resolve() == source_attestation
valid = valid and source_reference.get("sha256") == hashlib.sha256(source_attestation.read_bytes()).hexdigest()
raise SystemExit(0 if valid else 1)
PY
    gate_status=$?
    set -e
    if [[ ${gate_status} -eq 0 ]]; then
      accepted=true
    else
      run_status=87
    fi
  fi
  if [[ ${run_status} -ne 0 ]]; then
    overall_status=1
  fi
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${seed}" "${run_status}" "${run_dir}" "${events_sha256}" \
    "${engineering_gates_sha256}" "${accepted}" \
    "${attempt_run_set_id}" "${attempt_expected_git_sha}" "${attempt_id}" \
    "${attempt_job_id}" "${attempt_restart_count}" "${attempt_started_at}" \
    "${git_head_before}" "${source_git_head}" \
    "${superproject_dirty_before}" "${source_superproject_dirty}" \
    "${submodule_commit_before}" "${source_submodule_commit}" \
    "${submodule_dirty_before}" "${source_submodule_dirty}" \
    "${submodule_gitlink_before}" "${source_submodule_gitlink}" \
    "${submodule_diff_sha256_before}" "${source_submodule_diff_sha256}" \
    "${reviewed_commit_before}" "${source_reviewed_commit}" \
    "${reviewed_source_diff_sha256_before}" "${source_reviewed_diff_sha256}" \
    "${reviewed_tree_sha256_before}" "${source_reviewed_tree_sha256}" \
    >> "${status_file}"
done

capture_attestation
if ! attestation_matches; then
  overall_status=1
fi
set +e
"${core_python}" - "${run_set_dir}" "${attempt_manifest}" "${source_attestation}" \
  "${recovery_evidence}" "${qualification_evidence}" "${status_file}" <<'PY'
import sys
from pathlib import Path

from scripts.qualification_attempt import load_attempt_manifest, validate_source_qualification

run_set, manifest_raw, source_raw, recovery_raw, evidence_raw, status_raw = sys.argv[1:]
validate_source_qualification(
    run_set_dir=Path(run_set),
    manifest=load_attempt_manifest(Path(manifest_raw).parent),
    source_attestation_path=Path(source_raw),
    recovery_path=Path(recovery_raw),
    qualification_evidence_path=Path(evidence_raw),
    status_path=Path(status_raw),
)
PY
provenance_status=$?
set -e
if [[ ${provenance_status} -ne 0 ]]; then
  echo "qualification attempt provenance validation failed" >&2
  overall_status=1
fi
{
  echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "final_git_head=${source_git_head}"
  echo "final_submodule_commit=${source_submodule_commit}"
  echo "final_submodule_dirty=${source_submodule_dirty}"
  echo "final_reviewed_source_commit=${source_reviewed_commit}"
  echo "final_reviewed_source_tree_sha256=${source_reviewed_tree_sha256}"
  echo "exit_code=${overall_status}"
} >> "${run_set_dir}/qualification-metadata.txt"
echo "three_seed_qualification status=finished exit_code=${overall_status} run_set=${run_set_dir}"
exit "${overall_status}"
