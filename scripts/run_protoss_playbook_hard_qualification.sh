#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

if [[ $# -ne 4 || "$3" != "--expected-git-sha" ]]; then
  echo "usage: $0 <source-three-seed-run-set> <hard-qualification-run-set> --expected-git-sha <sha>" >&2
  exit 2
fi

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_root="/mnt/scratch/users/tbczhang/outputs/RTSCortex"
source_run_set="$(readlink -f "$1")"
run_set_dir="$2"
expected_git_sha="$4"
sc2_patch="4.10"
core_python="${RTSCORTEX_CORE_PYTHON:-/mnt/fastscratch/users/tbczhang/envs/rtscortex-core/bin/python}"
core_cli="${RTSCORTEX_CORE_CLI:-/mnt/fastscratch/users/tbczhang/envs/rtscortex-core/bin/rtscortex}"
config="${repo_dir}/configs/experiments/live_simple64_hima_protoss_ensemble_cortex_v0_5_hard_qualification_shadow.yaml"
readiness_config="${repo_dir}/configs/experiments/live_simple64_hima_protoss_ensemble_cortex_v0_5_frozen_playbook_natural_terminal.yaml"
engineering_baseline="${repo_dir}/configs/acceptance/protoss_natural_terminal_v1.json"
source_status="${source_run_set}/qualification-status.tsv"
working_playbook="${output_root}/cortex-playbook-hard-qualification-shadow-working.sqlite3"
lock_path="${output_root}/protoss-playbook-paired.lock"
seeds=(0 1 2)

if [[ ! "${expected_git_sha}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "expected Git revision must be a full lowercase SHA" >&2
  exit 2
fi
if [[ ! -f "${source_status}" ]]; then
  echo "source three-seed status does not exist: ${source_status}" >&2
  exit 2
fi
if [[ ! -x "${core_python}" || ! -x "${core_cli}" ]]; then
  echo "RTSCortex core Python/CLI is missing or not executable" >&2
  exit 2
fi
export PYTHONPATH="${repo_dir}/src:${repo_dir}/integrations/llm_pysc2/src${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${run_set_dir}"
run_set_dir="$(readlink -f "${run_set_dir}")"
baseline="${run_set_dir}/playbook.soft-baseline.sqlite3"
probe_output="${run_set_dir}/playbook.hard-shadow-probe.sqlite3"
plan="${run_set_dir}/hard-qualification-plan.json"
report="${run_set_dir}/hard-qualification-report.json"
manifest="${run_set_dir}/hard-qualification-manifest.json"
production_baseline="${run_set_dir}/playbook.production.sqlite3"
readiness_evidence="${run_set_dir}/playbook-hard-readiness.json"
recovery_evidence="${run_set_dir}/recovery-canary.json"
qualification_evidence="${run_set_dir}/qualification-evidence.json"
reviewed_source_root="${run_set_dir}/reviewed-source"
status_file="${run_set_dir}/hard-qualification-status.tsv"

cd "${repo_dir}"
exec 9>"${lock_path}"
if ! flock -n 9; then
  echo "another Protoss Playbook experiment owns ${lock_path}" >&2
  exit 1
fi

git_head="$(git rev-parse HEAD)"
superproject_dirty="$(test -n "$(git status --porcelain --ignore-submodules=dirty)" && echo true || echo false)"
submodule_commit="$(git -C third_party/LLM-PySC2 rev-parse HEAD)"
submodule_dirty="$(test -n "$(git -C third_party/LLM-PySC2 status --porcelain)" && echo true || echo false)"
submodule_gitlink="$(git ls-tree HEAD third_party/LLM-PySC2 | awk '{print $3}')"
submodule_diff_sha256="$(git -C third_party/LLM-PySC2 diff --binary | sha256sum | awk '{print $1}')"
if [[ "${git_head}" != "${expected_git_sha}" \
  || "${superproject_dirty}" != "false" \
  || "${submodule_dirty}" != "false" \
  || "${submodule_commit}" != "${submodule_gitlink}" ]]; then
  echo "hard qualification requires clean ${expected_git_sha} and exact clean gitlink ${submodule_gitlink}" >&2
  exit 2
fi

git status --short > "${run_set_dir}/source-status.txt"
"${core_python}" scripts/run_recovery_acceptance_canary.py \
  --expected-git-sha "${expected_git_sha}" \
  --output "${recovery_evidence}"

"${core_python}" - \
  "${expected_git_sha}" \
  "${recovery_evidence}" \
  "${engineering_baseline}" \
  "${qualification_evidence}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

expected_git_sha, recovery_raw, baseline_raw, output_raw = sys.argv[1:]
recovery = Path(recovery_raw).resolve()
baseline = Path(baseline_raw).resolve()
payload = {
    "format_version": "1.0",
    "evidence_kind": "three-seed-qualification",
    "diagnostic_only": False,
    "expected_git_sha": expected_git_sha,
    "recovery_evidence": {
        "path": str(recovery),
        "sha256": hashlib.sha256(recovery.read_bytes()).hexdigest(),
    },
    "natural_run_baseline": {
        "path": str(baseline),
        "sha256": hashlib.sha256(baseline.read_bytes()).hexdigest(),
    },
}
Path(output_raw).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

"${core_python}" scripts/prepare_reviewed_llm_pysc2_runtime.py \
  --source third_party/LLM-PySC2 \
  --output-root "${reviewed_source_root}" \
  --patch-directory integrations/llm_pysc2/patches \
  --expected-gitlink "${submodule_gitlink}"
export RTSCORTEX_REVIEWED_SOURCE_ROOT="${reviewed_source_root}"
reviewed_source_manifest="${reviewed_source_root}/reviewed-source.json"
reviewed_source_manifest_sha256="$(sha256sum "${reviewed_source_manifest}" | awk '{print $1}')"
reviewed_llm_pysc2="${reviewed_source_root}/third_party/LLM-PySC2"
reviewed_source_diff_sha256="$(git -C "${reviewed_llm_pysc2}" diff --binary | sha256sum | awk '{print $1}')"
reviewed_source_tree_sha256="$(
  "${core_python}" -m scripts.hash_reviewed_source_tree \
    "${reviewed_llm_pysc2}" --field reviewed_tree_sha256
)"

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
    && [[ "${source_reviewed_commit}" == "${submodule_gitlink}" ]] \
    && [[ "${source_reviewed_diff_sha256}" == "${reviewed_source_diff_sha256}" ]] \
    && [[ "${source_reviewed_tree_sha256}" == "${reviewed_source_tree_sha256}" ]]
}

mapfile -t source_runs < <(awk -F $'\t' 'NR > 1 {print $3}' "${source_status}")
if [[ ${#source_runs[@]} -ne 3 ]]; then
  echo "source qualification must contain exactly three runs" >&2
  exit 2
fi
source_run_args=()
for source_run in "${source_runs[@]}"; do
  source_run_args+=(--source-run-dir "${source_run}")
done
"${core_python}" -m scripts.prepare_playbook_hard_qualification \
  "${source_run_args[@]}" \
  --source-status "${source_status}" \
  --baseline-output "${baseline}" \
  --probe-output "${probe_output}" \
  --plan-output "${plan}" \
  --expected-git-sha "${expected_git_sha}" \
  --sc2-patch "${sc2_patch}" \
  --max-probes-per-batch 8 \
  --max-probe-batches 2
eligible_parent_count="$(jq -r '.eligible_parent_count' "${plan}")"
batch_count="$(jq -r '.batches | length' "${plan}")"
batch_sizes="$(jq -r '[.batches[].batch_size] | @json' "${plan}")"

{
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "expected_git_sha=${expected_git_sha}"
  echo "sc2_patch=${sc2_patch}"
  echo "source_run_set=${source_run_set}"
  echo "source_status=${source_status}"
  echo "qualification_plan=${plan}"
  echo "eligible_parent_count=${eligible_parent_count}"
  echo "batch_count=${batch_count}"
  echo "batch_sizes=${batch_sizes}"
  echo "reviewed_source_manifest=${reviewed_source_manifest}"
  echo "reviewed_source_manifest_sha256=${reviewed_source_manifest_sha256}"
  echo "reviewed_source_tree_sha256=${reviewed_source_tree_sha256}"
} > "${run_set_dir}/hard-qualification-metadata.txt"

status_header=(
  experiment_kind mode batch_id batch_index seed probe_path probe_sha256 arm subject_arm
  arm_order exit_code run_dir playbook_before_sha256 playbook_after_sha256
  playbook_before_snapshot playbook_after_snapshot git_head_before git_head_after
  superproject_dirty_before superproject_dirty_after submodule_commit_before
  submodule_commit_after submodule_dirty_before submodule_dirty_after
  submodule_gitlink_before submodule_gitlink_after submodule_diff_sha256_before
  submodule_diff_sha256_after reviewed_source_commit_before reviewed_source_commit_after
  reviewed_source_diff_sha256_before reviewed_source_diff_sha256_after
  reviewed_source_tree_sha256_before reviewed_source_tree_sha256_after
)
(
  IFS=$'\t'
  echo "${status_header[*]}"
) > "${status_file}"

write_terminal_rejection() {
  local reason="$1"
  local runner_exit_code="$2"
  set +e
  "${core_python}" -m scripts.analyze_playbook_hard_qualification \
    --plan "${plan}" \
    --baseline "${baseline}" \
    --status "${status_file}" \
    --engineering-baseline "${engineering_baseline}" \
    --recovery-evidence "${recovery_evidence}" \
    --expected-git-sha "${expected_git_sha}" \
    --sc2-patch "${sc2_patch}" \
    --working-database "${working_playbook}" \
    --report-output "${report}" \
    --manifest-output "${manifest}" \
    --terminal-rejection-reason "${reason}" \
    --runner-exit-code "${runner_exit_code}"
  set -e
}

failure_stage="post_plan_initialization"
terminal_rejection_on_exit() {
  local runner_exit_code="$1"
  trap - EXIT
  if [[ ${runner_exit_code} -ne 0 && ! -f "${report}" ]]; then
    write_terminal_rejection "runner_failed:${failure_stage}" "${runner_exit_code}"
  fi
  exit "${runner_exit_code}"
}
trap 'terminal_rejection_on_exit $?' EXIT

while IFS=$'\t' read -r batch_id batch_index probe_path probe_sha256; do
  failure_stage="${batch_id}:probe_validation"
  if [[ ! -f "${probe_path}" \
    || "$(sha256sum "${probe_path}" | awk '{print $1}')" != "${probe_sha256}" ]]; then
    echo "hard qualification probe hash mismatch for ${batch_id}" >&2
    exit 87
  fi
  for seed in "${seeds[@]}"; do
    failure_stage="${batch_id}:seed-${seed}"
    rm -f "${working_playbook}" "${working_playbook}-shm" "${working_playbook}-wal"
    cp "${probe_path}" "${working_playbook}"
    before_snapshot="${run_set_dir}/${batch_id}.seed-${seed}.before.sqlite3"
    after_snapshot="${run_set_dir}/${batch_id}.seed-${seed}.after.sqlite3"
    cp "${working_playbook}" "${before_snapshot}"
    before_sha256="$(sha256sum "${working_playbook}" | awk '{print $1}')"
    capture_attestation
    git_head_before="${source_git_head}"
    dirty_before="${source_superproject_dirty}"
    submodule_commit_before="${source_submodule_commit}"
    submodule_dirty_before="${source_submodule_dirty}"
    submodule_gitlink_before="${source_submodule_gitlink}"
    submodule_diff_before="${source_submodule_diff_sha256}"
    reviewed_commit_before="${source_reviewed_commit}"
    reviewed_diff_before="${source_reviewed_diff_sha256}"
    reviewed_tree_before="${source_reviewed_tree_sha256}"
    log_path="${run_set_dir}/${batch_id}.seed-${seed}.log"
    run_dir=""
    if ! attestation_matches; then
      echo "source attestation changed before ${batch_id} seed ${seed}" >&2
      run_status=86
    else
      set +e
      SC2PATH="/mnt/scratch/users/tbczhang/StarCraftII" \
        HF_HUB_OFFLINE=1 \
        TRANSFORMERS_OFFLINE=1 \
        TOKENIZERS_PARALLELISM=false \
        "${core_cli}" run \
          --config "${config}" \
          --seed "${seed}" \
          --qualification-evidence "${qualification_evidence}" \
        2>&1 | tee "${log_path}"
      run_status=${PIPESTATUS[0]}
      set -e
      run_dir="$(sed -n -e 's/^Run directory: //p' -e 's/^Artifacts: //p' "${log_path}" | tail -n 1)"
    fi
    capture_attestation
    if ! attestation_matches; then
      echo "source attestation changed during ${batch_id} seed ${seed}" >&2
      run_status=86
    fi
    after_sha256="$(sha256sum "${working_playbook}" | awk '{print $1}')"
    cp "${working_playbook}" "${after_snapshot}"
    fields=(
      "qualification" "hard_shadow" "${batch_id}" "${batch_index}" "${seed}"
      "${probe_path}" "${probe_sha256}" "shadow_probe" "shadow_probe" "0,1,2"
      "${run_status}" "${run_dir}" "${before_sha256}" "${after_sha256}"
      "${before_snapshot}" "${after_snapshot}" "${git_head_before}" "${source_git_head}"
      "${dirty_before}" "${source_superproject_dirty}" "${submodule_commit_before}"
      "${source_submodule_commit}" "${submodule_dirty_before}" "${source_submodule_dirty}"
      "${submodule_gitlink_before}" "${source_submodule_gitlink}"
      "${submodule_diff_before}" "${source_submodule_diff_sha256}"
      "${reviewed_commit_before}" "${source_reviewed_commit}"
      "${reviewed_diff_before}" "${source_reviewed_diff_sha256}"
      "${reviewed_tree_before}" "${source_reviewed_tree_sha256}"
    )
    (
      IFS=$'\t'
      echo "${fields[*]}"
    ) >> "${status_file}"
    if [[ ${run_status} -ne 0 ]]; then
      echo "hard qualification ${batch_id} seed ${seed} failed with ${run_status}" >&2
      write_terminal_rejection "runner_failed:${batch_id}:seed-${seed}" "${run_status}"
      exit "${run_status}"
    fi
  done
done < <(
  jq -r '.batches[] | [.batch_id, .batch_index, .probe_baseline_path, .probe_baseline_sha256] | @tsv' "${plan}"
)

failure_stage="aggregate_analysis"
"${core_python}" -m scripts.analyze_playbook_hard_qualification \
  --plan "${plan}" \
  --baseline "${baseline}" \
  --status "${status_file}" \
  --engineering-baseline "${engineering_baseline}" \
  --recovery-evidence "${recovery_evidence}" \
  --expected-git-sha "${expected_git_sha}" \
  --sc2-patch "${sc2_patch}" \
  --working-database "${working_playbook}" \
  --report-output "${report}" \
  --manifest-output "${manifest}"

failure_stage="selected_parent_promotion"
selected_parent_rule_id="$(jq -r '.parent_rule_id' "${manifest}")"
cp "${baseline}" "${production_baseline}"
"${core_cli}" playbook qualify-hard \
  --database "${production_baseline}" \
  --parent-rule-id "${selected_parent_rule_id}" \
  --expected-git-sha "${expected_git_sha}" \
  --sc2-patch "${sc2_patch}" \
  --qualification-manifest "${manifest}" \
  --evaluation-seed 3 \
  --evaluation-seed 4 \
  --evaluation-seed 5 \
  > "${run_set_dir}/qualified-hard-rule.json"

failure_stage="hard_readiness"
"${core_cli}" playbook hard-readiness \
  --database "${production_baseline}" \
  --config "${readiness_config}" \
  --expected-git-sha "${expected_git_sha}" \
  --sc2-patch "${sc2_patch}" \
  --evaluation-seed 3 \
  --evaluation-seed 4 \
  --evaluation-seed 5 \
  --output "${readiness_evidence}"
production_baseline_sha256="$(sha256sum "${production_baseline}" | awk '{print $1}')"

{
  echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "selected_parent_rule_id=${selected_parent_rule_id}"
  echo "production_baseline=${production_baseline}"
  echo "production_baseline_sha256=${production_baseline_sha256}"
  echo "readiness_evidence=${readiness_evidence}"
  echo "exit_code=0"
} >> "${run_set_dir}/hard-qualification-metadata.txt"
trap - EXIT
echo "hard_rule_qualification status=accepted run_set=${run_set_dir} baseline=${production_baseline}"
