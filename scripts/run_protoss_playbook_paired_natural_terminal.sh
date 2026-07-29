#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 || "$3" != "--expected-git-sha" || "$5" != "--counterfactual-canary" ]]; then
  echo "usage: $0 <baseline-playbook.sqlite3> <run-set-dir> --expected-git-sha <sha> --counterfactual-canary <json>" >&2
  exit 2
fi

repo_dir="/mnt/scratch/users/tbczhang/projects/RTSCortex"
output_root="/mnt/scratch/users/tbczhang/outputs/RTSCortex"
baseline_source="$(readlink -f "$1")"
run_set_dir="$2"
expected_git_sha="$4"
counterfactual_canary="$(readlink -f "$6")"
frozen_config="${repo_dir}/configs/experiments/live_simple64_hima_protoss_ensemble_cortex_v0_5_frozen_playbook_natural_terminal.yaml"
evolving_config="${repo_dir}/configs/experiments/live_simple64_hima_protoss_ensemble_cortex_v0_5_natural_terminal.yaml"
shadow_config="${repo_dir}/configs/experiments/live_simple64_hima_protoss_ensemble_cortex_v0_5_shadow_calibration_natural_terminal.yaml"
frozen_playbook="${output_root}/cortex-playbook-frozen-working.sqlite3"
evolving_playbook="${output_root}/cortex-playbook-evolving-working.sqlite3"
shadow_playbook="${output_root}/cortex-playbook-shadow-working.sqlite3"
engineering_baseline="${repo_dir}/configs/acceptance/protoss_natural_terminal_v1.json"
lock_path="${output_root}/protoss-playbook-paired.lock"

if [[ ! -f "${baseline_source}" ]]; then
  echo "baseline Playbook does not exist: ${baseline_source}" >&2
  exit 2
fi

mkdir -p "${run_set_dir}"
run_set_dir="$(readlink -f "${run_set_dir}")"
recovery_evidence="${run_set_dir}/recovery-canary.json"
baseline_snapshot="${run_set_dir}/playbook.baseline.sqlite3"

exec 9>"${lock_path}"
if ! flock -n 9; then
  echo "another Protoss Playbook experiment owns ${lock_path}" >&2
  exit 1
fi

if [[ "${baseline_source}" != "${baseline_snapshot}" ]]; then
  cp "${baseline_source}" "${baseline_snapshot}"
fi
baseline_sha256="$(sha256sum "${baseline_snapshot}" | awk '{print $1}')"

uv run python - "${counterfactual_canary}" "${expected_git_sha}" "${baseline_sha256}" <<'PY'
import sys
import json
from pathlib import Path
from scripts.analyze_playbook_experiment import counterfactual_canary_is_valid

artifact = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if not counterfactual_canary_is_valid(
    artifact,
    baseline_sha256=sys.argv[3],
    expected_git_sha=sys.argv[2],
):
    raise SystemExit("counterfactual canary is missing, rejected, or source-mismatched")
PY

cd "${repo_dir}"
git status --short > "${run_set_dir}/source-status.txt"
git diff --binary > "${run_set_dir}/source.diff"
git_head="$(git rev-parse HEAD)"
superproject_dirty="$(test -n "$(git status --porcelain --ignore-submodules=dirty)" && echo true || echo false)"
submodule_commit="$(git -C third_party/LLM-PySC2 rev-parse HEAD)"
submodule_dirty="$(test -n "$(git -C third_party/LLM-PySC2 status --porcelain)" && echo true || echo false)"
submodule_diff_sha256="$(git -C third_party/LLM-PySC2 diff --binary | sha256sum | awk '{print $1}')"
if [[ "${git_head}" != "${expected_git_sha}" || "${superproject_dirty}" != "false" ]]; then
  echo "formal acceptance requires clean ${expected_git_sha}; got head=${git_head} dirty=${superproject_dirty}" >&2
  exit 2
fi

capture_source_attestation() {
  source_git_head="$(git rev-parse HEAD)"
  source_superproject_dirty="$(test -n "$(git status --porcelain --ignore-submodules=dirty)" && echo true || echo false)"
  source_submodule_commit="$(git -C third_party/LLM-PySC2 rev-parse HEAD)"
  source_submodule_dirty="$(test -n "$(git -C third_party/LLM-PySC2 status --porcelain)" && echo true || echo false)"
  source_submodule_diff_sha256="$(git -C third_party/LLM-PySC2 diff --binary | sha256sum | awk '{print $1}')"
}

source_matches_baseline() {
  [[ "${source_git_head}" == "${expected_git_sha}" ]] \
    && [[ "${source_superproject_dirty}" == "false" ]] \
    && [[ "${source_submodule_commit}" == "${submodule_commit}" ]] \
    && [[ "${source_submodule_dirty}" == "${submodule_dirty}" ]] \
    && [[ "${source_submodule_diff_sha256}" == "${submodule_diff_sha256}" ]]
}

uv run python scripts/run_recovery_acceptance_canary.py \
  --expected-git-sha "${expected_git_sha}" \
  --output "${recovery_evidence}"
{
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "git_head=${git_head}"
  echo "superproject_dirty=${superproject_dirty}"
  echo "submodule_commit=${submodule_commit}"
  echo "submodule_dirty=${submodule_dirty}"
  echo "submodule_diff_sha256=${submodule_diff_sha256}"
  echo "expected_git_sha=${expected_git_sha}"
  echo "baseline_sha256=${baseline_sha256}"
  echo "seeds=0,1,2"
  echo "experiment_modes=independent_paired,sequential_learning"
  echo "arm_order=counterbalanced_by_seed"
  echo "counterfactual_calibration=matched_shadow_twin_per_behavior_run"
  echo "counterfactual_canary=${counterfactual_canary}"
} > "${run_set_dir}/experiment-metadata.txt"

status_file="${run_set_dir}/experiment-status.tsv"
printf "experiment_kind\tmode\tseed\tarm\tsubject_arm\tarm_order\texit_code\trun_dir\tplaybook_before_sha256\tplaybook_after_sha256\tplaybook_before_snapshot\tplaybook_after_snapshot\tgit_head_before\tgit_head_after\tsuperproject_dirty_before\tsuperproject_dirty_after\tsubmodule_commit_before\tsubmodule_commit_after\tsubmodule_dirty_before\tsubmodule_dirty_after\tsubmodule_diff_sha256_before\tsubmodule_diff_sha256_after\n" \
  > "${status_file}"
overall_status=0

reset_playbook() {
  local target="$1"
  rm -f "${target}" "${target}-shm" "${target}-wal"
  cp "${baseline_snapshot}" "${target}"
}

run_arm() {
  local mode="$1"
  local seed="$2"
  local arm="$3"
  local order="$4"
  local config working_playbook
  if [[ "${arm}" == "frozen" ]]; then
    config="${frozen_config}"
    working_playbook="${frozen_playbook}"
  else
    config="${evolving_config}"
    working_playbook="${evolving_playbook}"
  fi
  local arm_dir="${run_set_dir}/${mode}/${arm}"
  mkdir -p "${arm_dir}"
  capture_source_attestation
  if ! source_matches_baseline; then
    echo "source attestation changed before ${mode}/${arm}/seed-${seed}" >&2
    exit 2
  fi
  local git_head_before="${source_git_head}"
  local dirty_before="${source_superproject_dirty}"
  local submodule_commit_before="${source_submodule_commit}"
  local submodule_dirty_before="${source_submodule_dirty}"
  local submodule_diff_before="${source_submodule_diff_sha256}"
  local before_sha256 log_path run_status run_dir after_sha256 before_snapshot after_snapshot
  before_sha256="$(sha256sum "${working_playbook}" | awk '{print $1}')"
  before_snapshot="${arm_dir}/seed-${seed}.before.sqlite3"
  cp "${working_playbook}" "${before_snapshot}"
  log_path="${arm_dir}/seed-${seed}.log"
  set +e
  SC2PATH="/mnt/scratch/users/tbczhang/StarCraftII" \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    uv run rtscortex run \
      --config "${config}" \
      --seed "${seed}" \
      --console \
      --console-port 8765 \
    2>&1 | tee "${log_path}"
  run_status=${PIPESTATUS[0]}
  set -e
  capture_source_attestation
  local git_head_after="${source_git_head}"
  local dirty_after="${source_superproject_dirty}"
  local submodule_commit_after="${source_submodule_commit}"
  local submodule_dirty_after="${source_submodule_dirty}"
  local submodule_diff_after="${source_submodule_diff_sha256}"
  if ! source_matches_baseline; then
    echo "source attestation changed during ${mode}/${arm}/seed-${seed}" >&2
    run_status=86
  fi
  run_dir="$(sed -n 's/^Run directory: //p' "${log_path}" | tail -n 1)"
  after_sha256="$(sha256sum "${working_playbook}" | awk '{print $1}')"
  after_snapshot="${arm_dir}/seed-${seed}.after.sqlite3"
  cp "${working_playbook}" "${after_snapshot}"
  printf "behavior\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${mode}" "${seed}" "${arm}" "${arm}" "${order}" "${run_status}" "${run_dir}" \
    "${before_sha256}" "${after_sha256}" "${before_snapshot}" "${after_snapshot}" \
    "${git_head_before}" "${git_head_after}" "${dirty_before}" "${dirty_after}" \
    "${submodule_commit_before}" "${submodule_commit_after}" \
    "${submodule_dirty_before}" "${submodule_dirty_after}" \
    "${submodule_diff_before}" "${submodule_diff_after}" \
    >> "${status_file}"
  if [[ ${run_status} -ne 0 ]]; then
    overall_status=1
  fi
  run_shadow_calibration \
    "${mode}" "${seed}" "${arm}" "${order}" "${before_snapshot}"
}

run_shadow_calibration() {
  local mode="$1"
  local seed="$2"
  local subject_arm="$3"
  local order="$4"
  local behavior_before_snapshot="$5"
  reset_playbook "${shadow_playbook}"
  cp "${behavior_before_snapshot}" "${shadow_playbook}"
  local arm_dir="${run_set_dir}/${mode}/shadow-${subject_arm}"
  mkdir -p "${arm_dir}"
  capture_source_attestation
  if ! source_matches_baseline; then
    echo "source attestation changed before ${mode}/shadow-${subject_arm}/seed-${seed}" >&2
    exit 2
  fi
  local git_head_before="${source_git_head}"
  local dirty_before="${source_superproject_dirty}"
  local submodule_commit_before="${source_submodule_commit}"
  local submodule_dirty_before="${source_submodule_dirty}"
  local submodule_diff_before="${source_submodule_diff_sha256}"
  local before_sha256 log_path run_status run_dir after_sha256 before_snapshot after_snapshot
  before_sha256="$(sha256sum "${shadow_playbook}" | awk '{print $1}')"
  before_snapshot="${arm_dir}/seed-${seed}.before.sqlite3"
  cp "${shadow_playbook}" "${before_snapshot}"
  log_path="${arm_dir}/seed-${seed}.log"
  set +e
  SC2PATH="/mnt/scratch/users/tbczhang/StarCraftII" \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    uv run rtscortex run \
      --config "${shadow_config}" \
      --seed "${seed}" \
      --console \
      --console-port 8765 \
    2>&1 | tee "${log_path}"
  run_status=${PIPESTATUS[0]}
  set -e
  capture_source_attestation
  local git_head_after="${source_git_head}"
  local dirty_after="${source_superproject_dirty}"
  local submodule_commit_after="${source_submodule_commit}"
  local submodule_dirty_after="${source_submodule_dirty}"
  local submodule_diff_after="${source_submodule_diff_sha256}"
  if ! source_matches_baseline; then
    echo "source attestation changed during ${mode}/shadow-${subject_arm}/seed-${seed}" >&2
    run_status=86
  fi
  run_dir="$(sed -n 's/^Run directory: //p' "${log_path}" | tail -n 1)"
  after_sha256="$(sha256sum "${shadow_playbook}" | awk '{print $1}')"
  after_snapshot="${arm_dir}/seed-${seed}.after.sqlite3"
  cp "${shadow_playbook}" "${after_snapshot}"
  printf "calibration\t%s\t%s\tshadow\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${mode}" "${seed}" "${subject_arm}" "${order}" "${run_status}" "${run_dir}" \
    "${before_sha256}" "${after_sha256}" "${before_snapshot}" "${after_snapshot}" \
    "${git_head_before}" "${git_head_after}" "${dirty_before}" "${dirty_after}" \
    "${submodule_commit_before}" "${submodule_commit_after}" \
    "${submodule_dirty_before}" "${submodule_dirty_after}" \
    "${submodule_diff_before}" "${submodule_diff_after}" \
    >> "${status_file}"
  if [[ ${run_status} -ne 0 ]]; then
    overall_status=1
  fi
}

# Experiment A: both arms start every seed from byte-identical baseline state.
for seed in 0 1 2; do
  reset_playbook "${frozen_playbook}"
  reset_playbook "${evolving_playbook}"
  if (( seed % 2 == 0 )); then
    order="frozen,evolving"
  else
    order="evolving,frozen"
  fi
  IFS=',' read -r first second <<< "${order}"
  run_arm "independent_paired" "${seed}" "${first}" "${order}"
  run_arm "independent_paired" "${seed}" "${second}" "${order}"
done

# Experiment B: only evolving deliberately carries evidence across seeds.
reset_playbook "${evolving_playbook}"
for seed in 0 1 2; do
  reset_playbook "${frozen_playbook}"
  if (( seed % 2 == 0 )); then
    order="frozen,evolving"
  else
    order="evolving,frozen"
  fi
  IFS=',' read -r first second <<< "${order}"
  run_arm "sequential_learning" "${seed}" "${first}" "${order}"
  run_arm "sequential_learning" "${seed}" "${second}" "${order}"
done

capture_source_attestation
if ! source_matches_baseline; then
  echo "source attestation changed before final analysis" >&2
  overall_status=1
fi
{
  echo "final_git_head=${source_git_head}"
  echo "final_superproject_dirty=${source_superproject_dirty}"
  echo "final_submodule_commit=${source_submodule_commit}"
  echo "final_submodule_dirty=${source_submodule_dirty}"
  echo "final_submodule_diff_sha256=${source_submodule_diff_sha256}"
} >> "${run_set_dir}/experiment-metadata.txt"

set +e
uv run python scripts/analyze_playbook_experiment.py \
  "${run_set_dir}" \
  --baseline-sha256 "${baseline_sha256}" \
  --expected-git-sha "${expected_git_sha}" \
  --engineering-baseline "${engineering_baseline}" \
  --recovery-evidence "${recovery_evidence}" \
  --counterfactual-canary "${counterfactual_canary}"
analysis_status=$?
set -e
if [[ ${analysis_status} -ne 0 ]]; then
  overall_status=1
fi

{
  echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "analysis_exit_code=${analysis_status}"
  echo "exit_code=${overall_status}"
} >> "${run_set_dir}/experiment-metadata.txt"
echo "playbook_experiment status=finished exit_code=${overall_status} report=${run_set_dir}/comparison.json"
exit "${overall_status}"
