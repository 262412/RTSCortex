#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 8 || "$3" != "--expected-git-sha" || "$5" != "--execution-seed" || "$7" != "--evaluation-seeds" ]]; then
  echo "usage: $0 <baseline-playbook.sqlite3> <run-set-dir> --expected-git-sha <sha> --execution-seed <seed> --evaluation-seeds 3,4,5" >&2
  exit 2
fi

repo_dir="/mnt/scratch/users/tbczhang/projects/RTSCortex"
output_root="/mnt/scratch/users/tbczhang/outputs/RTSCortex"
baseline_source="$(readlink -f "$1")"
run_set_dir="$2"
expected_git_sha="$4"
seed="$6"
evaluation_seed_csv="$8"
IFS=',' read -r -a evaluation_seeds <<< "${evaluation_seed_csv}"
if [[ ${#evaluation_seeds[@]} -ne 3 ]]; then
  echo "production canary requires the formal three-seed evaluation set" >&2
  exit 2
fi
declare -A unique_evaluation_seeds=()
execution_seed_is_held_out=false
for evaluation_seed in "${evaluation_seeds[@]}"; do
  if [[ ! "${evaluation_seed}" =~ ^[0-9]+$ || -n "${unique_evaluation_seeds[${evaluation_seed}]:-}" ]]; then
    echo "evaluation seeds must be three distinct non-negative integers" >&2
    exit 2
  fi
  unique_evaluation_seeds["${evaluation_seed}"]=1
  if [[ "${evaluation_seed}" == "${seed}" ]]; then
    execution_seed_is_held_out=true
  fi
done
if [[ ! "${seed}" =~ ^[0-9]+$ || "${execution_seed_is_held_out}" != "true" ]]; then
  echo "execution seed must be one member of the full held-out evaluation set" >&2
  exit 2
fi
active_config="${repo_dir}/configs/experiments/live_simple64_hima_protoss_ensemble_cortex_v0_5_natural_terminal.yaml"
shadow_config="${repo_dir}/configs/experiments/live_simple64_hima_protoss_ensemble_cortex_v0_5_shadow_calibration_natural_terminal.yaml"
engineering_baseline="${repo_dir}/configs/acceptance/protoss_natural_terminal_v1.json"
active_playbook="${output_root}/cortex-playbook-evolving-working.sqlite3"
shadow_playbook="${output_root}/cortex-playbook-counterfactual-shadow.sqlite3"
lock_path="${output_root}/protoss-playbook-paired.lock"

mkdir -p "${run_set_dir}"
run_set_dir="$(readlink -f "${run_set_dir}")"
baseline_snapshot="${run_set_dir}/playbook.baseline.sqlite3"
cp "${baseline_source}" "${baseline_snapshot}"
baseline_sha256="$(sha256sum "${baseline_snapshot}" | awk '{print $1}')"
recovery_evidence="${run_set_dir}/recovery-canary.json"
readiness_evidence="${run_set_dir}/playbook-hard-readiness.json"
status_file="${run_set_dir}/experiment-status.tsv"

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
  echo "counterfactual canary requires clean ${expected_git_sha} and exact clean gitlink ${submodule_gitlink}" >&2
  exit 2
fi

readiness_seed_args=()
for evaluation_seed in "${evaluation_seeds[@]}"; do
  readiness_seed_args+=(--evaluation-seed "${evaluation_seed}")
done
uv run rtscortex playbook hard-readiness \
  --database "${baseline_snapshot}" \
  --config "${active_config}" \
  --expected-git-sha "${expected_git_sha}" \
  --sc2-patch "4.10" \
  "${readiness_seed_args[@]}" \
  --output "${readiness_evidence}"

uv run python scripts/run_recovery_acceptance_canary.py \
  --expected-git-sha "${expected_git_sha}" \
  --output "${recovery_evidence}"

printf "experiment_kind\tmode\tseed\tarm\tsubject_arm\tarm_order\texit_code\trun_dir\tplaybook_before_sha256\tplaybook_after_sha256\tplaybook_before_snapshot\tplaybook_after_snapshot\tgit_head_before\tgit_head_after\tsuperproject_dirty_before\tsuperproject_dirty_after\tsubmodule_commit_before\tsubmodule_commit_after\tsubmodule_dirty_before\tsubmodule_dirty_after\tsubmodule_gitlink_before\tsubmodule_gitlink_after\tsubmodule_diff_sha256_before\tsubmodule_diff_sha256_after\n" > "${status_file}"

run_canary_arm() {
  local kind="$1"
  local arm="$2"
  local subject_arm="$3"
  local config="$4"
  local working_playbook="$5"
  local arm_dir="${run_set_dir}/${arm}"
  mkdir -p "${arm_dir}"
  rm -f "${working_playbook}" "${working_playbook}-shm" "${working_playbook}-wal"
  cp "${baseline_snapshot}" "${working_playbook}"
  local before_snapshot="${arm_dir}/before.sqlite3"
  local after_snapshot="${arm_dir}/after.sqlite3"
  cp "${working_playbook}" "${before_snapshot}"
  local before_sha256
  before_sha256="$(sha256sum "${working_playbook}" | awk '{print $1}')"
  local log_path="${arm_dir}/seed-${seed}.log"
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
  local run_status=${PIPESTATUS[0]}
  set -e
  local run_dir
  run_dir="$(
    sed -n \
      -e 's/^Run directory: //p' \
      -e 's/^Artifacts: //p' \
      "${log_path}" \
      | tail -n 1
  )"
  local after_sha256
  after_sha256="$(sha256sum "${working_playbook}" | awk '{print $1}')"
  cp "${working_playbook}" "${after_snapshot}"
  local git_head_after
  git_head_after="$(git rev-parse HEAD)"
  local dirty_after
  dirty_after="$(test -n "$(git status --porcelain --ignore-submodules=dirty)" && echo true || echo false)"
  local submodule_commit_after
  submodule_commit_after="$(git -C third_party/LLM-PySC2 rev-parse HEAD)"
  local submodule_dirty_after
  submodule_dirty_after="$(test -n "$(git -C third_party/LLM-PySC2 status --porcelain)" && echo true || echo false)"
  local submodule_gitlink_after
  submodule_gitlink_after="$(git ls-tree HEAD third_party/LLM-PySC2 | awk '{print $3}')"
  local submodule_diff_after
  submodule_diff_after="$(git -C third_party/LLM-PySC2 diff --binary | sha256sum | awk '{print $1}')"
  printf "%s\tcausal_canary\t%s\t%s\t%s\tactive,shadow\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${kind}" "${seed}" "${arm}" "${subject_arm}" "${run_status}" "${run_dir}" \
    "${before_sha256}" "${after_sha256}" "${before_snapshot}" "${after_snapshot}" \
    "${git_head}" "${git_head_after}" "${superproject_dirty}" "${dirty_after}" \
    "${submodule_commit}" "${submodule_commit_after}" "${submodule_dirty}" \
    "${submodule_dirty_after}" "${submodule_gitlink}" "${submodule_gitlink_after}" \
    "${submodule_diff_sha256}" "${submodule_diff_after}" \
    >> "${status_file}"
}

run_canary_arm behavior active "" "${active_config}" "${active_playbook}"
run_canary_arm calibration shadow active "${shadow_config}" "${shadow_playbook}"

uv run python -m scripts.analyze_playbook_counterfactual_canary \
  "${run_set_dir}" \
  --baseline-sha256 "${baseline_sha256}" \
  --expected-git-sha "${expected_git_sha}" \
  --engineering-baseline "${engineering_baseline}" \
  --recovery-evidence "${recovery_evidence}" \
  --readiness-evidence "${readiness_evidence}" \
  --output "${run_set_dir}/counterfactual-canary.json"
