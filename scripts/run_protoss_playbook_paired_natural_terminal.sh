#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <baseline-playbook.sqlite3> <run-set-dir>" >&2
  exit 2
fi

repo_dir="/mnt/scratch/users/tbczhang/projects/RTSCortex"
output_root="/mnt/scratch/users/tbczhang/outputs/RTSCortex"
baseline_source="$(readlink -f "$1")"
run_set_dir="$2"
frozen_config="${repo_dir}/configs/experiments/live_simple64_hima_protoss_ensemble_cortex_v0_5_frozen_playbook_natural_terminal.yaml"
evolving_config="${repo_dir}/configs/experiments/live_simple64_hima_protoss_ensemble_cortex_v0_5_natural_terminal.yaml"
frozen_playbook="${output_root}/cortex-playbook-frozen-working.sqlite3"
evolving_playbook="${output_root}/cortex-playbook-evolving-working.sqlite3"
lock_path="${output_root}/protoss-playbook-paired.lock"

if [[ ! -f "${baseline_source}" ]]; then
  echo "baseline Playbook does not exist: ${baseline_source}" >&2
  exit 2
fi

mkdir -p "${run_set_dir}"
run_set_dir="$(readlink -f "${run_set_dir}")"
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

cd "${repo_dir}"
git status --short > "${run_set_dir}/source-status.txt"
git diff --binary > "${run_set_dir}/source.diff"
{
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "git_head=$(git rev-parse HEAD)"
  echo "git_dirty=$(test -n "$(git status --porcelain)" && echo true || echo false)"
  echo "baseline_sha256=${baseline_sha256}"
  echo "seeds=0,1,2"
  echo "experiment_modes=independent_paired,sequential_learning"
  echo "arm_order=counterbalanced_by_seed"
} > "${run_set_dir}/experiment-metadata.txt"

status_file="${run_set_dir}/experiment-status.tsv"
printf "mode\tseed\tarm\tarm_order\texit_code\trun_dir\tplaybook_before_sha256\tplaybook_after_sha256\tplaybook_before_snapshot\tplaybook_after_snapshot\n" \
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
  run_dir="$(sed -n 's/^Run directory: //p' "${log_path}" | tail -n 1)"
  after_sha256="$(sha256sum "${working_playbook}" | awk '{print $1}')"
  after_snapshot="${arm_dir}/seed-${seed}.after.sqlite3"
  cp "${working_playbook}" "${after_snapshot}"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${mode}" "${seed}" "${arm}" "${order}" "${run_status}" "${run_dir}" \
    "${before_sha256}" "${after_sha256}" "${before_snapshot}" "${after_snapshot}" \
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

uv run python scripts/analyze_playbook_experiment.py \
  "${run_set_dir}" \
  --baseline-sha256 "${baseline_sha256}"

{
  echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "exit_code=${overall_status}"
} >> "${run_set_dir}/experiment-metadata.txt"
echo "playbook_experiment status=finished exit_code=${overall_status} report=${run_set_dir}/comparison.json"
exit "${overall_status}"
