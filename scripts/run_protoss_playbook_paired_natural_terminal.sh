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

mkdir -p "${run_set_dir}/frozen" "${run_set_dir}/evolving"
run_set_dir="$(readlink -f "${run_set_dir}")"
baseline_snapshot="${run_set_dir}/playbook.baseline.sqlite3"

exec 9>"${lock_path}"
if ! flock -n 9; then
  echo "another Protoss Playbook paired experiment owns ${lock_path}" >&2
  exit 1
fi

if [[ "${baseline_source}" != "${baseline_snapshot}" ]]; then
  cp "${baseline_source}" "${baseline_snapshot}"
fi
cp "${frozen_config}" "${run_set_dir}/frozen.config.yaml"
cp "${evolving_config}" "${run_set_dir}/evolving.config.yaml"

cd "${repo_dir}"
git status --short > "${run_set_dir}/source-status.txt"
git diff --binary > "${run_set_dir}/source.diff"
{
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "git_head=$(git rev-parse HEAD)"
  echo "git_dirty=$(test -n "$(git status --porcelain)" && echo true || echo false)"
  echo "baseline_sha256=$(sha256sum "${baseline_snapshot}" | awk '{print $1}')"
  echo "frozen_config_sha256=$(sha256sum "${frozen_config}" | awk '{print $1}')"
  echo "evolving_config_sha256=$(sha256sum "${evolving_config}" | awk '{print $1}')"
  echo "seeds=0,1,2"
  echo "arm_order=frozen,evolving"
} > "${run_set_dir}/experiment-metadata.txt"

rm -f \
  "${evolving_playbook}" \
  "${evolving_playbook}-shm" \
  "${evolving_playbook}-wal"
cp "${baseline_snapshot}" "${evolving_playbook}"

status_file="${run_set_dir}/paired-status.tsv"
printf "seed\tarm\texit_code\trun_dir\tplaybook_before_sha256\tplaybook_after_sha256\n" \
  > "${status_file}"
overall_status=0

for seed in 0 1 2; do
  for arm in frozen evolving; do
    if [[ "${arm}" == "frozen" ]]; then
      config="${frozen_config}"
      working_playbook="${frozen_playbook}"
      rm -f \
        "${working_playbook}" \
        "${working_playbook}-shm" \
        "${working_playbook}-wal"
      cp "${baseline_snapshot}" "${working_playbook}"
    else
      config="${evolving_config}"
      working_playbook="${evolving_playbook}"
    fi

    before_sha256="$(sha256sum "${working_playbook}" | awk '{print $1}')"
    log_path="${run_set_dir}/${arm}/seed-${seed}.log"
    echo "seed=${seed} arm=${arm} status=starting utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
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
    cp "${working_playbook}" "${run_set_dir}/${arm}/seed-${seed}.playbook.sqlite3"
    printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
      "${seed}" \
      "${arm}" \
      "${run_status}" \
      "${run_dir}" \
      "${before_sha256}" \
      "${after_sha256}" \
      >> "${status_file}"
    echo "seed=${seed} arm=${arm} status=finished exit_code=${run_status} utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    if [[ ${run_status} -ne 0 ]]; then
      overall_status=1
    fi
  done
done

{
  echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "exit_code=${overall_status}"
} >> "${run_set_dir}/experiment-metadata.txt"
echo "paired_experiment status=finished exit_code=${overall_status} status_file=${status_file}"
exit "${overall_status}"
