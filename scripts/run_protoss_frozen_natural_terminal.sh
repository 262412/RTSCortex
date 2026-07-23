#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <frozen-playbook.sqlite3> <run-set-dir>" >&2
  exit 2
fi

repo_dir="/mnt/scratch/users/tbczhang/projects/RTSCortex"
config_path="${repo_dir}/configs/experiments/live_simple64_hima_protoss_ensemble_cortex_v0_5_frozen_playbook_natural_terminal.yaml"
frozen_playbook="$1"
run_set_dir="$2"
working_playbook="/mnt/scratch/users/tbczhang/outputs/RTSCortex/cortex-playbook-frozen-working.sqlite3"

mkdir -p "${run_set_dir}"
cd "${repo_dir}"

for seed in 0 1 2; do
  rm -f \
    "${working_playbook}" \
    "${working_playbook}-shm" \
    "${working_playbook}-wal"
  cp "${frozen_playbook}" "${working_playbook}"

  echo "seed=${seed} status=starting utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  set +e
  SC2PATH="/mnt/scratch/users/tbczhang/StarCraftII" \
    uv run rtscortex run \
      --config "${config_path}" \
      --seed "${seed}" \
      --console \
      --console-port 8765 \
    2>&1 | tee "${run_set_dir}/seed-${seed}.log"
  run_status=${PIPESTATUS[0]}
  set -e
  echo "seed=${seed} status=finished exit_code=${run_status} utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  if [[ ${run_status} -ne 0 ]]; then
    exit "${run_status}"
  fi
done
