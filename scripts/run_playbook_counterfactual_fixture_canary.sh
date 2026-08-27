#!/usr/bin/env bash
set -euo pipefail
set -o noclobber
export PYTHONDONTWRITEBYTECODE=1

if [[ $# -ne 5 ]]; then
  echo "usage: $0 <run-set-dir> --expected-git-sha <sha> --seed <seed>" >&2
  exit 2
fi
if [[ "$2" != "--expected-git-sha" || "$4" != "--seed" ]]; then
  echo "usage: $0 <run-set-dir> --expected-git-sha <sha> --seed <seed>" >&2
  exit 2
fi

repo_dir="/mnt/scratch/users/tbczhang/projects/RTSCortex"
output_root="/mnt/scratch/users/tbczhang/outputs/RTSCortex"
run_set_dir="$1"
expected_git_sha="$3"
seed="$5"
active_config="${repo_dir}/configs/experiments/live_simple64_scripted_playbook_fixture_active.yaml"
shadow_config="${repo_dir}/configs/experiments/live_simple64_scripted_playbook_fixture_shadow.yaml"
active_playbook="${output_root}/cortex-playbook-canary-fixture-active.sqlite3"
shadow_playbook="${output_root}/cortex-playbook-canary-fixture-shadow.sqlite3"
baseline_snapshot="${run_set_dir}/playbook.canary-fixture.sqlite3"
readiness_evidence="${run_set_dir}/playbook-hard-readiness.json"
recovery_placeholder="${run_set_dir}/recovery-not-required.json"
reviewed_source_root="${run_set_dir}/reviewed-source"
status_file="${run_set_dir}/experiment-status.tsv"

claim_python="${RTSCORTEX_CLAIM_PYTHON:-python3}"
"${claim_python}" "${repo_dir}/scripts/qualification_attempt.py" claim \
  "${run_set_dir}" \
  --expected-git-sha "${expected_git_sha}" \
  --slurm-job-id "${SLURM_JOB_ID:-unknown}" \
  --slurm-restart-count "${SLURM_RESTART_COUNT:-0}"
run_set_dir="$(readlink -f "${run_set_dir}")"
baseline_snapshot="${run_set_dir}/playbook.canary-fixture.sqlite3"
readiness_evidence="${run_set_dir}/playbook-hard-readiness.json"
recovery_placeholder="${run_set_dir}/recovery-not-required.json"
status_file="${run_set_dir}/experiment-status.tsv"
attempt_manifest="${run_set_dir}/attempt-manifest.json"

cd "${repo_dir}"
git_head="$(git rev-parse HEAD)"
superproject_dirty="$(test -n "$(git status --porcelain --ignore-submodules=dirty)" && echo true || echo false)"
submodule_commit="$(git -C third_party/LLM-PySC2 rev-parse HEAD)"
submodule_dirty="$(test -n "$(git -C third_party/LLM-PySC2 status --porcelain)" && echo true || echo false)"
submodule_gitlink="$(git ls-tree HEAD third_party/LLM-PySC2 | awk '{print $3}')"
if [[ "${git_head}" != "${expected_git_sha}" \
  || "${superproject_dirty}" != "false" \
  || "${submodule_dirty}" != "false" \
  || "${submodule_commit}" != "${submodule_gitlink}" ]]; then
  echo "fixture canary requires clean ${expected_git_sha} and exact clean gitlink ${submodule_gitlink}" >&2
  exit 2
fi

uv run rtscortex playbook create-canary-fixture \
  --database "${baseline_snapshot}" \
  --expected-git-sha "${expected_git_sha}" \
  --sc2-patch "4.10"
baseline_sha256="$(sha256sum "${baseline_snapshot}" | awk '{print $1}')"

uv run rtscortex playbook hard-readiness \
  --database "${baseline_snapshot}" \
  --config "${active_config}" \
  --expected-git-sha "${expected_git_sha}" \
  --sc2-patch "4.10" \
  --evaluation-seed "${seed}" \
  --allow-canary-fixture \
  --output "${readiness_evidence}"
export RTSCORTEX_PLAYBOOK_HARD_READINESS_PATH="${readiness_evidence}"

uv run python scripts/prepare_reviewed_llm_pysc2_runtime.py \
  --source third_party/LLM-PySC2 \
  --output-root "${reviewed_source_root}" \
  --patch-directory integrations/llm_pysc2/patches \
  --expected-gitlink "${submodule_gitlink}"
export RTSCORTEX_REVIEWED_SOURCE_ROOT="${reviewed_source_root}"
reviewed_llm_pysc2="${reviewed_source_root}/third_party/LLM-PySC2"
reviewed_source_diff_sha256="$(
  git -C "${reviewed_llm_pysc2}" diff --binary | sha256sum | awk '{print $1}'
)"
reviewed_source_tree_sha256="$(
  uv run python -m scripts.hash_reviewed_source_tree \
    "${reviewed_llm_pysc2}" --field reviewed_tree_sha256
)"

uv run python - "${attempt_manifest}" "${recovery_placeholder}" <<'PY'
import sys
from pathlib import Path

from scripts.qualification_attempt import attempt_fields, create_only_json, load_attempt_manifest

manifest = load_attempt_manifest(Path(sys.argv[1]).parent)
create_only_json(
    sys.argv[2],
    {
        "accepted": False,
        "skipped": True,
        "reason": "bounded fixture canary does not claim recovery acceptance",
        **attempt_fields(manifest),
        "attempt": attempt_fields(manifest),
    },
)
PY
printf "experiment_kind\tmode\tseed\tarm\tsubject_arm\tarm_order\texit_code\trun_dir\tplaybook_before_sha256\tplaybook_after_sha256\tplaybook_before_snapshot\tplaybook_after_snapshot\tgit_head_before\tgit_head_after\tsuperproject_dirty_before\tsuperproject_dirty_after\tsubmodule_commit_before\tsubmodule_commit_after\tsubmodule_dirty_before\tsubmodule_dirty_after\tsubmodule_gitlink_before\tsubmodule_gitlink_after\tsubmodule_diff_sha256_before\tsubmodule_diff_sha256_after\treviewed_source_commit_before\treviewed_source_commit_after\treviewed_source_diff_sha256_before\treviewed_source_diff_sha256_after\treviewed_source_tree_sha256_before\treviewed_source_tree_sha256_after\n" \
  > "${status_file}"

run_arm() {
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
  : > "${log_path}"
  local submodule_diff_sha256
  submodule_diff_sha256="$(git -C third_party/LLM-PySC2 diff --binary | sha256sum | awk '{print $1}')"
  local reviewed_commit_before
  reviewed_commit_before="$(git -C "${reviewed_llm_pysc2}" rev-parse HEAD)"
  local reviewed_diff_before
  reviewed_diff_before="$(
    git -C "${reviewed_llm_pysc2}" diff --binary | sha256sum | awk '{print $1}'
  )"
  local reviewed_tree_before
  reviewed_tree_before="$(
    uv run python -m scripts.hash_reviewed_source_tree \
      "${reviewed_llm_pysc2}" --field reviewed_tree_sha256
  )"
  if [[ "${reviewed_commit_before}" != "${submodule_gitlink}" \
    || "${reviewed_diff_before}" != "${reviewed_source_diff_sha256}" \
    || "${reviewed_tree_before}" != "${reviewed_source_tree_sha256}" ]]; then
    echo "reviewed Worker source changed before fixture ${arm}/seed-${seed}" >&2
    exit 2
  fi
  set +e
  SC2PATH="/mnt/scratch/users/tbczhang/StarCraftII" \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    uv run rtscortex run --config "${config}" --seed "${seed}" \
    2>&1 | tee -a "${log_path}"
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
  local dirty_after
  local submodule_commit_after
  local submodule_dirty_after
  local submodule_gitlink_after
  local submodule_diff_after
  git_head_after="$(git rev-parse HEAD)"
  dirty_after="$(test -n "$(git status --porcelain --ignore-submodules=dirty)" && echo true || echo false)"
  submodule_commit_after="$(git -C third_party/LLM-PySC2 rev-parse HEAD)"
  submodule_dirty_after="$(test -n "$(git -C third_party/LLM-PySC2 status --porcelain)" && echo true || echo false)"
  submodule_gitlink_after="$(git ls-tree HEAD third_party/LLM-PySC2 | awk '{print $3}')"
  submodule_diff_after="$(git -C third_party/LLM-PySC2 diff --binary | sha256sum | awk '{print $1}')"
  local reviewed_commit_after
  reviewed_commit_after="$(git -C "${reviewed_llm_pysc2}" rev-parse HEAD)"
  local reviewed_diff_after
  reviewed_diff_after="$(
    git -C "${reviewed_llm_pysc2}" diff --binary | sha256sum | awk '{print $1}'
  )"
  local reviewed_tree_after
  reviewed_tree_after="$(
    uv run python -m scripts.hash_reviewed_source_tree \
      "${reviewed_llm_pysc2}" --field reviewed_tree_sha256
  )"
  if [[ "${reviewed_commit_after}" != "${submodule_gitlink}" \
    || "${reviewed_diff_after}" != "${reviewed_source_diff_sha256}" \
    || "${reviewed_tree_after}" != "${reviewed_source_tree_sha256}" ]]; then
    echo "reviewed Worker source changed during fixture ${arm}/seed-${seed}" >&2
    run_status=86
  fi
  local fields=(
    "${kind}" "fixture" "${seed}" "${arm}" "${subject_arm}" "active,shadow"
    "${run_status}" "${run_dir}" "${before_sha256}" "${after_sha256}"
    "${before_snapshot}" "${after_snapshot}" "${git_head}" "${git_head_after}"
    "${superproject_dirty}" "${dirty_after}" "${submodule_commit}"
    "${submodule_commit_after}" "${submodule_dirty}" "${submodule_dirty_after}"
    "${submodule_gitlink}" "${submodule_gitlink_after}"
    "${submodule_diff_sha256}" "${submodule_diff_after}"
    "${reviewed_commit_before}" "${reviewed_commit_after}"
    "${reviewed_diff_before}" "${reviewed_diff_after}"
    "${reviewed_tree_before}" "${reviewed_tree_after}"
  )
  (
    IFS=$'\t'
    echo "${fields[*]}"
  ) >> "${status_file}"
}

run_arm behavior active "" "${active_config}" "${active_playbook}"
run_arm calibration shadow active "${shadow_config}" "${shadow_playbook}"

uv run python -m scripts.analyze_playbook_counterfactual_canary \
  "${run_set_dir}" \
  --baseline-sha256 "${baseline_sha256}" \
  --expected-git-sha "${expected_git_sha}" \
  --engineering-baseline "${repo_dir}/configs/acceptance/protoss_natural_terminal_v1.json" \
  --recovery-evidence "${recovery_placeholder}" \
  --readiness-evidence "${readiness_evidence}" \
  --canary-kind fixture \
  --output "${run_set_dir}/counterfactual-canary.json"
