#!/usr/bin/env bash
set -euo pipefail
set -o noclobber
export PYTHONDONTWRITEBYTECODE=1

if [[ $# -ne 5 || "$2" != "--expected-git-sha" || "$4" != "--seed" ]]; then
  echo "usage: $0 <run-set-dir> --expected-git-sha <40-char-sha> --seed <nonnegative-seed>" >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_dir="$(cd -- "${script_dir}/.." && pwd -P)"
run_set_dir="$1"
expected_git_sha="$3"
seed="$5"
config_path="${repo_dir}/configs/experiments/live_simple64_hima_protoss_authoritative_build_circuit_canary.yaml"

if [[ ! "${expected_git_sha}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "expected-git-sha must be a full 40-character lowercase Git SHA" >&2
  exit 2
fi
if [[ ! "${seed}" =~ ^[0-9]+$ ]]; then
  echo "seed must be a nonnegative integer" >&2
  exit 2
fi
if [[ ! -f "${config_path}" ]]; then
  echo "missing dedicated canary config: ${config_path}" >&2
  exit 2
fi

claim_python="${RTSCORTEX_CLAIM_PYTHON:-python3}"
"${claim_python}" "${repo_dir}/scripts/qualification_attempt.py" claim \
  "${run_set_dir}" \
  --expected-git-sha "${expected_git_sha}" \
  --slurm-job-id "${SLURM_JOB_ID:-unknown}" \
  --slurm-restart-count "${SLURM_RESTART_COUNT:-0}"
run_set_dir="$(readlink -f "${run_set_dir}")"
reviewed_source_root="${run_set_dir}/reviewed-source"
status_path="${run_set_dir}/canary-attestation.json"
report_path="${run_set_dir}/authoritative-build-circuit-canary.json"
log_path="${run_set_dir}/run.log"
attempt_manifest="${run_set_dir}/attempt-manifest.json"
: > "${log_path}"

cd "${repo_dir}"
export PYTHONPATH="${repo_dir}/src:${repo_dir}/integrations/llm_pysc2/src${PYTHONPATH:+:${PYTHONPATH}}"
git_head_before="$(git rev-parse HEAD)"
superproject_dirty_before="$(test -n "$(git status --porcelain --ignore-submodules=dirty)" && echo true || echo false)"
submodule_commit_before="$(git -C third_party/LLM-PySC2 rev-parse HEAD)"
submodule_dirty_before="$(test -n "$(git -C third_party/LLM-PySC2 status --porcelain)" && echo true || echo false)"
submodule_gitlink_before="$(git ls-tree HEAD third_party/LLM-PySC2 | awk '{print $3}')"
submodule_diff_sha256_before="$(git -C third_party/LLM-PySC2 diff --binary | sha256sum | awk '{print $1}')"
if [[ "${git_head_before}" != "${expected_git_sha}" \
  || "${superproject_dirty_before}" != "false" \
  || "${submodule_dirty_before}" != "false" \
  || "${submodule_commit_before}" != "${submodule_gitlink_before}" ]]; then
  echo "authoritative Build canary requires clean expected SHA and exact clean submodule gitlink" >&2
  exit 2
fi

uv run python scripts/prepare_reviewed_llm_pysc2_runtime.py \
  --source third_party/LLM-PySC2 \
  --output-root "${reviewed_source_root}" \
  --patch-directory integrations/llm_pysc2/patches \
  --expected-gitlink "${submodule_gitlink_before}" \
  > "${run_set_dir}/reviewed-source-preparation.json"
reviewed_llm_pysc2="${reviewed_source_root}/third_party/LLM-PySC2"
reviewed_source_commit_before="$(git -C "${reviewed_llm_pysc2}" rev-parse HEAD)"
reviewed_source_diff_sha256_before="$(git -C "${reviewed_llm_pysc2}" diff --binary | sha256sum | awk '{print $1}')"
reviewed_source_tree_sha256_before="$(uv run python -m scripts.hash_reviewed_source_tree "${reviewed_llm_pysc2}" --field reviewed_tree_sha256)"
if [[ "${reviewed_source_commit_before}" != "${submodule_gitlink_before}" ]]; then
  echo "reviewed Worker source commit does not match the exact submodule gitlink" >&2
  exit 2
fi

set +e
SC2PATH="/mnt/scratch/users/tbczhang/StarCraftII" \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  TOKENIZERS_PARALLELISM=false \
  RTSCORTEX_REVIEWED_SOURCE_ROOT="$(readlink -f "${reviewed_source_root}")" \
  uv run rtscortex run --config "${config_path}" --seed "${seed}" \
  2>&1 | tee -a "${log_path}"
run_status=${PIPESTATUS[0]}
set -e

run_dir="$(sed -n -e 's/^Run directory: //p' -e 's/^Artifacts: //p' "${log_path}" | tail -n 1)"
if [[ -n "${run_dir}" ]]; then
  run_dir="$(readlink -f "${run_dir}")"
fi
canonical_run_link="${run_set_dir}/canonical-run"
if [[ -z "${run_dir}" || ! -d "${run_dir}" \
  || ! -f "${run_dir}/authoritative-build-circuit-canary.jsonl" \
  || ! -f "${run_dir}/events.jsonl" \
  || ! -f "${run_dir}/summary.json" ]]; then
  run_status=86
else
  if [[ -e "${canonical_run_link}" || -L "${canonical_run_link}" ]]; then
    echo "canonical-run symlink already exists in run set" >&2
    run_status=86
  else
    ln -s "${run_dir}" "${canonical_run_link}"
  fi
fi
canary_journal_sha256=""
events_sha256=""
summary_sha256=""
if [[ -n "${run_dir}" && -f "${run_dir}/authoritative-build-circuit-canary.jsonl" ]]; then
  canary_journal_sha256="$(sha256sum "${run_dir}/authoritative-build-circuit-canary.jsonl" | awk '{print $1}')"
fi
if [[ -n "${run_dir}" && -f "${run_dir}/events.jsonl" ]]; then
  events_sha256="$(sha256sum "${run_dir}/events.jsonl" | awk '{print $1}')"
fi
if [[ -n "${run_dir}" && -f "${run_dir}/summary.json" ]]; then
  summary_sha256="$(sha256sum "${run_dir}/summary.json" | awk '{print $1}')"
fi

git_head_after="$(git rev-parse HEAD)"
superproject_dirty_after="$(test -n "$(git status --porcelain --ignore-submodules=dirty)" && echo true || echo false)"
submodule_commit_after="$(git -C third_party/LLM-PySC2 rev-parse HEAD)"
submodule_dirty_after="$(test -n "$(git -C third_party/LLM-PySC2 status --porcelain)" && echo true || echo false)"
submodule_gitlink_after="$(git ls-tree HEAD third_party/LLM-PySC2 | awk '{print $3}')"
submodule_diff_sha256_after="$(git -C third_party/LLM-PySC2 diff --binary | sha256sum | awk '{print $1}')"
reviewed_source_commit_after="$(git -C "${reviewed_llm_pysc2}" rev-parse HEAD)"
reviewed_source_diff_sha256_after="$(git -C "${reviewed_llm_pysc2}" diff --binary | sha256sum | awk '{print $1}')"
reviewed_source_tree_sha256_after="$(uv run python -m scripts.hash_reviewed_source_tree "${reviewed_llm_pysc2}" --field reviewed_tree_sha256)"
if [[ "${git_head_after}" != "${expected_git_sha}" \
  || "${superproject_dirty_after}" != "false" \
  || "${submodule_dirty_after}" != "false" \
  || "${submodule_commit_after}" != "${submodule_gitlink_after}" \
  || "${submodule_gitlink_after}" != "${submodule_gitlink_before}" \
  || "${reviewed_source_commit_after}" != "${submodule_gitlink_before}" \
  || "${reviewed_source_diff_sha256_after}" != "${reviewed_source_diff_sha256_before}" \
  || "${reviewed_source_tree_sha256_after}" != "${reviewed_source_tree_sha256_before}" ]]; then
  run_status=86
fi

export CANARY_ATTESTATION_PATH="${status_path}"
export CANARY_ATTEMPT_MANIFEST="${attempt_manifest}"
export CANARY_EXPECTED_GIT_SHA="${expected_git_sha}"
export CANARY_SEED="${seed}"
export CANARY_RUN_DIR="${run_dir}"
export CANARY_CANONICAL_RUN_LINK="${canonical_run_link}"
export CANARY_RUN_STATUS="${run_status}"
export CANARY_JOURNAL_SHA256="${canary_journal_sha256}"
export CANARY_EVENTS_SHA256="${events_sha256}"
export CANARY_SUMMARY_SHA256="${summary_sha256}"
export CANARY_GIT_HEAD_BEFORE="${git_head_before}"
export CANARY_GIT_HEAD_AFTER="${git_head_after}"
export CANARY_SUPERPROJECT_DIRTY_BEFORE="${superproject_dirty_before}"
export CANARY_SUPERPROJECT_DIRTY_AFTER="${superproject_dirty_after}"
export CANARY_SUBMODULE_COMMIT_BEFORE="${submodule_commit_before}"
export CANARY_SUBMODULE_COMMIT_AFTER="${submodule_commit_after}"
export CANARY_SUBMODULE_DIRTY_BEFORE="${submodule_dirty_before}"
export CANARY_SUBMODULE_DIRTY_AFTER="${submodule_dirty_after}"
export CANARY_SUBMODULE_GITLINK_BEFORE="${submodule_gitlink_before}"
export CANARY_SUBMODULE_GITLINK_AFTER="${submodule_gitlink_after}"
export CANARY_SUBMODULE_DIFF_BEFORE="${submodule_diff_sha256_before}"
export CANARY_SUBMODULE_DIFF_AFTER="${submodule_diff_sha256_after}"
export CANARY_REVIEWED_COMMIT_BEFORE="${reviewed_source_commit_before}"
export CANARY_REVIEWED_COMMIT_AFTER="${reviewed_source_commit_after}"
export CANARY_REVIEWED_DIFF_BEFORE="${reviewed_source_diff_sha256_before}"
export CANARY_REVIEWED_DIFF_AFTER="${reviewed_source_diff_sha256_after}"
export CANARY_REVIEWED_TREE_BEFORE="${reviewed_source_tree_sha256_before}"
export CANARY_REVIEWED_TREE_AFTER="${reviewed_source_tree_sha256_after}"
export CANARY_CONFIG_PATH="${config_path}"
export CANARY_CONFIG_SHA256="$(sha256sum "${config_path}" | awk '{print $1}')"

uv run python - <<'PY'
import os
from pathlib import Path

from scripts.qualification_attempt import attempt_fields, create_only_json, load_attempt_manifest

def value(name: str) -> str:
    return os.environ.get(name, "")

attempt = load_attempt_manifest(Path(value("CANARY_ATTEMPT_MANIFEST")).parent)
attestation = {
    "schema_version": "1.0",
    "artifact_kind": "authoritative-build-circuit-canary",
    **attempt_fields(attempt),
    "attempt": attempt_fields(attempt),
    "diagnostic_only": True,
    "expected_git_sha": value("CANARY_EXPECTED_GIT_SHA"),
    "seed": int(value("CANARY_SEED")),
    "exit_code": int(value("CANARY_RUN_STATUS")),
    "run_dir": value("CANARY_RUN_DIR"),
    "canonical_run_link": value("CANARY_CANONICAL_RUN_LINK"),
    "config_path": value("CANARY_CONFIG_PATH"),
    "config_sha256": value("CANARY_CONFIG_SHA256"),
    "canary_journal_path": str(Path(value("CANARY_RUN_DIR")) / "authoritative-build-circuit-canary.jsonl"),
    "canary_journal_sha256": value("CANARY_JOURNAL_SHA256"),
    "events_path": str(Path(value("CANARY_RUN_DIR")) / "events.jsonl"),
    "events_sha256": value("CANARY_EVENTS_SHA256"),
    "summary_path": str(Path(value("CANARY_RUN_DIR")) / "summary.json"),
    "summary_sha256": value("CANARY_SUMMARY_SHA256"),
    "source_attestation": {
        "git_head_before": value("CANARY_GIT_HEAD_BEFORE"),
        "git_head_after": value("CANARY_GIT_HEAD_AFTER"),
        "superproject_dirty_before": value("CANARY_SUPERPROJECT_DIRTY_BEFORE"),
        "superproject_dirty_after": value("CANARY_SUPERPROJECT_DIRTY_AFTER"),
        "submodule_commit_before": value("CANARY_SUBMODULE_COMMIT_BEFORE"),
        "submodule_commit_after": value("CANARY_SUBMODULE_COMMIT_AFTER"),
        "submodule_dirty_before": value("CANARY_SUBMODULE_DIRTY_BEFORE"),
        "submodule_dirty_after": value("CANARY_SUBMODULE_DIRTY_AFTER"),
        "submodule_gitlink_before": value("CANARY_SUBMODULE_GITLINK_BEFORE"),
        "submodule_gitlink_after": value("CANARY_SUBMODULE_GITLINK_AFTER"),
        "submodule_diff_sha256_before": value("CANARY_SUBMODULE_DIFF_BEFORE"),
        "submodule_diff_sha256_after": value("CANARY_SUBMODULE_DIFF_AFTER"),
        "reviewed_source_commit_before": value("CANARY_REVIEWED_COMMIT_BEFORE"),
        "reviewed_source_commit_after": value("CANARY_REVIEWED_COMMIT_AFTER"),
        "reviewed_source_diff_sha256_before": value("CANARY_REVIEWED_DIFF_BEFORE"),
        "reviewed_source_diff_sha256_after": value("CANARY_REVIEWED_DIFF_AFTER"),
        "reviewed_source_tree_sha256_before": value("CANARY_REVIEWED_TREE_BEFORE"),
        "reviewed_source_tree_sha256_after": value("CANARY_REVIEWED_TREE_AFTER"),
    },
}
create_only_json(value("CANARY_ATTESTATION_PATH"), attestation)
PY

set +e
uv run python -m scripts.analyze_authoritative_build_circuit_canary \
  "${run_set_dir}" \
  --expected-git-sha "${expected_git_sha}" \
  --seed "${seed}" \
  --output "${report_path}"
analysis_status=$?
set -e
if [[ ${analysis_status} -ne 0 ]]; then
  exit ${analysis_status}
fi
exit 0
