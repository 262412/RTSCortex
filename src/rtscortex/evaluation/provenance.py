"""Reproducibility metadata and configuration snapshots for evaluations."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import yaml

from rtscortex.config import ExperimentConfig

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROMPT_VERSIONS = {
    "planning": "planning-v1",
    "reflection": "reflection-v1",
}


def write_experiment_snapshot(
    config: ExperimentConfig,
    output_dir: Path,
    *,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Write the resolved YAML configuration and machine-readable provenance."""

    output_dir.mkdir(parents=True, exist_ok=True)
    config_payload = config.model_dump(mode="json")
    (output_dir / "config.yaml").write_text(
        yaml.safe_dump(config_payload, sort_keys=False),
        encoding="utf-8",
    )
    provenance = build_provenance(config, project_root=project_root)
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return provenance


def build_provenance(
    config: ExperimentConfig,
    *,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    config_json = json.dumps(
        config.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    upstream_root = project_root / "third_party" / "LLM-PySC2"
    prompt_source = PACKAGE_ROOT / "agents" / "modules.py"
    prompt_source_sha256 = hashlib.sha256(prompt_source.read_bytes()).hexdigest()

    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "configuration": {
            "sha256": hashlib.sha256(config_json.encode()).hexdigest(),
            "seeds": config.evaluation.seeds,
            "agent_variant": config.agent.variant,
        },
        "code": {
            "rtscortex_commit": _git_value(project_root, "rev-parse", "HEAD"),
            "rtscortex_dirty": _git_dirty(project_root),
            "llm_pysc2_commit": _git_value(upstream_root, "rev-parse", "HEAD"),
            "llm_pysc2_dirty": _git_dirty(upstream_root),
        },
        "provider": {
            "kind": config.provider.kind,
            "model": config.provider.model,
            "base_url": config.provider.base_url,
            "prompt_cost_per_million_tokens": (config.provider.prompt_cost_per_million_tokens),
            "completion_cost_per_million_tokens": (
                config.provider.completion_cost_per_million_tokens
            ),
        },
        "prompts": {
            name: {
                "version": prompt_version,
                "source": "src/rtscortex/agents/modules.py",
                "source_sha256": prompt_source_sha256,
            }
            for name, prompt_version in PROMPT_VERSIONS.items()
        },
        "environment": {
            "rtscortex_version": version("rtscortex"),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "adapter": config.environment.adapter,
            "adapter_version": _adapter_version(config, upstream_root),
            "scenario": config.environment.scenario,
            "sc2_path": (
                None if config.environment.sc2_path is None else str(config.environment.sc2_path)
            ),
        },
    }


def _git_value(root: Path, *arguments: str) -> str | None:
    if not root.is_dir():
        return None
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _git_dirty(root: Path) -> bool | None:
    status = _git_value(root, "status", "--porcelain")
    return None if status is None else bool(status)


def _adapter_version(config: ExperimentConfig, upstream_root: Path) -> str | None:
    if config.environment.adapter == "mock":
        return "mock-v1"
    return _git_value(upstream_root, "rev-parse", "HEAD")
