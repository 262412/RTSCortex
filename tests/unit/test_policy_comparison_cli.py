from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from typer.testing import CliRunner

from rtscortex.cli import app as cli_module

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORPUS_MANIFEST = PROJECT_ROOT / "benchmarks/policy/protoss_v0_2/manifest.yaml"


def test_policy_corpus_verify_cli_reports_balanced_fixture_counts() -> None:
    result = CliRunner().invoke(
        cli_module.app,
        ["policy-corpus", "verify", str(CORPUS_MANIFEST)],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["valid"] is True
    assert payload["fixture_count"] == 48
    assert payload["stratum_counts"] == {
        "early": 8,
        "technology": 8,
        "production": 8,
        "combat": 8,
        "blocked": 8,
        "in_progress": 8,
    }


def test_policy_corpus_build_cli_delegates_to_typed_builder(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    config_path = tmp_path / "corpus.yaml"
    config_path.write_text("sources: []\n", encoding="utf-8")
    output_dir = tmp_path / "built"
    manifest_path = output_dir / "manifest.yaml"
    fixtures_path = output_dir / "fixtures.jsonl"
    calls: list[tuple[Path, Path]] = []

    def fake_build(config: Path, output: Path) -> object:
        calls.append((config, output))
        return SimpleNamespace(
            manifest=SimpleNamespace(
                fixture_count=48,
                stratum_counts={},
                seeds=[0, 1, 2],
            ),
            manifest_path=manifest_path,
            fixtures_path=fixtures_path,
        )

    monkeypatch.setattr(cli_module, "build_policy_corpus_from_file", fake_build)

    result = CliRunner().invoke(
        cli_module.app,
        [
            "policy-corpus",
            "build",
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == [(config_path.resolve(), output_dir.resolve())]
    assert f"Manifest: {manifest_path}" in result.output
    assert f"Fixtures: {fixtures_path}" in result.output


def test_policy_compare_cli_writes_offline_reports_and_candidate_artifacts(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "comparison.yaml"
    output_dir = tmp_path / "comparison-output"
    config_path.write_text(
        f"""
format_version: "0.2"
corpus_manifest: {CORPUS_MANIFEST}
output_root: {tmp_path}
qwen:
  enabled: false
hima:
  enabled: [a, b, c]
  allow_unlicensed_weights: false
  model_paths: {{}}
hiernet:
  enabled: false
""",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        cli_module.app,
        [
            "policy-compare",
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert f"Comparison: {output_dir / 'comparison.json'}" in result.output
    assert f"Report: {output_dir / 'report.md'}" in result.output
    assert (output_dir / "config.snapshot.yaml").is_file()
    assert (output_dir / "corpus.snapshot.yaml").is_file()
    assert (output_dir / "candidates/hima-protoss-a/records.jsonl").is_file()
    comparison = json.loads((output_dir / "comparison.json").read_text())
    assert len(comparison["fixture_ids"]) == 48
    assert len(comparison["records"]) == 240
