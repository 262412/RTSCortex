from __future__ import annotations

import asyncio
import json
import socket
import subprocess
import sys
from pathlib import Path

import pytest

import rtscortex.policy.comparison as comparison_module
import rtscortex.policy.comparison_worker as comparison_worker_module
from rtscortex.policy.comparison import (
    PolicyComparisonConfig,
    PolicyComparisonError,
    load_policy_comparison_config,
    run_policy_comparison,
    validate_hima_model_path,
)
from rtscortex.policy.corpus import load_policy_corpus
from rtscortex.policy.hima.subagent import HIMA_PINNED_REVISIONS
from rtscortex.policy.models import (
    PolicyAvailability,
    PolicyAvailabilityStatus,
    PolicyShadowComparison,
)
from rtscortex.policy.shadow import PolicyShadowRunner
from rtscortex.policy.subagents import (
    HIMA_PROTOSS_SPECS,
    HIMA_RACE_SPECS,
    PolicySubagentRegistration,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORPUS_MANIFEST = PROJECT_ROOT / "benchmarks/policy/protoss_v0_2/manifest.yaml"
ZERG_CORPUS_MANIFEST = PROJECT_ROOT / "benchmarks/policy/zerg_v0_3/manifest.yaml"


def _offline_config(
    tmp_path: Path,
    *,
    manifest_path: Path = CORPUS_MANIFEST,
) -> PolicyComparisonConfig:
    return PolicyComparisonConfig.model_validate(
        {
            "corpus_manifest": manifest_path,
            "output_root": tmp_path,
            "qwen": {"enabled": False},
            "hima": {"enabled": ["a", "b", "c"]},
            "hiernet": {"enabled": False},
        }
    )


def _snapshot_path(root: Path, *, model_id: str, revision: str) -> Path:
    path = root / f"models--{model_id.replace('/', '--')}" / "snapshots" / revision
    path.mkdir(parents=True)
    return path


def test_comparison_config_is_strict_and_resolves_project_relative_paths(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "comparison.yaml"
    config_path.write_text(
        """
format_version: "0.2"
corpus_manifest: benchmarks/corpus.yaml
output_root: outputs
qwen:
  enabled: false
hima:
  enabled: [a, c]
  python_executable: env/bin/python
hiernet:
  enabled: false
""",
        encoding="utf-8",
    )

    config = load_policy_comparison_config(config_path, base_dir=tmp_path)

    assert config.corpus_manifest == (tmp_path / "benchmarks/corpus.yaml").resolve()
    assert config.output_root == (tmp_path / "outputs").resolve()
    assert config.hima.python_executable == (tmp_path / "env/bin/python").resolve()
    assert config.hima.enabled == ["a", "c"]

    with pytest.raises(ValueError, match="requires experiment_config"):
        PolicyComparisonConfig.model_validate(
            {
                "corpus_manifest": "manifest.yaml",
                "qwen": {"enabled": True},
            }
        )
    with pytest.raises(ValueError, match="extra"):
        PolicyComparisonConfig.model_validate(
            {
                "corpus_manifest": "manifest.yaml",
                "qwen": {"enabled": False},
                "surprise": True,
            }
        )


def test_comparison_config_preserves_hima_venv_python_symlink(tmp_path: Path) -> None:
    venv_python = tmp_path / "hima-venv/bin/python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(Path(sys.executable).resolve())
    config = PolicyComparisonConfig.model_validate(
        {
            "corpus_manifest": "manifest.yaml",
            "qwen": {"enabled": False},
            "hima": {"python_executable": "hima-venv/bin/python"},
        }
    )

    resolved = config.resolved(base_dir=tmp_path)

    assert resolved.hima.python_executable == venv_python.absolute()
    assert resolved.hima.python_executable != venv_python.resolve()


def test_hima_checkpoint_provenance_rejects_swapped_model_and_revision(
    tmp_path: Path,
) -> None:
    model_a = HIMA_PROTOSS_SPECS[0]
    model_b = HIMA_PROTOSS_SPECS[1]
    expected_a = HIMA_PINNED_REVISIONS[model_a.model_id]
    expected_b = HIMA_PINNED_REVISIONS[model_b.model_id]
    checkpoint = _snapshot_path(
        tmp_path,
        model_id=model_a.model_id,
        revision=expected_a,
    )

    assert validate_hima_model_path(model_a.model_id, checkpoint) == expected_a
    with pytest.raises(PolicyComparisonError, match="provenance mismatch"):
        validate_hima_model_path(model_b.model_id, checkpoint)

    wrong_revision = _snapshot_path(
        tmp_path / "wrong",
        model_id=model_a.model_id,
        revision=expected_b,
    )
    with pytest.raises(PolicyComparisonError, match="provenance mismatch"):
        validate_hima_model_path(model_a.model_id, wrong_revision)


def test_offline_comparison_writes_complete_no_download_artifact_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported_model_modules_before = {
        name
        for name in sys.modules
        if name == "transformers"
        or name.startswith("transformers.")
        or name == "huggingface_hub"
        or name.startswith("huggingface_hub.")
    }

    def unexpected_network(
        self: socket.socket,
        address: object,
    ) -> None:
        del self, address
        raise AssertionError("offline comparison must not open a network connection")

    def unexpected_process(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        del args, kwargs
        raise AssertionError("offline comparison must not start HIMA")

    monkeypatch.setattr(socket.socket, "connect", unexpected_network)
    monkeypatch.setattr(comparison_module, "_execute_process", unexpected_process)
    output_dir = tmp_path / "offline-result"

    artifacts = run_policy_comparison(_offline_config(tmp_path), output_dir=output_dir)

    assert artifacts.output_dir == output_dir.resolve()
    assert artifacts.comparison.candidate_ids == [
        "qwen3-8b-current",
        "hima-protoss-a",
        "hima-protoss-b",
        "hima-protoss-c",
        "hiernet-sc2-protoss",
    ]
    assert len(artifacts.comparison.fixture_ids) == 48
    assert len(artifacts.comparison.records) == 48 * 5
    assert artifacts.reports.comparison_path.is_file()
    assert artifacts.reports.report_path.is_file()
    assert artifacts.config_snapshot_path.is_file()
    assert artifacts.corpus_snapshot_path.read_text(encoding="utf-8") == (
        CORPUS_MANIFEST.read_text(encoding="utf-8")
    )
    for candidate_id in artifacts.comparison.candidate_ids:
        candidate_dir = output_dir / "candidates" / candidate_id
        assert len((candidate_dir / "records.jsonl").read_text().splitlines()) == 48
        provenance = json.loads((candidate_dir / "provenance.json").read_text())
        assert provenance["network_downloads_allowed"] is False
        assert provenance["fixture_ids"] == artifacts.comparison.fixture_ids
    assert {
        name
        for name in sys.modules
        if name == "transformers"
        or name.startswith("transformers.")
        or name == "huggingface_hub"
        or name.startswith("huggingface_hub.")
    } == imported_model_modules_before


def test_zerg_corpus_selects_zerg_specialists_and_race_provenance(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "zerg-offline-result"

    artifacts = run_policy_comparison(
        _offline_config(tmp_path, manifest_path=ZERG_CORPUS_MANIFEST),
        output_dir=output_dir,
    )

    assert artifacts.comparison.candidate_ids == [
        "qwen3-8b-current",
        "hima-zerg-a",
        "hima-zerg-b",
        "hima-zerg-c",
        "hiernet-sc2-protoss",
    ]
    provenance = json.loads(
        (output_dir / "candidates/hima-zerg-a/provenance.json").read_text()
    )
    assert provenance["vocabulary_version"] == "hima-zerg-63-v1"
    assert provenance["parser_version"] == "hima-zerg-parser-v2"
    report = artifacts.reports.report_path.read_text()
    assert "hima-zerg-parser-v2" in report
    assert "hima-protoss-parser-v5" not in report


def test_comparison_worker_uses_the_manifest_race_for_zerg(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = HIMA_RACE_SPECS["zerg"][0]
    revision = HIMA_PINNED_REVISIONS[spec.model_id]
    checkpoint = _snapshot_path(tmp_path, model_id=spec.model_id, revision=revision)
    request_path = tmp_path / "request.json"
    response_path = tmp_path / "response.json"
    request_path.write_text(
        comparison_module.HIMAWorkerRequest(
            manifest_path=ZERG_CORPUS_MANIFEST,
            model_id=spec.model_id,
            model_path=checkpoint,
            device="cpu",
            allow_unlicensed_weights=True,
        ).model_dump_json(),
        encoding="utf-8",
    )

    class FakeGenerator:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        async def generate(self, *, user_message: str) -> str:
            assert '"unit"' in user_message
            return 'Actions: ["Drone"]'

    monkeypatch.setattr(
        comparison_worker_module,
        "TransformersHIMAGenerator",
        FakeGenerator,
    )

    comparison_worker_module.run_worker(request_path, response_path)

    comparison = PolicyShadowComparison.model_validate_json(
        response_path.read_text()
    )
    assert comparison.candidate_ids == ["hima-zerg-a"]
    assert len(comparison.records) == 48
    assert all(record.spec.race == "Zerg" for record in comparison.records)


def test_hima_candidates_run_sequentially_and_one_failure_is_isolated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    python = tmp_path / "hima-python"
    python.write_text("", encoding="utf-8")
    paths: dict[str, Path] = {}
    for variant, spec in zip(("a", "b", "c"), HIMA_PROTOSS_SPECS, strict=True):
        path = _snapshot_path(
            tmp_path / variant,
            model_id=spec.model_id,
            revision=HIMA_PINNED_REVISIONS[spec.model_id],
        )
        paths[variant] = path

    calls: list[str] = []

    def fake_process(
        command: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        del cwd, timeout
        assert env["HF_HUB_OFFLINE"] == "1"
        assert env["TRANSFORMERS_OFFLINE"] == "1"
        request_path = Path(command[command.index("--request") + 1])
        response_path = Path(command[command.index("--response") + 1])
        request = comparison_module.HIMAWorkerRequest.model_validate_json(
            request_path.read_text(encoding="utf-8")
        )
        calls.append(request.model_id)
        if request.model_id.endswith("Protoss-b"):
            return subprocess.CompletedProcess(command, 7, "", "synthetic crash")
        spec = next(item for item in HIMA_PROTOSS_SPECS if item.model_id == request.model_id)
        fixtures = load_policy_corpus(request.manifest_path)
        comparison = asyncio.run(
            PolicyShadowRunner().compare(
                fixtures,
                [
                    PolicySubagentRegistration(
                        spec=spec,
                        availability=PolicyAvailability(
                            status=PolicyAvailabilityStatus.UNAVAILABLE,
                            reason="fake worker result",
                        ),
                    )
                ],
            )
        )
        response_path.write_text(comparison.model_dump_json(), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(comparison_module, "_execute_process", fake_process)
    config = _offline_config(tmp_path).model_copy(
        update={
            "hima": _offline_config(tmp_path).hima.model_copy(
                update={
                    "python_executable": python,
                    "allow_unlicensed_weights": True,
                    "model_paths": paths,
                }
            )
        }
    )
    output_dir = tmp_path / "isolated-result"

    result = run_policy_comparison(config, output_dir=output_dir)

    assert calls == [spec.model_id for spec in HIMA_PROTOSS_SPECS]
    b_records = [
        record
        for record in result.comparison.records
        if record.spec.subagent_id == "hima-protoss-b"
    ]
    assert len(b_records) == 48
    assert all(record.status.value == "failed" for record in b_records)
    assert all("synthetic crash" in (record.error or "") for record in b_records)
    assert all(
        record.status.value == "unavailable"
        for record in result.comparison.records
        if record.spec.subagent_id in {"hima-protoss-a", "hima-protoss-c"}
    )


def test_run_rejects_existing_output_directory(tmp_path: Path) -> None:
    output_dir = tmp_path / "existing"
    output_dir.mkdir()
    with pytest.raises(PolicyComparisonError, match="already exists"):
        run_policy_comparison(_offline_config(tmp_path), output_dir=output_dir)
