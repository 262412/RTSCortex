"""Offline policy-comparison configuration and candidate-isolated orchestration."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, NoReturn

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from rtscortex.config import load_config
from rtscortex.policy.corpus import load_policy_corpus
from rtscortex.policy.hima.models import (
    HIMA_ADAPTER_VERSION,
    HIMA_PARSER_VERSION,
    HIMA_UPSTREAM_REVISION,
    HIMA_VOCABULARY_VERSION,
)
from rtscortex.policy.hima.subagent import HIMA_PINNED_REVISIONS
from rtscortex.policy.models import (
    PolicyAvailability,
    PolicyAvailabilityStatus,
    PolicyObservationFixture,
    PolicyShadowComparison,
    PolicySubagentSpec,
)
from rtscortex.policy.report import (
    PolicyComparisonReportArtifacts,
    write_policy_comparison_reports,
)
from rtscortex.policy.shadow import PolicyShadowRunner
from rtscortex.policy.subagents import (
    HIERNET_SC2_SPEC,
    HIMA_PROTOSS_SPECS,
    QWEN3_8B_SPEC,
    LLMPlanningPolicySubagent,
    PolicySubagentRegistration,
)
from rtscortex.providers import OpenAICompatibleProvider

PolicyVariant = Literal["a", "b", "c"]
HIMA_VARIANTS: tuple[PolicyVariant, ...] = ("a", "b", "c")
COMPARISON_FORMAT_VERSION: Literal["0.2"] = "0.2"


class PolicyComparisonError(ValueError):
    """Raised when a comparison cannot be configured or safely executed."""


class ComparisonConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class QwenComparisonConfig(ComparisonConfigModel):
    enabled: bool = True
    experiment_config: Path | None = None

    @model_validator(mode="after")
    def require_experiment_config(self) -> QwenComparisonConfig:
        if self.enabled and self.experiment_config is None:
            raise ValueError("enabled Qwen comparison requires experiment_config")
        return self


class HIMAComparisonConfig(ComparisonConfigModel):
    enabled: list[PolicyVariant] = Field(default_factory=list)
    python_executable: Path = Path(sys.executable)
    device: str = Field(default="cuda:0", min_length=1)
    allow_unlicensed_weights: bool = False
    model_paths: dict[PolicyVariant, Path] = Field(default_factory=dict)
    timeout_seconds: float = Field(default=1_800.0, gt=0.0)

    @model_validator(mode="after")
    def require_unique_variants(self) -> HIMAComparisonConfig:
        if len(self.enabled) != len(set(self.enabled)):
            raise ValueError("HIMA enabled variants must be unique")
        return self


class HierNetComparisonConfig(ComparisonConfigModel):
    enabled: bool = False


class PolicyComparisonConfig(ComparisonConfigModel):
    format_version: Literal["0.2"] = COMPARISON_FORMAT_VERSION
    corpus_manifest: Path
    output_root: Path = Path("~/scratch/outputs/RTSCortex")
    qwen: QwenComparisonConfig = Field(default_factory=QwenComparisonConfig)
    hima: HIMAComparisonConfig = Field(default_factory=HIMAComparisonConfig)
    hiernet: HierNetComparisonConfig = Field(default_factory=HierNetComparisonConfig)

    def resolved(self, *, base_dir: Path) -> PolicyComparisonConfig:
        """Expand user paths and resolve project-relative paths from ``base_dir``."""

        qwen_path = self.qwen.experiment_config
        paths = {
            variant: _resolve_path(path, base_dir=base_dir)
            for variant, path in self.hima.model_paths.items()
        }
        return self.model_copy(
            update={
                "corpus_manifest": _resolve_path(self.corpus_manifest, base_dir=base_dir),
                "output_root": _resolve_path(self.output_root, base_dir=base_dir),
                "qwen": self.qwen.model_copy(
                    update={
                        "experiment_config": (
                            _resolve_path(qwen_path, base_dir=base_dir)
                            if qwen_path is not None
                            else None
                        )
                    }
                ),
                "hima": self.hima.model_copy(
                    update={
                        "python_executable": _resolve_path(
                            self.hima.python_executable,
                            base_dir=base_dir,
                        ),
                        "model_paths": paths,
                    }
                ),
            }
        )


class HIMAWorkerRequest(ComparisonConfigModel):
    manifest_path: Path
    model_id: str
    model_path: Path
    device: str
    allow_unlicensed_weights: Literal[True]


@dataclass(frozen=True)
class PolicyComparisonRunArtifacts:
    output_dir: Path
    comparison: PolicyShadowComparison
    reports: PolicyComparisonReportArtifacts
    config_snapshot_path: Path
    corpus_snapshot_path: Path


@dataclass(frozen=True)
class _CandidateResult:
    comparison: PolicyShadowComparison
    execution_backend: str
    error: str | None = None
    revision: str | None = None
    license_acknowledged: bool | None = None


class _FailingPolicySubagent:
    def __init__(self, spec: PolicySubagentSpec, error: str) -> None:
        self.spec = spec
        self._error = error

    async def propose(self, fixture: PolicyObservationFixture) -> NoReturn:
        del fixture
        raise RuntimeError(self._error)


def load_policy_comparison_config(
    path: Path,
    *,
    base_dir: Path | None = None,
) -> PolicyComparisonConfig:
    """Load and resolve one strict Policy Comparison v0.2 YAML file."""

    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise PolicyComparisonError(f"cannot read comparison config: {error}") from error
    if not isinstance(payload, Mapping):
        raise PolicyComparisonError("policy comparison config must contain a YAML mapping")
    config = PolicyComparisonConfig.model_validate(payload)
    return config.resolved(base_dir=(base_dir or path.parent).resolve())


def run_policy_comparison(
    config: PolicyComparisonConfig,
    *,
    output_dir: Path | None = None,
) -> PolicyComparisonRunArtifacts:
    """Evaluate every configured candidate on one immutable corpus.

    HIMA candidates are always evaluated candidate-outer in separate subprocesses.
    Missing or unlicensed weights are reported without importing Transformers or
    attempting network access.
    """

    manifest_path = config.corpus_manifest.expanduser().resolve()
    if not manifest_path.is_file():
        raise PolicyComparisonError(f"corpus manifest does not exist: {manifest_path}")
    target = (
        output_dir.expanduser().resolve()
        if output_dir is not None
        else config.output_root.expanduser().resolve() / _comparison_run_id()
    )
    if target.exists():
        raise PolicyComparisonError(f"comparison output directory already exists: {target}")
    fixtures = load_policy_corpus(manifest_path)
    target.mkdir(parents=True)

    candidates: list[_CandidateResult] = []
    candidates.append(_run_qwen_candidate(config, fixtures))
    for variant, spec in zip(HIMA_VARIANTS, HIMA_PROTOSS_SPECS, strict=True):
        candidates.append(
            _run_hima_candidate(
                config,
                fixtures,
                manifest_path=manifest_path,
                output_dir=target,
                variant=variant,
                spec=spec,
            )
        )
    candidates.append(_run_hiernet_candidate(config, fixtures))

    comparison = _merge_candidate_comparisons(fixtures, candidates)
    reports = write_policy_comparison_reports(comparison, target)
    config_snapshot_path = target / "config.snapshot.yaml"
    corpus_snapshot_path = target / "corpus.snapshot.yaml"
    config_snapshot_path.write_text(
        yaml.safe_dump(
            config.model_dump(mode="json"),
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    corpus_snapshot_path.write_text(
        manifest_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    for result in candidates:
        _write_candidate_artifacts(result, target / "candidates")
    return PolicyComparisonRunArtifacts(
        output_dir=target,
        comparison=comparison,
        reports=reports,
        config_snapshot_path=config_snapshot_path,
        corpus_snapshot_path=corpus_snapshot_path,
    )


def validate_hima_model_path(model_id: str, model_path: Path) -> str:
    """Reject missing, swapped, or unpinned HIMA checkpoints without loading them."""

    if model_id not in HIMA_PINNED_REVISIONS:
        raise PolicyComparisonError(f"unsupported HIMA model ID: {model_id}")
    if not model_path.is_absolute():
        raise PolicyComparisonError("HIMA model path must be absolute")
    resolved = model_path.resolve()
    if not resolved.is_dir():
        raise PolicyComparisonError(f"HIMA model directory does not exist: {resolved}")
    expected_revision = HIMA_PINNED_REVISIONS[model_id]
    expected_repo_token = f"models--{model_id.replace('/', '--')}"
    if (
        resolved.name == expected_revision
        and resolved.parent.name == "snapshots"
        and resolved.parent.parent.name == expected_repo_token
    ):
        return expected_revision
    raise PolicyComparisonError(
        "HIMA checkpoint provenance mismatch: expected exact snapshot "
        f"{expected_repo_token}/snapshots/{expected_revision}, received {resolved}"
    )


def _run_qwen_candidate(
    config: PolicyComparisonConfig,
    fixtures: Sequence[PolicyObservationFixture],
) -> _CandidateResult:
    if not config.qwen.enabled:
        return _static_candidate(
            fixtures,
            QWEN3_8B_SPEC,
            PolicyAvailability(
                status=PolicyAvailabilityStatus.SKIPPED,
                reason="Qwen candidate is disabled in the comparison config",
            ),
            execution_backend="not_started",
        )
    assert config.qwen.experiment_config is not None
    if not config.qwen.experiment_config.is_file():
        raise PolicyComparisonError(
            f"Qwen experiment config does not exist: {config.qwen.experiment_config}"
        )
    experiment = load_config(config.qwen.experiment_config)
    if experiment.provider.kind != "openai_compatible":
        raise PolicyComparisonError(
            "Qwen comparison requires provider.kind=openai_compatible"
        )
    spec = QWEN3_8B_SPEC.model_copy(update={"model_id": experiment.provider.model})
    provider = OpenAICompatibleProvider(
        base_url=experiment.provider.base_url,
        model=experiment.provider.model,
        api_key_env=experiment.provider.api_key_env,
        timeout_seconds=experiment.provider.timeout_seconds,
        max_tokens=experiment.provider.max_tokens,
        enable_thinking=experiment.provider.enable_thinking,
    )

    async def compare() -> PolicyShadowComparison:
        try:
            return await PolicyShadowRunner().compare(
                fixtures,
                [
                    PolicySubagentRegistration(
                        spec=spec,
                        availability=PolicyAvailability(
                            status=PolicyAvailabilityStatus.AVAILABLE
                        ),
                        subagent=LLMPlanningPolicySubagent(provider, spec=spec),
                    )
                ],
            )
        finally:
            await provider.close()

    return _CandidateResult(
        comparison=asyncio.run(compare()),
        execution_backend="openai_compatible",
    )


def _run_hima_candidate(
    config: PolicyComparisonConfig,
    fixtures: Sequence[PolicyObservationFixture],
    *,
    manifest_path: Path,
    output_dir: Path,
    variant: PolicyVariant,
    spec: PolicySubagentSpec,
) -> _CandidateResult:
    if variant not in config.hima.enabled:
        return _static_candidate(
            fixtures,
            spec,
            PolicyAvailability(
                status=PolicyAvailabilityStatus.SKIPPED,
                reason=f"HIMA Protoss-{variant} is disabled in the comparison config",
            ),
            execution_backend="not_started",
            revision=HIMA_PINNED_REVISIONS[spec.model_id],
            license_acknowledged=config.hima.allow_unlicensed_weights,
        )
    model_path = config.hima.model_paths.get(variant)
    if model_path is None:
        return _static_candidate(
            fixtures,
            spec,
            PolicyAvailability(
                status=PolicyAvailabilityStatus.UNAVAILABLE,
                reason="local HIMA model path is not configured; no download attempted",
            ),
            execution_backend="not_started",
            revision=HIMA_PINNED_REVISIONS[spec.model_id],
            license_acknowledged=config.hima.allow_unlicensed_weights,
        )
    if not config.hima.allow_unlicensed_weights:
        return _static_candidate(
            fixtures,
            spec,
            PolicyAvailability(
                status=PolicyAvailabilityStatus.UNAVAILABLE,
                reason=(
                    "HIMA weights have no declared license and "
                    "allow_unlicensed_weights is false; no model loaded"
                ),
            ),
            execution_backend="not_started",
            revision=HIMA_PINNED_REVISIONS[spec.model_id],
            license_acknowledged=False,
        )
    try:
        revision = validate_hima_model_path(spec.model_id, model_path)
    except PolicyComparisonError as error:
        return _static_candidate(
            fixtures,
            spec,
            PolicyAvailability(
                status=PolicyAvailabilityStatus.UNAVAILABLE,
                reason=str(error),
            ),
            execution_backend="provenance_rejected",
            revision=HIMA_PINNED_REVISIONS[spec.model_id],
            license_acknowledged=True,
        )
    python = config.hima.python_executable
    if not python.is_file():
        return _static_candidate(
            fixtures,
            spec,
            PolicyAvailability(
                status=PolicyAvailabilityStatus.UNAVAILABLE,
                reason=f"HIMA Python executable does not exist: {python}",
            ),
            execution_backend="not_started",
            revision=revision,
            license_acknowledged=True,
        )

    with tempfile.TemporaryDirectory(prefix=f".hima-{variant}-", dir=output_dir) as raw_temp:
        temporary_dir = Path(raw_temp)
        request_path = temporary_dir / "request.json"
        response_path = temporary_dir / "response.json"
        request = HIMAWorkerRequest(
            manifest_path=manifest_path,
            model_id=spec.model_id,
            model_path=model_path,
            device=config.hima.device,
            allow_unlicensed_weights=True,
        )
        request_path.write_text(request.model_dump_json(indent=2) + "\n", encoding="utf-8")
        command = [
            str(python),
            "-m",
            "rtscortex.policy.comparison_worker",
            "--request",
            str(request_path),
            "--response",
            str(response_path),
        ]
        environment = _offline_subprocess_environment()
        completed = _execute_process(
            command,
            cwd=_project_root(),
            env=environment,
            timeout=config.hima.timeout_seconds,
        )
        if completed.returncode != 0:
            detail = _subprocess_error(completed)
            return _failed_candidate(
                fixtures,
                spec,
                f"HIMA subprocess exited with code {completed.returncode}: {detail}",
                revision=revision,
            )
        if not response_path.is_file():
            return _failed_candidate(
                fixtures,
                spec,
                "HIMA subprocess succeeded without a response artifact",
                revision=revision,
            )
        try:
            comparison = PolicyShadowComparison.model_validate_json(
                response_path.read_text(encoding="utf-8")
            )
            _validate_candidate_response(comparison, fixtures, spec)
        except Exception as error:
            return _failed_candidate(
                fixtures,
                spec,
                f"invalid HIMA subprocess response: {type(error).__name__}: {error}",
                revision=revision,
            )
    return _CandidateResult(
        comparison=comparison,
        execution_backend="isolated_subprocess",
        revision=revision,
        license_acknowledged=True,
    )


def _run_hiernet_candidate(
    config: PolicyComparisonConfig,
    fixtures: Sequence[PolicyObservationFixture],
) -> _CandidateResult:
    reason = "adapter_not_implemented"
    if not config.hiernet.enabled:
        reason += "; HierNet candidate is disabled"
    return _static_candidate(
        fixtures,
        HIERNET_SC2_SPEC,
        PolicyAvailability(
            status=PolicyAvailabilityStatus.UNAVAILABLE,
            reason=reason,
        ),
        execution_backend="not_started",
    )


def _static_candidate(
    fixtures: Sequence[PolicyObservationFixture],
    spec: PolicySubagentSpec,
    availability: PolicyAvailability,
    *,
    execution_backend: str,
    revision: str | None = None,
    license_acknowledged: bool | None = None,
) -> _CandidateResult:
    comparison = asyncio.run(
        PolicyShadowRunner().compare(
            fixtures,
            [PolicySubagentRegistration(spec=spec, availability=availability)],
        )
    )
    return _CandidateResult(
        comparison=comparison,
        execution_backend=execution_backend,
        revision=revision,
        license_acknowledged=license_acknowledged,
    )


def _failed_candidate(
    fixtures: Sequence[PolicyObservationFixture],
    spec: PolicySubagentSpec,
    error: str,
    *,
    revision: str | None,
) -> _CandidateResult:
    availability = PolicyAvailability(status=PolicyAvailabilityStatus.AVAILABLE)
    subagent = _FailingPolicySubagent(spec, error)
    comparison = asyncio.run(
        PolicyShadowRunner().compare(
            fixtures,
            [
                PolicySubagentRegistration(
                    spec=spec,
                    availability=availability,
                    subagent=subagent,
                )
            ],
        )
    )
    return _CandidateResult(
        comparison=comparison,
        execution_backend="isolated_subprocess_failed",
        error=error,
        revision=revision,
        license_acknowledged=True,
    )


def _merge_candidate_comparisons(
    fixtures: Sequence[PolicyObservationFixture],
    candidates: Sequence[_CandidateResult],
) -> PolicyShadowComparison:
    fixture_ids = [fixture.fixture_id for fixture in fixtures]
    candidate_ids: list[str] = []
    records_by_key = {}
    summaries = []
    for result in candidates:
        comparison = result.comparison
        if comparison.fixture_ids != fixture_ids or comparison.fixtures != list(fixtures):
            raise PolicyComparisonError("candidate comparison did not preserve the corpus")
        if len(comparison.candidate_ids) != 1 or len(comparison.summaries) != 1:
            raise PolicyComparisonError("candidate comparison must contain exactly one policy")
        candidate_id = comparison.candidate_ids[0]
        if candidate_id in candidate_ids:
            raise PolicyComparisonError(f"duplicate candidate response: {candidate_id}")
        candidate_ids.append(candidate_id)
        summaries.append(comparison.summaries[0])
        for record in comparison.records:
            key = (record.fixture_id, candidate_id)
            if key in records_by_key:
                raise PolicyComparisonError(f"duplicate candidate record: {key}")
            records_by_key[key] = record
    expected = {
        (fixture_id, candidate_id)
        for fixture_id in fixture_ids
        for candidate_id in candidate_ids
    }
    if set(records_by_key) != expected:
        raise PolicyComparisonError("candidate comparison response is incomplete")
    records = [
        records_by_key[(fixture_id, candidate_id)]
        for fixture_id in fixture_ids
        for candidate_id in candidate_ids
    ]
    return PolicyShadowComparison(
        fixture_ids=fixture_ids,
        fixtures=list(fixtures),
        candidate_ids=candidate_ids,
        records=records,
        summaries=summaries,
    )


def _write_candidate_artifacts(result: _CandidateResult, root: Path) -> None:
    comparison = result.comparison
    candidate_id = comparison.candidate_ids[0]
    candidate_dir = root / candidate_id
    candidate_dir.mkdir(parents=True, exist_ok=False)
    records = [
        record
        for record in comparison.records
        if record.spec.subagent_id == candidate_id
    ]
    records_path = candidate_dir / "records.jsonl"
    records_path.write_text(
        "".join(record.model_dump_json() + "\n" for record in records),
        encoding="utf-8",
    )
    first = records[0] if records else None
    status_counts = Counter(record.status.value for record in records)
    provenance: dict[str, object] = {
        "format_version": COMPARISON_FORMAT_VERSION,
        "candidate_id": candidate_id,
        "spec": first.spec.model_dump(mode="json") if first is not None else None,
        "model_id": first.spec.model_id if first is not None else None,
        "provider_kind": first.spec.provider_kind.value if first is not None else None,
        "availability": (
            first.availability.model_dump(mode="json") if first is not None else None
        ),
        "execution_backend": result.execution_backend,
        "revision": result.revision,
        "license_acknowledged": result.license_acknowledged,
        "fixture_ids": comparison.fixture_ids,
        "status_counts": dict(sorted(status_counts.items())),
        "error": result.error,
        "shadow_only": True,
        "network_downloads_allowed": False,
    }
    if candidate_id.startswith("hima-protoss-"):
        provenance.update(
            {
                "hima_upstream_revision": HIMA_UPSTREAM_REVISION,
                "adapter_version": HIMA_ADAPTER_VERSION,
                "parser_version": HIMA_PARSER_VERSION,
                "vocabulary_version": HIMA_VOCABULARY_VERSION,
            }
        )
    (candidate_dir / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _validate_candidate_response(
    comparison: PolicyShadowComparison,
    fixtures: Sequence[PolicyObservationFixture],
    spec: PolicySubagentSpec,
) -> None:
    fixture_ids = [fixture.fixture_id for fixture in fixtures]
    if comparison.candidate_ids != [spec.subagent_id]:
        raise PolicyComparisonError("worker returned the wrong candidate ID")
    if comparison.fixture_ids != fixture_ids or comparison.fixtures != list(fixtures):
        raise PolicyComparisonError("worker returned a different corpus or fixture order")
    if len(comparison.records) != len(fixtures):
        raise PolicyComparisonError("worker returned an incomplete candidate result")
    if any(record.spec != spec for record in comparison.records):
        raise PolicyComparisonError("worker returned records for a different model spec")


def _execute_process(
    command: list[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(  # noqa: S603 - executable is an explicit configured local path
            command,
            cwd=cwd,
            env=dict(env),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return subprocess.CompletedProcess(command, 124, stdout="", stderr=str(error))


def _offline_subprocess_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
        }
    )
    source_root = str(_project_root() / "src")
    previous = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        os.pathsep.join((source_root, previous)) if previous else source_root
    )
    return environment


def _subprocess_error(completed: subprocess.CompletedProcess[str]) -> str:
    detail = (completed.stderr or completed.stdout or "no diagnostic output").strip()
    return detail[-4_000:]


def _comparison_run_id() -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"policy-comparison-{stamp}"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_path(path: Path, *, base_dir: Path) -> Path:
    expanded = path.expanduser()
    return expanded.resolve() if expanded.is_absolute() else (base_dir / expanded).resolve()


__all__ = [
    "COMPARISON_FORMAT_VERSION",
    "HIMA_VARIANTS",
    "HIMAComparisonConfig",
    "HIMAWorkerRequest",
    "HierNetComparisonConfig",
    "PolicyComparisonConfig",
    "PolicyComparisonError",
    "PolicyComparisonRunArtifacts",
    "QwenComparisonConfig",
    "load_policy_comparison_config",
    "run_policy_comparison",
    "validate_hima_model_path",
]
