"""Offline selection-policy metrics over minimized fast-executor corpora."""

from __future__ import annotations

import math
import time
from collections import Counter
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from rtscortex.cortex.dataset import (
    ExecutorCorpusManifest,
    ExecutorCorpusSample,
    ExecutorSplit,
    load_executor_corpus,
)


class SavedCandidatePolicy(Protocol):
    """A policy that ranks already-compiled candidates without reconstructing SC2."""

    policy_id: str
    policy_version: str

    def select(self, sample: ExecutorCorpusSample) -> str | None: ...


class DeterministicSavedCandidatePolicy:
    """The persisted-context equivalent of DeterministicCandidateExecutor."""

    policy_id = "deterministic-saved-candidate-policy"
    policy_version = "0.1.0"

    def select(self, sample: ExecutorCorpusSample) -> str | None:
        candidate = min(
            sample.candidates,
            key=lambda item: (
                0 if item.features.advances_goal else 1,
                item.features.action_rank,
                item.features.actor_rank,
                item.features.argument_rank,
                item.features.compile_ordinal,
            ),
            default=None,
        )
        return None if candidate is None else candidate.candidate_id


class ExecutorCorpusBenchmark(BaseModel):
    """CPU ranking metrics; this deliberately does not claim live tick latency."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    benchmark_kind: str = "saved_candidate_ranking"
    reconstructs_full_observation: bool = False
    policy_id: str
    policy_version: str
    corpus_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    split: ExecutorSplit | Literal["all"]
    sample_count: int = Field(ge=0)
    repetitions: int = Field(ge=1)
    measured_calls: int = Field(ge=0)
    agreement_count: int = Field(ge=0)
    disagreement_count: int = Field(ge=0)
    agreement_rate: float = Field(ge=0.0, le=1.0)
    overall_agreement_count: int = Field(ge=0)
    overall_disagreement_count: int = Field(ge=0)
    overall_agreement_rate: float = Field(ge=0.0, le=1.0)
    abstain_match_count: int = Field(ge=0)
    selected_labels: int = Field(ge=0)
    abstained_labels: int = Field(ge=0)
    latency_us_p50: float = Field(ge=0.0)
    latency_us_p95: float = Field(ge=0.0)
    latency_us_p99: float = Field(ge=0.0)
    prediction_distribution: dict[str, int]
    disagreement_by_action: dict[str, int]


def benchmark_executor_corpus(
    manifest_path: Path,
    *,
    repetitions: int = 100,
    policy: SavedCandidatePolicy | None = None,
    split: ExecutorSplit | Literal["all"] = ExecutorSplit.TEST,
) -> ExecutorCorpusBenchmark:
    """Measure a saved-candidate ranker and compare it with recorded labels.

    The corpus intentionally omits raw observations and executable arguments. This
    benchmark therefore measures only candidate-ranking cost and agreement, never
    Bridge, validation, or end-to-end Runtime latency.
    """

    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    manifest = ExecutorCorpusManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    selected_split: ExecutorSplit | None = None if split == "all" else ExecutorSplit(split)
    samples = load_executor_corpus(manifest_path, split=selected_split)
    if not samples:
        raise ValueError(f"executor benchmark split {split!s} contains no samples")
    active_policy = policy or DeterministicSavedCandidatePolicy()
    predictions = [active_policy.select(sample) for sample in samples]
    # Warm the exact policy path without mixing warm-up calls into latency metrics.
    if samples:
        active_policy.select(samples[0])
    latencies: list[float] = []
    for _ in range(repetitions):
        for sample in samples:
            started = time.perf_counter_ns()
            active_policy.select(sample)
            latencies.append((time.perf_counter_ns() - started) / 1_000.0)

    overall_agreement = 0
    selected_agreement = 0
    abstain_matches = 0
    disagreements: Counter[str] = Counter()
    distribution: Counter[str] = Counter()
    selected_labels = 0
    for sample, prediction in zip(samples, predictions, strict=True):
        expected = sample.label.selected_candidate_id
        selected_labels += expected is not None
        distribution["abstain" if prediction is None else "selected"] += 1
        if prediction == expected:
            overall_agreement += 1
            if expected is None:
                abstain_matches += 1
            else:
                selected_agreement += 1
            continue
        expected_action = "abstain"
        if expected is not None:
            expected_action = next(
                candidate.action_name
                for candidate in sample.candidates
                if candidate.candidate_id == expected
            )
        disagreements[expected_action] += 1
    sample_count = len(samples)
    selected_disagreement = selected_labels - selected_agreement
    return ExecutorCorpusBenchmark(
        policy_id=active_policy.policy_id,
        policy_version=active_policy.policy_version,
        corpus_fingerprint=manifest.corpus_fingerprint,
        split=split,
        sample_count=sample_count,
        repetitions=repetitions,
        measured_calls=len(latencies),
        agreement_count=selected_agreement,
        disagreement_count=selected_disagreement,
        agreement_rate=(selected_agreement / selected_labels if selected_labels else 0.0),
        overall_agreement_count=overall_agreement,
        overall_disagreement_count=sample_count - overall_agreement,
        overall_agreement_rate=overall_agreement / sample_count,
        abstain_match_count=abstain_matches,
        selected_labels=selected_labels,
        abstained_labels=sample_count - selected_labels,
        latency_us_p50=_percentile(latencies, 0.50),
        latency_us_p95=_percentile(latencies, 0.95),
        latency_us_p99=_percentile(latencies, 0.99),
        prediction_distribution=dict(sorted(distribution.items())),
        disagreement_by_action=dict(sorted(disagreements.items())),
    )


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight
