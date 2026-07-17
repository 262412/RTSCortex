"""Deterministic, privacy-minimized corpora for the Cortex fast executor."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from rtscortex.contracts import (
    ExecutionReport,
    ExecutionStage,
    ExecutionStatus,
    ObservationEnvelope,
)
from rtscortex.cortex.candidates import observation_fingerprint
from rtscortex.cortex.models import (
    CandidateFeatures,
    CandidateSelection,
    CandidateSelectionStatus,
    CommandLineage,
    CortexIntent,
    CortexRole,
    ExecutableCandidate,
    MacroIntent,
)
from rtscortex.memory import StoredEvent, read_event_log

EXECUTOR_CORPUS_SCHEMA_VERSION: Literal["0.2"] = "0.2"
EXECUTOR_CORPUS_BUILDER_VERSION = "0.2.2"
DEFAULT_EXECUTOR_SPLIT_SEED = "rtscortex-fast-executor-v0.1"
_TERMINAL_STATUSES = frozenset(status.value for status in ExecutionStatus)
_SAFE_CODE = re.compile(r"^[A-Za-z0-9_.:/-]{1,120}$")
_INTENT_ADAPTER: TypeAdapter[CortexIntent] = TypeAdapter(CortexIntent)


class ExecutorCorpusError(ValueError):
    """The source journal or generated executor corpus is invalid."""


class ExecutorSplit(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class _CorpusModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CompactEconomyFeatures(_CorpusModel):
    minerals: int = Field(ge=0)
    vespene: int = Field(ge=0)
    supply_used: int = Field(ge=0)
    supply_cap: int = Field(ge=0)
    workers: int = Field(ge=0)
    army_supply: int = Field(ge=0)


class CompactObservationFeatures(_CorpusModel):
    """Decision features only: no RGB, text prompt, unit tag, or coordinates."""

    game_loop: int = Field(ge=0)
    economy: CompactEconomyFeatures
    own_unit_counts: dict[str, int]
    own_structure_counts: dict[str, int]
    visible_enemy_counts: dict[str, int]
    production_counts: dict[str, int]
    upgrades: list[str]
    available_actions: list[str]
    alert_count: int = Field(ge=0)


class ExecutorCandidateSample(_CorpusModel):
    candidate_id: str = Field(pattern=r"^candidate:[0-9a-f]{64}$")
    action_name: str = Field(min_length=1)
    features: CandidateFeatures


class ExecutorSelectionLabel(_CorpusModel):
    selection_id: str = Field(pattern=r"^selection:[0-9a-f]{64}$")
    status: CandidateSelectionStatus
    selected_candidate_id: str | None = Field(
        default=None,
        pattern=r"^candidate:[0-9a-f]{64}$",
    )
    fallback_reason: str | None = None
    executor_id: str = Field(min_length=1)
    executor_version: str = Field(min_length=1)
    recorded_latency_ms: float = Field(ge=0.0)

    @model_validator(mode="after")
    def validate_selection(self) -> ExecutorSelectionLabel:
        if self.status is CandidateSelectionStatus.SELECTED and self.selected_candidate_id is None:
            raise ValueError("selected labels require selected_candidate_id")
        if self.status is CandidateSelectionStatus.ABSTAINED:
            if self.selected_candidate_id is not None:
                raise ValueError("abstained labels cannot select a candidate")
            if not self.fallback_reason:
                raise ValueError("abstained labels require fallback_reason")
        return self


class ExecutorTerminalOutcome(_CorpusModel):
    status: ExecutionStatus
    execution_stage: ExecutionStage | None = None
    failure_code: str | None = None
    action_name: str | None = None


class ExecutorCorpusSample(_CorpusModel):
    schema_version: Literal["0.2"] = EXECUTOR_CORPUS_SCHEMA_VERSION
    sample_id: str = Field(pattern=r"^executor-sample:[0-9a-f]{64}$")
    split: ExecutorSplit
    run_id: str
    episode_id: str
    step_id: int = Field(ge=0)
    game_loop: int = Field(ge=0)
    observation_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    semantic_feature_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    observation_features: CompactObservationFeatures
    intent_id: str = Field(pattern=r"^intent:[0-9a-f]{64}$")
    intent_action_names: list[str] = Field(min_length=1)
    source_role: CortexRole
    source_id: str = Field(min_length=1)
    source_version: str = Field(min_length=1)
    has_macro_plan: bool = False
    has_situation_assessment: bool = False
    candidates: list[ExecutorCandidateSample]
    label: ExecutorSelectionLabel
    command_id: str | None = Field(default=None, pattern=r"^command:[0-9a-f]{64}$")
    terminal_outcome: ExecutorTerminalOutcome | None = None

    @model_validator(mode="after")
    def validate_candidate_domain(self) -> ExecutorCorpusSample:
        candidate_ids = [candidate.candidate_id for candidate in self.candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("candidate IDs must be unique")
        selected = self.label.selected_candidate_id
        if selected is not None and selected not in set(candidate_ids):
            raise ValueError("selected candidate must belong to the candidate domain")
        if len(self.intent_action_names) != len(set(self.intent_action_names)):
            raise ValueError("intent action names must be unique")
        if any(
            candidate.action_name not in self.intent_action_names for candidate in self.candidates
        ):
            raise ValueError("candidate actions must belong to the intent action domain")
        compile_ordinals = [candidate.features.compile_ordinal for candidate in self.candidates]
        if len(compile_ordinals) != len(set(compile_ordinals)):
            raise ValueError("candidate compile ordinals must be unique")
        return self


class ExecutorCorpusSource(_CorpusModel):
    journal_path: str
    source_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    event_count: int = Field(ge=0)
    selection_event_count: int = Field(ge=0)
    protocol_v11_observation_count: int = Field(ge=0)
    run_ids: list[str]
    episode_keys: list[str]


class ExecutorCorpusArtifact(_CorpusModel):
    file: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sample_count: int = Field(ge=0)


class ExecutorCorpusConservation(_CorpusModel):
    selection_events: int = Field(ge=0)
    included_samples: int = Field(ge=0)
    excluded_selections: int = Field(ge=0)
    selected_labels: int = Field(ge=0)
    abstained_labels: int = Field(ge=0)
    lineages_linked: int = Field(ge=0)
    lineages_missing: int = Field(ge=0)
    terminal_outcomes_linked: int = Field(ge=0)
    terminal_outcomes_missing: int = Field(ge=0)


class ExecutorCorpusDuplicates(_CorpusModel):
    duplicate_selection_events: int = Field(ge=0)
    duplicate_sample_fingerprints: int = Field(ge=0)
    repeated_observation_fingerprints: int = Field(ge=0)
    repeated_semantic_feature_fingerprints: int = Field(ge=0)


class ExecutorCorpusDistributions(_CorpusModel):
    split: dict[str, int]
    selection_status: dict[str, int]
    source_role: dict[str, int]
    selected_action: dict[str, int]
    candidate_action: dict[str, int]
    terminal_status: dict[str, int]


class ExecutorCorpusManifest(_CorpusModel):
    schema_version: Literal["0.2"] = EXECUTOR_CORPUS_SCHEMA_VERSION
    builder_version: str = EXECUTOR_CORPUS_BUILDER_VERSION
    corpus_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    split_seed: str = Field(min_length=1)
    split_policy: str = (
        "sha256(seed|run_id|episode_id): train<80, validation<90, test<100; "
        "semantic duplicates crossing splits are excluded"
    )
    redacted_fields: list[str] = Field(
        default_factory=lambda: [
            "rgb_screen",
            "rgb_minimap",
            "image_uri",
            "text_observation",
            "candidate_arguments",
            "candidate_actor_instances",
            "unit_tags",
            "coordinates",
            "prompts",
            "credentials",
            "raw_failure_reason",
            "runtime_observation_fingerprint",
            "runtime_candidate_id",
            "runtime_selection_id",
            "runtime_intent_id",
            "runtime_command_id",
            "runtime_macro_plan_id",
            "runtime_situation_assessment_id",
            "raw_journal_sha256",
        ]
    )
    sources: list[ExecutorCorpusSource]
    artifacts: dict[ExecutorSplit, ExecutorCorpusArtifact]
    conservation: ExecutorCorpusConservation
    exclusion_reasons: dict[str, int]
    duplicates: ExecutorCorpusDuplicates
    distributions: ExecutorCorpusDistributions

    @model_validator(mode="after")
    def validate_artifact_set(self) -> ExecutorCorpusManifest:
        if set(self.artifacts) != set(ExecutorSplit):
            raise ValueError("manifest must define train, validation, and test artifacts")
        return self


class ExecutorCorpusVerification(_CorpusModel):
    valid: bool
    sample_count: int = Field(ge=0)
    split_counts: dict[str, int]
    errors: list[str]


@dataclass(frozen=True)
class ExecutorCorpusBuildResult:
    manifest_path: Path
    split_paths: dict[ExecutorSplit, Path]
    manifest: ExecutorCorpusManifest
    samples: tuple[ExecutorCorpusSample, ...]


@dataclass(frozen=True)
class _LoadedJournal:
    path: Path
    source_fingerprint: str
    events: tuple[StoredEvent, ...]


@dataclass(frozen=True)
class _EventIndexes:
    observations: dict[tuple[str, str, int], StoredEvent]
    intents: dict[tuple[str, str, str], StoredEvent]
    candidate_sets: dict[tuple[str, str, str], StoredEvent]
    lineages: dict[tuple[str, str, str], tuple[StoredEvent, ...]]
    executions: dict[tuple[str, str, str], StoredEvent]
    terminal_lifecycles: dict[tuple[str, str, str], StoredEvent]


_EpisodeKey: TypeAlias = tuple[str, str]


def build_executor_corpus(
    sources: Sequence[Path],
    output_dir: Path,
    *,
    split_seed: str = DEFAULT_EXECUTOR_SPLIT_SEED,
) -> ExecutorCorpusBuildResult:
    """Export one minimized sample for each valid executor selection event."""

    if not sources:
        raise ExecutorCorpusError("at least one run directory or events.jsonl is required")
    if not split_seed:
        raise ExecutorCorpusError("split_seed cannot be empty")
    journals = tuple(_load_journal(source) for source in sources)
    indexes = _index_events(event for journal in journals for event in journal.events)
    selection_events = [
        event
        for journal in journals
        for event in journal.events
        if event.event_type == "executor_selection"
    ]
    samples, exclusions, duplicate_selections = _build_samples(
        selection_events,
        indexes,
        split_seed=split_seed,
    )
    samples, cross_split_exclusions = _exclude_cross_split_semantic_duplicates(samples)
    exclusions.update(cross_split_exclusions)
    ordered = tuple(sorted(samples, key=_sample_sort_key))
    if not ordered:
        details = ", ".join(f"{reason}={count}" for reason, count in sorted(exclusions.items()))
        suffix = f" ({details})" if details else ""
        raise ExecutorCorpusError(f"no valid executor samples were found{suffix}")
    output_dir.mkdir(parents=True, exist_ok=True)
    split_paths, artifacts, encoded = _write_splits(output_dir, ordered)
    duplicate_sample_fingerprints = len(ordered) - len(
        {_sample_content_fingerprint(sample) for sample in ordered}
    )
    repeated_observation_fingerprints = len(ordered) - len(
        {sample.observation_fingerprint for sample in ordered}
    )
    repeated_semantic_feature_fingerprints = len(ordered) - len(
        {sample.semantic_feature_fingerprint for sample in ordered}
    )
    manifest = ExecutorCorpusManifest(
        corpus_fingerprint=_sha256_bytes(b"".join(encoded[split] for split in ExecutorSplit)),
        split_seed=split_seed,
        sources=[_source_manifest(journal, output_dir) for journal in journals],
        artifacts=artifacts,
        conservation=_conservation(selection_events, ordered),
        exclusion_reasons=dict(sorted(exclusions.items())),
        duplicates=ExecutorCorpusDuplicates(
            duplicate_selection_events=duplicate_selections,
            duplicate_sample_fingerprints=duplicate_sample_fingerprints,
            repeated_observation_fingerprints=repeated_observation_fingerprints,
            repeated_semantic_feature_fingerprints=(repeated_semantic_feature_fingerprints),
        ),
        distributions=_distributions(ordered),
    )
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    verification = verify_executor_corpus(manifest_path)
    if not verification.valid:
        raise ExecutorCorpusError(
            "built executor corpus failed verification: " + "; ".join(verification.errors)
        )
    return ExecutorCorpusBuildResult(
        manifest_path=manifest_path,
        split_paths=split_paths,
        manifest=manifest,
        samples=ordered,
    )


def load_executor_corpus(
    manifest_path: Path,
    *,
    split: ExecutorSplit | None = None,
    verify: bool = True,
) -> list[ExecutorCorpusSample]:
    """Load canonical samples from one or all episode-isolated splits."""

    manifest = _load_manifest(manifest_path)
    if verify:
        result = verify_executor_corpus(manifest_path)
        if not result.valid:
            raise ExecutorCorpusError("invalid executor corpus: " + "; ".join(result.errors))
    splits = (split,) if split is not None else tuple(ExecutorSplit)
    samples: list[ExecutorCorpusSample] = []
    for split_name in splits:
        artifact = manifest.artifacts[split_name]
        samples.extend(_read_samples(_artifact_path(manifest_path, artifact)))
    return samples


def verify_executor_corpus(
    manifest_path: Path,
    *,
    verify_sources: bool = False,
) -> ExecutorCorpusVerification:
    """Verify hashes, schemas, split isolation, conservation, and provenance."""

    errors: list[str] = []
    try:
        manifest = _load_manifest(manifest_path)
    except Exception as error:
        return ExecutorCorpusVerification(
            valid=False,
            sample_count=0,
            split_counts={},
            errors=[f"manifest cannot be decoded: {type(error).__name__}: {error}"],
        )

    samples: list[ExecutorCorpusSample] = []
    encoded_by_split: dict[ExecutorSplit, bytes] = {}
    split_counts: Counter[str] = Counter()
    episode_splits: dict[_EpisodeKey, set[ExecutorSplit]] = defaultdict(set)
    semantic_splits: dict[str, set[ExecutorSplit]] = defaultdict(set)
    for split_name in ExecutorSplit:
        artifact = manifest.artifacts.get(split_name)
        if artifact is None:
            errors.append(f"manifest is missing {split_name.value} artifact")
            continue
        try:
            path = _artifact_path(manifest_path, artifact)
        except ExecutorCorpusError as error:
            errors.append(str(error))
            continue
        if not path.is_file():
            errors.append(f"artifact is missing: {path}")
            continue
        encoded = path.read_bytes()
        encoded_by_split[split_name] = encoded
        if _sha256_bytes(encoded) != artifact.sha256:
            errors.append(f"{split_name.value} SHA256 does not match manifest")
        try:
            split_samples = _read_samples(path)
        except Exception as error:
            errors.append(f"{split_name.value} cannot be decoded: {type(error).__name__}: {error}")
            continue
        if len(split_samples) != artifact.sample_count:
            errors.append(f"{split_name.value} sample count does not match manifest")
        canonical = b"".join(
            (_canonical_json(sample.model_dump(mode="json")) + "\n").encode("utf-8")
            for sample in split_samples
        )
        if encoded != canonical:
            errors.append(f"{split_name.value} is not canonically encoded")
        if split_samples != sorted(split_samples, key=_sample_sort_key):
            errors.append(f"{split_name.value} samples are not in canonical order")
        for sample in split_samples:
            if sample.split is not split_name:
                errors.append(f"sample {sample.sample_id} is stored in the wrong split")
            if sample.semantic_feature_fingerprint != _sha256_json(
                sample.observation_features.model_dump(mode="json")
            ):
                errors.append(f"sample {sample.sample_id} has an invalid semantic hash")
            if sample.sample_id != _expected_sample_id(sample):
                errors.append(f"sample {sample.sample_id} has an invalid sample ID")
            episode_splits[(sample.run_id, sample.episode_id)].add(sample.split)
            semantic_splits[sample.semantic_feature_fingerprint].add(sample.split)
        samples.extend(split_samples)
        split_counts[split_name.value] += len(split_samples)

    for episode_key, assigned in episode_splits.items():
        if len(assigned) != 1:
            errors.append(f"episode {episode_key!r} spans multiple splits")
    for fingerprint, assigned in semantic_splits.items():
        if len(assigned) != 1:
            errors.append(f"semantic feature fingerprint {fingerprint} spans multiple splits")
    for sample in samples:
        expected_split = executor_episode_split(
            sample.run_id,
            sample.episode_id,
            seed=manifest.split_seed,
        )
        if sample.split is not expected_split:
            errors.append(f"sample {sample.sample_id} violates the split policy")
    _verify_sample_uniqueness(samples, errors)
    duplicates = _duplicate_metrics(samples, manifest)
    if duplicates != manifest.duplicates:
        errors.append("duplicate metrics do not match corpus samples")
    if sum(manifest.exclusion_reasons.values()) != (manifest.conservation.excluded_selections):
        errors.append("exclusion reasons do not conserve excluded selections")
    if len(samples) + manifest.conservation.excluded_selections != (
        manifest.conservation.selection_events
    ):
        errors.append("selection conservation does not hold")
    if len(samples) != manifest.conservation.included_samples:
        errors.append("included sample count does not match conservation")
    sample_conservation = _conservation_from_samples(samples).model_copy(
        update={
            "selection_events": manifest.conservation.selection_events,
            "excluded_selections": manifest.conservation.excluded_selections,
        }
    )
    if sample_conservation != manifest.conservation:
        errors.append("label or lineage conservation does not match samples")
    try:
        distributions = _distributions(samples)
    except KeyError:
        errors.append("sample distributions cannot be computed from invalid labels")
    else:
        if distributions != manifest.distributions:
            errors.append("sample distributions do not match manifest")
    corpus_bytes = b"".join(encoded_by_split.get(split, b"") for split in ExecutorSplit)
    if _sha256_bytes(corpus_bytes) != manifest.corpus_fingerprint:
        errors.append("corpus fingerprint does not match split artifacts")
    if verify_sources:
        _verify_sources(manifest_path, manifest, errors)
    return ExecutorCorpusVerification(
        valid=not errors,
        sample_count=len(samples),
        split_counts=dict(sorted(split_counts.items())),
        errors=errors,
    )


def executor_episode_split(
    run_id: str,
    episode_id: str,
    *,
    seed: str = DEFAULT_EXECUTOR_SPLIT_SEED,
) -> ExecutorSplit:
    """Assign a complete episode using a stable content hash."""

    digest = hashlib.sha256(f"{seed}|{run_id}|{episode_id}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    if bucket < 80:
        return ExecutorSplit.TRAIN
    if bucket < 90:
        return ExecutorSplit.VALIDATION
    return ExecutorSplit.TEST


def _exclude_cross_split_semantic_duplicates(
    samples: Sequence[ExecutorCorpusSample],
) -> tuple[list[ExecutorCorpusSample], Counter[str]]:
    """Keep episode splits stable while removing cross-split feature leakage."""

    grouped: dict[str, list[ExecutorCorpusSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.semantic_feature_fingerprint].append(sample)
    retained: list[ExecutorCorpusSample] = []
    excluded = Counter[str]()
    for group in grouped.values():
        owner = min(group, key=lambda sample: (sample.run_id, sample.episode_id))
        for sample in group:
            if sample.split is owner.split:
                retained.append(sample)
            else:
                excluded["cross_split_semantic_duplicate"] += 1
    return retained, excluded


def _load_journal(source: Path) -> _LoadedJournal:
    expanded = source.expanduser().resolve()
    path = expanded / "events.jsonl" if expanded.is_dir() else expanded
    if not path.is_file():
        raise ExecutorCorpusError(f"event journal does not exist: {path}")
    try:
        events = tuple(read_event_log(path))
    except Exception as error:
        raise ExecutorCorpusError(
            f"event journal cannot be decoded: {path}: {type(error).__name__}: {error}"
        ) from error
    return _LoadedJournal(
        path=path,
        source_fingerprint=_safe_source_fingerprint(events),
        events=events,
    )


def _index_events(events: Iterable[StoredEvent]) -> _EventIndexes:
    observations: dict[tuple[str, str, int], StoredEvent] = {}
    intents: dict[tuple[str, str, str], StoredEvent] = {}
    candidates: dict[tuple[str, str, str], StoredEvent] = {}
    lineages: dict[tuple[str, str, str], list[StoredEvent]] = defaultdict(list)
    executions: dict[tuple[str, str, str], StoredEvent] = {}
    lifecycles: dict[tuple[str, str, str], StoredEvent] = {}
    for event in events:
        episode_step = (event.run_id, event.episode_id, event.step_id)
        if event.event_type == "observation":
            _insert_unique_event(observations, episode_step, event, "observation")
        elif event.event_type in {"intent_emitted", "candidate_set_built"}:
            intent_id = _text(event.payload, "intent_id")
            if intent_id is not None:
                target = intents if event.event_type == "intent_emitted" else candidates
                _insert_unique_event(
                    target,
                    (event.run_id, event.episode_id, intent_id),
                    event,
                    event.event_type,
                )
        elif event.event_type == "command_lineage":
            payload = _object(event.payload.get("lineage", event.payload))
            selection_id = _text(payload, "selection_id")
            if selection_id is not None:
                lineages[(event.run_id, event.episode_id, selection_id)].append(event)
        elif event.event_type == "execution":
            command_id = _text(event.payload, "command_id")
            if command_id is not None:
                _insert_unique_event(
                    executions,
                    (event.run_id, event.episode_id, command_id),
                    event,
                    "execution",
                )
        elif event.event_type == "command_lifecycle":
            command_id = _command_id(event.payload)
            status = _text(event.payload, "status")
            if command_id is not None and status in _TERMINAL_STATUSES:
                _insert_unique_event(
                    lifecycles,
                    (event.run_id, event.episode_id, command_id),
                    event,
                    "terminal command_lifecycle",
                )
    return _EventIndexes(
        observations=observations,
        intents=intents,
        candidate_sets=candidates,
        lineages={key: tuple(value) for key, value in lineages.items()},
        executions=executions,
        terminal_lifecycles=lifecycles,
    )


def _insert_unique_event(
    target: dict[tuple[Any, ...], StoredEvent],
    key: tuple[Any, ...],
    event: StoredEvent,
    label: str,
) -> None:
    previous = target.get(key)
    if previous is not None:
        raise ExecutorCorpusError(
            f"duplicate {label} identity {key!r} in event IDs "
            f"{previous.event_id} and {event.event_id}"
        )
    target[key] = event


def _build_samples(
    selection_events: Sequence[StoredEvent],
    indexes: _EventIndexes,
    *,
    split_seed: str,
) -> tuple[list[ExecutorCorpusSample], Counter[str], int]:
    samples: list[ExecutorCorpusSample] = []
    exclusions: Counter[str] = Counter()
    seen_selection_ids: set[str] = set()
    seen_safe_selection_ids: set[str] = set()
    duplicate_selections = 0
    for event in selection_events:
        selection_id = _text(event.payload, "selection_id")
        if selection_id is not None and selection_id in seen_selection_ids:
            exclusions["duplicate_selection_id"] += 1
            duplicate_selections += 1
            continue
        if selection_id is not None:
            seen_selection_ids.add(selection_id)
        try:
            sample = _sample_from_selection(event, indexes, split_seed=split_seed)
        except (ExecutorCorpusError, ValueError, TypeError, KeyError) as error:
            exclusions[_exclusion_code(error)] += 1
            continue
        if sample.label.selection_id in seen_safe_selection_ids:
            exclusions["duplicate_selection_id"] += 1
            duplicate_selections += 1
            continue
        seen_safe_selection_ids.add(sample.label.selection_id)
        samples.append(sample)
    return samples, exclusions, duplicate_selections


def _sample_from_selection(
    event: StoredEvent,
    indexes: _EventIndexes,
    *,
    split_seed: str,
) -> ExecutorCorpusSample:
    selection = _parse_selection(event.payload)
    key = (event.run_id, event.episode_id, selection.intent_id)
    intent_event = indexes.intents.get(key)
    candidate_event = indexes.candidate_sets.get(key)
    observation_event = indexes.observations.get((event.run_id, event.episode_id, event.step_id))
    if intent_event is None:
        raise ExecutorCorpusError("missing_intent")
    if candidate_event is None:
        raise ExecutorCorpusError("missing_candidate_set")
    if observation_event is None:
        raise ExecutorCorpusError("missing_observation")
    observation = ObservationEnvelope.model_validate(observation_event.payload)
    if observation.protocol_version != "1.1":
        raise ExecutorCorpusError("unsupported_protocol")
    intent_payload = _object(intent_event.payload.get("intent", intent_event.payload))
    intent = _INTENT_ADAPTER.validate_python(intent_payload)
    if (
        (observation.run_id, observation.episode_id, observation.step_id)
        != (event.run_id, event.episode_id, event.step_id)
        or intent.intent_id != selection.intent_id
        or intent_event.step_id != event.step_id
        or candidate_event.step_id != event.step_id
        or (intent.run_id, intent.episode_id, intent.step_id)
        != (event.run_id, event.episode_id, event.step_id)
        or intent.created_game_loop != observation.game_loop
    ):
        raise ExecutorCorpusError("pipeline_identity_mismatch")
    candidates = [
        ExecutableCandidate.model_validate(candidate)
        for candidate in _list(candidate_event.payload.get("candidates"))
    ]
    if len({candidate.candidate_id for candidate in candidates}) != len(candidates):
        raise ExecutorCorpusError("duplicate_candidate_id")
    if any(candidate.intent_id != selection.intent_id for candidate in candidates):
        raise ExecutorCorpusError("candidate_intent_mismatch")
    expected_fingerprint = observation_fingerprint(observation)
    if any(candidate.observation_fingerprint != expected_fingerprint for candidate in candidates):
        raise ExecutorCorpusError("observation_fingerprint_mismatch")
    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    if (
        selection.status is CandidateSelectionStatus.SELECTED
        and selection.candidate_id not in candidate_by_id
    ):
        raise ExecutorCorpusError("selection_outside_candidate_set")
    # All joins above use the original runtime identifiers. Nothing below this
    # boundary persists them: corpus-local identifiers are derived exclusively
    # from the compact safe projection and semantic ranks.
    selected_candidate = (
        None if selection.candidate_id is None else candidate_by_id[selection.candidate_id]
    )
    lineage, terminal = _linked_outcome(
        event,
        selection,
        selected_candidate,
        indexes,
    )
    compact_features = _compact_observation(observation)
    semantic_feature_fingerprint = _sha256_json(compact_features.model_dump(mode="json"))
    safe_observation_fingerprint = _local_id_digest(
        {
            "kind": "observation",
            "run_id": event.run_id,
            "episode_id": event.episode_id,
            "step_id": event.step_id,
            "game_loop": observation.game_loop,
            "semantic_feature_fingerprint": semantic_feature_fingerprint,
        }
    )
    safe_intent_id = _local_id(
        "intent",
        {
            "observation_fingerprint": safe_observation_fingerprint,
            "source_role": intent.source_role.value,
            "source_id": intent.source_id,
            "source_version": intent.source_version,
            "action_names": intent.action_names,
            "priority": intent.priority,
            "ttl_game_loops": intent.ttl_game_loops,
            "target_kind": intent.target.kind.value,
        },
    )
    safe_candidate_by_raw_id: dict[str, str] = {}
    candidate_samples: list[ExecutorCandidateSample] = []
    for local_ordinal, candidate in enumerate(candidates):
        safe_candidate_id = _local_id(
            "candidate",
            {
                "intent_id": safe_intent_id,
                "local_ordinal": local_ordinal,
                "action_name": candidate.action_name,
                "features": candidate.features.model_dump(mode="json"),
            },
        )
        safe_candidate_by_raw_id[candidate.candidate_id] = safe_candidate_id
        candidate_samples.append(
            ExecutorCandidateSample(
                candidate_id=safe_candidate_id,
                action_name=candidate.action_name,
                features=candidate.features,
            )
        )
    safe_selected_candidate_id = (
        None if selection.candidate_id is None else safe_candidate_by_raw_id[selection.candidate_id]
    )
    safe_selection_id = _local_id(
        "selection",
        {
            "intent_id": safe_intent_id,
            "status": selection.status.value,
            "selected_candidate_id": safe_selected_candidate_id,
            "executor_id": selection.executor_id,
            "executor_version": selection.executor_version,
        },
    )
    label = ExecutorSelectionLabel(
        selection_id=safe_selection_id,
        status=selection.status,
        selected_candidate_id=safe_selected_candidate_id,
        fallback_reason=_safe_code(selection.fallback_reason),
        executor_id=selection.executor_id,
        executor_version=selection.executor_version,
        recorded_latency_ms=selection.latency_ms,
    )
    split = executor_episode_split(event.run_id, event.episode_id, seed=split_seed)
    safe_command_id = (
        None
        if lineage is None
        else _local_id(
            "command",
            {
                "selection_id": safe_selection_id,
                "selected_candidate_id": safe_selected_candidate_id,
                "action_name": (None if terminal is None else terminal.action_name),
            },
        )
    )
    sample = ExecutorCorpusSample(
        sample_id="executor-sample:" + "0" * 64,
        split=split,
        run_id=event.run_id,
        episode_id=event.episode_id,
        step_id=event.step_id,
        game_loop=observation.game_loop,
        observation_fingerprint=safe_observation_fingerprint,
        semantic_feature_fingerprint=semantic_feature_fingerprint,
        observation_features=compact_features,
        intent_id=safe_intent_id,
        intent_action_names=intent.action_names,
        source_role=intent.source_role,
        source_id=intent.source_id,
        source_version=intent.source_version,
        has_macro_plan=isinstance(intent, MacroIntent),
        has_situation_assessment=intent.situation_assessment_id is not None,
        candidates=candidate_samples,
        label=label,
        command_id=safe_command_id,
        terminal_outcome=terminal,
    )
    return sample.model_copy(update={"sample_id": _expected_sample_id(sample)})


def _parse_selection(payload: dict[str, Any]) -> CandidateSelection:
    candidate_id = payload.get("candidate_id", payload.get("selected_candidate_id"))
    return CandidateSelection.model_validate(
        {
            "selection_id": payload.get("selection_id"),
            "intent_id": payload.get("intent_id"),
            "status": payload.get("status"),
            "candidate_id": candidate_id,
            "executor_id": payload.get("executor_id"),
            "executor_version": payload.get("executor_version"),
            "confidence": payload.get("confidence"),
            "latency_ms": payload.get("latency_ms"),
            "fallback_reason": payload.get("fallback_reason"),
        }
    )


def _linked_outcome(
    event: StoredEvent,
    selection: CandidateSelection,
    selected_candidate: ExecutableCandidate | None,
    indexes: _EventIndexes,
) -> tuple[CommandLineage | None, ExecutorTerminalOutcome | None]:
    lineage_events = indexes.lineages.get(
        (event.run_id, event.episode_id, selection.selection_id),
        (),
    )
    if len(lineage_events) > 1:
        raise ExecutorCorpusError("multiple_command_lineages")
    if not lineage_events:
        return None, None
    lineage_payload = _object(lineage_events[0].payload.get("lineage", lineage_events[0].payload))
    lineage = CommandLineage.model_validate(lineage_payload)
    if lineage.intent_id != selection.intent_id or lineage.candidate_id != selection.candidate_id:
        raise ExecutorCorpusError("command_lineage_mismatch")
    command_key = (event.run_id, event.episode_id, lineage.command_id)
    execution_event = indexes.executions.get(command_key)
    if execution_event is not None:
        report = ExecutionReport.model_validate(execution_event.payload)
        if (
            (report.run_id, report.episode_id, report.command_id)
            != (event.run_id, event.episode_id, lineage.command_id)
            or report.step_id != execution_event.step_id
            or execution_event.run_id != event.run_id
            or execution_event.episode_id != event.episode_id
            or (
                selected_candidate is not None
                and (
                    report.action_name != selected_candidate.action_name
                    or report.actor != selected_candidate.actor
                )
            )
        ):
            raise ExecutorCorpusError("execution_identity_mismatch")
        return lineage, ExecutorTerminalOutcome(
            status=report.status,
            execution_stage=report.execution_stage,
            failure_code=_safe_code(report.failure_code),
            action_name=report.action_name,
        )
    lifecycle = indexes.terminal_lifecycles.get(command_key)
    if lifecycle is None:
        return lineage, None
    status_value = _text(lifecycle.payload, "status")
    if status_value is None:
        raise ExecutorCorpusError("terminal_lifecycle_missing_status")
    status = ExecutionStatus(status_value)
    command = _object(lifecycle.payload.get("command"))
    if (
        lifecycle.run_id != event.run_id
        or lifecycle.episode_id != event.episode_id
        or lifecycle.step_id < event.step_id
        or _text(command, "command_id") != lineage.command_id
        or (
            selected_candidate is not None
            and (
                _text(command, "name") != selected_candidate.action_name
                or _text(command, "actor") != selected_candidate.actor
            )
        )
    ):
        raise ExecutorCorpusError("terminal_lifecycle_identity_mismatch")
    return lineage, ExecutorTerminalOutcome(
        status=status,
        failure_code=(
            None if status is ExecutionStatus.SUCCEEDED else "lifecycle_terminal_without_execution"
        ),
        action_name=_text(command, "name"),
    )


def _compact_observation(observation: ObservationEnvelope) -> CompactObservationFeatures:
    economy = observation.state.economy
    return CompactObservationFeatures(
        game_loop=observation.game_loop,
        economy=CompactEconomyFeatures(**economy.model_dump()),
        own_unit_counts=_type_counts(unit.unit_type for unit in observation.state.own_units),
        own_structure_counts=_type_counts(
            unit.unit_type for unit in observation.state.own_structures
        ),
        visible_enemy_counts=_type_counts(
            unit.unit_type for unit in observation.state.visible_enemies
        ),
        production_counts=_type_counts(item.name for item in observation.state.production_queue),
        upgrades=sorted(set(observation.state.upgrades)),
        available_actions=sorted({action.name for action in observation.available_actions}),
        alert_count=len(observation.alerts),
    )


def _write_splits(
    output_dir: Path,
    samples: Sequence[ExecutorCorpusSample],
) -> tuple[
    dict[ExecutorSplit, Path],
    dict[ExecutorSplit, ExecutorCorpusArtifact],
    dict[ExecutorSplit, bytes],
]:
    paths: dict[ExecutorSplit, Path] = {}
    artifacts: dict[ExecutorSplit, ExecutorCorpusArtifact] = {}
    encoded: dict[ExecutorSplit, bytes] = {}
    for split in ExecutorSplit:
        split_samples = [sample for sample in samples if sample.split is split]
        data = b"".join(
            (_canonical_json(sample.model_dump(mode="json")) + "\n").encode("utf-8")
            for sample in split_samples
        )
        path = output_dir / f"{split.value}.jsonl"
        path.write_bytes(data)
        paths[split] = path
        encoded[split] = data
        artifacts[split] = ExecutorCorpusArtifact(
            file=path.name,
            sha256=_sha256_bytes(data),
            sample_count=len(split_samples),
        )
    return paths, artifacts, encoded


def _source_manifest(
    journal: _LoadedJournal,
    manifest_dir: Path,
) -> ExecutorCorpusSource:
    events = journal.events
    return ExecutorCorpusSource(
        journal_path=os.path.relpath(journal.path, start=manifest_dir.resolve()),
        source_fingerprint=journal.source_fingerprint,
        event_count=len(events),
        selection_event_count=sum(event.event_type == "executor_selection" for event in events),
        protocol_v11_observation_count=sum(
            event.event_type == "observation" and event.payload.get("protocol_version") == "1.1"
            for event in events
        ),
        run_ids=sorted({event.run_id for event in events}),
        episode_keys=sorted({f"{event.run_id}/{event.episode_id}" for event in events}),
    )


def _conservation(
    selection_events: Sequence[StoredEvent],
    samples: Sequence[ExecutorCorpusSample],
) -> ExecutorCorpusConservation:
    base = _conservation_from_samples(samples)
    return base.model_copy(
        update={
            "selection_events": len(selection_events),
            "excluded_selections": len(selection_events) - len(samples),
        }
    )


def _conservation_from_samples(
    samples: Sequence[ExecutorCorpusSample],
) -> ExecutorCorpusConservation:
    selected = sum(sample.label.status is CandidateSelectionStatus.SELECTED for sample in samples)
    lineages = sum(sample.command_id is not None for sample in samples)
    terminal = sum(sample.terminal_outcome is not None for sample in samples)
    return ExecutorCorpusConservation(
        selection_events=len(samples),
        included_samples=len(samples),
        excluded_selections=0,
        selected_labels=selected,
        abstained_labels=len(samples) - selected,
        lineages_linked=lineages,
        lineages_missing=len(samples) - lineages,
        terminal_outcomes_linked=terminal,
        terminal_outcomes_missing=len(samples) - terminal,
    )


def _distributions(samples: Sequence[ExecutorCorpusSample]) -> ExecutorCorpusDistributions:
    split: Counter[str] = Counter()
    status: Counter[str] = Counter()
    roles: Counter[str] = Counter()
    selected_actions: Counter[str] = Counter()
    candidate_actions: Counter[str] = Counter()
    terminal_status: Counter[str] = Counter()
    for sample in samples:
        split[sample.split.value] += 1
        status[sample.label.status.value] += 1
        roles[sample.source_role.value] += 1
        by_id = {candidate.candidate_id: candidate for candidate in sample.candidates}
        if sample.label.selected_candidate_id is not None:
            selected_actions[by_id[sample.label.selected_candidate_id].action_name] += 1
        for candidate in sample.candidates:
            candidate_actions[candidate.action_name] += 1
        if sample.terminal_outcome is not None:
            terminal_status[sample.terminal_outcome.status.value] += 1
    return ExecutorCorpusDistributions(
        split=dict(sorted(split.items())),
        selection_status=dict(sorted(status.items())),
        source_role=dict(sorted(roles.items())),
        selected_action=dict(sorted(selected_actions.items())),
        candidate_action=dict(sorted(candidate_actions.items())),
        terminal_status=dict(sorted(terminal_status.items())),
    )


def _verify_sample_uniqueness(
    samples: Sequence[ExecutorCorpusSample],
    errors: list[str],
) -> None:
    sample_ids = [sample.sample_id for sample in samples]
    selection_ids = [sample.label.selection_id for sample in samples]
    if len(sample_ids) != len(set(sample_ids)):
        errors.append("duplicate sample IDs detected")
    if len(selection_ids) != len(set(selection_ids)):
        errors.append("duplicate selection IDs detected")
    for sample in samples:
        candidate_ids = {candidate.candidate_id for candidate in sample.candidates}
        if (
            sample.label.selected_candidate_id is not None
            and sample.label.selected_candidate_id not in candidate_ids
        ):
            errors.append(f"sample {sample.sample_id} selects outside its candidate set")


def _duplicate_metrics(
    samples: Sequence[ExecutorCorpusSample],
    manifest: ExecutorCorpusManifest,
) -> ExecutorCorpusDuplicates:
    return ExecutorCorpusDuplicates(
        duplicate_selection_events=manifest.exclusion_reasons.get(
            "duplicate_selection_id",
            0,
        ),
        duplicate_sample_fingerprints=len(samples)
        - len({_sample_content_fingerprint(sample) for sample in samples}),
        repeated_observation_fingerprints=len(samples)
        - len({sample.observation_fingerprint for sample in samples}),
        repeated_semantic_feature_fingerprints=len(samples)
        - len({sample.semantic_feature_fingerprint for sample in samples}),
    )


def _verify_sources(
    manifest_path: Path,
    manifest: ExecutorCorpusManifest,
    errors: list[str],
) -> None:
    for source in manifest.sources:
        locator = Path(source.journal_path)
        path = locator if locator.is_absolute() else (manifest_path.parent / locator).resolve()
        if not path.is_file():
            errors.append(f"source journal is missing: {path}")
        else:
            try:
                events = tuple(read_event_log(path))
            except Exception as error:
                errors.append(
                    f"source journal cannot be decoded: {path}: {type(error).__name__}: {error}"
                )
                continue
            if _safe_source_fingerprint(events) != source.source_fingerprint:
                errors.append(f"source journal safe structure changed: {path}")


def _load_manifest(path: Path) -> ExecutorCorpusManifest:
    if not path.is_file():
        raise ExecutorCorpusError(f"manifest does not exist: {path}")
    return ExecutorCorpusManifest.model_validate_json(path.read_text(encoding="utf-8"))


def _artifact_path(
    manifest_path: Path,
    artifact: ExecutorCorpusArtifact,
) -> Path:
    candidate = (manifest_path.parent / artifact.file).resolve()
    if candidate.parent != manifest_path.parent.resolve():
        raise ExecutorCorpusError("artifact paths must stay beside the manifest")
    return candidate


def _read_samples(path: Path) -> list[ExecutorCorpusSample]:
    samples: list[ExecutorCorpusSample] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                samples.append(ExecutorCorpusSample.model_validate_json(line))
            except Exception as error:
                raise ExecutorCorpusError(
                    f"invalid sample at {path}:{line_number}: {error}"
                ) from error
    return samples


def _sample_sort_key(sample: ExecutorCorpusSample) -> tuple[str, str, str, int, str]:
    return (
        sample.split.value,
        sample.run_id,
        sample.episode_id,
        sample.step_id,
        sample.label.selection_id,
    )


def _sample_content_fingerprint(sample: ExecutorCorpusSample) -> str:
    payload = sample.model_dump(mode="json", exclude={"sample_id", "split"})
    return _sha256_json(payload)


def _expected_sample_id(sample: ExecutorCorpusSample) -> str:
    selected_action: str | None = None
    if sample.label.selected_candidate_id is not None:
        selected_action = next(
            (
                candidate.action_name
                for candidate in sample.candidates
                if candidate.candidate_id == sample.label.selected_candidate_id
            ),
            None,
        )
    identity = {
        "run_id": sample.run_id,
        "episode_id": sample.episode_id,
        "step_id": sample.step_id,
        "observation_fingerprint": sample.observation_fingerprint,
        "intent_id": sample.intent_id,
        "intent_action_names": sample.intent_action_names,
        "selection_id": sample.label.selection_id,
        "candidate_ids": [candidate.candidate_id for candidate in sample.candidates],
        "selected_candidate_id": sample.label.selected_candidate_id,
        "selected_action": selected_action,
    }
    return f"executor-sample:{_sha256_json(identity)}"


def _type_counts(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _command_id(payload: dict[str, Any]) -> str | None:
    direct = _text(payload, "command_id")
    if direct is not None:
        return direct
    return _text(_object(payload.get("command")), "command_id")


def _exclusion_code(error: Exception) -> str:
    if isinstance(error, ExecutorCorpusError) and str(error):
        return str(error)
    return f"invalid_{type(error).__name__.lower()}"


def _safe_code(value: str | None) -> str | None:
    if value is None or _SAFE_CODE.fullmatch(value):
        return value
    return "unstructured_code_redacted"


def _text(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    return value if isinstance(value, str) and value else None


def _object(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    if not isinstance(value, list):
        raise ExecutorCorpusError("invalid_candidate_set")
    return value


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _sha256_json(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _local_id(prefix: str, payload: object) -> str:
    return f"{prefix}:{_local_id_digest(payload)}"


def _local_id_digest(payload: object) -> str:
    return _sha256_json(
        {
            "corpus_schema_version": EXECUTOR_CORPUS_SCHEMA_VERSION,
            "payload": payload,
        }
    )


def _safe_source_fingerprint(events: Sequence[StoredEvent]) -> str:
    """Hash an allow-listed payload projection without raw IDs or executable values."""

    candidate_ordinals: dict[tuple[str, str, str], dict[str, int]] = {}
    for event in events:
        if event.event_type != "candidate_set_built":
            continue
        intent_id = _text(event.payload, "intent_id")
        raw_candidates = event.payload.get("candidates")
        if intent_id is None or not isinstance(raw_candidates, list):
            continue
        candidate_ordinals[(event.run_id, event.episode_id, intent_id)] = {
            candidate_id: ordinal
            for ordinal, raw_candidate in enumerate(raw_candidates)
            if isinstance(raw_candidate, dict)
            and (candidate_id := _text(raw_candidate, "candidate_id")) is not None
        }

    return _sha256_json(
        [
            {
                "ordinal": ordinal,
                "event_type": event.event_type,
                "run_id": event.run_id,
                "episode_id": event.episode_id,
                "step_id": event.step_id,
                "safe_payload": _safe_event_projection(event, candidate_ordinals),
            }
            for ordinal, event in enumerate(events)
        ]
    )


def _safe_event_projection(
    event: StoredEvent,
    candidate_ordinals: dict[tuple[str, str, str], dict[str, int]],
) -> object:
    try:
        if event.event_type == "observation":
            observation = ObservationEnvelope.model_validate(event.payload)
            return {
                "protocol_version": observation.protocol_version,
                "features": _compact_observation(observation).model_dump(mode="json"),
            }
        if event.event_type == "intent_emitted":
            payload = _object(event.payload.get("intent", event.payload))
            intent = _INTENT_ADAPTER.validate_python(payload)
            return {
                "source_role": intent.source_role.value,
                "source_id": intent.source_id,
                "source_version": intent.source_version,
                "action_names": intent.action_names,
                "priority": intent.priority,
                "ttl_game_loops": intent.ttl_game_loops,
                "target_kind": intent.target.kind.value,
                "has_macro_plan": isinstance(intent, MacroIntent),
                "has_situation_assessment": intent.situation_assessment_id is not None,
            }
        if event.event_type == "candidate_set_built":
            raw_candidates = _list(event.payload.get("candidates"))
            return [
                {
                    "local_ordinal": ordinal,
                    "action_name": candidate.action_name,
                    "features": candidate.features.model_dump(mode="json"),
                }
                for ordinal, raw_candidate in enumerate(raw_candidates)
                for candidate in [ExecutableCandidate.model_validate(raw_candidate)]
            ]
        if event.event_type == "executor_selection":
            selection = _parse_selection(event.payload)
            ordinal_map = candidate_ordinals.get(
                (event.run_id, event.episode_id, selection.intent_id),
                {},
            )
            return {
                "status": selection.status.value,
                "selected_candidate_ordinal": (
                    None
                    if selection.candidate_id is None
                    else ordinal_map.get(selection.candidate_id, "outside_domain")
                ),
                "executor_id": selection.executor_id,
                "executor_version": selection.executor_version,
                "fallback_reason": _safe_code(selection.fallback_reason),
                "latency_ms": selection.latency_ms,
            }
        if event.event_type == "command_lineage":
            payload = _object(event.payload.get("lineage", event.payload))
            lineage = CommandLineage.model_validate(payload)
            ordinal_map = candidate_ordinals.get(
                (event.run_id, event.episode_id, lineage.intent_id),
                {},
            )
            return {
                "selected_candidate_ordinal": ordinal_map.get(
                    lineage.candidate_id,
                    "outside_domain",
                ),
                "source_role": lineage.source_role.value,
                "source_id": lineage.source_id,
                "source_version": lineage.source_version,
                "executor_id": lineage.executor_id,
                "executor_version": lineage.executor_version,
                "has_macro_plan": lineage.macro_plan_id is not None,
                "selected_game_loop": lineage.selected_game_loop,
            }
        if event.event_type == "execution":
            report = ExecutionReport.model_validate(event.payload)
            return {
                "status": report.status.value,
                "success": report.success,
                "execution_stage": (
                    None if report.execution_stage is None else report.execution_stage.value
                ),
                "failure_code": _safe_code(report.failure_code),
                "action_name": report.action_name,
                "source": None if report.source is None else report.source.value,
            }
        if event.event_type == "command_lifecycle":
            command = _object(event.payload.get("command"))
            return {
                "status": _safe_code(_text(event.payload, "status")),
                "reason": _safe_code(_text(event.payload, "reason")),
                "action_name": _text(command, "name"),
                "source": _safe_code(_text(command, "source")),
                "priority": command.get("priority"),
                "ttl_game_loops": command.get("ttl_game_loops"),
                "created_game_loop": command.get("created_game_loop"),
            }
    except (ValueError, TypeError, KeyError):
        return {"projection_status": "invalid"}
    return {"projection_status": "metadata_only"}
