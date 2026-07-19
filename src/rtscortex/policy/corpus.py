"""Deterministic, provenance-rich corpora for offline policy comparison."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from rtscortex.contracts import ExecutionReport, ObservationEnvelope, ProductionItem, UnitState
from rtscortex.memory import StoredEvent, read_event_log
from rtscortex.policy.models import (
    PolicyFixtureSource,
    PolicyFixtureStratum,
    PolicyObservationFixture,
)
from rtscortex.progress import (
    GoalProgressReport,
    GoalProgressStatus,
    GoalProgressVerifier,
    GoalRequirement,
    GoalRequirementKind,
    GoalSpec,
)
from rtscortex.races import ActionDomain, RaceId, RaceProfileData, race_profile

CORPUS_FORMAT_VERSION: Literal["0.2"] = "0.2"
CORPUS_PROTOCOL_VERSION: Literal["1.1"] = "1.1"
CORPUS_STRATA: tuple[PolicyFixtureStratum, ...] = (
    PolicyFixtureStratum.EARLY,
    PolicyFixtureStratum.TECHNOLOGY,
    PolicyFixtureStratum.PRODUCTION,
    PolicyFixtureStratum.COMBAT,
    PolicyFixtureStratum.BLOCKED,
    PolicyFixtureStratum.IN_PROGRESS,
)

_IN_PROGRESS_STATUSES = frozenset(
    {"constructing", "in_progress", "pending", "queued", "warping_in"}
)
_CONDITION_PHASES: tuple[PolicyFixtureStratum, ...] = (
    PolicyFixtureStratum.EARLY,
    PolicyFixtureStratum.TECHNOLOGY,
    PolicyFixtureStratum.PRODUCTION,
    PolicyFixtureStratum.COMBAT,
)
_PREVIOUS_ACTION_WINDOW_GAME_LOOPS = int(60 * 22.4)
_DEFENSIVE_ALERT_NAMES = frozenset(
    {"underattack", "buildingunderattack", "unitunderattack"}
)


def _canonical_static(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def _is_defensive_alert(value: str) -> bool:
    return _canonical_static(value) in _DEFENSIVE_ALERT_NAMES


@dataclass(frozen=True)
class _CorpusRaceSemantics:
    race: RaceId
    profile: RaceProfileData
    runtime_to_hima_short_action: Mapping[str, str]
    in_progress_actions: tuple[tuple[GoalRequirementKind, str, str], ...]
    production_structures: frozenset[str]
    technology_structures: frozenset[str]
    technology_actions: frozenset[str]
    production_actions: frozenset[str]
    production_readiness_structures: frozenset[str]


_TECHNOLOGY_TARGET_OVERRIDES: Mapping[RaceId, frozenset[str]] = {
    RaceId.PROTOSS: frozenset(
        {
            "cyberneticscore",
            "forge",
            "twilightcouncil",
            "templararchive",
            "darkshrine",
            "roboticsbay",
            "fleetbeacon",
        }
    ),
    RaceId.TERRAN: frozenset(
        {
            "factory",
            "starport",
            "engineeringbay",
            "barrackstechlab",
            "factorytechlab",
            "starporttechlab",
        }
    ),
    RaceId.ZERG: frozenset(
        {"lair", "hive", "evolutionchamber", "hydraliskden"}
    ),
}


def _corpus_race_semantics(race: RaceId | str) -> _CorpusRaceSemantics:
    profile = race_profile(race).data
    technology_structures = _TECHNOLOGY_TARGET_OVERRIDES[profile.race]
    specs_by_name = {spec.name: spec for spec in profile.progress_action_specs}
    runtime_to_hima_short_action: dict[str, str] = {}
    for mapping in profile.macro_action_mappings:
        for runtime_action in mapping.runtime_actions:
            spec = specs_by_name.get(runtime_action)
            if spec is not None:
                short_action = spec.effect_target
                if runtime_action.startswith("Research_"):
                    short_action = f"{short_action}Research"
            else:
                parts = runtime_action.split("_")
                short_action = next(
                    (
                        part
                        for part in parts[1:]
                        if part not in {"Screen", "Near", "Minimap"}
                    ),
                    mapping.semantic_action.split(maxsplit=1)[-1],
                )
            runtime_to_hima_short_action[runtime_action] = short_action
    in_progress_actions = tuple(
        (spec.effect_kind, spec.effect_target, spec.name)
        for spec in profile.progress_action_specs
        if spec.effect_kind in {GoalRequirementKind.STRUCTURE, GoalRequirementKind.UNIT}
    )
    technology_actions = frozenset(
        spec.name
        for spec in profile.progress_action_specs
        if profile.action_domains.get(spec.name) is ActionDomain.TECHNOLOGY
        or _canonical_static(spec.effect_target) in technology_structures
    )
    production_actions = frozenset(
        spec.name
        for spec in profile.progress_action_specs
        if profile.action_domains.get(spec.name) is ActionDomain.PRODUCTION
        and spec.name not in technology_actions
        and spec.name.startswith(("Train_", "Warp_"))
    )
    production_readiness_structures = frozenset(
        _canonical_static(prerequisite.target)
        for spec in profile.progress_action_specs
        if spec.name in production_actions
        for prerequisite in spec.prerequisites
        if prerequisite.kind is GoalRequirementKind.STRUCTURE
    )
    return _CorpusRaceSemantics(
        race=profile.race,
        profile=profile,
        runtime_to_hima_short_action=runtime_to_hima_short_action,
        in_progress_actions=in_progress_actions,
        production_structures=frozenset(
            _canonical_static(structure) for structure in profile.production_structures
        ),
        technology_structures=technology_structures,
        technology_actions=technology_actions,
        production_actions=production_actions,
        production_readiness_structures=production_readiness_structures,
    )


_PROTOSS_CORPUS_SEMANTICS = _corpus_race_semantics(RaceId.PROTOSS)
# Compatibility aliases for the v0.2 Protoss corpus and existing callers.
_PRODUCTION_STRUCTURES = _PROTOSS_CORPUS_SEMANTICS.production_structures
_TECHNOLOGY_STRUCTURES = _PROTOSS_CORPUS_SEMANTICS.technology_structures
_RUNTIME_TO_HIMA_SHORT_ACTION = dict(
    _PROTOSS_CORPUS_SEMANTICS.runtime_to_hima_short_action
)
_IN_PROGRESS_ACTIONS = _PROTOSS_CORPUS_SEMANTICS.in_progress_actions


class PolicyCorpusError(ValueError):
    """Base error for an invalid or incomplete corpus."""


class PolicyCorpusInsufficientStates(PolicyCorpusError):
    """Raised when real journals cannot satisfy the configured quotas."""


class CorpusModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PolicyCorpusSourceConfig(CorpusModel):
    source_id: str = Field(min_length=1, max_length=120)
    journal_path: str = Field(min_length=1)
    seed: int | None = None
    map_name: str = Field(default="Simple64", min_length=1, max_length=120)


class PolicyCorpusBuildConfig(CorpusModel):
    format_version: Literal["0.2"] = CORPUS_FORMAT_VERSION
    corpus_id: str = Field(default="protoss-v0.2", min_length=1, max_length=120)
    race: RaceId = RaceId.PROTOSS
    protocol_version: Literal["1.1"] = CORPUS_PROTOCOL_VERSION
    fixtures_per_stratum: int = Field(default=8, ge=1)
    minimum_game_loop_gap: int = Field(default=224, ge=0)
    max_per_episode_per_stratum: int = Field(default=4, ge=1)
    minimum_episodes_per_stratum: int = Field(default=2, ge=1)
    minimum_seeds: int = Field(default=3, ge=1)
    minimum_condition_fixtures_per_phase: int = Field(default=2, ge=0)
    sources: list[PolicyCorpusSourceConfig] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_sampling_constraints(self) -> PolicyCorpusBuildConfig:
        source_ids = [source.source_id for source in self.sources]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("policy corpus source IDs must be unique")
        capacity = self.max_per_episode_per_stratum * self.minimum_episodes_per_stratum
        if capacity < self.fixtures_per_stratum:
            raise ValueError(
                "episode coverage cannot satisfy fixtures_per_stratum with the configured "
                "per-episode maximum"
            )
        configured_seeds = {source.seed for source in self.sources if source.seed is not None}
        if len(configured_seeds) < self.minimum_seeds:
            raise ValueError("configured sources do not cover minimum_seeds")
        required_condition_fixtures = self.minimum_condition_fixtures_per_phase * len(
            _CONDITION_PHASES
        )
        if required_condition_fixtures > self.fixtures_per_stratum:
            raise ValueError("condition phase coverage exceeds fixtures_per_stratum")
        return self


class PolicyCorpusManifestSource(CorpusModel):
    source_id: str
    journal_path: str
    journal_sha256: str
    seed: int | None
    map_name: str
    observation_count: int = Field(ge=0)
    run_ids: list[str]
    episode_keys: list[str]


class PolicyCorpusManifest(CorpusModel):
    format_version: Literal["0.2"] = CORPUS_FORMAT_VERSION
    corpus_id: str
    race: RaceId = RaceId.PROTOSS
    protocol_version: Literal["1.1"] = CORPUS_PROTOCOL_VERSION
    fixtures_file: str = "fixtures.jsonl"
    fixture_count: int = Field(ge=0)
    fixtures_sha256: str
    fixtures_per_stratum: int = Field(ge=1)
    minimum_game_loop_gap: int = Field(ge=0)
    max_per_episode_per_stratum: int = Field(ge=1)
    minimum_episodes_per_stratum: int = Field(ge=1)
    minimum_seeds: int = Field(ge=1)
    minimum_condition_fixtures_per_phase: int = Field(ge=0)
    stratum_counts: dict[PolicyFixtureStratum, int]
    condition_phase_counts: dict[PolicyFixtureStratum, dict[str, int]]
    seeds: list[int]
    episode_keys: list[str]
    fixture_ids: list[str]
    sources: list[PolicyCorpusManifestSource]


class PolicyCorpusVerification(CorpusModel):
    valid: bool
    fixture_count: int = Field(ge=0)
    stratum_counts: dict[PolicyFixtureStratum, int]
    seeds: list[int]
    episode_keys: list[str]
    errors: list[str]


@dataclass(frozen=True)
class PolicyCorpusBuildResult:
    manifest_path: Path
    fixtures_path: Path
    manifest: PolicyCorpusManifest
    fixtures: tuple[PolicyObservationFixture, ...]


@dataclass(frozen=True)
class _LoadedSource:
    config: PolicyCorpusSourceConfig
    configured_path: str
    path: Path
    journal_sha256: str
    observations: tuple[StoredEvent, ...]
    successful_executions: tuple[StoredEvent, ...]


@dataclass(frozen=True)
class _FixtureCandidate:
    source_id: str
    fixture: PolicyObservationFixture


@dataclass(frozen=True)
class _Classification:
    primary: PolicyFixtureStratum
    phase_tags: tuple[str, ...]
    condition_tags: tuple[str, ...]
    blocker_tags: tuple[str, ...]
    evidence: tuple[str, ...]


def load_policy_corpus_config(path: Path) -> PolicyCorpusBuildConfig:
    """Load a strict corpus-source configuration."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise PolicyCorpusError("policy corpus config must contain a YAML mapping")
    return PolicyCorpusBuildConfig.model_validate(payload)


def build_policy_corpus_from_file(
    config_path: Path,
    output_dir: Path,
) -> PolicyCorpusBuildResult:
    """Build a corpus from a YAML config, resolving relative source paths beside it."""

    config = load_policy_corpus_config(config_path)
    return build_policy_corpus(config, output_dir, base_dir=config_path.parent)


def build_policy_corpus(
    config: PolicyCorpusBuildConfig,
    output_dir: Path,
    *,
    base_dir: Path | None = None,
) -> PolicyCorpusBuildResult:
    """Select balanced real observations and persist a deterministic corpus."""

    semantics = _corpus_race_semantics(config.race)
    sources = tuple(
        _load_source(source, semantics=semantics, base_dir=base_dir)
        for source in config.sources
    )
    candidates = _collect_candidates(
        sources,
        protocol_version=config.protocol_version,
        corpus_id=config.corpus_id,
        semantics=semantics,
    )
    selected = _select_balanced(candidates, config)

    output_dir.mkdir(parents=True, exist_ok=True)
    fixtures_path = output_dir / "fixtures.jsonl"
    fixture_bytes = _encode_fixtures(selected)
    fixtures_path.write_bytes(fixture_bytes)

    manifest_sources = [_manifest_source(source) for source in sources]
    stratum_counts = Counter(fixture.primary_stratum for fixture in selected)
    condition_phase_counts = _condition_phase_counts(selected)
    seeds = sorted(
        {
            fixture.source.seed
            for fixture in selected
            if fixture.source is not None and fixture.source.seed is not None
        }
    )
    episode_keys = sorted(
        {
            _episode_key(fixture.observation.run_id, fixture.observation.episode_id)
            for fixture in selected
        }
    )
    manifest = PolicyCorpusManifest(
        corpus_id=config.corpus_id,
        race=config.race,
        fixture_count=len(selected),
        fixtures_sha256=_sha256_bytes(fixture_bytes),
        fixtures_per_stratum=config.fixtures_per_stratum,
        minimum_game_loop_gap=config.minimum_game_loop_gap,
        max_per_episode_per_stratum=config.max_per_episode_per_stratum,
        minimum_episodes_per_stratum=config.minimum_episodes_per_stratum,
        minimum_seeds=config.minimum_seeds,
        minimum_condition_fixtures_per_phase=(config.minimum_condition_fixtures_per_phase),
        stratum_counts={stratum: stratum_counts[stratum] for stratum in CORPUS_STRATA},
        condition_phase_counts=condition_phase_counts,
        seeds=seeds,
        episode_keys=episode_keys,
        fixture_ids=[fixture.fixture_id for fixture in selected],
        sources=manifest_sources,
    )
    manifest_path = output_dir / "manifest.yaml"
    manifest_payload = manifest.model_dump(mode="json")
    manifest_path.write_text(
        yaml.safe_dump(manifest_payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    verification = verify_policy_corpus(manifest_path)
    if not verification.valid:
        rendered = "; ".join(verification.errors)
        raise PolicyCorpusError(f"built corpus failed verification: {rendered}")
    return PolicyCorpusBuildResult(
        manifest_path=manifest_path,
        fixtures_path=fixtures_path,
        manifest=manifest,
        fixtures=tuple(selected),
    )


def load_policy_corpus(
    manifest_path: Path,
    *,
    verify: bool = True,
) -> list[PolicyObservationFixture]:
    """Load corpus fixtures in their canonical order."""

    manifest = _load_manifest(manifest_path)
    fixtures_path = _fixtures_path(manifest_path, manifest)
    fixtures = _read_fixtures(fixtures_path)
    if verify:
        result = verify_policy_corpus(manifest_path)
        if not result.valid:
            rendered = "; ".join(result.errors)
            raise PolicyCorpusError(f"invalid policy corpus: {rendered}")
    return fixtures


def load_policy_corpus_manifest(manifest_path: Path) -> PolicyCorpusManifest:
    """Load one strict corpus manifest, including its active race."""

    return _load_manifest(manifest_path)


def verify_policy_corpus(
    manifest_path: Path,
    *,
    verify_sources: bool = False,
) -> PolicyCorpusVerification:
    """Verify fixture integrity, coverage, provenance, and deterministic hashes."""

    manifest = _load_manifest(manifest_path)
    fixtures_path = _fixtures_path(manifest_path, manifest)
    errors: list[str] = []
    if not fixtures_path.is_file():
        errors.append(f"fixtures file is missing: {fixtures_path}")
        return PolicyCorpusVerification(
            valid=False,
            fixture_count=0,
            stratum_counts={},
            seeds=[],
            episode_keys=[],
            errors=errors,
        )
    fixture_bytes = fixtures_path.read_bytes()
    if _sha256_bytes(fixture_bytes) != manifest.fixtures_sha256:
        errors.append("fixtures SHA256 does not match manifest")
    try:
        fixtures = _read_fixtures(fixtures_path)
    except Exception as error:
        errors.append(f"fixtures cannot be decoded: {type(error).__name__}: {error}")
        fixtures = []

    _verify_fixtures(fixtures, manifest, errors)
    if verify_sources:
        _verify_source_journals(manifest, fixtures, errors)

    counts = Counter(
        fixture.primary_stratum for fixture in fixtures if fixture.primary_stratum is not None
    )
    seeds = sorted(
        {
            fixture.source.seed
            for fixture in fixtures
            if fixture.source is not None and fixture.source.seed is not None
        }
    )
    episode_keys = sorted(
        {
            _episode_key(fixture.observation.run_id, fixture.observation.episode_id)
            for fixture in fixtures
        }
    )
    return PolicyCorpusVerification(
        valid=not errors,
        fixture_count=len(fixtures),
        stratum_counts={stratum: counts[stratum] for stratum in CORPUS_STRATA},
        seeds=seeds,
        episode_keys=episode_keys,
        errors=errors,
    )


def state_fingerprint(observation: ObservationEnvelope) -> str:
    """Hash strategic state while excluding IDs, positions, timestamps, and prose."""

    economy = observation.state.economy
    payload = {
        "economy": {
            "minerals_50": economy.minerals // 50,
            "vespene_25": economy.vespene // 25,
            "supply_used": economy.supply_used,
            "supply_cap": economy.supply_cap,
            "workers": economy.workers,
            "army_supply": economy.army_supply,
        },
        "production_queue": _aggregate_production(observation.state.production_queue),
        "own_units": _aggregate_units(observation.state.own_units),
        "own_structures": _aggregate_units(observation.state.own_structures),
        "visible_enemies": _aggregate_units(observation.state.visible_enemies),
        "upgrades": sorted({_canonical(value) for value in observation.state.upgrades}),
        "alerts": sorted({_canonical(value) for value in observation.alerts}),
        "available_actions": sorted({action.name for action in observation.available_actions}),
    }
    return _sha256_bytes(_canonical_json(payload))


def _collect_candidates(
    sources: Sequence[_LoadedSource],
    *,
    protocol_version: str,
    corpus_id: str,
    semantics: _CorpusRaceSemantics,
) -> dict[PolicyFixtureStratum, list[_FixtureCandidate]]:
    verifier = GoalProgressVerifier(action_specs=semantics.profile.progress_action_specs)
    candidates: dict[PolicyFixtureStratum, list[_FixtureCandidate]] = {
        stratum: [] for stratum in CORPUS_STRATA
    }
    for source in sources:
        for event in source.observations:
            observation = ObservationEnvelope.model_validate(event.payload)
            if observation.protocol_version != protocol_version:
                continue
            phase = _observation_phase(observation, semantics.race)
            phase_goal = _phase_goal(verifier, observation, phase, semantics.race)
            phase_report = verifier.verify(observation, phase_goal)
            previous_actions = _previous_actions_for_observation(
                source,
                event,
                observation,
                semantics,
            )
            phase_classification = _classify_observation(
                observation,
                phase_report,
                primary=phase,
                phase=phase,
            )
            candidates[phase].append(
                _make_candidate(
                    source=source,
                    event=event,
                    observation=observation,
                    goal=phase_goal,
                    report=phase_report,
                    classification=phase_classification,
                    previous_actions=previous_actions,
                    corpus_id=corpus_id,
                )
            )

            progress_goal = _in_progress_goal(observation, phase, semantics.race)
            if progress_goal is not None:
                progress_report = verifier.verify(observation, progress_goal)
                if progress_report.status is not GoalProgressStatus.IN_PROGRESS:
                    raise PolicyCorpusError(
                        "deterministic in-progress goal did not verify as in_progress: "
                        f"{source.config.source_id}:event-{event.event_id}"
                    )
                progress_classification = _classify_observation(
                    observation,
                    progress_report,
                    primary=PolicyFixtureStratum.IN_PROGRESS,
                    phase=phase,
                )
                candidates[PolicyFixtureStratum.IN_PROGRESS].append(
                    _make_candidate(
                        source=source,
                        event=event,
                        observation=observation,
                        goal=progress_goal,
                        report=progress_report,
                        classification=progress_classification,
                        previous_actions=previous_actions,
                        corpus_id=corpus_id,
                    )
                )
            elif phase_report.status is GoalProgressStatus.BLOCKED:
                blocked_classification = _classify_observation(
                    observation,
                    phase_report,
                    primary=PolicyFixtureStratum.BLOCKED,
                    phase=phase,
                )
                candidates[PolicyFixtureStratum.BLOCKED].append(
                    _make_candidate(
                        source=source,
                        event=event,
                        observation=observation,
                        goal=phase_goal,
                        report=phase_report,
                        classification=blocked_classification,
                        previous_actions=previous_actions,
                        corpus_id=corpus_id,
                    )
                )
    return candidates


def _make_candidate(
    *,
    source: _LoadedSource,
    event: StoredEvent,
    observation: ObservationEnvelope,
    goal: GoalSpec,
    report: GoalProgressReport,
    classification: _Classification,
    previous_actions: list[str],
    corpus_id: str,
) -> _FixtureCandidate:
    observation_sha256 = _sha256_bytes(_canonical_json(observation.model_dump(mode="json")))
    fixture_source = PolicyFixtureSource(
        run_id=observation.run_id,
        episode_id=observation.episode_id,
        event_id=event.event_id,
        seed=source.config.seed,
        map_name=source.config.map_name,
        game_loop=observation.game_loop,
        protocol_version=observation.protocol_version,
        journal_sha256=source.journal_sha256,
        observation_sha256=observation_sha256,
    )
    fixture = PolicyObservationFixture(
        fixture_id=(
            f"{corpus_id}:{classification.primary.value}:{source.config.source_id}:"
            f"event-{event.event_id}"
        ),
        observation=observation,
        previous_actions=previous_actions,
        goal_spec=goal,
        goal_progress=report,
        primary_stratum=classification.primary,
        phase_tags=list(classification.phase_tags),
        condition_tags=list(classification.condition_tags),
        blocker_tags=list(classification.blocker_tags),
        selection_evidence=list(classification.evidence),
        source=fixture_source,
        state_fingerprint=state_fingerprint(observation),
    )
    return _FixtureCandidate(source_id=source.config.source_id, fixture=fixture)


def _select_balanced(
    candidates: Mapping[PolicyFixtureStratum, Sequence[_FixtureCandidate]],
    config: PolicyCorpusBuildConfig,
) -> list[PolicyObservationFixture]:
    selected: list[PolicyObservationFixture] = []
    fingerprints: set[str] = set()
    failures: list[str] = []
    for stratum in CORPUS_STRATA:
        grouped: dict[tuple[str, str, str], list[_FixtureCandidate]] = defaultdict(list)
        for candidate in candidates[stratum]:
            observation = candidate.fixture.observation
            grouped[(candidate.source_id, observation.run_id, observation.episode_id)].append(
                candidate
            )
        spaced = {
            key: _apply_game_loop_gap(items, config.minimum_game_loop_gap)
            for key, items in grouped.items()
        }
        if stratum in {
            PolicyFixtureStratum.BLOCKED,
            PolicyFixtureStratum.IN_PROGRESS,
        }:
            chosen, phase_failures = _select_condition_stratum(
                spaced,
                quota=config.fixtures_per_stratum,
                phase_quota=config.minimum_condition_fixtures_per_phase,
                max_per_episode=config.max_per_episode_per_stratum,
                used_fingerprints=fingerprints,
            )
            failures.extend(f"{stratum.value}: {failure}" for failure in phase_failures)
        else:
            chosen = _round_robin_stratum(
                spaced,
                quota=config.fixtures_per_stratum,
                max_per_episode=config.max_per_episode_per_stratum,
                used_fingerprints=fingerprints,
            )
        episode_count = len(
            {(fixture.observation.run_id, fixture.observation.episode_id) for fixture in chosen}
        )
        if len(chosen) != config.fixtures_per_stratum:
            failures.append(
                f"{stratum.value}: selected {len(chosen)}/{config.fixtures_per_stratum} "
                f"from {len(grouped)} episodes"
            )
        elif episode_count < config.minimum_episodes_per_stratum:
            failures.append(
                f"{stratum.value}: selected states cover only {episode_count}/"
                f"{config.minimum_episodes_per_stratum} episodes"
            )
        selected.extend(chosen)
        fingerprints.update(
            fixture.state_fingerprint for fixture in chosen if fixture.state_fingerprint is not None
        )
    if failures:
        raise PolicyCorpusInsufficientStates("; ".join(failures))
    return selected


def _select_condition_stratum(
    grouped: Mapping[tuple[str, str, str], Sequence[_FixtureCandidate]],
    *,
    quota: int,
    phase_quota: int,
    max_per_episode: int,
    used_fingerprints: set[str],
) -> tuple[list[PolicyObservationFixture], list[str]]:
    selected: list[PolicyObservationFixture] = []
    selected_fingerprints: set[str] = set()
    episode_counts: Counter[tuple[str, str, str]] = Counter()
    failures: list[str] = []
    for phase in _CONDITION_PHASES:
        phase_groups = {
            key: [
                candidate for candidate in candidates if phase.value in candidate.fixture.phase_tags
            ]
            for key, candidates in grouped.items()
        }
        phase_groups = {key: values for key, values in phase_groups.items() if values}
        phase_selected = _round_robin_stratum(
            phase_groups,
            quota=phase_quota,
            max_per_episode=max_per_episode,
            used_fingerprints=used_fingerprints | selected_fingerprints,
            episode_counts=episode_counts,
        )
        selected.extend(phase_selected)
        selected_fingerprints.update(
            fixture.state_fingerprint
            for fixture in phase_selected
            if fixture.state_fingerprint is not None
        )
        if len(phase_selected) != phase_quota:
            failures.append(f"phase {phase.value} selected {len(phase_selected)}/{phase_quota}")

    remaining = quota - len(selected)
    if remaining > 0:
        fill = _round_robin_stratum(
            grouped,
            quota=remaining,
            max_per_episode=max_per_episode,
            used_fingerprints=used_fingerprints | selected_fingerprints,
            episode_counts=episode_counts,
        )
        selected.extend(fill)
    return selected, failures


def _apply_game_loop_gap(
    candidates: Sequence[_FixtureCandidate],
    minimum_gap: int,
) -> list[_FixtureCandidate]:
    ordered = sorted(
        candidates,
        key=lambda item: (
            item.fixture.observation.game_loop,
            item.fixture.source.event_id if item.fixture.source is not None else -1,
        ),
    )
    selected: list[_FixtureCandidate] = []
    last_game_loop: int | None = None
    for candidate in ordered:
        game_loop = candidate.fixture.observation.game_loop
        if last_game_loop is not None and game_loop - last_game_loop < minimum_gap:
            continue
        selected.append(candidate)
        last_game_loop = game_loop
    return selected


def _round_robin_stratum(
    grouped: Mapping[tuple[str, str, str], Sequence[_FixtureCandidate]],
    *,
    quota: int,
    max_per_episode: int,
    used_fingerprints: set[str],
    episode_counts: Counter[tuple[str, str, str]] | None = None,
) -> list[PolicyObservationFixture]:
    group_keys = sorted(grouped)
    offsets = {key: 0 for key in group_keys}
    counts = episode_counts if episode_counts is not None else Counter()
    local_fingerprints: set[str] = set()
    selected: list[PolicyObservationFixture] = []
    while len(selected) < quota:
        made_progress = False
        for key in group_keys:
            if len(selected) >= quota:
                break
            if counts[key] >= max_per_episode:
                continue
            items = grouped[key]
            while offsets[key] < len(items):
                candidate = items[offsets[key]]
                offsets[key] += 1
                fingerprint = candidate.fixture.state_fingerprint
                if fingerprint is None:
                    continue
                if fingerprint in used_fingerprints or fingerprint in local_fingerprints:
                    continue
                selected.append(candidate.fixture)
                local_fingerprints.add(fingerprint)
                counts[key] += 1
                made_progress = True
                break
        if not made_progress:
            break
    return selected


def _classify_observation(
    observation: ObservationEnvelope,
    report: GoalProgressReport,
    *,
    primary: PolicyFixtureStratum,
    phase: PolicyFixtureStratum,
) -> _Classification:
    action_names = {action.name for action in observation.available_actions}
    incomplete_entities = [
        unit
        for unit in (*observation.state.own_units, *observation.state.own_structures)
        if _canonical(unit.status or "") in _IN_PROGRESS_STATUSES
    ]
    defensive_alerts = [
        alert for alert in observation.alerts if _is_defensive_alert(alert)
    ]
    condition_tags = [report.status.value]
    if report.defensive_hold_required:
        condition_tags.append("defensive_hold_required")
    blocker_tags = tuple(sorted({blocker.kind.value for blocker in report.blockers}))
    evidence = [
        f"phase={phase.value}",
        f"goal_progress={report.status.value}",
        f"goal={report.goal_id}",
    ]
    if observation.state.visible_enemies:
        evidence.append(f"visible_enemies={len(observation.state.visible_enemies)}")
    if "Attack_Unit" in action_names:
        evidence.append("available_action=Attack_Unit")
    if defensive_alerts:
        evidence.append(f"defensive_alerts={len(defensive_alerts)}")
    structure_names = {
        _canonical(structure.unit_type) for structure in observation.state.own_structures
    }
    if "cyberneticscore" in structure_names:
        evidence.append("structure=CyberneticsCore")
    if observation.state.upgrades:
        evidence.append(f"upgrades={len(observation.state.upgrades)}")
    technology_actions = sorted(
        name for name in action_names if name.startswith("Research_") or "CyberneticsCore" in name
    )
    evidence.extend(f"available_action={name}" for name in technology_actions)
    if observation.state.production_queue:
        evidence.append(f"production_queue={len(observation.state.production_queue)}")
    if incomplete_entities:
        evidence.append(f"incomplete_entities={len(incomplete_entities)}")
    return _Classification(
        primary=primary,
        phase_tags=(phase.value,),
        condition_tags=tuple(condition_tags),
        blocker_tags=blocker_tags,
        evidence=tuple(evidence),
    )


def _observation_phase(
    observation: ObservationEnvelope,
    race: RaceId = RaceId.PROTOSS,
) -> PolicyFixtureStratum:
    """Assign one deterministic strategic phase without using combat-unit presence."""

    semantics = _corpus_race_semantics(race)
    action_names = {action.name for action in observation.available_actions}
    structure_names = {
        _canonical(structure.unit_type) for structure in observation.state.own_structures
    }
    completed_structure_names = {
        _canonical(structure.unit_type)
        for structure in observation.state.own_structures
        if _canonical(structure.status or "") not in _IN_PROGRESS_STATUSES
    }
    defensive_alert = any(_is_defensive_alert(alert) for alert in observation.alerts)
    if observation.state.visible_enemies or "Attack_Unit" in action_names or defensive_alert:
        return PolicyFixtureStratum.COMBAT
    has_incomplete_technology_structure = any(
        _canonical(structure.unit_type) in semantics.technology_structures
        and _canonical(structure.status or "") in _IN_PROGRESS_STATUSES
        for structure in observation.state.own_structures
    )
    if action_names & semantics.technology_actions or has_incomplete_technology_structure:
        return PolicyFixtureStratum.TECHNOLOGY
    has_incomplete_production_structure = any(
        _canonical(structure.unit_type) in semantics.production_structures
        and _canonical(structure.status or "") in _IN_PROGRESS_STATUSES
        for structure in observation.state.own_structures
    )
    if (
        observation.state.production_queue
        or action_names & semantics.production_actions
        or has_incomplete_production_structure
        or completed_structure_names & semantics.production_readiness_structures
    ):
        return PolicyFixtureStratum.PRODUCTION
    if observation.state.upgrades or structure_names & semantics.technology_structures:
        return PolicyFixtureStratum.TECHNOLOGY
    return PolicyFixtureStratum.EARLY


def _phase_goal(
    verifier: GoalProgressVerifier,
    observation: ObservationEnvelope,
    phase: PolicyFixtureStratum,
    race: RaceId = RaceId.PROTOSS,
) -> GoalSpec:
    """Choose one measurable next-state goal appropriate to the assigned phase."""

    completed_structures = {
        _canonical(structure.unit_type)
        for structure in observation.state.own_structures
        if _canonical(structure.status or "") not in _IN_PROGRESS_STATUSES
    }
    upgrades = {_canonical(upgrade) for upgrade in observation.state.upgrades}
    action, use_baseline = _phase_action(
        race,
        phase,
        completed_structures=completed_structures,
        upgrades=upgrades,
        defensive_hold=any(_is_defensive_alert(alert) for alert in observation.alerts),
    )
    return verifier.goal_from_action_names(
        goal_id=f"{race.value}-{phase.value}-next-v1",
        strategic_goal=f"Advance the {phase.value} phase with {action}",
        action_names=[action],
        observation=observation if use_baseline else None,
    )


def _phase_action(
    race: RaceId,
    phase: PolicyFixtureStratum,
    *,
    completed_structures: set[str],
    upgrades: set[str],
    defensive_hold: bool,
) -> tuple[str, bool]:
    if phase not in _CONDITION_PHASES:
        raise ValueError(f"unsupported policy corpus phase: {phase.value}")
    if race is RaceId.PROTOSS:
        if phase is PolicyFixtureStratum.EARLY:
            if "pylon" not in completed_structures:
                return "Build_Pylon_Screen", False
            if "gateway" not in completed_structures:
                return "Build_Gateway_Screen", False
            return "Train_Zealot", True
        if phase is PolicyFixtureStratum.TECHNOLOGY:
            if "cyberneticscore" not in completed_structures:
                return "Build_CyberneticsCore_Screen", False
            if "warpgate" not in upgrades and "warpgateresearch" not in upgrades:
                return "Research_WarpGate", False
            return "Train_Stalker", True
        if phase is PolicyFixtureStratum.PRODUCTION:
            return "Train_Stalker", True
        if defensive_hold:
            return "Build_ShieldBattery_Screen", False
        action = (
            "Train_Stalker" if "cyberneticscore" in completed_structures else "Train_Zealot"
        )
        return action, True
    if race is RaceId.TERRAN:
        if phase is PolicyFixtureStratum.EARLY:
            if "supplydepot" not in completed_structures:
                return "Build_SupplyDepot_Screen", False
            if "barracks" not in completed_structures:
                return "Build_Barracks_Screen", False
            return "Train_Marine", True
        if phase is PolicyFixtureStratum.TECHNOLOGY:
            if "factory" not in completed_structures:
                return "Build_Factory_Screen", False
            if "starport" not in completed_structures:
                return "Build_Starport_Screen", False
            return "Train_SiegeTank", True
        if phase is PolicyFixtureStratum.PRODUCTION:
            return "Train_SiegeTank" if "factory" in completed_structures else "Train_Marine", True
        return ("Build_Bunker_Screen", False) if defensive_hold else ("Train_Marine", True)
    if phase is PolicyFixtureStratum.EARLY:
        if "spawningpool" not in completed_structures:
            return "Build_SpawningPool_Screen", False
        return "Train_Zergling", True
    if phase is PolicyFixtureStratum.TECHNOLOGY:
        if "lair" not in completed_structures and "hive" not in completed_structures:
            return "Morph_Lair", False
        if "hydraliskden" not in completed_structures:
            return "Build_HydraliskDen_Screen", False
        return "Train_Hydralisk", True
    if phase is PolicyFixtureStratum.PRODUCTION:
        return (
            ("Train_Roach", True)
            if "roachwarren" in completed_structures
            else ("Train_Zergling", True)
        )
    if defensive_hold:
        return "Build_SpineCrawler_Screen", False
    return (
        ("Train_Roach", True)
        if "roachwarren" in completed_structures
        else ("Train_Zergling", True)
    )


def _in_progress_goal(
    observation: ObservationEnvelope,
    phase: PolicyFixtureStratum,
    race: RaceId = RaceId.PROTOSS,
) -> GoalSpec | None:
    """Describe a real observed construction, unit, or queue item as a goal."""

    semantics = _corpus_race_semantics(race)
    for kind, target, action_name in semantics.in_progress_actions:
        completed, incomplete = _effect_counts(observation, kind, target)
        if incomplete:
            return _single_requirement_goal(
                phase=phase,
                race=race,
                kind=kind,
                target=target,
                required_count=completed + 1,
                action_name=action_name,
            )
    if not observation.state.production_queue:
        return None
    item = sorted(
        observation.state.production_queue,
        key=lambda candidate: (_canonical(candidate.name), candidate.progress),
    )[0]
    return _single_requirement_goal(
        phase=phase,
        race=race,
        kind=GoalRequirementKind.UNIT,
        target=item.name,
        required_count=1,
        action_name=None,
    )


def _single_requirement_goal(
    *,
    phase: PolicyFixtureStratum,
    race: RaceId,
    kind: GoalRequirementKind,
    target: str,
    required_count: int,
    action_name: str | None,
) -> GoalSpec:
    canonical_target = _canonical(target)
    return GoalSpec(
        goal_id=f"{race.value}-{phase.value}-in-progress-{canonical_target}-v1",
        strategic_goal=f"Complete the in-progress {target}",
        requirements=[
            GoalRequirement(
                requirement_id=f"in-progress:{kind.value}:{canonical_target}",
                kind=kind,
                target=target,
                count=required_count,
                action_name=action_name,
                description=f"Complete the observed in-progress {target}",
            )
        ],
    )


def _effect_counts(
    observation: ObservationEnvelope,
    kind: GoalRequirementKind,
    target: str,
) -> tuple[int, int]:
    canonical_target = _canonical(target)
    entities = (
        observation.state.own_structures
        if kind is GoalRequirementKind.STRUCTURE
        else observation.state.own_units
    )
    matching = [entity for entity in entities if _canonical(entity.unit_type) == canonical_target]
    incomplete = sum(
        1 for entity in matching if _canonical(entity.status or "") in _IN_PROGRESS_STATUSES
    )
    return len(matching) - incomplete, incomplete


def _verify_fixtures(
    fixtures: Sequence[PolicyObservationFixture],
    manifest: PolicyCorpusManifest,
    errors: list[str],
) -> None:
    semantics = _corpus_race_semantics(manifest.race)
    if len(fixtures) != manifest.fixture_count:
        errors.append(
            f"fixture count {len(fixtures)} does not match manifest {manifest.fixture_count}"
        )
    fixture_ids = [fixture.fixture_id for fixture in fixtures]
    if fixture_ids != manifest.fixture_ids:
        errors.append("fixture order or IDs do not match manifest")
    if len(fixture_ids) != len(set(fixture_ids)):
        errors.append("fixture IDs are not unique")
    fingerprints = [fixture.state_fingerprint for fixture in fixtures]
    if None in fingerprints or len(fingerprints) != len(set(fingerprints)):
        errors.append("state fingerprints are missing or duplicated")

    source_hashes = {source.journal_sha256 for source in manifest.sources}
    counts = Counter[PolicyFixtureStratum]()
    episode_counts: Counter[tuple[PolicyFixtureStratum, str, str]] = Counter()
    loops: dict[tuple[PolicyFixtureStratum, str, str], list[int]] = defaultdict(list)
    stratum_episodes: dict[PolicyFixtureStratum, set[tuple[str, str]]] = defaultdict(set)
    verifier = GoalProgressVerifier(action_specs=semantics.profile.progress_action_specs)
    for fixture in fixtures:
        observation = fixture.observation
        stratum = fixture.primary_stratum
        source = fixture.source
        if observation.protocol_version != manifest.protocol_version:
            errors.append(f"{fixture.fixture_id}: protocol is not {manifest.protocol_version}")
        if stratum is None:
            errors.append(f"{fixture.fixture_id}: primary_stratum is missing")
            continue
        counts[stratum] += 1
        key = (stratum, observation.run_id, observation.episode_id)
        episode_counts[key] += 1
        loops[key].append(observation.game_loop)
        stratum_episodes[stratum].add((observation.run_id, observation.episode_id))
        if fixture.state_fingerprint != state_fingerprint(observation):
            errors.append(f"{fixture.fixture_id}: state fingerprint mismatch")
        unknown_previous_actions = set(fixture.previous_actions) - set(
            semantics.runtime_to_hima_short_action.values()
        )
        if unknown_previous_actions:
            errors.append(
                f"{fixture.fixture_id}: previous actions are not official HIMA short names"
            )
        if source is None:
            errors.append(f"{fixture.fixture_id}: source provenance is missing")
            continue
        observation_hash = _sha256_bytes(_canonical_json(observation.model_dump(mode="json")))
        if source.observation_sha256 != observation_hash:
            errors.append(f"{fixture.fixture_id}: observation SHA256 mismatch")
        if source.journal_sha256 not in source_hashes:
            errors.append(f"{fixture.fixture_id}: journal SHA256 is absent from manifest")
        if (
            source.run_id != observation.run_id
            or source.episode_id != observation.episode_id
            or source.game_loop != observation.game_loop
            or source.protocol_version != observation.protocol_version
        ):
            errors.append(f"{fixture.fixture_id}: source provenance does not match observation")
        if len(fixture.phase_tags) != 1:
            errors.append(f"{fixture.fixture_id}: fixture must have exactly one phase tag")
            continue
        try:
            phase = PolicyFixtureStratum(fixture.phase_tags[0])
        except ValueError:
            errors.append(f"{fixture.fixture_id}: unknown phase tag")
            continue
        if phase not in _CONDITION_PHASES:
            errors.append(f"{fixture.fixture_id}: condition tag is not a strategic phase")
            continue
        if stratum is PolicyFixtureStratum.IN_PROGRESS:
            expected_goal = _in_progress_goal(observation, phase, manifest.race)
            if expected_goal is None:
                errors.append(f"{fixture.fixture_id}: no in-progress state evidence remains")
                continue
        else:
            expected_goal = _phase_goal(verifier, observation, phase, manifest.race)
        expected_report = verifier.verify(observation, expected_goal)
        if fixture.goal_spec != expected_goal or fixture.goal_progress != expected_report:
            errors.append(f"{fixture.fixture_id}: goal progress is not deterministic")
        if (
            stratum is PolicyFixtureStratum.BLOCKED
            and expected_report.status is not GoalProgressStatus.BLOCKED
        ):
            errors.append(f"{fixture.fixture_id}: blocked fixture goal is no longer blocked")
        if (
            stratum is PolicyFixtureStratum.IN_PROGRESS
            and expected_report.status is not GoalProgressStatus.IN_PROGRESS
        ):
            errors.append(
                f"{fixture.fixture_id}: in-progress fixture goal is no longer in progress"
            )
        if stratum in _CONDITION_PHASES and stratum is not phase:
            errors.append(f"{fixture.fixture_id}: primary phase and phase tag differ")
        expected_classification = _classify_observation(
            observation,
            expected_report,
            primary=stratum,
            phase=phase,
        )
        if (
            fixture.primary_stratum != expected_classification.primary
            or fixture.phase_tags != list(expected_classification.phase_tags)
            or fixture.condition_tags != list(expected_classification.condition_tags)
            or fixture.blocker_tags != list(expected_classification.blocker_tags)
            or fixture.selection_evidence != list(expected_classification.evidence)
        ):
            errors.append(f"{fixture.fixture_id}: stratum metadata is not deterministic")

    for stratum in CORPUS_STRATA:
        expected = manifest.fixtures_per_stratum
        if counts[stratum] != expected:
            errors.append(
                f"{stratum.value}: fixture count {counts[stratum]} does not equal {expected}"
            )
        if len(stratum_episodes[stratum]) < manifest.minimum_episodes_per_stratum:
            errors.append(
                f"{stratum.value}: episode coverage is below "
                f"{manifest.minimum_episodes_per_stratum}"
            )
    for key, count in episode_counts.items():
        if count > manifest.max_per_episode_per_stratum:
            errors.append(f"{key[0].value}:{key[1]}:{key[2]} exceeds per-episode maximum")
    for key, game_loops in loops.items():
        ordered = sorted(game_loops)
        if any(
            current - previous < manifest.minimum_game_loop_gap
            for previous, current in zip(ordered, ordered[1:], strict=False)
        ):
            errors.append(f"{key[0].value}:{key[1]}:{key[2]} violates game-loop gap")

    actual_counts = {stratum: counts[stratum] for stratum in CORPUS_STRATA}
    if actual_counts != manifest.stratum_counts:
        errors.append("stratum counts do not match manifest")
    actual_condition_phase_counts = _condition_phase_counts(fixtures)
    if actual_condition_phase_counts != manifest.condition_phase_counts:
        errors.append("condition phase counts do not match manifest")
    for condition in (
        PolicyFixtureStratum.BLOCKED,
        PolicyFixtureStratum.IN_PROGRESS,
    ):
        for phase in _CONDITION_PHASES:
            count = actual_condition_phase_counts[condition][phase.value]
            if count < manifest.minimum_condition_fixtures_per_phase:
                errors.append(
                    f"{condition.value}: phase {phase.value} coverage {count} is below "
                    f"{manifest.minimum_condition_fixtures_per_phase}"
                )
    seeds = {
        fixture.source.seed
        for fixture in fixtures
        if fixture.source is not None and fixture.source.seed is not None
    }
    if len(seeds) < manifest.minimum_seeds:
        errors.append(f"corpus covers only {len(seeds)}/{manifest.minimum_seeds} seeds")
    if sorted(seeds) != manifest.seeds:
        errors.append("seed coverage does not match manifest")
    episode_keys = sorted(
        {
            _episode_key(fixture.observation.run_id, fixture.observation.episode_id)
            for fixture in fixtures
        }
    )
    if episode_keys != manifest.episode_keys:
        errors.append("episode coverage does not match manifest")


def _verify_source_journals(
    manifest: PolicyCorpusManifest,
    fixtures: Sequence[PolicyObservationFixture],
    errors: list[str],
) -> None:
    semantics = _corpus_race_semantics(manifest.race)
    fixtures_by_journal: dict[str, list[PolicyObservationFixture]] = defaultdict(list)
    for fixture in fixtures:
        if fixture.source is not None:
            fixtures_by_journal[fixture.source.journal_sha256].append(fixture)
    for source in manifest.sources:
        path = _expand_path(source.journal_path, base_dir=None)
        if not path.is_file():
            errors.append(f"source journal is missing: {source.source_id}")
            continue
        if _sha256_file(path) != source.journal_sha256:
            errors.append(f"source journal SHA256 mismatch: {source.source_id}")
            continue
        source_events = tuple(read_event_log(path))
        observations = {
            (event.run_id, event.episode_id, event.event_id): event
            for event in source_events
            if event.event_type == "observation"
        }
        loaded_source = _LoadedSource(
            config=PolicyCorpusSourceConfig(
                source_id=source.source_id,
                journal_path=source.journal_path,
                seed=source.seed,
                map_name=source.map_name,
            ),
            configured_path=source.journal_path,
            path=path,
            journal_sha256=source.journal_sha256,
            observations=tuple(observations.values()),
            successful_executions=tuple(
                event
                for event in source_events
                if event.event_type == "execution"
                and _successful_hima_action(event, semantics) is not None
            ),
        )
        if len(observations) != source.observation_count:
            errors.append(f"source observation count mismatch: {source.source_id}")
        for fixture in fixtures_by_journal[source.journal_sha256]:
            assert fixture.source is not None
            key = (
                fixture.source.run_id,
                fixture.source.episode_id,
                fixture.source.event_id,
            )
            event = observations.get(key)
            if event is None:
                errors.append(f"{fixture.fixture_id}: source observation is missing")
                continue
            observation = ObservationEnvelope.model_validate(event.payload)
            observation_hash = _sha256_bytes(_canonical_json(observation.model_dump(mode="json")))
            if observation_hash != fixture.source.observation_sha256:
                errors.append(f"{fixture.fixture_id}: source observation payload changed")
            expected_previous_actions = _previous_actions_for_observation(
                loaded_source,
                event,
                observation,
                semantics,
            )
            if fixture.previous_actions != expected_previous_actions:
                errors.append(f"{fixture.fixture_id}: previous action history changed")
            if fixture.source.seed != source.seed or fixture.source.map_name != source.map_name:
                errors.append(f"{fixture.fixture_id}: source config metadata mismatch")


def _load_source(
    config: PolicyCorpusSourceConfig,
    *,
    semantics: _CorpusRaceSemantics,
    base_dir: Path | None,
) -> _LoadedSource:
    path = _expand_path(config.journal_path, base_dir=base_dir)
    if not path.is_file():
        raise PolicyCorpusError(f"source journal does not exist: {path}")
    events = tuple(read_event_log(path))
    observations = tuple(event for event in events if event.event_type == "observation")
    if not observations:
        raise PolicyCorpusError(f"source journal contains no observations: {path}")
    return _LoadedSource(
        config=config,
        configured_path=config.journal_path,
        path=path,
        journal_sha256=_sha256_file(path),
        observations=observations,
        successful_executions=tuple(
            event
            for event in events
            if event.event_type == "execution"
            and _successful_hima_action(event, semantics) is not None
        ),
    )


def _previous_actions_for_observation(
    source: _LoadedSource,
    observation_event: StoredEvent,
    observation: ObservationEnvelope,
    semantics: _CorpusRaceSemantics,
) -> list[str]:
    earliest_loop = max(
        0,
        observation.game_loop - _PREVIOUS_ACTION_WINDOW_GAME_LOOPS,
    )
    actions: list[tuple[int, int, str]] = []
    for execution_event in source.successful_executions:
        if (
            execution_event.run_id != observation.run_id
            or execution_event.episode_id != observation.episode_id
        ):
            continue
        normalized = _successful_hima_action(execution_event, semantics)
        if normalized is None:
            continue
        game_loop, action = normalized
        if game_loop < earliest_loop or game_loop > observation.game_loop:
            continue
        if (
            game_loop == observation.game_loop
            and execution_event.event_id >= observation_event.event_id
        ):
            continue
        actions.append((game_loop, execution_event.event_id, action))
    return [action for _, _, action in sorted(actions)]


def _successful_hima_action(
    event: StoredEvent,
    semantics: _CorpusRaceSemantics,
) -> tuple[int, str] | None:
    try:
        report = ExecutionReport.model_validate(event.payload)
    except ValueError:
        return None
    if not report.success or report.action_name is None:
        return None
    action = semantics.runtime_to_hima_short_action.get(report.action_name)
    if action is None:
        return None
    game_loops = [
        entry.game_loop for entry in report.primitive_trace if entry.game_loop is not None
    ]
    evidence = report.effect_evidence
    if evidence is not None:
        game_loops.extend(
            game_loop
            for game_loop in (
                evidence.dispatch_game_loop,
                evidence.accepted_game_loop,
                evidence.confirmed_game_loop,
            )
            if game_loop is not None
        )
    if not game_loops:
        return None
    return max(game_loops), action


def _manifest_source(source: _LoadedSource) -> PolicyCorpusManifestSource:
    run_ids = sorted({event.run_id for event in source.observations})
    episode_keys = sorted(
        {_episode_key(event.run_id, event.episode_id) for event in source.observations}
    )
    return PolicyCorpusManifestSource(
        source_id=source.config.source_id,
        journal_path=source.configured_path,
        journal_sha256=source.journal_sha256,
        seed=source.config.seed,
        map_name=source.config.map_name,
        observation_count=len(source.observations),
        run_ids=run_ids,
        episode_keys=episode_keys,
    )


def _condition_phase_counts(
    fixtures: Sequence[PolicyObservationFixture],
) -> dict[PolicyFixtureStratum, dict[str, int]]:
    counts: dict[PolicyFixtureStratum, Counter[str]] = {
        PolicyFixtureStratum.BLOCKED: Counter(),
        PolicyFixtureStratum.IN_PROGRESS: Counter(),
    }
    for fixture in fixtures:
        stratum = fixture.primary_stratum
        if stratum not in counts:
            continue
        counts[stratum].update(fixture.phase_tags)
    return {
        condition: {phase.value: counts[condition][phase.value] for phase in _CONDITION_PHASES}
        for condition in counts
    }


def _load_manifest(path: Path) -> PolicyCorpusManifest:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise PolicyCorpusError("policy corpus manifest must contain a YAML mapping")
    return PolicyCorpusManifest.model_validate(payload)


def _fixtures_path(manifest_path: Path, manifest: PolicyCorpusManifest) -> Path:
    configured = Path(manifest.fixtures_file)
    return configured if configured.is_absolute() else manifest_path.parent / configured


def _read_fixtures(path: Path) -> list[PolicyObservationFixture]:
    fixtures: list[PolicyObservationFixture] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                fixtures.append(PolicyObservationFixture.model_validate_json(line))
    return fixtures


def _encode_fixtures(fixtures: Sequence[PolicyObservationFixture]) -> bytes:
    lines = [
        _canonical_json(fixture.model_dump(mode="json")).decode("utf-8") for fixture in fixtures
    ]
    return (("\n".join(lines) + "\n") if lines else "").encode()


def _aggregate_units(units: Sequence[UnitState]) -> list[dict[str, str | int]]:
    counts = Counter(
        (_canonical(unit.unit_type), _canonical(unit.status or "unknown")) for unit in units
    )
    return [
        {"unit_type": unit_type, "status": status, "count": count}
        for (unit_type, status), count in sorted(counts.items())
    ]


def _aggregate_production(items: Sequence[ProductionItem]) -> list[dict[str, str | int]]:
    counts = Counter((_canonical(item.name), min(10, int(item.progress * 10))) for item in items)
    return [
        {"name": name, "progress_tenth": progress, "count": count}
        for (name, progress), count in sorted(counts.items())
    ]


def _canonical(value: str) -> str:
    return value.strip().casefold().replace(" ", "_")


def _canonical_json(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _episode_key(run_id: str, episode_id: str) -> str:
    return f"{run_id}:{episode_id}"


def _expand_path(value: str, *, base_dir: Path | None) -> Path:
    expanded = Path(os.path.expandvars(value)).expanduser()
    if not expanded.is_absolute() and base_dir is not None:
        expanded = base_dir / expanded
    return expanded.resolve()
