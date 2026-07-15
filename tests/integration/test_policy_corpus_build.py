from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from rtscortex.contracts import (
    ActionArgumentType,
    ActionSource,
    AvailableAction,
    EconomyState,
    ExecutionReport,
    ExecutionStage,
    ExecutionStatus,
    ObservationEnvelope,
    PrimitiveOrigin,
    PrimitiveTraceEntry,
    SC2State,
    UnitState,
)
from rtscortex.memory import EventStore
from rtscortex.policy.corpus import (
    CORPUS_STRATA,
    PolicyCorpusBuildConfig,
    PolicyCorpusInsufficientStates,
    PolicyCorpusSourceConfig,
    build_policy_corpus,
    load_policy_corpus,
    verify_policy_corpus,
)
from rtscortex.policy.models import PolicyFixtureStratum

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PINNED_CORPUS_MANIFEST = (
    PROJECT_ROOT / "benchmarks/policy/protoss_v0_2/manifest.yaml"
)


def test_build_load_and_verify_balanced_corpus(tmp_path: Path) -> None:
    first = _write_source(tmp_path, source_number=1)
    second = _write_source(tmp_path, source_number=2)
    config = _config(first, second)

    first_result = build_policy_corpus(config, tmp_path / "corpus-first")
    second_result = build_policy_corpus(config, tmp_path / "corpus-second")
    verification = verify_policy_corpus(first_result.manifest_path, verify_sources=True)
    fixtures = load_policy_corpus(first_result.manifest_path)

    assert verification.valid
    assert verification.errors == []
    assert verification.fixture_count == 12
    assert verification.stratum_counts == {stratum: 2 for stratum in CORPUS_STRATA}
    assert len({fixture.state_fingerprint for fixture in fixtures}) == 12
    assert all(fixture.source is not None for fixture in fixtures)
    assert all(fixture.observation.protocol_version == "1.1" for fixture in fixtures)
    assert any(fixture.previous_actions == ["Pylon"] for fixture in fixtures)
    in_progress = [
        fixture
        for fixture in fixtures
        if fixture.primary_stratum is PolicyFixtureStratum.IN_PROGRESS
    ]
    assert all("in_progress" in fixture.condition_tags for fixture in in_progress)
    assert first_result.fixtures_path.read_bytes() == second_result.fixtures_path.read_bytes()
    assert first_result.manifest.fixtures_sha256 == second_result.manifest.fixtures_sha256


def test_verifier_detects_tampered_fixture_file(tmp_path: Path) -> None:
    first = _write_source(tmp_path, source_number=1)
    second = _write_source(tmp_path, source_number=2)
    result = build_policy_corpus(_config(first, second), tmp_path / "corpus")
    result.fixtures_path.write_bytes(result.fixtures_path.read_bytes() + b"\n")

    verification = verify_policy_corpus(result.manifest_path)

    assert not verification.valid
    assert "fixtures SHA256 does not match manifest" in verification.errors


def test_builder_reports_missing_real_stratum_without_fabrication(tmp_path: Path) -> None:
    first = _write_source(tmp_path, source_number=1)
    second = _write_source(
        tmp_path,
        source_number=2,
        omitted=PolicyFixtureStratum.TECHNOLOGY,
    )

    with pytest.raises(PolicyCorpusInsufficientStates, match="technology: selected 1/2"):
        build_policy_corpus(_config(first, second), tmp_path / "corpus")


def test_builder_reports_missing_condition_phase_without_fabrication(
    tmp_path: Path,
) -> None:
    first = _write_source(tmp_path, source_number=1)
    second = _write_source(tmp_path, source_number=2)
    config = PolicyCorpusBuildConfig(
        corpus_id="insufficient-condition-phases",
        fixtures_per_stratum=8,
        minimum_game_loop_gap=224,
        max_per_episode_per_stratum=4,
        minimum_episodes_per_stratum=2,
        minimum_seeds=2,
        minimum_condition_fixtures_per_phase=2,
        sources=[
            PolicyCorpusSourceConfig(
                source_id="source-1",
                journal_path=str(first),
                seed=1,
            ),
            PolicyCorpusSourceConfig(
                source_id="source-2",
                journal_path=str(second),
                seed=2,
            ),
        ],
    )

    with pytest.raises(
        PolicyCorpusInsufficientStates,
        match=r"in_progress: phase technology selected 0/2",
    ):
        build_policy_corpus(config, tmp_path / "corpus")


def test_checked_in_corpus_has_exact_stage_aware_condition_coverage() -> None:
    verification = verify_policy_corpus(PINNED_CORPUS_MANIFEST)
    fixtures = load_policy_corpus(PINNED_CORPUS_MANIFEST)

    assert verification.valid
    assert verification.fixture_count == 48
    assert verification.stratum_counts == {stratum: 8 for stratum in CORPUS_STRATA}
    for condition in (
        PolicyFixtureStratum.BLOCKED,
        PolicyFixtureStratum.IN_PROGRESS,
    ):
        phase_counts = Counter(
            fixture.phase_tags[0]
            for fixture in fixtures
            if fixture.primary_stratum is condition
        )
        assert phase_counts == {
            PolicyFixtureStratum.EARLY.value: 2,
            PolicyFixtureStratum.TECHNOLOGY.value: 2,
            PolicyFixtureStratum.PRODUCTION.value: 2,
            PolicyFixtureStratum.COMBAT.value: 2,
        }


def _config(first: Path, second: Path) -> PolicyCorpusBuildConfig:
    return PolicyCorpusBuildConfig(
        corpus_id="test-protoss-v0.2",
        fixtures_per_stratum=2,
        minimum_game_loop_gap=224,
        max_per_episode_per_stratum=1,
        minimum_episodes_per_stratum=2,
        minimum_seeds=2,
        minimum_condition_fixtures_per_phase=0,
        sources=[
            PolicyCorpusSourceConfig(
                source_id="source-1",
                journal_path=str(first),
                seed=1,
            ),
            PolicyCorpusSourceConfig(
                source_id="source-2",
                journal_path=str(second),
                seed=2,
            ),
        ],
    )


def _write_source(
    tmp_path: Path,
    *,
    source_number: int,
    omitted: PolicyFixtureStratum | None = None,
) -> Path:
    run_id = f"run-{source_number}"
    episode_id = "episode-0"
    root = tmp_path / run_id
    store = EventStore(root / "events.sqlite3", root / "events.jsonl")
    try:
        for step_id, stratum in enumerate(CORPUS_STRATA):
            if stratum is omitted:
                continue
            observation = _observation(
                run_id=run_id,
                episode_id=episode_id,
                step_id=step_id,
                game_loop=step_id * 300,
                source_number=source_number,
                stratum=stratum,
            )
            store.append_event(
                run_id=run_id,
                episode_id=episode_id,
                step_id=step_id,
                event_type="observation",
                payload=observation,
            )
            if step_id == 0:
                store.append_event(
                    run_id=run_id,
                    episode_id=episode_id,
                    step_id=step_id,
                    event_type="execution",
                    payload=ExecutionReport(
                        run_id=run_id,
                        episode_id=episode_id,
                        step_id=step_id,
                        command_id=f"build-pylon-{source_number}",
                        success=True,
                        action_name="Build_Pylon_Screen",
                        actor="Builder/Probe-1",
                        source=ActionSource.PLANNER,
                        status=ExecutionStatus.SUCCEEDED,
                        execution_stage=ExecutionStage.EFFECT_VERIFICATION,
                        primitive_trace=[
                            PrimitiveTraceEntry(
                                function="Build_Pylon_screen",
                                origin=PrimitiveOrigin.TRANSLATOR,
                                ordinal=0,
                                total=1,
                                game_loop=50,
                                accepted=True,
                            )
                        ],
                    ),
                )
    finally:
        store.close()
    return root / "events.jsonl"


def _observation(
    *,
    run_id: str,
    episode_id: str,
    step_id: int,
    game_loop: int,
    source_number: int,
    stratum: PolicyFixtureStratum,
) -> ObservationEnvelope:
    nexus = UnitState(
        unit_id=f"nexus-{source_number}",
        unit_type="Nexus",
        alliance="self",
        status="idle",
    )
    pylon = UnitState(
        unit_id=f"pylon-{source_number}",
        unit_type="Pylon",
        alliance="self",
        status="idle",
    )
    gateway = UnitState(
        unit_id=f"gateway-{source_number}",
        unit_type="Gateway",
        alliance="self",
        status="idle",
    )
    structures = [nexus]
    units = [
        UnitState(
            unit_id=f"probe-{source_number}",
            unit_type="Probe",
            alliance="self",
            status="active",
        )
    ]
    enemies: list[UnitState] = []
    actions: list[AvailableAction] = [AvailableAction(name="No_Operation")]
    minerals = 50 + source_number * 50

    if stratum is PolicyFixtureStratum.EARLY:
        actions.append(_position_action("Build_Pylon_Screen", source_number))
    elif stratum is PolicyFixtureStratum.TECHNOLOGY:
        structures.extend([pylon, gateway])
        actions.append(_position_action("Build_CyberneticsCore_Screen", source_number))
    elif stratum is PolicyFixtureStratum.PRODUCTION:
        structures.extend([pylon, gateway])
        actions.append(AvailableAction(name="Train_Zealot", actor_scopes=["gateway"]))
    elif stratum is PolicyFixtureStratum.COMBAT:
        enemy = UnitState(
            unit_id=f"0x{source_number + 10:x}",
            unit_type="Zergling",
            alliance="enemy",
        )
        enemies.append(enemy)
        actions.append(
            AvailableAction(
                name="Attack_Unit",
                argument_names=["tag"],
                argument_types=[ActionArgumentType.TAG],
                actor_scopes=["army"],
                argument_candidates=[[enemy.unit_id]],
            )
        )
    elif stratum is PolicyFixtureStratum.BLOCKED:
        minerals = 0
    elif stratum is PolicyFixtureStratum.IN_PROGRESS:
        structures.append(
            UnitState(
                unit_id=f"assimilator-{source_number}",
                unit_type="Assimilator",
                alliance="self",
                status="constructing",
            )
        )
        minerals = 0

    return ObservationEnvelope(
        run_id=run_id,
        episode_id=episode_id,
        step_id=step_id,
        game_loop=game_loop,
        state=SC2State(
            economy=EconomyState(
                minerals=minerals,
                supply_used=10 + source_number,
                supply_cap=15,
                workers=10 + source_number,
            ),
            own_units=units,
            own_structures=structures,
            visible_enemies=enemies,
        ),
        available_actions=actions,
    )


def _position_action(name: str, source_number: int) -> AvailableAction:
    return AvailableAction(
        name=name,
        argument_names=["position"],
        argument_types=[ActionArgumentType.POSITION],
        actor_scopes=["builder"],
        argument_candidates=[[[10 + source_number, 20]]],
    )
