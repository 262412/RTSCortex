from __future__ import annotations

from pathlib import Path

import pytest

from rtscortex.contracts import AvailableAction, EconomyState, UnitState
from rtscortex.policy.corpus import (
    _IN_PROGRESS_ACTIONS,
    _RUNTIME_TO_HIMA_SHORT_ACTION,
    PolicyCorpusBuildConfig,
    PolicyCorpusSourceConfig,
    _corpus_race_semantics,
    _observation_phase,
    _phase_goal,
    load_policy_corpus_config,
    state_fingerprint,
)
from rtscortex.policy.models import PolicyFixtureStratum
from rtscortex.progress import GoalProgressStatus, GoalProgressVerifier, GoalRequirementKind
from rtscortex.races import RaceId
from tests.helpers import make_observation


def test_state_fingerprint_excludes_volatile_observation_identity() -> None:
    first = make_observation(include_enemy=False)
    moved = first.model_copy(
        update={
            "run_id": "another-run",
            "episode_id": "another-episode",
            "step_id": 99,
            "game_loop": 9_999,
            "text_observation": "Different prose must not affect strategic state.",
            "state": first.state.model_copy(
                update={
                    "own_units": [
                        first.state.own_units[0].model_copy(
                            update={"unit_id": "another-tag", "position": (91.0, 17.0)}
                        )
                    ]
                }
            ),
        }
    )

    assert state_fingerprint(first) == state_fingerprint(moved)


def test_state_fingerprint_tracks_economic_bins_and_action_frontier() -> None:
    observation = make_observation(include_enemy=False)
    same_mineral_bin = observation.model_copy(
        update={
            "state": observation.state.model_copy(
                update={"economy": EconomyState(minerals=99, supply_used=2, supply_cap=15)}
            )
        }
    )
    next_mineral_bin = observation.model_copy(
        update={
            "state": observation.state.model_copy(
                update={"economy": EconomyState(minerals=100, supply_used=2, supply_cap=15)}
            )
        }
    )
    extra_action = observation.model_copy(update={"available_actions": []})

    assert state_fingerprint(observation) == state_fingerprint(same_mineral_bin)
    assert state_fingerprint(observation) != state_fingerprint(next_mineral_bin)
    assert state_fingerprint(observation) != state_fingerprint(extra_action)


def test_policy_corpus_config_rejects_impossible_episode_capacity() -> None:
    with pytest.raises(ValueError, match="episode coverage cannot satisfy"):
        PolicyCorpusBuildConfig(
            fixtures_per_stratum=8,
            max_per_episode_per_stratum=3,
            minimum_episodes_per_stratum=2,
            minimum_seeds=1,
            sources=[
                PolicyCorpusSourceConfig(
                    source_id="one",
                    journal_path="events.jsonl",
                    seed=0,
                )
            ],
        )


def test_policy_corpus_config_rejects_impossible_condition_phase_coverage() -> None:
    with pytest.raises(ValueError, match="condition phase coverage exceeds"):
        PolicyCorpusBuildConfig(
            fixtures_per_stratum=7,
            max_per_episode_per_stratum=7,
            minimum_episodes_per_stratum=1,
            minimum_seeds=1,
            minimum_condition_fixtures_per_phase=2,
            sources=[
                PolicyCorpusSourceConfig(
                    source_id="one",
                    journal_path="events.jsonl",
                    seed=0,
                )
            ],
        )


def test_combat_unit_presence_alone_does_not_define_production_phase() -> None:
    observation = make_observation(include_enemy=False)
    with_zealot = observation.model_copy(
        update={
            "state": observation.state.model_copy(
                update={
                    "own_units": [
                        *observation.state.own_units,
                        UnitState(
                            unit_id="zealot-1",
                            unit_type="Zealot",
                            alliance="self",
                            status="active",
                        ),
                    ]
                }
            )
        }
    )
    with_train_frontier = with_zealot.model_copy(
        update={
            "available_actions": [
                *with_zealot.available_actions,
                AvailableAction(name="Train_Zealot", actor_scopes=["gateway"]),
            ]
        }
    )

    assert _observation_phase(with_zealot) is PolicyFixtureStratum.EARLY
    assert _observation_phase(with_train_frontier) is PolicyFixtureStratum.PRODUCTION


def test_load_policy_corpus_config_is_strict(tmp_path: Path) -> None:
    config_path = tmp_path / "sources.yaml"
    config_path.write_text(
        """
format_version: "0.2"
corpus_id: test
protocol_version: "1.1"
fixtures_per_stratum: 1
minimum_game_loop_gap: 0
max_per_episode_per_stratum: 1
minimum_episodes_per_stratum: 1
minimum_seeds: 1
minimum_condition_fixtures_per_phase: 0
sources:
  - source_id: source-a
    journal_path: events.jsonl
    seed: 7
    map_name: Simple64
""".lstrip(),
        encoding="utf-8",
    )

    config = load_policy_corpus_config(config_path)

    assert config.corpus_id == "test"
    assert config.race is RaceId.PROTOSS
    assert config.sources[0].seed == 7


def test_zerg_worker_frontier_does_not_misclassify_opening_as_production() -> None:
    observation = make_observation(include_enemy=False).model_copy(
        update={
            "available_actions": [
                AvailableAction(name="Train_Drone", actor_scopes=["larva"]),
                AvailableAction(name="Train_Overlord", actor_scopes=["larva"]),
            ]
        }
    )

    assert _observation_phase(observation, RaceId.ZERG) is PolicyFixtureStratum.EARLY


def test_canonical_unit_under_attack_alert_defines_combat_phase() -> None:
    observation = make_observation(
        include_enemy=False,
        alerts=["unit_under_attack"],
    )

    assert _observation_phase(observation, RaceId.TERRAN) is PolicyFixtureStratum.COMBAT


def test_zerg_phase_classification_separates_technology_and_army_production() -> None:
    observation = make_observation(include_enemy=False)
    with_lair = observation.model_copy(
        update={
            "state": observation.state.model_copy(
                update={
                    "own_structures": [
                        UnitState(
                            unit_id="lair-1",
                            unit_type="Lair",
                            alliance="self",
                            status="active",
                        )
                    ]
                }
            ),
            "available_actions": [],
        }
    )
    with_roach_frontier = observation.model_copy(
        update={"available_actions": [AvailableAction(name="Train_Roach", actor_scopes=["larva"])]}
    )
    with_idle_spawning_pool = observation.model_copy(
        update={
            "state": observation.state.model_copy(
                update={
                    "own_structures": [
                        UnitState(
                            unit_id="pool-1",
                            unit_type="SpawningPool",
                            alliance="self",
                            status="active",
                        )
                    ]
                }
            ),
            "available_actions": [],
        }
    )

    assert _observation_phase(with_lair, RaceId.ZERG) is PolicyFixtureStratum.TECHNOLOGY
    assert _observation_phase(with_roach_frontier, RaceId.ZERG) is PolicyFixtureStratum.PRODUCTION
    assert (
        _observation_phase(with_idle_spawning_pool, RaceId.ZERG) is PolicyFixtureStratum.PRODUCTION
    )


def test_zerg_previous_action_names_come_from_the_active_race_profile() -> None:
    mapping = _corpus_race_semantics(RaceId.ZERG).runtime_to_hima_short_action

    assert mapping["Train_Drone"] == "Drone"
    assert mapping["Build_SpawningPool_Screen"] == "SpawningPool"
    assert "Build_Pylon_Screen" not in mapping


def test_zerg_idle_spawning_pool_can_form_a_blocked_production_fixture() -> None:
    semantics = _corpus_race_semantics(RaceId.ZERG)
    base = make_observation(include_enemy=False)
    observation = base.model_copy(
        update={
            "state": base.state.model_copy(
                update={
                    "economy": EconomyState(minerals=0, supply_used=15, supply_cap=15),
                    "own_structures": [
                        UnitState(
                            unit_id="pool-1",
                            unit_type="SpawningPool",
                            alliance="self",
                            status="active",
                        )
                    ],
                }
            ),
            "available_actions": [],
        }
    )
    verifier = GoalProgressVerifier(action_specs=semantics.profile.progress_action_specs)
    goal = _phase_goal(
        verifier,
        observation,
        PolicyFixtureStratum.PRODUCTION,
        RaceId.ZERG,
    )

    assert verifier.verify(observation, goal).status is GoalProgressStatus.BLOCKED


def test_future_corpora_recognize_oracle_phoenix_and_shield_battery_events() -> None:
    assert {
        action: _RUNTIME_TO_HIMA_SHORT_ACTION[action]
        for action in (
            "Train_Oracle",
            "Train_Phoenix",
            "Build_ShieldBattery_Screen",
        )
    } == {
        "Train_Oracle": "Oracle",
        "Train_Phoenix": "Phoenix",
        "Build_ShieldBattery_Screen": "ShieldBattery",
    }
    assert {
        (kind, target, action)
        for kind, target, action in _IN_PROGRESS_ACTIONS
        if action
        in {
            "Train_Oracle",
            "Train_Phoenix",
            "Build_ShieldBattery_Screen",
        }
    } == {
        (GoalRequirementKind.UNIT, "Oracle", "Train_Oracle"),
        (GoalRequirementKind.UNIT, "Phoenix", "Train_Phoenix"),
        (
            GoalRequirementKind.STRUCTURE,
            "ShieldBattery",
            "Build_ShieldBattery_Screen",
        ),
    }
