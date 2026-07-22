from __future__ import annotations

from typing import Any

from rtscortex.contracts import EpisodeOutcome, EpisodeResult
from rtscortex.memory import StoredEvent
from rtscortex.playbook import (
    StrategicConsequenceAttributor,
    StrategicConsequenceType,
)


def _event(
    event_id: int,
    event_type: str,
    game_loop: int,
    payload: dict[str, Any],
) -> StoredEvent:
    return StoredEvent(
        event_id=event_id,
        run_id="run",
        episode_id="episode",
        step_id=game_loop,
        event_type=event_type,
        created_at="2026-01-01T00:00:00+00:00",
        payload={"game_loop": game_loop, **payload},
    )


def _situation(
    event_id: int,
    game_loop: int,
    *,
    phase: str = "combat",
    threat: str = "none",
    economy: str = "stable",
    readiness: str = "ready",
    own_value: int = 1_000,
    enemy_value: int = 600,
    own_units: int = 8,
    enemy_units: int = 5,
    bases: int = 2,
    production: int = 4,
    enemy_visible: bool = True,
) -> StoredEvent:
    return _event(
        event_id,
        "situation_assessed",
        game_loop,
        {
            "phase": phase,
            "threat_level": threat,
            "economy_status": economy,
            "army_readiness": readiness,
            "own_force": {
                "estimated_resource_value": own_value,
                "total_units": own_units,
            },
            "visible_enemy_force": {
                "estimated_resource_value": enemy_value,
                "total_units": enemy_units,
            },
            "bases": {
                "own_base_count": bases,
                "own_production_capacity": production,
            },
            "scouting": {"enemy_visible": enemy_visible},
        },
    )


def _observation(
    event_id: int,
    game_loop: int,
    *,
    minerals: int = 500,
    health: float = 1.0,
) -> StoredEvent:
    return _event(
        event_id,
        "observation",
        game_loop,
        {
            "state": {
                "economy": {"minerals": minerals, "army_supply": 12},
                "own_units": [
                    {"unit_type": "Stalker", "health_fraction": health},
                    {"unit_type": "Zealot", "health_fraction": health},
                ],
            }
        },
    )


def _decision(
    event_id: int,
    game_loop: int,
    *,
    command_id: str,
    role: str,
    action: str,
    succeeded: bool = True,
) -> list[StoredEvent]:
    return [
        _event(
            event_id,
            "command_lineage",
            game_loop,
            {
                "command_id": command_id,
                "semantic_action": action,
                "lineage": {
                    "selected_game_loop": game_loop,
                    "responsibility": role,
                },
            },
        ),
        _event(
            event_id + 1,
            "execution",
            game_loop + 1,
            {"command_id": command_id, "success": succeeded},
        ),
    ]


def _result(outcome: EpisodeOutcome = EpisodeOutcome.DEFEAT, *, seed: int = 0) -> EpisodeResult:
    return EpisodeResult(
        run_id="run",
        episode_id="episode",
        scenario="Simple64",
        seed=seed,
        outcome=outcome,
        steps=20_000,
    )


def _types(
    events: list[StoredEvent],
    outcome: EpisodeOutcome = EpisodeOutcome.DEFEAT,
) -> set[StrategicConsequenceType]:
    return {
        consequence.consequence_type
        for consequence in StrategicConsequenceAttributor().attribute(
            events,
            _result(outcome),
            agent_race="protoss",
            opponent_race="zerg",
        )
    }


def test_attribution_requires_a_completed_match() -> None:
    events = [
        _situation(1, 1_000, threat="high"),
        _situation(2, 1_200, threat="high"),
    ]

    assert _types(events, EpisodeOutcome.TRUNCATED) == set()


def test_attribution_does_not_invent_missing_force_or_base_facts() -> None:
    events = [
        _event(
            1,
            "situation_assessed",
            6_000,
            {
                "phase": "production",
                "threat_level": "none",
                "economy_status": "floating",
                "army_readiness": "ready",
            },
        ),
        _event(
            2,
            "situation_assessed",
            7_000,
            {
                "phase": "production",
                "threat_level": "none",
                "economy_status": "floating",
                "army_readiness": "ready",
            },
        ),
    ]

    assert _types(events) == set()


def test_attribution_detects_unanswered_persistent_threat() -> None:
    events = [
        _situation(1, 1_000, threat="high"),
        *_decision(
            2,
            1_040,
            command_id="defense-attempt",
            role="defense",
            action="Attack_Unit",
        ),
        _situation(4, 1_120, threat="critical"),
    ]

    consequences = StrategicConsequenceAttributor().attribute(
        events,
        _result(),
        agent_race="protoss",
        opponent_race="zerg",
    )

    threat = next(
        item
        for item in consequences
        if item.consequence_type is StrategicConsequenceType.THREAT_UNANSWERED
    )
    assert threat.evidence["executed_response_count"] == 1


def test_attribution_detects_affordable_delayed_expansion() -> None:
    events = [
        _observation(1, 5_600, minerals=650),
        _situation(2, 5_600, phase="production", bases=1, production=2),
        _situation(3, 6_050, phase="production", bases=1, production=2),
    ]

    assert StrategicConsequenceType.EXPANSION_DELAYED in _types(events)


def test_attribution_detects_floating_underproduction() -> None:
    events = [
        _situation(
            1,
            6_000,
            phase="production",
            economy="floating",
            bases=2,
            production=1,
        ),
        _situation(
            2,
            6_500,
            phase="production",
            economy="floating",
            bases=2,
            production=1,
        ),
    ]

    assert StrategicConsequenceType.PRODUCTION_IMBALANCE in _types(events)


def test_attribution_detects_failed_timing_attack() -> None:
    events = [
        _situation(1, 8_000, own_value=1_200, enemy_value=900),
        *_decision(
            2,
            8_020,
            command_id="attack",
            role="offense",
            action="Attack_Unit",
        ),
        _situation(4, 8_500, own_value=700, enemy_value=800, readiness="forming"),
    ]

    assert StrategicConsequenceType.TIMING_ATTACK_FAILED in _types(events)


def test_attribution_detects_unnecessary_retreat() -> None:
    events = [
        _situation(1, 9_000, own_value=1_200, enemy_value=800, threat="low"),
        _observation(2, 9_000, health=0.95),
        *_decision(
            3,
            9_020,
            command_id="retreat",
            role="retreat",
            action="Move_Minimap",
        ),
        _situation(5, 9_300, own_value=1_150, enemy_value=800, threat="low"),
    ]

    assert StrategicConsequenceType.UNNECESSARY_RETREAT in _types(events)


def test_attribution_detects_advantage_not_converted() -> None:
    events = [
        _situation(1, 10_000, own_value=1_600, enemy_value=700),
        *_decision(
            2,
            10_100,
            command_id="ineffective-pressure",
            role="offense",
            action="Attack_Unit",
        ),
        _situation(4, 10_900, own_value=1_500, enemy_value=700),
    ]

    consequences = StrategicConsequenceAttributor().attribute(
        events,
        _result(),
        agent_race="protoss",
        opponent_race="zerg",
    )

    unconverted = next(
        item
        for item in consequences
        if item.consequence_type is StrategicConsequenceType.ADVANTAGE_NOT_CONVERTED
    )
    assert unconverted.evidence["offensive_actions_executed"] == 1


def test_attribution_detects_verified_key_decision_in_victory() -> None:
    events = [
        _situation(1, 11_000, threat="high", own_value=1_000, enemy_value=900),
        *_decision(
            2,
            11_020,
            command_id="defend",
            role="defense",
            action="Attack_Unit",
        ),
        _situation(4, 11_500, threat="none", own_value=900, enemy_value=400),
    ]

    assert StrategicConsequenceType.SUCCESSFUL_KEY_DECISION in _types(
        events,
        EpisodeOutcome.VICTORY,
    )
