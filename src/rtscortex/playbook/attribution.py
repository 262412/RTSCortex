"""Deterministic post-game attribution of strategic consequences."""

from __future__ import annotations

import hashlib
from collections.abc import Container, Sequence
from dataclasses import dataclass
from typing import Any

from rtscortex.contracts import EpisodeOutcome, EpisodeResult
from rtscortex.game_phase import GamePhase
from rtscortex.memory import StoredEvent
from rtscortex.playbook.models import (
    DecisionQuality,
    PlaybookRuleEffect,
    StrategicConditionSnapshot,
    StrategicConsequence,
    StrategicConsequenceType,
)

_THREAT_RANK = {"none": 0, "low": 1, "high": 2, "critical": 3}
_WORKERS = {"drone", "mule", "probe", "scv"}
_EXPANSION_ACTION = {
    "protoss": "BUILD NEXUS",
    "terran": "BUILD COMMANDCENTER",
    "zerg": "BUILD HATCHERY",
}
_PRODUCTION_TOKENS = {
    "BARRACKS",
    "FACTORY",
    "GATEWAY",
    "HATCHERY",
    "ROBOTICS",
    "STARGATE",
    "STARPORT",
}


@dataclass(frozen=True, slots=True)
class _SituationPoint:
    event_id: int
    step_id: int
    game_loop: int
    phase: GamePhase
    threat: str
    economy: str
    readiness: str
    own_value: int
    enemy_value: int
    own_units: int
    enemy_units: int
    own_bases: int
    production_capacity: int
    enemy_visible: bool
    force_known: bool
    bases_known: bool


@dataclass(frozen=True, slots=True)
class _ObservationPoint:
    event_id: int
    game_loop: int
    minerals: int
    army_supply: int
    average_army_health: float | None


@dataclass(frozen=True, slots=True)
class _DecisionPoint:
    event_id: int
    step_id: int
    game_loop: int
    command_id: str
    role: str | None
    semantic_action: str
    succeeded: bool


@dataclass(frozen=True, slots=True)
class _Trace:
    situations: tuple[_SituationPoint, ...]
    observations: tuple[_ObservationPoint, ...]
    decisions: tuple[_DecisionPoint, ...]


class StrategicConsequenceAttributor:
    """Extract bounded strategic lessons from terminal or explicitly censored episodes."""

    attributor_id = "deterministic-strategic-consequence-attributor"
    attributor_version = "1.0.0"
    response_window_game_loops = 112
    persistence_window_game_loops = 448
    strategic_window_game_loops = 896
    expansion_deadline_game_loops = 5_600

    def attribute(
        self,
        events: Sequence[StoredEvent],
        result: EpisodeResult,
        *,
        agent_race: str,
        opponent_race: str,
    ) -> tuple[StrategicConsequence, ...]:
        if result.outcome not in {
            EpisodeOutcome.VICTORY,
            EpisodeOutcome.DEFEAT,
            EpisodeOutcome.DRAW,
            EpisodeOutcome.TRUNCATED,
        }:
            return ()
        trace = _build_trace(events)
        if len(trace.situations) < 2:
            return ()
        detected = [
            *self._unanswered_threat(trace, result, agent_race, opponent_race),
            *self._delayed_expansion(trace, result, agent_race, opponent_race),
            *self._production_imbalance(trace, result, agent_race, opponent_race),
            *self._failed_timing_attack(trace, result, agent_race, opponent_race),
            *self._unnecessary_retreat(trace, result, agent_race, opponent_race),
            *self._unconverted_advantage(trace, result, agent_race, opponent_race),
            *self._successful_key_decisions(trace, result, agent_race, opponent_race),
        ]
        unique: dict[str, StrategicConsequence] = {}
        for consequence in detected:
            unique.setdefault(consequence.consequence_id, consequence)
        return tuple(
            sorted(
                unique.values(),
                key=lambda item: (
                    item.start_game_loop,
                    item.consequence_type.value,
                    item.consequence_id,
                ),
            )
        )

    def _unanswered_threat(
        self,
        trace: _Trace,
        result: EpisodeResult,
        agent_race: str,
        opponent_race: str,
    ) -> list[StrategicConsequence]:
        points = trace.situations
        for index, start in enumerate(points):
            if _THREAT_RANK[start.threat] < _THREAT_RANK["high"]:
                continue
            end = _first_after(
                points[index + 1 :],
                start.game_loop + self.response_window_game_loops,
            )
            if end is None or _THREAT_RANK[end.threat] < _THREAT_RANK["high"]:
                continue
            responses = [
                decision
                for decision in trace.decisions
                if decision.succeeded
                and decision.role in {"defense", "focus_fire", "retreat"}
                and start.game_loop <= decision.game_loop <= end.game_loop
            ]
            response_count = len(responses)
            response_summary = (
                "no defensive response was executed"
                if response_count == 0
                else f"{response_count} defensive responses were executed but did not resolve it"
            )
            return [
                self._make(
                    result,
                    agent_race,
                    opponent_race,
                    StrategicConsequenceType.THREAT_UNANSWERED,
                    DecisionQuality.STRATEGIC_ERROR,
                    PlaybookRuleEffect.PREFER,
                    start,
                    end,
                    role="defense",
                    semantic_action=None,
                    objective="Answer persistent high-priority threats before resuming macro play.",
                    explanation=(
                        f"Threat remained {start.threat}/{end.threat} for "
                        f"{end.game_loop - start.game_loop} loops; {response_summary}."
                    ),
                    evidence={
                        "duration_game_loops": end.game_loop - start.game_loop,
                        "start_threat": start.threat,
                        "end_threat": end.threat,
                        "executed_response_count": response_count,
                    },
                    confidence=0.9,
                    source_event_ids=(
                        start.event_id,
                        *(decision.event_id for decision in responses),
                        end.event_id,
                    ),
                )
            ]
        return []

    def _delayed_expansion(
        self,
        trace: _Trace,
        result: EpisodeResult,
        agent_race: str,
        opponent_race: str,
    ) -> list[StrategicConsequence]:
        expansion = _EXPANSION_ACTION.get(agent_race.casefold())
        if expansion is None:
            return []
        points = trace.situations
        for index, start in enumerate(points):
            if (
                not start.bases_known
                or start.game_loop < self.expansion_deadline_game_loops
                or start.own_bases > 1
            ):
                continue
            observation = _nearest_observation(trace.observations, start.game_loop)
            affordable = start.economy == "floating" or (
                observation is not None and observation.minerals >= 400
            )
            if not affordable:
                continue
            end = _first_after(
                points[index + 1 :],
                start.game_loop + self.persistence_window_game_loops,
            )
            if end is None or not end.bases_known or end.own_bases > 1:
                continue
            expansion_attempts = [
                decision
                for decision in trace.decisions
                if decision.succeeded
                and _action_key(decision.semantic_action) == _action_key(expansion)
                and start.game_loop <= decision.game_loop <= end.game_loop
            ]
            return [
                self._make(
                    result,
                    agent_race,
                    opponent_race,
                    StrategicConsequenceType.EXPANSION_DELAYED,
                    DecisionQuality.STRATEGIC_ERROR,
                    PlaybookRuleEffect.PREFER,
                    start,
                    end,
                    role="economy",
                    semantic_action=expansion,
                    objective="Convert available resources into a timely second base.",
                    explanation=(
                        f"The economy remained on {start.own_bases} base after loop "
                        f"{start.game_loop} while an expansion was affordable."
                    ),
                    evidence={
                        "deadline_game_loop": self.expansion_deadline_game_loops,
                        "minerals": None if observation is None else observation.minerals,
                        "own_bases": start.own_bases,
                        "expansion_action_executed": bool(expansion_attempts),
                    },
                    confidence=0.85,
                    source_event_ids=(
                        start.event_id,
                        *(decision.event_id for decision in expansion_attempts),
                        end.event_id,
                    ),
                )
            ]
        return []

    def _production_imbalance(
        self,
        trace: _Trace,
        result: EpisodeResult,
        agent_race: str,
        opponent_race: str,
    ) -> list[StrategicConsequence]:
        points = trace.situations
        for index, start in enumerate(points):
            expected_capacity = max(2, start.own_bases * 2)
            if (
                not start.bases_known
                or start.game_loop < self.expansion_deadline_game_loops
                or start.economy != "floating"
                or start.production_capacity >= expected_capacity
            ):
                continue
            end = _first_after(
                points[index + 1 :],
                start.game_loop + self.persistence_window_game_loops,
            )
            if (
                end is None
                or not end.bases_known
                or end.production_capacity >= expected_capacity
            ):
                continue
            production_attempts = [
                decision
                for decision in trace.decisions
                if decision.succeeded
                and decision.role == "production"
                and _is_production_structure(decision.semantic_action)
                and start.game_loop <= decision.game_loop <= end.game_loop
            ]
            return [
                self._make(
                    result,
                    agent_race,
                    opponent_race,
                    StrategicConsequenceType.PRODUCTION_IMBALANCE,
                    DecisionQuality.STRATEGIC_ERROR,
                    PlaybookRuleEffect.PREFER,
                    start,
                    end,
                    role="production",
                    semantic_action=None,
                    objective="Add production capacity before banking unusable resources.",
                    explanation=(
                        f"A floating economy sustained only {start.production_capacity} "
                        f"production structures; at least {expected_capacity} were expected."
                    ),
                    evidence={
                        "actual_capacity": start.production_capacity,
                        "expected_capacity": expected_capacity,
                        "own_bases": start.own_bases,
                        "production_structure_actions_executed": len(production_attempts),
                    },
                    confidence=0.85,
                    source_event_ids=(
                        start.event_id,
                        *(decision.event_id for decision in production_attempts),
                        end.event_id,
                    ),
                )
            ]
        return []

    def _failed_timing_attack(
        self,
        trace: _Trace,
        result: EpisodeResult,
        agent_race: str,
        opponent_race: str,
    ) -> list[StrategicConsequence]:
        for decision in trace.decisions:
            if not decision.succeeded or decision.role not in {"offense", "focus_fire"}:
                continue
            start = _latest_at_or_before(trace.situations, decision.game_loop)
            end = _first_after(
                trace.situations,
                decision.game_loop + self.persistence_window_game_loops // 2,
                maximum=decision.game_loop + self.strategic_window_game_loops,
            )
            if (
                start is None
                or end is None
                or not start.force_known
                or not end.force_known
                or start.readiness not in {"ready", "engaged"}
                or _THREAT_RANK[start.threat] > _THREAT_RANK["low"]
                or start.own_value < 400
            ):
                continue
            own_loss = max(0, start.own_value - end.own_value)
            enemy_loss = max(0, start.enemy_value - end.enemy_value)
            collapsed = own_loss >= max(200, int(start.own_value * 0.3))
            unfavorable = enemy_loss * 4 < own_loss * 3
            if not collapsed or not unfavorable:
                continue
            return [
                self._make(
                    result,
                    agent_race,
                    opponent_race,
                    StrategicConsequenceType.TIMING_ATTACK_FAILED,
                    DecisionQuality.STRATEGIC_ERROR,
                    PlaybookRuleEffect.AVOID,
                    start,
                    end,
                    role=decision.role,
                    semantic_action=decision.semantic_action,
                    objective="Delay the timing attack until the projected trade is favorable.",
                    explanation=(
                        f"The attack lost {own_loss} estimated value while removing only "
                        f"{enemy_loss} within {end.game_loop - start.game_loop} loops."
                    ),
                    evidence={
                        "command_id": decision.command_id,
                        "own_value_loss": own_loss,
                        "enemy_value_loss": enemy_loss,
                    },
                    confidence=0.9,
                    source_event_ids=(start.event_id, decision.event_id, end.event_id),
                )
            ]
        return []

    def _unnecessary_retreat(
        self,
        trace: _Trace,
        result: EpisodeResult,
        agent_race: str,
        opponent_race: str,
    ) -> list[StrategicConsequence]:
        for decision in trace.decisions:
            if not decision.succeeded or decision.role != "retreat":
                continue
            start = _latest_at_or_before(trace.situations, decision.game_loop)
            observation = _nearest_observation(trace.observations, decision.game_loop)
            advantaged = (
                start is not None
                and start.force_known
                and start.enemy_visible
                and start.enemy_value > 0
                and start.own_value >= int(start.enemy_value * 1.25)
            )
            if (
                start is None
                or observation is None
                or observation.average_army_health is None
                or start.readiness not in {"ready", "engaged"}
                or _THREAT_RANK[start.threat] > _THREAT_RANK["low"]
                or observation.average_army_health < 0.8
                or not advantaged
            ):
                continue
            return [
                self._make(
                    result,
                    agent_race,
                    opponent_race,
                    StrategicConsequenceType.UNNECESSARY_RETREAT,
                    DecisionQuality.STRATEGIC_ERROR,
                    PlaybookRuleEffect.AVOID,
                    start,
                    start,
                    role="retreat",
                    semantic_action=decision.semantic_action,
                    objective="Hold or continue pressure while the healthy army has a clear edge.",
                    explanation=(
                        "A retreat was executed with low threat, healthy units, and at least "
                        "a 25% observed force-value advantage."
                    ),
                    evidence={
                        "command_id": decision.command_id,
                        "average_army_health": observation.average_army_health,
                        "own_value": start.own_value,
                        "enemy_value": start.enemy_value,
                    },
                    confidence=0.9,
                    source_event_ids=(start.event_id, decision.event_id, observation.event_id),
                )
            ]
        return []

    def _unconverted_advantage(
        self,
        trace: _Trace,
        result: EpisodeResult,
        agent_race: str,
        opponent_race: str,
    ) -> list[StrategicConsequence]:
        if result.outcome is EpisodeOutcome.VICTORY:
            return []
        points = trace.situations
        for index, start in enumerate(points):
            advantaged = (
                start.force_known
                and start.enemy_visible
                and start.own_units >= 6
                and start.enemy_value > 0
                and start.own_value >= int(start.enemy_value * 1.5)
                and start.own_value - start.enemy_value >= 400
            )
            if (
                not advantaged
                or start.readiness not in {"ready", "engaged"}
                or _THREAT_RANK[start.threat] > _THREAT_RANK["low"]
            ):
                continue
            end = _first_after(
                points[index + 1 :],
                start.game_loop + self.strategic_window_game_loops,
            )
            if (
                end is None
                or not end.force_known
                or end.enemy_value <= 0
                or end.own_value < int(end.enemy_value * 1.5)
                or end.own_value - end.enemy_value < 400
            ):
                continue
            offensive_actions = [
                decision
                for decision in trace.decisions
                if decision.succeeded
                and decision.role in {"offense", "focus_fire"}
                and start.game_loop <= decision.game_loop <= end.game_loop
            ]
            return [
                self._make(
                    result,
                    agent_race,
                    opponent_race,
                    StrategicConsequenceType.ADVANTAGE_NOT_CONVERTED,
                    DecisionQuality.STRATEGIC_ERROR,
                    PlaybookRuleEffect.PREFER,
                    start,
                    end,
                    role="offense",
                    semantic_action=None,
                    objective=(
                        "Convert a verified army advantage into pressure or objective damage."
                    ),
                    explanation=(
                        f"A {start.own_value - start.enemy_value} value advantage persisted for "
                        f"{end.game_loop - start.game_loop} loops without offensive execution."
                    ),
                    evidence={
                        "own_value": start.own_value,
                        "enemy_value": start.enemy_value,
                        "duration_game_loops": end.game_loop - start.game_loop,
                        "offensive_actions_executed": len(offensive_actions),
                    },
                    confidence=0.85,
                    source_event_ids=(
                        start.event_id,
                        *(decision.event_id for decision in offensive_actions),
                        end.event_id,
                    ),
                )
            ]
        return []

    def _successful_key_decisions(
        self,
        trace: _Trace,
        result: EpisodeResult,
        agent_race: str,
        opponent_race: str,
    ) -> list[StrategicConsequence]:
        if result.outcome is not EpisodeOutcome.VICTORY:
            return []
        ranked: list[tuple[int, StrategicConsequence]] = []
        for decision in trace.decisions:
            if not decision.succeeded or decision.role is None:
                continue
            start = _latest_at_or_before(trace.situations, decision.game_loop)
            end = _first_after(
                trace.situations,
                decision.game_loop + self.persistence_window_game_loops // 2,
                maximum=decision.game_loop + self.strategic_window_game_loops,
            )
            if start is None or end is None:
                continue
            score, evidence = _verified_advantage(decision, start, end)
            if score <= 0:
                continue
            ranked.append(
                (
                    score,
                    self._make(
                        result,
                        agent_race,
                        opponent_race,
                        StrategicConsequenceType.SUCCESSFUL_KEY_DECISION,
                        DecisionQuality.ADVANTAGE_GAINED,
                        PlaybookRuleEffect.PREFER,
                        start,
                        end,
                        role=decision.role,
                        semantic_action=decision.semantic_action,
                        objective="Repeat this verified decision in a matching strategic state.",
                        explanation=(
                            f"{decision.semantic_action} was followed by a measurable strategic "
                            "advantage and the episode ended in victory."
                        ),
                        evidence={"command_id": decision.command_id, **evidence},
                        confidence=0.9,
                        source_event_ids=(start.event_id, decision.event_id, end.event_id),
                    ),
                )
            )
        ranked.sort(key=lambda item: (-item[0], item[1].start_game_loop, item[1].consequence_id))
        return [item[1] for item in ranked[:3]]

    def _make(
        self,
        result: EpisodeResult,
        agent_race: str,
        opponent_race: str,
        consequence_type: StrategicConsequenceType,
        quality: DecisionQuality,
        effect: PlaybookRuleEffect,
        start: _SituationPoint,
        end: _SituationPoint,
        *,
        role: str | None,
        semantic_action: str | None,
        objective: str,
        explanation: str,
        evidence: dict[str, object],
        confidence: float,
        source_event_ids: tuple[int, ...] | None = None,
    ) -> StrategicConsequence:
        identity = "|".join(
            (
                result.run_id,
                result.episode_id,
                consequence_type.value,
                str(start.game_loop),
                role or "",
                semantic_action or "",
            )
        )
        digest = hashlib.sha256(identity.encode()).hexdigest()
        return StrategicConsequence(
            consequence_id=f"consequence:{digest}",
            run_id=result.run_id,
            episode_id=result.episode_id,
            consequence_type=consequence_type,
            quality=quality,
            effect=effect,
            role=role,  # type: ignore[arg-type]
            semantic_action=semantic_action,
            objective=objective,
            start_game_loop=start.game_loop,
            end_game_loop=end.game_loop,
            source_event_ids=source_event_ids or (start.event_id, end.event_id),
            condition=StrategicConditionSnapshot(
                phase=start.phase,
                threat_level=start.threat,  # type: ignore[arg-type]
                economy_status=start.economy,  # type: ignore[arg-type]
                army_readiness=start.readiness,  # type: ignore[arg-type]
            ),
            explanation=explanation,
            evidence={
                **evidence,
                "agent_race": agent_race,
                "opponent_race": opponent_race,
                "attributor_id": self.attributor_id,
                "attributor_version": self.attributor_version,
            },
            confidence=(
                min(confidence, 0.7)
                if result.outcome is EpisodeOutcome.TRUNCATED
                else confidence
            ),
            censored=result.outcome is EpisodeOutcome.TRUNCATED,
        )


def _build_trace(events: Sequence[StoredEvent]) -> _Trace:
    situations: list[_SituationPoint] = []
    observations: list[_ObservationPoint] = []
    lineages: dict[str, tuple[StoredEvent, dict[str, Any]]] = {}
    executions: dict[str, bool] = {}
    for event in events:
        payload = event.payload
        if event.event_type == "situation_assessed":
            situations.append(_situation_point(event))
        elif event.event_type == "observation":
            observations.append(_observation_point(event))
        elif event.event_type == "command_lineage":
            command_id = str(payload.get("command_id") or "")
            if command_id:
                lineages[command_id] = (event, payload)
        elif event.event_type == "execution":
            command_id = str(payload.get("command_id") or "")
            if command_id:
                executions[command_id] = payload.get("success") is True
    decisions: list[_DecisionPoint] = []
    for command_id, (event, payload) in lineages.items():
        lineage = _object(payload.get("lineage"))
        loop = _integer(lineage.get("selected_game_loop"), default=event.step_id)
        decisions.append(
            _DecisionPoint(
                event_id=event.event_id,
                step_id=event.step_id,
                game_loop=loop,
                command_id=command_id,
                role=_text(lineage.get("responsibility")),
                semantic_action=str(payload.get("semantic_action") or "unknown"),
                succeeded=executions.get(command_id, False),
            )
        )
    return _Trace(
        situations=tuple(sorted(situations, key=lambda point: point.game_loop)),
        observations=tuple(sorted(observations, key=lambda point: point.game_loop)),
        decisions=tuple(sorted(decisions, key=lambda point: (point.game_loop, point.command_id))),
    )


def _situation_point(event: StoredEvent) -> _SituationPoint:
    payload = event.payload
    own = _object(payload.get("own_force"))
    enemy = _object(payload.get("visible_enemy_force"))
    bases = _object(payload.get("bases"))
    scouting = _object(payload.get("scouting"))
    return _SituationPoint(
        event_id=event.event_id,
        step_id=event.step_id,
        game_loop=_integer(payload.get("game_loop"), default=event.step_id),
        phase=_phase(payload.get("phase")),
        threat=_choice(payload.get("threat_level"), _THREAT_RANK, "none"),
        economy=_choice(
            payload.get("economy_status"),
            {"constrained", "stable", "floating"},
            "stable",
        ),
        readiness=_choice(
            payload.get("army_readiness"),
            {"empty", "forming", "ready", "engaged"},
            "forming",
        ),
        own_value=_integer(own.get("estimated_resource_value")),
        enemy_value=_integer(enemy.get("estimated_resource_value")),
        own_units=_integer(own.get("total_units")),
        enemy_units=_integer(enemy.get("total_units")),
        own_bases=_integer(bases.get("own_base_count"), default=1),
        production_capacity=_integer(bases.get("own_production_capacity")),
        enemy_visible=bool(scouting.get("enemy_visible", enemy.get("total_units", 0))),
        force_known=bool(own) and bool(enemy),
        bases_known=bool(bases),
    )


def _observation_point(event: StoredEvent) -> _ObservationPoint:
    state = _object(event.payload.get("state"))
    economy = _object(state.get("economy"))
    own_units = state.get("own_units")
    army_health: list[float] = []
    if isinstance(own_units, list):
        for raw in own_units:
            unit = _object(raw)
            if str(unit.get("unit_type") or "").casefold() in _WORKERS:
                continue
            health = unit.get("health_fraction")
            if isinstance(health, (int, float)):
                army_health.append(float(health))
    return _ObservationPoint(
        event_id=event.event_id,
        game_loop=_integer(event.payload.get("game_loop"), default=event.step_id),
        minerals=_integer(economy.get("minerals")),
        army_supply=_integer(economy.get("army_supply")),
        average_army_health=(sum(army_health) / len(army_health) if army_health else None),
    )


def _verified_advantage(
    decision: _DecisionPoint,
    start: _SituationPoint,
    end: _SituationPoint,
) -> tuple[int, dict[str, object]]:
    if decision.role in {"defense", "focus_fire", "retreat"}:
        reduction = _THREAT_RANK[start.threat] - _THREAT_RANK[end.threat]
        if reduction >= 2 and end.own_value > 0:
            return 400 + reduction * 100, {"threat_reduction": reduction}
    if (
        decision.role == "economy"
        and start.bases_known
        and end.bases_known
        and end.own_bases > start.own_bases
    ):
        return 500, {"base_count_delta": end.own_bases - start.own_bases}
    if decision.role in {"production", "technology"}:
        capacity_delta = (
            end.production_capacity - start.production_capacity
            if start.bases_known and end.bases_known
            else 0
        )
        force_delta = (
            end.own_value - start.own_value
            if start.force_known and end.force_known
            else 0
        )
        if capacity_delta > 0 or force_delta >= 300:
            return 300 + capacity_delta * 100 + max(0, force_delta), {
                "production_capacity_delta": capacity_delta,
                "own_force_value_delta": force_delta,
            }
    if (
        decision.role in {"offense", "focus_fire"}
        and start.force_known
        and end.force_known
        and start.enemy_value > 0
    ):
        enemy_loss = start.enemy_value - end.enemy_value
        own_loss = max(0, start.own_value - end.own_value)
        if enemy_loss >= max(200, int(start.enemy_value * 0.3)) and enemy_loss > own_loss:
            return 400 + enemy_loss - own_loss, {
                "enemy_value_loss": enemy_loss,
                "own_value_loss": own_loss,
            }
    return 0, {}


def _first_after(
    points: tuple[_SituationPoint, ...] | list[_SituationPoint],
    minimum: int,
    *,
    maximum: int | None = None,
) -> _SituationPoint | None:
    return next(
        (
            point
            for point in points
            if point.game_loop >= minimum
            and (maximum is None or point.game_loop <= maximum)
        ),
        None,
    )


def _latest_at_or_before(
    points: tuple[_SituationPoint, ...],
    game_loop: int,
) -> _SituationPoint | None:
    return next((point for point in reversed(points) if point.game_loop <= game_loop), None)


def _nearest_observation(
    points: tuple[_ObservationPoint, ...],
    game_loop: int,
) -> _ObservationPoint | None:
    nearest = min(points, key=lambda point: abs(point.game_loop - game_loop), default=None)
    if nearest is None or abs(nearest.game_loop - game_loop) > 112:
        return None
    return nearest


def _is_production_structure(action: str) -> bool:
    normalized = _action_key(action)
    return normalized.startswith("BUILD") and any(
        token in normalized for token in _PRODUCTION_TOKENS
    )


def _action_key(value: str) -> str:
    return "".join(character for character in value.upper() if character.isalnum())


def _phase(value: object) -> GamePhase:
    try:
        return GamePhase(str(value))
    except ValueError:
        return GamePhase.EARLY


def _object(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _integer(value: object, *, default: int = 0) -> int:
    return int(value) if isinstance(value, (int, float)) else default


def _choice(value: object, choices: Container[str], default: str) -> str:
    return str(value) if isinstance(value, str) and value in choices else default
