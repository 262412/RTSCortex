"""Compile semantic Cortex intents into observation-bound executable candidates."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from typing import Any

from rtscortex.contracts import (
    ActionCommand,
    ActionSource,
    AvailableAction,
    CommandLifecycleSnapshot,
    ObservationEnvelope,
)
from rtscortex.cortex.models import (
    CandidateFeatures,
    CandidateSelection,
    CandidateSelectionStatus,
    CortexIntent,
    ExecutableCandidate,
    FastExecutorContext,
    IntentTargetKind,
    ReflexIntent,
    TacticalIntent,
)
from rtscortex.progress import GoalProgressReport


class CandidateCompilationError(ValueError):
    """An intent or selection cannot be safely bound to the current observation."""


class CandidateCompiler:
    """Enumerate only exact actors and arguments already exposed by the worker."""

    def compile(
        self,
        observation: ObservationEnvelope,
        intent: CortexIntent,
        *,
        goal_progress: GoalProgressReport | None = None,
        busy_actors: Iterable[str] = (),
        recent_commands: Sequence[CommandLifecycleSnapshot] = (),
    ) -> FastExecutorContext:
        _validate_intent_observation(intent, observation)
        busy = tuple(dict.fromkeys(busy_actors))
        busy_set = set(busy)
        advancing_actions = (
            frozenset(goal_progress.advancing_actions) if goal_progress is not None else frozenset()
        )
        observation_hash = observation_fingerprint(observation)
        candidates: list[ExecutableCandidate] = []
        seen_candidate_ids: set[str] = set()
        for action_rank, action_name in enumerate(intent.action_names):
            matching_actions = [
                action for action in observation.available_actions if action.name == action_name
            ]
            for action in matching_actions:
                actors = _candidate_actors(action, intent.actor_scopes)
                arguments = _candidate_arguments(action, intent, observation)
                for actor_rank, actor in enumerate(actors):
                    if actor in busy_set:
                        continue
                    for argument_rank, argument_set in enumerate(arguments):
                        candidate = _build_candidate(
                            observation,
                            intent,
                            observation_hash=observation_hash,
                            action_name=action_name,
                            actor=actor,
                            arguments=argument_set,
                            action_rank=action_rank,
                            actor_rank=actor_rank,
                            argument_rank=argument_rank,
                            compile_ordinal=len(candidates),
                            advances_goal=action_name in advancing_actions,
                        )
                        if candidate.candidate_id in seen_candidate_ids:
                            continue
                        if not _candidate_is_semantically_valid(candidate, observation):
                            continue
                        seen_candidate_ids.add(candidate.candidate_id)
                        candidates.append(candidate)
        return FastExecutorContext(
            observation=observation,
            intent=intent,
            goal_progress=goal_progress,
            busy_actors=list(busy),
            recent_commands=list(recent_commands),
            candidates=candidates,
        )

    def materialize(
        self,
        context: FastExecutorContext,
        selection: CandidateSelection,
        *,
        command_id: str,
    ) -> ActionCommand:
        if selection.intent_id != context.intent.intent_id:
            raise CandidateCompilationError("selection belongs to a different intent")
        if selection.status is not CandidateSelectionStatus.SELECTED:
            raise CandidateCompilationError("an abstained selection cannot become a command")
        current_fingerprint = observation_fingerprint(context.observation)
        selected = next(
            (
                candidate
                for candidate in context.candidates
                if candidate.candidate_id == selection.candidate_id
            ),
            None,
        )
        if selected is None:
            raise CandidateCompilationError("selected candidate is not present in the context")
        if selected.observation_fingerprint != current_fingerprint:
            raise CandidateCompilationError("selected candidate is stale")
        command = _command_from_candidate(
            selected,
            context.intent,
            command_id=command_id,
        )
        if not _candidate_is_semantically_valid(selected, context.observation):
            raise CandidateCompilationError("selected candidate is no longer semantically legal")
        return command


def observation_fingerprint(observation: ObservationEnvelope) -> str:
    """Hash decision-relevant observation state without timestamps or image references."""

    payload = observation.model_dump(
        mode="json",
        exclude={"observed_at", "image_uri"},
    )
    return _sha256(payload)


def _validate_intent_observation(
    intent: CortexIntent,
    observation: ObservationEnvelope,
) -> None:
    expected = (
        observation.run_id,
        observation.episode_id,
        observation.step_id,
        observation.game_loop,
    )
    actual = (
        intent.run_id,
        intent.episode_id,
        intent.step_id,
        intent.created_game_loop,
    )
    if actual != expected:
        raise CandidateCompilationError("intent is stale or belongs to another observation")


def _candidate_actors(action: AvailableAction, intent_actors: list[str]) -> list[str]:
    available = list(dict.fromkeys(action.actor_scopes))
    requested = list(dict.fromkeys(intent_actors))
    if requested:
        available_set = set(available)
        return [actor for actor in requested if actor in available_set]
    return available


def _candidate_arguments(
    action: AvailableAction,
    intent: CortexIntent,
    observation: ObservationEnvelope,
) -> list[list[Any]]:
    if not action.argument_names:
        return [[]]
    if action.argument_candidates is None:
        return []
    candidates = [list(arguments) for arguments in action.argument_candidates]
    if not isinstance(intent, TacticalIntent):
        return candidates
    if action.name == "Move_Minimap" and intent.target.kind is IntentTargetKind.RETREAT_REGION:
        return candidates[-1:]
    if (
        action.name == "Move_Minimap"
        and intent.target.kind is IntentTargetKind.ENEMY
        and len(candidates) > 1
    ):
        return candidates[:-1]
    if action.name != "Attack_Unit" or intent.target.unit_type is None:
        return candidates
    enemy_by_tag = {
        _normalize_tag(enemy.unit_id): enemy
        for enemy in observation.state.visible_enemies
        if enemy.unit_type == intent.target.unit_type
    }
    return sorted(
        (
            arguments
            for arguments in candidates
            if arguments and _normalize_tag(arguments[0]) in enemy_by_tag
        ),
        key=lambda arguments: (
            enemy_by_tag[_normalize_tag(arguments[0])].health_fraction,
            _normalize_tag(arguments[0]),
        ),
    )


def _build_candidate(
    observation: ObservationEnvelope,
    intent: CortexIntent,
    *,
    observation_hash: str,
    action_name: str,
    actor: str,
    arguments: list[Any],
    action_rank: int,
    actor_rank: int,
    argument_rank: int,
    compile_ordinal: int,
    advances_goal: bool,
) -> ExecutableCandidate:
    identity = {
        "run_id": observation.run_id,
        "episode_id": observation.episode_id,
        "step_id": observation.step_id,
        "game_loop": observation.game_loop,
        "intent_id": intent.intent_id,
        "action": action_name,
        "actor": actor,
        "arguments": arguments,
    }
    return ExecutableCandidate(
        candidate_id=f"candidate:{_sha256(identity)}",
        observation_fingerprint=observation_hash,
        intent_id=intent.intent_id,
        action_name=action_name,
        actor=actor,
        arguments=arguments,
        features=CandidateFeatures(
            action_rank=action_rank,
            actor_rank=actor_rank,
            argument_rank=argument_rank,
            compile_ordinal=compile_ordinal,
            advances_goal=advances_goal,
        ),
    )


def _candidate_is_semantically_valid(
    candidate: ExecutableCandidate,
    observation: ObservationEnvelope,
) -> bool:
    if candidate.action_name != "Attack_Unit":
        return True
    if (
        not (candidate.actor == "army" or candidate.actor.startswith("CombatGroup"))
        or not candidate.arguments
    ):
        return False
    target = _normalize_tag(candidate.arguments[0])
    return target in {_normalize_tag(enemy.unit_id) for enemy in observation.state.visible_enemies}


def _command_from_candidate(
    candidate: ExecutableCandidate,
    intent: CortexIntent,
    *,
    command_id: str,
) -> ActionCommand:
    source = ActionSource.REFLEX if isinstance(intent, ReflexIntent) else ActionSource.PLANNER
    return ActionCommand(
        command_id=command_id,
        actor=candidate.actor,
        name=candidate.action_name,
        arguments=candidate.arguments,
        priority=intent.priority,
        ttl_game_loops=intent.ttl_game_loops,
        created_game_loop=intent.created_game_loop,
        source=source,
    )


def _sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalize_tag(value: object) -> str:
    if isinstance(value, int) and not isinstance(value, bool):
        return hex(value)
    return str(value).casefold()
