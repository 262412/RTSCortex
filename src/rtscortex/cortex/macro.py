"""Pure projection helpers for live HIMA macro-policy results."""

from __future__ import annotations

import json
from collections.abc import Iterable
from hashlib import sha256

from rtscortex.contracts import ObservationEnvelope
from rtscortex.cortex.models import MacroPlan, MacroStep, MacroStepStatus
from rtscortex.policy.hima.live import HIMALiveProposalResponse
from rtscortex.policy.hima.mapping import (
    HIMA_RUNTIME_MAPPINGS,
    HIMAMacroActionMapper,
    HIMAMacroMapping,
    hima_runtime_mappings,
)
from rtscortex.policy.hima.race_vocabulary import resolve_race_hima_action
from rtscortex.policy.hima.vocabulary import resolve_hima_action
from rtscortex.policy.models import (
    MacroPolicyProposal,
    PolicyActionAssessment,
    PolicyActionClassification,
    PolicyObservationFixture,
)
from rtscortex.progress import GoalProgressVerifier, GoalRequirementKind, GoalSpec
from rtscortex.races import PROTOSS_PROFILE_DATA, RaceProfileData

_MAPPINGS_BY_MACRO = {mapping.macro_action: mapping for mapping in HIMA_RUNTIME_MAPPINGS}
_RUNTIME_TO_HIMA_TOKEN = {
    runtime_action: action.upstream_name
    for mapping in HIMA_RUNTIME_MAPPINGS
    if (action := resolve_hima_action(mapping.macro_action)) is not None
    for runtime_action in mapping.runtime_actions
}
_MAX_STRATEGIC_GOAL_CHARACTERS = 240


def macro_plan_from_hima(
    response: HIMALiveProposalResponse,
    observation: ObservationEnvelope,
    ttl_game_loops: int,
    *,
    current_observation: ObservationEnvelope | None = None,
    profile: RaceProfileData = PROTOSS_PROFILE_DATA,
) -> MacroPlan:
    """Project one correlated HIMA response into a typed, immutable macro plan."""

    if ttl_game_loops < 1:
        raise ValueError("ttl_game_loops must be positive")
    if (
        response.run_id,
        response.episode_id,
        response.step_id,
        response.game_loop,
    ) != (
        observation.run_id,
        observation.episode_id,
        observation.step_id,
        observation.game_loop,
    ):
        raise ValueError("HIMA response does not match the source observation")

    projection_observation = current_observation or observation
    if (
        projection_observation.run_id,
        projection_observation.episode_id,
    ) != (observation.run_id, observation.episode_id):
        raise ValueError("current observation does not match the source episode")

    proposal = response.proposal
    assessments = HIMAMacroActionMapper(profile).assess(
        proposal,
        _live_fixture(projection_observation),
    )
    assessment_by_step = {
        (assessment.ordinal, assessment.source_action): assessment for assessment in assessments
    }
    mappings_by_macro = _mappings_by_macro(profile)
    progress_specs = {spec.name: spec for spec in profile.progress_action_specs}
    projected_counts: dict[tuple[GoalRequirementKind, str], int] = {}
    steps: list[MacroStep] = []
    for proposal_step in sorted(proposal.steps, key=lambda item: item.ordinal):
        repeat = proposal_step.repeat
        target_satisfied = False
        mapping = mappings_by_macro.get(proposal_step.canonical_action)
        if proposal_step.target_count is not None and mapping is not None:
            spec = next(
                (
                    progress_specs[action]
                    for action in mapping.runtime_actions
                    if action in progress_specs
                ),
                None,
            )
            if spec is not None:
                key = (spec.effect_kind, spec.effect_target.casefold())
                current = projected_counts.setdefault(
                    key,
                    _state_effect_count(
                        projection_observation,
                        spec.effect_kind,
                        spec.effect_target,
                    ),
                )
                repeat = max(proposal_step.target_count - current, 0)
                target_satisfied = repeat == 0
                projected_counts[key] = max(current, proposal_step.target_count)
        macro_step = _macro_step(
            proposal_step.ordinal,
            proposal_step.canonical_action,
            max(1, repeat),
            assessment_by_step.get(
                (proposal_step.ordinal, proposal_step.canonical_action)
            ),
            mappings_by_macro=mappings_by_macro,
            managed_worker_action=f"TRAIN {profile.worker_type.upper()}",
            controller_managed_actions=frozenset(profile.controller_managed_actions),
        )
        if target_satisfied:
            macro_step = macro_step.model_copy(
                update={
                    "status": MacroStepStatus.OBSOLETE,
                    "reason": "cumulative_target_already_satisfied",
                }
            )
        steps.append(macro_step)
    metadata = proposal.generation_metadata
    raw_proposal = response.model_dump(mode="json")
    plan_digest = sha256(
        json.dumps(
            {
                "response": raw_proposal,
                "ttl_game_loops": ttl_game_loops,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return MacroPlan(
        plan_id=f"macro-plan:{plan_digest}",
        run_id=projection_observation.run_id,
        episode_id=projection_observation.episode_id,
        source_step_id=projection_observation.step_id,
        created_game_loop=projection_observation.game_loop,
        expires_game_loop=projection_observation.game_loop + ttl_game_loops,
        strategic_objective=_bounded_strategic_objective(proposal.strategic_objective),
        steps=steps,
        source_model_id=metadata.model_id if metadata is not None else "hima-live",
        source_model_revision=(metadata.model_revision if metadata is not None else "not_recorded"),
        adapter_version=proposal.adapter_version,
        parser_version=proposal.parser_version,
        vocabulary_version=proposal.vocabulary_version,
        raw_proposal=raw_proposal,
    )


def macro_goal_spec(
    plan: MacroPlan,
    observation: ObservationEnvelope,
    verifier: GoalProgressVerifier | None = None,
    profile: RaceProfileData = PROTOSS_PROFILE_DATA,
) -> GoalSpec | None:
    """Build the measurable prefix of a plan without crossing a hard blocker.

    The goal is intended to be created once when the plan is accepted.  Rebuilding it
    after effects have appeared would move the verifier's observation baselines.
    """

    _validate_plan_episode(plan, observation)
    parse_blocker_ordinal = _hard_parse_blocker_ordinal(plan, observation, profile)
    action_names: list[str] = []
    for step in sorted(plan.steps, key=lambda item: item.ordinal):
        if parse_blocker_ordinal is not None and step.ordinal >= parse_blocker_ordinal:
            break
        if step.status is MacroStepStatus.BLOCKED:
            break
        if step.status is MacroStepStatus.OBSOLETE:
            continue
        if not step.runtime_actions:
            # The sole soft unsupported action is TRAIN PROBE, which RTSCortex manages
            # automatically.  Other unsupported actions are marked BLOCKED above.
            continue
        action_names.extend([step.runtime_actions[0]] * step.repeat)
    if not action_names:
        return None
    progress_verifier = verifier or GoalProgressVerifier(profile.progress_action_specs)
    return progress_verifier.goal_from_action_names(
        goal_id=f"{plan.plan_id}:goal",
        strategic_goal=plan.strategic_objective,
        action_names=action_names,
        observation=observation,
    )


def _hard_parse_blocker_ordinal(
    plan: MacroPlan,
    observation: ObservationEnvelope,
    profile: RaceProfileData,
) -> int | None:
    """Return the first parser failure encoded in the source HIMA response."""

    proposal_payload = plan.raw_proposal.get("proposal")
    if proposal_payload is None:
        selected = plan.raw_proposal.get("selected")
        if isinstance(selected, dict):
            proposal_payload = selected.get("proposal")
    if not isinstance(proposal_payload, dict):
        return None
    proposal = MacroPolicyProposal.model_validate(proposal_payload)
    assessments = HIMAMacroActionMapper(profile).assess(
        proposal,
        _live_fixture(observation),
    )
    return min(
        (
            assessment.ordinal
            for assessment in assessments
            if assessment.classification is PolicyActionClassification.PARSE_ERROR
        ),
        default=None,
    )


def runtime_frontier(
    proposal: MacroPolicyProposal,
    observation: ObservationEnvelope,
    previous_actions: Iterable[str] = (),
    profile: RaceProfileData = PROTOSS_PROFILE_DATA,
) -> PolicyActionAssessment | None:
    """Return the current dependency-safe Runtime frontier for a HIMA proposal.

    Managed Probe production and already-obsolete mapped steps are transparent.  A
    parse error or any other unsupported HIMA action is a hard blocker and cannot be
    skipped merely because a later mapped action is legal.
    """

    fixture = _live_fixture(observation, previous_actions=previous_actions)
    assessments = HIMAMacroActionMapper(profile).assess(proposal, fixture)
    mapped_frontier = next(
        (assessment for assessment in assessments if assessment.is_runtime_frontier),
        None,
    )
    hard_blockers = [
        assessment
        for assessment in assessments
        if assessment.classification is PolicyActionClassification.PARSE_ERROR
        or (
            assessment.classification is PolicyActionClassification.UNSUPPORTED_BY_RUNTIME
            and assessment.reason_code != "managed_automatically"
        )
    ]
    earliest_blocker = min(
        hard_blockers,
        key=lambda item: (item.ordinal, item.source_action),
        default=None,
    )
    if earliest_blocker is not None and (
        mapped_frontier is None or earliest_blocker.ordinal <= mapped_frontier.ordinal
    ):
        return earliest_blocker.model_copy(
            update={"is_runtime_frontier": True, "is_frontier": True}
        )
    return mapped_frontier


def hima_previous_action_for_runtime_action(
    runtime_action: str,
    profile: RaceProfileData = PROTOSS_PROFILE_DATA,
) -> str | None:
    """Return the exact official HIMA token for one confirmed Runtime action."""

    if profile is PROTOSS_PROFILE_DATA:
        return _RUNTIME_TO_HIMA_TOKEN.get(runtime_action)
    race = profile.race.value
    return next(
        (
            action.upstream_name
            for mapping in hima_runtime_mappings(race)
            if runtime_action in mapping.runtime_actions
            if (action := resolve_race_hima_action(mapping.macro_action, race=race)) is not None
        ),
        None,
    )


def hima_previous_actions_for_runtime_actions(
    runtime_actions: Iterable[str],
    profile: RaceProfileData = PROTOSS_PROFILE_DATA,
) -> list[str]:
    """Project supported Runtime actions to official HIMA tokens in input order."""

    return [
        token
        for runtime_action in runtime_actions
        if (token := hima_previous_action_for_runtime_action(runtime_action, profile)) is not None
    ]


def _macro_step(
    ordinal: int,
    semantic_action: str,
    repeat: int,
    assessment: PolicyActionAssessment | None,
    *,
    mappings_by_macro: dict[str, HIMAMacroMapping],
    managed_worker_action: str,
    controller_managed_actions: frozenset[str],
) -> MacroStep:
    mapping = mappings_by_macro.get(semantic_action)
    if mapping is None:
        managed = semantic_action == managed_worker_action
        return MacroStep(
            ordinal=ordinal,
            semantic_action=semantic_action,
            repeat=repeat,
            status=(MacroStepStatus.OBSOLETE if managed else MacroStepStatus.BLOCKED),
            reason="managed_automatically" if managed else "unsupported_by_runtime",
        )

    if mapping.runtime_actions and all(
        action in controller_managed_actions for action in mapping.runtime_actions
    ):
        return MacroStep(
            ordinal=ordinal,
            semantic_action=semantic_action,
            runtime_actions=list(mapping.runtime_actions),
            repeat=repeat,
            status=MacroStepStatus.OBSOLETE,
            reason="managed_automatically",
        )

    classification = assessment.classification if assessment is not None else None
    status = MacroStepStatus.PENDING
    if classification is PolicyActionClassification.MAPPED_DEFERRED:
        status = MacroStepStatus.DEFERRED
    elif classification is PolicyActionClassification.ILLEGAL_ACTION:
        status = MacroStepStatus.BLOCKED
    elif classification is PolicyActionClassification.OBSOLETE:
        status = MacroStepStatus.OBSOLETE
    return MacroStep(
        ordinal=ordinal,
        semantic_action=semantic_action,
        runtime_actions=list(mapping.runtime_actions),
        repeat=repeat,
        status=status,
        reason=assessment.reason_code if assessment is not None else None,
    )


def _mappings_by_macro(profile: RaceProfileData) -> dict[str, HIMAMacroMapping]:
    if profile is PROTOSS_PROFILE_DATA:
        return _MAPPINGS_BY_MACRO
    return {mapping.macro_action: mapping for mapping in hima_runtime_mappings(profile.race.value)}


def _state_effect_count(
    observation: ObservationEnvelope,
    effect_kind: GoalRequirementKind,
    target: str,
) -> int:
    canonical_target = target.casefold().replace("_", "")
    if effect_kind is GoalRequirementKind.STRUCTURE:
        values = observation.state.own_structures
    elif effect_kind is GoalRequirementKind.UNIT:
        values = observation.state.own_units
    else:
        return sum(
            upgrade.casefold().replace("_", "") == canonical_target
            for upgrade in observation.state.upgrades
        )
    return sum(
        unit.unit_type.casefold().replace("_", "") == canonical_target
        for unit in values
    )


def _live_fixture(
    observation: ObservationEnvelope,
    *,
    previous_actions: Iterable[str] = (),
) -> PolicyObservationFixture:
    return PolicyObservationFixture(
        fixture_id=(
            f"live:{observation.run_id}:{observation.episode_id}:step-{observation.step_id}"
        ),
        observation=observation,
        previous_actions=list(previous_actions),
    )


def _validate_plan_episode(
    plan: MacroPlan,
    observation: ObservationEnvelope,
) -> None:
    if (plan.run_id, plan.episode_id) != (
        observation.run_id,
        observation.episode_id,
    ):
        raise ValueError("macro plan and goal observation must share an episode")
    if observation.game_loop < plan.created_game_loop:
        raise ValueError("goal observation cannot predate the macro plan")


def _bounded_strategic_objective(value: str) -> str:
    """Normalize free-form HIMA prose to the public GoalSpec boundary."""

    objective = " ".join(value.split()) or "Follow HIMA macro proposal"
    if len(objective) <= _MAX_STRATEGIC_GOAL_CHARACTERS:
        return objective
    prefix = objective[: _MAX_STRATEGIC_GOAL_CHARACTERS - 3]
    boundary = prefix.rsplit(" ", 1)[0].rstrip(" ,.;:-")
    if not boundary:
        boundary = prefix.rstrip(" ,.;:-")
    return f"{boundary}..."
