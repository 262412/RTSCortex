from __future__ import annotations

import pytest

from rtscortex.contracts import (
    ActionArgumentType,
    AvailableAction,
    ObservationEnvelope,
)
from rtscortex.cortex import (
    MacroStepStatus,
    hima_previous_action_for_runtime_action,
    hima_previous_actions_for_runtime_actions,
    macro_goal_spec,
    macro_plan_from_hima,
    runtime_frontier,
)
from rtscortex.policy.hima import HIMALiveProposalResponse, HIMAProposalParser
from rtscortex.policy.models import PolicyActionClassification
from tests.helpers import make_observation


def _pylon_observation() -> ObservationEnvelope:
    base = make_observation(include_enemy=False, game_loop=224)
    return base.model_copy(
        update={
            "available_actions": [
                AvailableAction(
                    name="Build_Pylon_Screen",
                    argument_names=["screen"],
                    argument_types=[ActionArgumentType.POSITION],
                    actor_scopes=["Builder/Probe-1"],
                    argument_candidates=[[[65, 90]]],
                )
            ]
        }
    )


def _response(raw_output: str) -> HIMALiveProposalResponse:
    observation = _pylon_observation()
    return HIMALiveProposalResponse(
        request_id="request-1",
        run_id=observation.run_id,
        episode_id=observation.episode_id,
        step_id=observation.step_id,
        game_loop=observation.game_loop,
        projection_hash="a" * 64,
        proposal=HIMAProposalParser().parse(raw_output),
    )


def test_macro_plan_projection_is_deterministic_and_preserves_step_semantics() -> None:
    observation = _pylon_observation()
    response = _response("Actions: ['Probe', 'Pylon', 'Gateway']")

    first = macro_plan_from_hima(response, observation, ttl_game_loops=448)
    repeated = macro_plan_from_hima(response, observation, ttl_game_loops=448)

    assert first == repeated
    assert first.plan_id.startswith("macro-plan:")
    assert first.created_game_loop == 224
    assert first.expires_game_loop == 672
    assert first.source_model_id == "hima-live"
    assert first.source_model_revision == "not_recorded"
    assert first.raw_proposal["request_id"] == "request-1"
    assert [step.semantic_action for step in first.steps] == [
        "TRAIN PROBE",
        "BUILD PYLON",
        "BUILD GATEWAY",
    ]
    assert first.steps[0].status is MacroStepStatus.OBSOLETE
    assert first.steps[0].reason == "managed_automatically"
    assert first.steps[1].status is MacroStepStatus.PENDING
    assert first.steps[1].runtime_actions == ["Build_Pylon_Screen"]
    assert first.steps[2].status is MacroStepStatus.PENDING
    assert first.steps[2].reason == "future_horizon_not_evaluated"


def test_macro_plan_rejects_uncorrelated_response_and_invalid_ttl() -> None:
    observation = _pylon_observation()
    response = _response("Actions: ['Pylon']")

    with pytest.raises(ValueError, match="positive"):
        macro_plan_from_hima(response, observation, ttl_game_loops=0)
    with pytest.raises(ValueError, match="source observation"):
        macro_plan_from_hima(
            response.model_copy(update={"step_id": observation.step_id + 1}),
            observation,
            ttl_game_loops=448,
        )


def test_macro_plan_bounds_long_hima_objective_before_goal_projection() -> None:
    observation = _pylon_observation()
    response = _response("Actions: ['Pylon']")
    long_objective = (
        "Transition into a balanced Protoss force with continuous production, "
        "air control, map vision, resilient static defense, a secure expansion, "
        "and enough reserves to replace losses while preserving pressure against "
        "every enemy counterattack and technology transition. "
        "Maintain this posture until the enemy economy collapses."
    )
    response = response.model_copy(
        update={
            "proposal": response.proposal.model_copy(
                update={"strategic_objective": long_objective}
            )
        }
    )

    plan = macro_plan_from_hima(response, observation, ttl_game_loops=448)
    goal = macro_goal_spec(plan, observation)

    assert len(plan.strategic_objective) <= 240
    assert plan.strategic_objective.endswith("...")
    assert goal is not None
    assert goal.strategic_goal == plan.strategic_objective


def test_runtime_frontier_skips_managed_probe() -> None:
    proposal = _response("Actions: ['Probe', 'Pylon']").proposal

    frontier = runtime_frontier(
        proposal,
        _pylon_observation(),
        previous_actions=("Probe",),
    )

    assert frontier is not None
    assert frontier.source_action == "BUILD PYLON"
    assert frontier.runtime_action == "Build_Pylon_Screen"
    assert frontier.classification is PolicyActionClassification.MAPPED_LEGAL_NOW
    assert frontier.is_runtime_frontier


def test_runtime_frontier_does_not_skip_unsupported_dependency() -> None:
    proposal = _response("Actions: ['Probe', 'Sentry', 'Pylon']").proposal

    frontier = runtime_frontier(proposal, _pylon_observation())

    assert frontier is not None
    assert frontier.source_action == "TRAIN SENTRY"
    assert (
        frontier.classification
        is PolicyActionClassification.UNSUPPORTED_BY_RUNTIME
    )
    assert frontier.reason_code == "not_implemented"
    assert frontier.runtime_action is None
    assert frontier.is_runtime_frontier


def test_runtime_frontier_treats_parse_error_as_hard_blocker() -> None:
    proposal = _response("Actions: ['Orthotomist', 'Pylon']").proposal

    frontier = runtime_frontier(proposal, _pylon_observation())

    assert frontier is not None
    assert frontier.source_action == "Orthotomist"
    assert frontier.classification is PolicyActionClassification.PARSE_ERROR
    assert frontier.reason_code == "unknown_action_token"


def test_macro_goal_uses_measurable_prefix_and_stops_at_hard_blocker() -> None:
    observation = _pylon_observation()
    response = _response(
        "Actions: ['Probe', 'Pylon', 'Pylon', 'Sentry', 'Gateway']"
    )
    plan = macro_plan_from_hima(response, observation, ttl_game_loops=448)

    goal = macro_goal_spec(plan, observation)

    assert goal is not None
    assert [item.action_name for item in goal.requirements] == [
        "Build_Pylon_Screen",
        "Build_Pylon_Screen",
    ]
    assert [item.count for item in goal.requirements] == [1, 2]
    assert plan.steps[3].status is MacroStepStatus.BLOCKED
    assert plan.steps[4].semantic_action == "BUILD GATEWAY"


def test_macro_goal_stops_before_unknown_token_parse_diagnostic() -> None:
    observation = _pylon_observation()
    plan = macro_plan_from_hima(
        _response("Actions: ['Pylon', 'Orthotomist', 'Gateway']"),
        observation,
        ttl_game_loops=448,
    )

    goal = macro_goal_spec(plan, observation)

    assert goal is not None
    assert [item.action_name for item in goal.requirements] == [
        "Build_Pylon_Screen"
    ]
    assert [step.ordinal for step in plan.steps] == [0, 2]
    assert plan.steps[1].semantic_action == "BUILD GATEWAY"


def test_macro_goal_treats_proposal_level_truncation_as_initial_blocker() -> None:
    observation = _pylon_observation()
    response = _response("Actions: ['Pylon']").model_copy(
        update={
            "proposal": HIMAProposalParser().parse(
                "Actions: ['Pylon']",
                truncated=True,
            )
        }
    )
    plan = macro_plan_from_hima(response, observation, ttl_game_loops=448)

    assert macro_goal_spec(plan, observation) is None


def test_macro_goal_accepts_later_state_but_rejects_another_episode() -> None:
    observation = _pylon_observation()
    plan = macro_plan_from_hima(
        _response("Actions: ['Pylon']"),
        observation,
        ttl_game_loops=448,
    )

    later = observation.model_copy(
        update={
            "step_id": observation.step_id + 1,
            "game_loop": observation.game_loop + 1,
        }
    )
    assert macro_goal_spec(plan, later) is not None

    with pytest.raises(ValueError, match="share an episode"):
        macro_goal_spec(
            plan,
            observation.model_copy(update={"episode_id": "another-episode"}),
        )


def test_runtime_actions_map_back_to_exact_official_hima_tokens() -> None:
    assert hima_previous_action_for_runtime_action("Build_Pylon_Screen") == "Pylon"
    assert (
        hima_previous_action_for_runtime_action("Research_WarpGate")
        == "WarpGateResearch"
    )
    assert hima_previous_action_for_runtime_action("Attack_Unit") is None
    assert hima_previous_actions_for_runtime_actions(
        ["Build_Pylon_Screen", "Attack_Unit", "Train_Oracle"]
    ) == ["Pylon", "Oracle"]
