"""SC2-native specialist runtime with a bounded, low-latency executor."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from pydantic import BaseModel

from rtscortex.config import ExperimentConfig
from rtscortex.contracts import (
    ActionBatch,
    ActionCommand,
    ActionSource,
    EpisodeResult,
    ExecutionReport,
    ExecutionStatus,
    IdleReason,
    ObservationEnvelope,
)
from rtscortex.contracts.interfaces import LLMProvider
from rtscortex.cortex import (
    CandidateCompiler,
    CandidateSelectionStatus,
    CommandLineage,
    CortexRole,
    DeterministicCandidateExecutor,
    DeterministicSituationAnalyzer,
    DeterministicTacticalAgent,
    MacroIntent,
    MacroPlan,
    MacroStep,
    MacroStepStatus,
    ReflexIntent,
    SituationAssessment,
    TacticalIntent,
    hima_previous_action_for_runtime_action,
    macro_goal_spec,
    macro_plan_from_hima,
    runtime_frontier,
)
from rtscortex.cortex.race_brain import (
    HIMAEnsemblePolicyClient,
    MacroPolicyHealth,
    MacroPolicyResponse,
    RaceBrainHealth,
    RaceBrainProposalResponse,
    RaceBrainStrategicContext,
    selected_hima_response,
)
from rtscortex.memory import EventStore
from rtscortex.playbook import (
    CortexPlaybookReviewer,
    LessonStatus,
    PlaybookContext,
    PlaybookQuery,
    PlaybookSelection,
    PlaybookStore,
)
from rtscortex.policy.hima import (
    HIMAInputContext,
    HIMALiveProposalResponse,
)
from rtscortex.policy.models import (
    MacroPolicyProposal,
    PolicyActionAssessment,
    PolicyActionClassification,
)
from rtscortex.progress import GoalProgressReport, GoalSpec
from rtscortex.reflex import ReflexEngine
from rtscortex.runtime.engine import (
    _ACTIONABLE_COMMAND_STATUSES,
    CommandStatus,
    RuntimeEngine,
)

_HIMA_PREVIOUS_ACTION_WINDOW_GAME_LOOPS = int(60 * 22.4)


def _deferred_frontier_requires_replan(frontier: PolicyActionAssessment) -> bool:
    reason = frontier.reason_code or ""
    return reason == "insufficient_supply" or reason.startswith("missing_prerequisite_")


def _macro_frontier_is_usable(frontier: PolicyActionAssessment | None) -> bool:
    if frontier is None:
        return False
    if frontier.classification is PolicyActionClassification.MAPPED_LEGAL_NOW:
        return True
    return (
        frontier.classification is PolicyActionClassification.MAPPED_DEFERRED
        and not _deferred_frontier_requires_replan(frontier)
    )


class MacroPolicyClient(Protocol):
    """The narrow transport surface required by the Cortex runtime."""

    async def health(self) -> MacroPolicyHealth: ...

    async def propose(
        self,
        context: HIMAInputContext,
        *,
        request_id: str | None = None,
    ) -> MacroPolicyResponse: ...

    async def close(self) -> None: ...


class MacroPolicySidecar(Protocol):
    """Lifecycle owner for a process-isolated macro specialist."""

    async def start(self) -> MacroPolicyHealth: ...

    async def restart(self) -> MacroPolicyHealth: ...

    async def close(self) -> None: ...


@dataclass(frozen=True)
class _PreparedCommand:
    command: ActionCommand
    lineage: CommandLineage
    semantic_action: str | None = None
    macro_step_ordinal: int | None = None


class CortexRuntimeEngine(RuntimeEngine):
    """Run specialist SC2 cognition while keeping execution deterministic and safe.

    The HIMA process may propose an ordered macro plan, but it never receives the
    action protocol and never dispatches a command.  Every command is rebuilt from
    the current observation's exact candidate domain, then passes through the same
    ProgressGuard, Validator, Arbiter, lifecycle, Bridge, and effect-verification
    path as the legacy runtime.
    """

    def __init__(
        self,
        *,
        config: ExperimentConfig,
        store: EventStore,
        provider: LLMProvider,
        macro_client: MacroPolicyClient | None = None,
        macro_sidecar: MacroPolicySidecar | None = None,
        macro_startup_failure: Exception | None = None,
        playbook_store: PlaybookStore | None = None,
        playbook_reviewer: CortexPlaybookReviewer | None = None,
    ) -> None:
        if config.agent.variant != "cortex":
            raise ValueError("CortexRuntimeEngine requires agent.variant=cortex")
        if macro_sidecar is not None and macro_client is None:
            raise ValueError("a macro sidecar requires its matching client")
        if (
            config.cortex.macro.kind in {"hima", "hima_ensemble"}
            and macro_client is None
            and macro_startup_failure is None
        ):
            raise ValueError("enabled cortex macro policy requires a live macro client")
        if macro_startup_failure is not None and config.cortex.macro.required:
            raise ValueError("required macro specialists cannot start in degraded mode")
        if config.cortex.macro.kind == "disabled" and macro_client is not None:
            raise ValueError("a macro client cannot be attached when the specialist is disabled")
        super().__init__(config=config, store=store, provider=provider)
        self.reflex = ReflexEngine(
            enabled=config.reflex.enabled,
            low_health_threshold=config.reflex.low_health_threshold,
        )
        self._macro_client = macro_client
        self._macro_sidecar = macro_sidecar
        self._macro_health: MacroPolicyHealth | None = None
        self._macro_startup_failure = macro_startup_failure
        self._macro_requests_suspended = macro_startup_failure is not None
        self._macro_health_announced_for: tuple[str, str] | None = None
        self._macro_task: asyncio.Task[MacroPolicyResponse] | None = None
        self._macro_source_observation: ObservationEnvelope | None = None
        self._macro_task_started_at: float | None = None
        self._macro_task_outcome_revision: int | None = None
        self._macro_recovery_task: asyncio.Task[None] | None = None
        self._macro_restart_attempts = 0
        self._macro_outcome_revision = 0
        self._next_macro_retry_game_loop: int | None = None
        self._macro_plan: MacroPlan | None = None
        self._macro_proposal: MacroPolicyProposal | None = None
        self._playbook_store = playbook_store
        self._playbook_reviewer = playbook_reviewer
        self._playbook_selection: PlaybookSelection | None = None
        self._playbook_selection_fingerprint: tuple[str, ...] | None = None
        self._macro_goal: GoalSpec | None = None
        self._macro_plan_frozen = False
        self._macro_inflight_command_id: str | None = None
        self._macro_command_steps: dict[str, tuple[str, str, int]] = {}
        self._command_lineages: dict[str, CommandLineage] = {}
        self._previous_hima_actions: list[tuple[int, str]] = []
        self._situation = DeterministicSituationAnalyzer(valid_for_game_loops=1)
        self._tactical = DeterministicTacticalAgent(
            retreat_health_threshold=(config.cortex.tactical.retreat_health_threshold),
            minimum_advance_army_supply=(config.cortex.tactical.minimum_advance_army_supply),
            reacquire_cooldown_game_loops=(config.cortex.tactical.reacquire_cooldown_game_loops),
        )
        self._candidate_compiler = CandidateCompiler()
        self._executor = DeterministicCandidateExecutor()

    async def start(self) -> None:
        """Load and validate the configured specialist before SC2 starts."""

        if self._macro_client is None:
            return
        try:
            if self._macro_sidecar is not None:
                self._macro_health = await self._macro_sidecar.start()
            else:
                self._macro_health = await self._macro_client.health()
        except Exception as error:
            if self.config.cortex.macro.required:
                raise
            self._macro_startup_failure = error
            if self._macro_sidecar is not None:
                await self._macro_sidecar.close()
            else:
                await self._macro_client.close()
            self._macro_sidecar = None
            self._macro_client = None

    async def tick(self, observation: ObservationEnvelope) -> ActionBatch:
        tick_started = time.perf_counter()
        await self._activate_episode(observation)
        self.store.append_event(
            run_id=observation.run_id,
            episode_id=observation.episode_id,
            step_id=observation.step_id,
            event_type="observation",
            payload=observation,
        )
        self._announce_specialist_health(observation)
        await self._collect_finished_macro(observation)
        self._note_alerts(observation)

        assessment = self._situation.assess(observation)
        self._record_cortex_event(observation, "situation_assessed", assessment)
        self._refresh_playbook(observation, assessment)

        goal_progress = self._macro_goal_progress(observation)
        self._record_goal_progress_if_changed(observation, goal_progress)

        if self._should_start_macro(observation):
            await self._begin_macro_cycle(observation, assessment)
            if (
                self.config.environment.pause_until_first_plan
                and self._macro_plan is None
                and self._macro_task is not None
            ):
                await self._wait_for_initial_macro(observation)
                goal_progress = self._macro_goal_progress(observation)
                self._record_goal_progress_if_changed(observation, goal_progress)

        prepared: list[_PreparedCommand] = []
        macro_prepared = self._prepare_macro_command(
            observation,
            assessment,
            goal_progress,
        )
        if macro_prepared is not None:
            prepared.append(macro_prepared)

        tactical_prepared = [
            item
            for intent in self._tactical.evaluate(observation, assessment)
            if (item := self._compile_intent(observation, intent)) is not None
        ]
        prepared.extend(tactical_prepared)

        reflex_started = time.perf_counter()
        raw_reflex = [
            command
            for command in self.reflex.evaluate(observation)
            if command.command_id not in self._command_states
        ]
        reflex_prepared = [
            item
            for command in raw_reflex
            if (item := self._prepare_reflex_command(observation, assessment, command)) is not None
        ]
        prepared.extend(reflex_prepared)
        reflex_latency_ms = (time.perf_counter() - reflex_started) * 1_000

        prepared_by_id = {item.command.command_id: item for item in prepared}
        macro_candidates = [
            item.command for item in prepared if item.lineage.source_role is CortexRole.MACRO
        ]
        tactical_candidates = [
            item.command for item in prepared if item.lineage.source_role is CortexRole.TACTICAL
        ]
        planner_candidates = [*macro_candidates, *tactical_candidates]
        reflex_candidates = [
            item.command for item in prepared if item.lineage.source_role is CortexRole.REFLEX
        ]
        for command in planner_candidates:
            self._transition_command(command, CommandStatus.PENDING, observation)

        guarded_macro = self.progress_guard.filter_commands(planner_candidates, goal_progress)
        guarded_reflex = self.progress_guard.filter_commands(reflex_candidates, goal_progress)
        rejected_commands = self._apply_validation_failures(
            [*guarded_macro.failures, *guarded_reflex.failures],
            observation,
        )
        (
            available_macro,
            available_reflex,
            busy_actor_rejections,
            busy_actor_candidates,
        ) = self._defer_busy_actor_commands(
            guarded_macro.accepted,
            guarded_reflex.accepted,
            observation,
        )
        rejected_commands.extend(busy_actor_rejections)
        candidate_outcome = self.validator.validate_candidates(
            [*available_macro, *available_reflex],
            observation,
        )
        rejected_commands.extend(
            self._apply_validation_failures(candidate_outcome.failures, observation)
        )
        arbitration = self.arbiter.arbitrate(
            [
                command
                for command in candidate_outcome.accepted
                if command.source is ActionSource.PLANNER
            ],
            [
                command
                for command in candidate_outcome.accepted
                if command.source is ActionSource.REFLEX
            ],
            game_loop=observation.game_loop,
        )
        outcome = self.validator.validate(arbitration.selected, observation)
        rejected_commands.extend(self._apply_validation_failures(outcome.failures, observation))
        accepted_commands = outcome.accepted
        accepted_ids = {command.command_id for command in accepted_commands}

        for command in planner_candidates:
            lifecycle = self._command_states.get(command.command_id)
            if (
                command.command_id not in accepted_ids
                and lifecycle is not None
                and lifecycle.status in _ACTIONABLE_COMMAND_STATUSES
                and command not in busy_actor_candidates
            ):
                self._transition_command(
                    command,
                    CommandStatus.SUPERSEDED,
                    observation,
                    reason="current observation candidate was not selected for dispatch",
                )

        for command in accepted_commands:
            prepared_command = prepared_by_id[command.command_id]
            self._record_command_lineage(observation, prepared_command)
            self._transition_command(command, CommandStatus.DISPATCHED, observation)
            if prepared_command.lineage.source_role is CortexRole.MACRO:
                plan_id = prepared_command.lineage.macro_plan_id
                if plan_id is None:
                    raise RuntimeError("macro command lineage is missing its plan ID")
                self._macro_inflight_command_id = command.command_id
                assert prepared_command.semantic_action is not None
                assert prepared_command.macro_step_ordinal is not None
                self._macro_command_steps[command.command_id] = (
                    plan_id,
                    prepared_command.semantic_action,
                    prepared_command.macro_step_ordinal,
                )
                self._set_macro_step_status(
                    prepared_command.macro_step_ordinal,
                    MacroStepStatus.DISPATCHED,
                    None,
                )

        idle_reason = None if accepted_commands else self._cortex_idle_reason()
        batch = ActionBatch(
            run_id=observation.run_id,
            episode_id=observation.episode_id,
            step_id=observation.step_id,
            decision_id=(
                f"{observation.run_id}:{observation.episode_id}:"
                f"{observation.step_id}:cortex-decision"
            ),
            strategic_goal=(
                "" if self._macro_plan is None else self._macro_plan.strategic_objective
            ),
            summary=self._decision_summary(goal_progress),
            planner_pending=self._macro_task is not None,
            idle_reason=idle_reason,
            commands=accepted_commands,
            rejected_commands=rejected_commands,
        )
        self.store.append_event(
            run_id=observation.run_id,
            episode_id=observation.episode_id,
            step_id=observation.step_id,
            event_type="decision",
            payload={
                "batch": batch.model_dump(mode="json"),
                "runtime_kind": "cortex",
                "reflex_latency_ms": reflex_latency_ms,
                "reflex_latency_target_ms": self.config.reflex.target_latency_ms,
                "tick_latency_ms": (time.perf_counter() - tick_started) * 1_000,
                "preemptions": [asdict(record) for record in arbitration.preemptions],
                "macro_candidates": [
                    command.model_dump(mode="json") for command in macro_candidates
                ],
                "tactical_candidates": [
                    command.model_dump(mode="json") for command in tactical_candidates
                ],
                "reflex_candidates": [
                    command.model_dump(mode="json") for command in reflex_candidates
                ],
                "busy_actor_candidates": [
                    command.model_dump(mode="json") for command in busy_actor_candidates
                ],
                "validated_candidates": [
                    command.model_dump(mode="json") for command in candidate_outcome.accepted
                ],
                "goal_progress": (
                    None if goal_progress is None else goal_progress.model_dump(mode="json")
                ),
            },
        )
        self._last_decision = batch
        for command in batch.commands:
            self._decision_by_command_id[command.command_id] = batch
        self._request_macro_if_exhausted()
        return batch

    async def _activate_episode(self, observation: ObservationEnvelope) -> None:
        episode_key = (observation.run_id, observation.episode_id)
        changed = self._episode_key != episode_key
        if changed and self._episode_key is not None:
            active_commands = [
                lifecycle.command.command_id
                for lifecycle in self._command_states.values()
                if lifecycle.status in {*_ACTIONABLE_COMMAND_STATUSES, CommandStatus.DISPATCHED}
            ]
            if active_commands:
                raise RuntimeError(
                    "cannot activate a new episode before end_episode terminalizes "
                    "active commands: " + ", ".join(sorted(active_commands))
                )
            if self._macro_requests_suspended and self.config.cortex.macro.required:
                raise RuntimeError(
                    "required HIMA macro specialist is suspended; restart the runtime "
                    "before activating a new episode"
                )
        if changed and self._episode_key is not None and self._macro_task is not None:
            # A sidecar request is single-flight and cancellation cannot stop GPU
            # inference. Drain it before resetting correlation state or starting the
            # next episode, then deliberately discard its episode-bound response.
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._macro_task
            self._macro_task = None
            self._macro_source_observation = None
            self._macro_task_started_at = None
            self._macro_task_outcome_revision = None
        await super()._activate_episode(observation)
        if not changed:
            return
        self._macro_health_announced_for = None
        self._macro_plan = None
        self._macro_proposal = None
        self._playbook_selection = None
        self._playbook_selection_fingerprint = None
        self._macro_goal = None
        self._macro_plan_frozen = False
        self._macro_inflight_command_id = None
        self._macro_command_steps = {}
        self._command_lineages = {}
        self._previous_hima_actions = []
        self._macro_source_observation = None
        self._macro_task_started_at = None
        self._macro_task_outcome_revision = None
        self._macro_outcome_revision = 0
        self._next_macro_retry_game_loop = None
        self._recover_cortex_episode(observation)

    def _recover_cortex_episode(self, observation: ObservationEnvelope) -> None:
        plan_event = self.store.last_event(
            observation.run_id,
            observation.episode_id,
            "macro_plan_accepted",
        )
        if plan_event is not None:
            plan_payload = plan_event.payload.get("plan", plan_event.payload)
            self._macro_plan = MacroPlan.model_validate(plan_payload)
            raw_response = self._macro_plan.raw_proposal
            if raw_response:
                if "selected" in raw_response:
                    coordinated = RaceBrainProposalResponse.model_validate(raw_response)
                    self._macro_proposal = coordinated.selected.proposal
                else:
                    response = HIMALiveProposalResponse.model_validate(raw_response)
                    self._macro_proposal = response.proposal
            goal_payload = plan_event.payload.get("goal_spec")
            if goal_payload is not None:
                self._macro_goal = GoalSpec.model_validate(goal_payload)
            self._last_plan_accepted_game_loop = int(
                plan_event.payload.get(
                    "accepted_game_loop",
                    self._macro_plan.created_game_loop,
                )
            )

        for event in self.store.events_of_type(
            observation.run_id,
            observation.episode_id,
            "command_lineage",
        ):
            payload = event.payload
            lineage = CommandLineage.model_validate(payload.get("lineage", payload))
            self._command_lineages[lineage.command_id] = lineage
            ordinal = payload.get("macro_step_ordinal")
            semantic = payload.get("semantic_action")
            if (
                lineage.macro_plan_id is not None
                and isinstance(ordinal, int)
                and isinstance(semantic, str)
            ):
                self._macro_command_steps[lineage.command_id] = (
                    lineage.macro_plan_id,
                    semantic,
                    ordinal,
                )

        recovered_macro_outcomes: list[tuple[str, bool]] = []
        for event in self.store.events_of_type(
            observation.run_id,
            observation.episode_id,
            "execution",
        ):
            report = ExecutionReport.model_validate(event.payload)
            recovered_lineage = self._command_lineages.get(report.command_id)
            if recovered_lineage is None or recovered_lineage.source_role is not CortexRole.MACRO:
                continue
            if report.status is ExecutionStatus.SUCCEEDED and report.action_name is not None:
                loop = self._execution_game_loop(report)
                token = hima_previous_action_for_runtime_action(report.action_name)
                if token is not None:
                    self._previous_hima_actions.append((loop, token))
            recovered_macro_outcomes.append(
                (report.command_id, report.status is ExecutionStatus.SUCCEEDED)
            )

        if self._macro_plan is not None:
            for command_id, succeeded in recovered_macro_outcomes:
                metadata = self._macro_command_steps.get(command_id)
                if metadata is not None and metadata[0] == self._macro_plan.plan_id:
                    self._advance_macro_step(
                        metadata[2],
                        succeeded=succeeded,
                        persist=False,
                    )
                    if not succeeded:
                        self._macro_plan_frozen = True
                        self._urgent_replan_requested = True

        dispatched: list[str] = []
        for lifecycle in self._command_states.values():
            if lifecycle.status is not CommandStatus.DISPATCHED:
                continue
            recovered_lineage = self._command_lineages.get(lifecycle.command.command_id)
            if recovered_lineage is not None and recovered_lineage.source_role is CortexRole.MACRO:
                dispatched.append(lifecycle.command.command_id)
        if len(dispatched) > 1:
            raise RuntimeError("recovered more than one in-flight macro command")
        self._macro_inflight_command_id = dispatched[0] if dispatched else None
        if self._macro_inflight_command_id is not None:
            metadata = self._macro_command_steps.get(self._macro_inflight_command_id)
            if (
                metadata is not None
                and self._macro_plan is not None
                and metadata[0] == self._macro_plan.plan_id
            ):
                self._set_macro_step_status(
                    metadata[2],
                    MacroStepStatus.DISPATCHED,
                    None,
                )

    def _announce_specialist_health(self, observation: ObservationEnvelope) -> None:
        episode_key = (observation.run_id, observation.episode_id)
        if self._macro_health_announced_for == episode_key:
            return
        if self._macro_startup_failure is not None:
            self._macro_health_announced_for = episode_key
            error = self._macro_startup_failure
            self._record_cortex_event(
                observation,
                "specialist_failed",
                {
                    "role": CortexRole.MACRO.value,
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "stage": "startup",
                    "fallback": "deterministic_reflex",
                },
            )
            return
        if self._macro_health is None:
            return
        self._macro_health_announced_for = episode_key
        self._record_cortex_event(
            observation,
            "specialist_ready",
            {
                "role": CortexRole.MACRO.value,
                "model_id": self._macro_health.model_id,
                "model_revision": self._macro_health.model_revision,
                "status": self._macro_health.status,
                "members": (
                    [member.model_dump(mode="json") for member in self._macro_health.members]
                    if isinstance(self._macro_health, RaceBrainHealth)
                    else None
                ),
            },
        )

    def _should_start_macro(self, observation: ObservationEnvelope) -> bool:
        if (
            self._macro_client is None
            or self._macro_task is not None
            or self._macro_requests_suspended
        ):
            return False
        if (
            self._next_macro_retry_game_loop is not None
            and observation.game_loop < self._next_macro_retry_game_loop
        ):
            return False
        if self._macro_inflight_command_id is not None:
            return False
        if self._urgent_replan_requested or self._macro_plan is None:
            return True
        if self._last_planner_started_game_loop is None:
            return True
        return (
            observation.game_loop - self._last_planner_started_game_loop
            >= self.config.cortex.macro.interval_game_loops
        )

    async def _begin_macro_cycle(
        self,
        observation: ObservationEnvelope,
        assessment: SituationAssessment,
    ) -> None:
        assert self._macro_client is not None
        macro_client = self._macro_client
        self._last_planner_started_game_loop = observation.game_loop
        self._next_macro_retry_game_loop = None
        self._urgent_replan_requested = False
        self._last_planner_failure = None
        self._macro_source_observation = observation
        self._macro_task_started_at = time.perf_counter()
        self._macro_task_outcome_revision = self._macro_outcome_revision
        previous_actions = self._recent_hima_actions(observation.game_loop)
        request_id = hashlib.sha256(
            (
                f"{observation.run_id}|{observation.episode_id}|"
                f"{observation.step_id}|{observation.game_loop}"
            ).encode()
        ).hexdigest()
        self._record_planner_event(
            observation,
            "planner_started",
            {
                "started_game_loop": observation.game_loop,
                "runtime_kind": "cortex",
                "specialist": CortexRole.MACRO.value,
                "previous_action_count": len(previous_actions),
            },
        )

        async def request() -> MacroPolicyResponse:
            context = HIMAInputContext(
                observation=observation,
                previous_actions=tuple(previous_actions),
            )
            if isinstance(macro_client, HIMAEnsemblePolicyClient):
                return await asyncio.wait_for(
                    macro_client.propose(
                        context,
                        request_id=request_id,
                        strategic_context=RaceBrainStrategicContext(
                            situation=assessment,
                            playbook=self._playbook_selection,
                        ),
                    ),
                    timeout=self.config.cortex.macro.timeout_seconds,
                )
            return await asyncio.wait_for(
                macro_client.propose(context, request_id=request_id),
                timeout=self.config.cortex.macro.timeout_seconds,
            )

        self._macro_task = asyncio.create_task(request())

    async def _wait_for_initial_macro(self, observation: ObservationEnvelope) -> None:
        assert self._macro_task is not None
        with contextlib.suppress(Exception):
            await self._macro_task
        await self._collect_finished_macro(observation)
        if self._macro_plan is None and self.config.cortex.macro.required:
            raise RuntimeError("required HIMA macro specialist failed before SC2 could start")

    async def _collect_finished_macro(self, observation: ObservationEnvelope) -> None:
        task = self._macro_task
        if task is None or not task.done():
            return
        if self._macro_inflight_command_id is not None:
            # Keep the completed proposal correlated with its source, but do not let
            # it replace the plan that still owns an in-flight command.
            return
        source_observation = self._macro_source_observation
        started_at = self._macro_task_started_at
        source_outcome_revision = self._macro_task_outcome_revision
        self._macro_task = None
        self._macro_source_observation = None
        self._macro_task_started_at = None
        self._macro_task_outcome_revision = None
        latency_ms = 0.0 if started_at is None else (time.perf_counter() - started_at) * 1_000
        revalidate_after_outcome = source_outcome_revision != self._macro_outcome_revision
        policy_response: MacroPolicyResponse | None = None
        response: HIMALiveProposalResponse | None = None
        try:
            policy_response = task.result()
            response = selected_hima_response(policy_response)
            if source_observation is None:
                raise RuntimeError("macro proposal completed without its source observation")
            if revalidate_after_outcome:
                self._record_cortex_event(
                    observation,
                    "macro_proposal_revalidated",
                    {
                        "role": CortexRole.MACRO.value,
                        "model_id": (
                            None if self._macro_health is None else self._macro_health.model_id
                        ),
                        "source_game_loop": source_observation.game_loop,
                        "current_game_loop": observation.game_loop,
                        "source_outcome_revision": source_outcome_revision,
                        "current_outcome_revision": self._macro_outcome_revision,
                    },
                )
            plan = macro_plan_from_hima(
                response,
                source_observation,
                self.config.cortex.macro.plan_ttl_game_loops,
                current_observation=(observation if revalidate_after_outcome else None),
            )
            if isinstance(policy_response, RaceBrainProposalResponse):
                raw = policy_response.model_dump(mode="json")
                plan_digest = hashlib.sha256(
                    json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
                plan = plan.model_copy(
                    update={
                        "plan_id": f"macro-plan:{plan_digest}",
                        "source_model_id": f"hima-{policy_response.race}-ensemble",
                        "source_model_revision": (
                            self._macro_health.model_revision
                            if isinstance(self._macro_health, RaceBrainHealth)
                            else "member_revisions_in_raw_proposal"
                        ),
                        "raw_proposal": raw,
                    }
                )
                self._record_cortex_event(
                    observation,
                    "race_brain_coordinated",
                    policy_response,
                )
            frontier = runtime_frontier(
                response.proposal,
                observation,
                self._recent_hima_actions(observation.game_loop),
            )
            fallback = self._fallback_frontier(
                response.proposal,
                observation,
                frontier,
            )
            if not plan.steps or not (_macro_frontier_is_usable(frontier) or fallback is not None):
                classification = (
                    "empty_plan"
                    if not plan.steps
                    else "none"
                    if frontier is None
                    else frontier.classification.value
                )
                reason = (
                    "no_macro_steps"
                    if not plan.steps
                    else "no_runtime_frontier"
                    if frontier is None
                    else frontier.reason_code or "unspecified"
                )
                self._reject_macro_plan(
                    response,
                    observation,
                    latency_ms=latency_ms,
                    classification=classification,
                    reason=reason,
                )
                return
            goal = macro_goal_spec(plan, observation, self.goal_progress_verifier)
            self._accept_macro_plan(
                plan,
                response.proposal,
                goal,
                observation,
                latency_ms=latency_ms,
            )
        except Exception as error:
            timed_out = self._is_timeout_error(error)
            self._last_planner_failure = (
                IdleReason.PLANNER_TIMEOUT if timed_out else IdleReason.NO_LEGAL_ACTION
            )
            if timed_out:
                self._macro_requests_suspended = True
                self._schedule_macro_recovery(observation)
            else:
                self._next_macro_retry_game_loop = (
                    observation.game_loop + self.config.cortex.macro.interval_game_loops
                )
            self._urgent_replan_requested = not timed_out
            payload = {
                "role": CortexRole.MACRO.value,
                "model_id": (None if self._macro_health is None else self._macro_health.model_id),
                "error_type": type(error).__name__,
                "message": str(error),
                "latency_ms": latency_ms,
                "requests_suspended": self._macro_requests_suspended,
                "generation_metadata": (
                    None
                    if response is None or response.proposal.generation_metadata is None
                    else response.proposal.generation_metadata.model_dump(mode="json")
                ),
            }
            self._record_cortex_event(observation, "specialist_failed", payload)
            self._record_cortex_event(observation, "macro_plan_rejected", payload)

    def _reject_macro_plan(
        self,
        response: HIMALiveProposalResponse,
        observation: ObservationEnvelope,
        *,
        latency_ms: float,
        classification: str,
        reason: str,
    ) -> None:
        metadata = response.proposal.generation_metadata
        self._last_planner_failure = IdleReason.NO_LEGAL_ACTION
        self._next_macro_retry_game_loop = (
            observation.game_loop + self.config.cortex.macro.interval_game_loops
        )
        self._urgent_replan_requested = True
        self._record_cortex_event(
            observation,
            "macro_plan_rejected",
            {
                "role": CortexRole.MACRO.value,
                "model_id": (
                    metadata.model_id
                    if metadata is not None
                    else None
                    if self._macro_health is None
                    else self._macro_health.model_id
                ),
                "model_revision": (
                    metadata.model_revision
                    if metadata is not None
                    else None
                    if self._macro_health is None
                    else self._macro_health.model_revision
                ),
                "classification": classification,
                "reason": reason,
                "latency_ms": latency_ms,
                "generation_metadata": (
                    None if metadata is None else metadata.model_dump(mode="json")
                ),
                "proposal": response.proposal.model_dump(
                    mode="json",
                    exclude={"raw_output"},
                ),
            },
        )

    def _accept_macro_plan(
        self,
        plan: MacroPlan,
        proposal: MacroPolicyProposal,
        goal: GoalSpec | None,
        observation: ObservationEnvelope,
        *,
        latency_ms: float,
    ) -> None:
        proposal_source_game_loop = plan.created_game_loop
        plan = plan.model_copy(
            update={
                "created_game_loop": observation.game_loop,
                "expires_game_loop": (
                    observation.game_loop + self.config.cortex.macro.plan_ttl_game_loops
                ),
            }
        )
        frontier_assessment = runtime_frontier(
            proposal,
            observation,
            self._recent_hima_actions(observation.game_loop),
        )
        is_revision = self._macro_plan is not None
        self._macro_plan = plan
        self._macro_proposal = proposal
        self._macro_goal = goal
        self._macro_plan_frozen = False
        self._next_macro_retry_game_loop = None
        self._last_plan_accepted_game_loop = observation.game_loop
        self._last_planner_failure = None
        self._urgent_replan_requested = False
        self._last_goal_progress_fingerprint = None
        self._record_cortex_event(
            observation,
            "macro_plan_accepted",
            {
                "plan": plan.model_dump(mode="json"),
                "plan_id": plan.plan_id,
                "source_model_id": plan.source_model_id,
                "source_model_revision": plan.source_model_revision,
                "accepted_game_loop": observation.game_loop,
                "proposal_source_game_loop": proposal_source_game_loop,
                "acceptance_delay_game_loops": max(
                    0, observation.game_loop - proposal_source_game_loop
                ),
                "is_revision": is_revision,
                "latency_ms": latency_ms,
                "generation_metadata": (
                    None
                    if proposal.generation_metadata is None
                    else proposal.generation_metadata.model_dump(mode="json")
                ),
                "runtime_frontier": (
                    None
                    if frontier_assessment is None
                    else frontier_assessment.runtime_action or frontier_assessment.source_action
                ),
                "goal_spec": None if goal is None else goal.model_dump(mode="json"),
            },
        )

    def _macro_goal_progress(
        self,
        observation: ObservationEnvelope,
    ) -> GoalProgressReport | None:
        if self._macro_goal is None:
            return None
        return self.goal_progress_verifier.verify(observation, self._macro_goal)

    def _prepare_macro_command(
        self,
        observation: ObservationEnvelope,
        assessment: SituationAssessment,
        goal_progress: GoalProgressReport | None,
    ) -> _PreparedCommand | None:
        if (
            self._macro_plan is None
            or self._macro_proposal is None
            or self._macro_plan_frozen
            or self._macro_inflight_command_id is not None
            or observation.game_loop >= self._macro_plan.expires_game_loop
        ):
            if (
                self._macro_plan is not None
                and observation.game_loop >= self._macro_plan.expires_game_loop
            ):
                self._macro_plan_frozen = True
                self._urgent_replan_requested = True
            return None
        remaining_steps = [
            step
            for step in self._macro_proposal.steps
            if not self._macro_step_is_complete(step.ordinal)
        ]
        if not remaining_steps:
            return None
        remaining_proposal = self._macro_proposal.model_copy(update={"steps": remaining_steps})
        frontier = runtime_frontier(
            remaining_proposal,
            observation,
            self._recent_hima_actions(observation.game_loop),
        )
        if frontier is None:
            return None
        blocked_frontier = frontier
        fallback = self._fallback_frontier(
            remaining_proposal,
            observation,
            blocked_frontier,
        )
        if fallback is not None:
            self._set_macro_step_status(
                blocked_frontier.ordinal,
                MacroStepStatus.DEFERRED,
                blocked_frontier.reason_code,
            )
            reason = (
                "supply_emergency"
                if fallback.source_action == "BUILD PYLON"
                and self._free_supply(observation)
                <= self.config.cortex.executor.supply_emergency_free_supply
                else "resource_fallback"
            )
            self._record_cortex_event(
                observation,
                "macro_frontier_preempted",
                {
                    "reason": reason,
                    "blocked_action": blocked_frontier.source_action,
                    "blocked_runtime_action": blocked_frontier.runtime_action,
                    "blocked_reason": blocked_frontier.reason_code,
                    "fallback_action": fallback.source_action,
                    "fallback_runtime_action": fallback.runtime_action,
                    "free_supply": self._free_supply(observation),
                },
            )
            frontier = fallback
        if frontier.classification is PolicyActionClassification.MAPPED_DEFERRED:
            if _deferred_frontier_requires_replan(frontier):
                self._set_macro_step_status(
                    frontier.ordinal,
                    MacroStepStatus.BLOCKED,
                    frontier.reason_code,
                )
                self._macro_plan_frozen = True
                self._next_macro_retry_game_loop = (
                    observation.game_loop + self.config.cortex.macro.interval_game_loops
                )
                self._urgent_replan_requested = True
                return None
            self._set_macro_step_status(
                frontier.ordinal,
                MacroStepStatus.DEFERRED,
                frontier.reason_code,
            )
            return None
        if frontier.classification in {
            PolicyActionClassification.PARSE_ERROR,
            PolicyActionClassification.UNSUPPORTED_BY_RUNTIME,
            PolicyActionClassification.ILLEGAL_ACTION,
        }:
            self._set_macro_step_status(
                frontier.ordinal,
                MacroStepStatus.BLOCKED,
                frontier.reason_code,
            )
            self._macro_plan_frozen = True
            self._next_macro_retry_game_loop = (
                observation.game_loop + self.config.cortex.macro.interval_game_loops
            )
            self._urgent_replan_requested = True
            return None
        if frontier.classification is PolicyActionClassification.OBSOLETE:
            self._set_macro_step_status(
                frontier.ordinal,
                MacroStepStatus.OBSOLETE,
                frontier.reason_code,
            )
            return None
        if (
            frontier.source_action == "BUILD PYLON"
            and self._free_supply(observation)
            >= self.config.cortex.executor.pylon_redundancy_free_supply
        ):
            self._set_macro_step_status(
                frontier.ordinal,
                MacroStepStatus.OBSOLETE,
                "supply_headroom_satisfied",
            )
            self._record_cortex_event(
                observation,
                "macro_step_deduplicated",
                {
                    "semantic_action": frontier.source_action,
                    "reason": "supply_headroom_satisfied",
                    "free_supply": self._free_supply(observation),
                },
            )
            self._request_macro_if_exhausted()
            return self._prepare_macro_command(
                observation,
                assessment,
                goal_progress,
            )
        if (
            frontier.classification is not PolicyActionClassification.MAPPED_LEGAL_NOW
            or frontier.runtime_action is None
        ):
            return None
        step = self._macro_step(frontier.ordinal)
        if step is None:
            raise RuntimeError("macro frontier has no matching typed plan step")
        intent = MacroIntent(
            intent_id=self._intent_id(
                observation,
                CortexRole.MACRO,
                f"{self._macro_plan.plan_id}:{frontier.ordinal}:{step.completed_repeats}",
            ),
            run_id=observation.run_id,
            episode_id=observation.episode_id,
            step_id=observation.step_id,
            created_game_loop=observation.game_loop,
            objective=self._macro_plan.strategic_objective,
            action_names=[frontier.runtime_action],
            priority=50,
            ttl_game_loops=self.config.runtime.planner_command_ttl_game_loops,
            source_id=self._macro_plan.source_model_id,
            source_version=self._macro_plan.source_model_revision,
            situation_assessment_id=assessment.assessment_id,
            macro_plan_id=self._macro_plan.plan_id,
        )
        return self._compile_intent(
            observation,
            intent,
            goal_progress=goal_progress,
            semantic_action=frontier.source_action,
            macro_step_ordinal=frontier.ordinal,
        )

    def _fallback_frontier(
        self,
        proposal: MacroPolicyProposal,
        observation: ObservationEnvelope,
        blocked_frontier: PolicyActionAssessment | None,
    ) -> PolicyActionAssessment | None:
        """Select a legal, bounded macro fallback without relaxing validation."""

        if (
            blocked_frontier is None
            or blocked_frontier.classification is not PolicyActionClassification.MAPPED_DEFERRED
        ):
            return None
        free_supply = self._free_supply(observation)
        if (
            blocked_frontier.source_action != "BUILD PYLON"
            and free_supply <= self.config.cortex.executor.supply_emergency_free_supply
        ):
            emergency = self._legal_proposal_step(
                proposal,
                observation,
                "BUILD PYLON",
            )
            if emergency is not None:
                return emergency
        nexus_count = sum(
            structure.unit_type == "Nexus" for structure in observation.state.own_structures
        )
        gas_saturated_before_expansion = (
            blocked_frontier.source_action == "BUILD ASSIMILATOR"
            and blocked_frontier.reason_code == "action_unavailable_now"
            and nexus_count < 2
            and observation.state.economy.minerals >= 400
        )
        if (
            blocked_frontier.reason_code != "insufficient_vespene"
            and not gas_saturated_before_expansion
        ):
            return None
        fallback_actions = []
        if gas_saturated_before_expansion or (
            observation.state.economy.minerals >= 800 and nexus_count < 2
        ):
            fallback_actions.append("BUILD NEXUS")
        fallback_actions.append("TRAIN ZEALOT")
        if free_supply <= self.config.cortex.executor.resource_fallback_pylon_free_supply:
            fallback_actions.append("BUILD PYLON")
        if "BUILD NEXUS" not in fallback_actions:
            fallback_actions.append("BUILD NEXUS")
        for action_name in fallback_actions:
            fallback = self._legal_proposal_step(
                proposal,
                observation,
                action_name,
            )
            if fallback is not None:
                return fallback
        return None

    def _legal_proposal_step(
        self,
        proposal: MacroPolicyProposal,
        observation: ObservationEnvelope,
        semantic_action: str,
    ) -> PolicyActionAssessment | None:
        previous_actions = self._recent_hima_actions(observation.game_loop)
        for step in sorted(proposal.steps, key=lambda item: item.ordinal):
            if step.canonical_action != semantic_action:
                continue
            isolated = proposal.model_copy(update={"steps": [step], "diagnostics": []})
            assessment = runtime_frontier(
                isolated,
                observation,
                previous_actions,
            )
            if (
                assessment is not None
                and assessment.classification is PolicyActionClassification.MAPPED_LEGAL_NOW
            ):
                return assessment
        return None

    @staticmethod
    def _free_supply(observation: ObservationEnvelope) -> int:
        economy = observation.state.economy
        return max(0, economy.supply_cap - economy.supply_used)

    def _schedule_macro_recovery(self, observation: ObservationEnvelope) -> None:
        if (
            self._macro_sidecar is None
            or self._macro_recovery_task is not None
            or self._macro_restart_attempts >= self.config.cortex.macro.restart_limit
        ):
            return
        self._macro_recovery_task = asyncio.create_task(self._recover_macro_specialist(observation))

    async def _recover_macro_specialist(
        self,
        observation: ObservationEnvelope,
    ) -> None:
        assert self._macro_sidecar is not None
        self._macro_restart_attempts += 1
        attempt = self._macro_restart_attempts
        try:
            try:
                self._macro_health = await self._macro_sidecar.restart()
            except Exception as error:
                self._record_cortex_event(
                    observation,
                    "specialist_recovery_failed",
                    {
                        "role": CortexRole.MACRO.value,
                        "restart_attempt": attempt,
                        "restart_limit": self.config.cortex.macro.restart_limit,
                        "error_type": type(error).__name__,
                        "message": str(error),
                    },
                )
                return
            self._macro_requests_suspended = False
            self._next_macro_retry_game_loop = None
            self._last_planner_failure = None
            self._urgent_replan_requested = True
            self._record_cortex_event(
                observation,
                "specialist_recovered",
                {
                    "role": CortexRole.MACRO.value,
                    "restart_attempt": attempt,
                    "restart_limit": self.config.cortex.macro.restart_limit,
                    "model_id": self._macro_health.model_id,
                    "model_revision": self._macro_health.model_revision,
                },
            )
        finally:
            self._macro_recovery_task = None

    def _prepare_reflex_command(
        self,
        observation: ObservationEnvelope,
        assessment: SituationAssessment,
        command: ActionCommand,
    ) -> _PreparedCommand | None:
        intent = ReflexIntent(
            intent_id=self._intent_id(
                observation,
                CortexRole.REFLEX,
                command.command_id,
            ),
            run_id=observation.run_id,
            episode_id=observation.episode_id,
            step_id=observation.step_id,
            created_game_loop=observation.game_loop,
            objective=(
                "Protect threatened units immediately"
                if command.name == "Retreat"
                else "Respond to the visible enemy threat"
            ),
            action_names=[command.name],
            actor_scopes=[command.actor],
            priority=command.priority,
            ttl_game_loops=command.ttl_game_loops,
            source_id="deterministic-reflex",
            source_version="0.1.0",
            situation_assessment_id=assessment.assessment_id,
        )
        return self._compile_intent(
            observation,
            intent,
            command_id=command.command_id,
        )

    def _compile_intent(
        self,
        observation: ObservationEnvelope,
        intent: MacroIntent | TacticalIntent | ReflexIntent,
        *,
        goal_progress: GoalProgressReport | None = None,
        command_id: str | None = None,
        semantic_action: str | None = None,
        macro_step_ordinal: int | None = None,
    ) -> _PreparedCommand | None:
        busy_actors = tuple(
            lifecycle.command.actor
            for lifecycle in self._command_states.values()
            if lifecycle.status is CommandStatus.DISPATCHED
        )
        self._record_cortex_event(
            observation,
            "intent_emitted",
            {
                "role": intent.source_role.value,
                "intent": intent.model_dump(mode="json"),
                "intent_id": intent.intent_id,
                "action_name": intent.action_names[0],
            },
        )
        active_snapshot = self._active_plan_snapshot()
        context = self._candidate_compiler.compile(
            observation,
            intent,
            goal_progress=goal_progress,
            busy_actors=busy_actors,
            recent_commands=() if active_snapshot is None else active_snapshot.commands,
        )
        self._record_cortex_event(
            observation,
            "candidate_set_built",
            {
                "intent_id": intent.intent_id,
                "role": intent.source_role.value,
                "candidate_count": len(context.candidates),
                "candidates": [
                    candidate.model_dump(mode="json") for candidate in context.candidates
                ],
            },
        )
        selection = self._executor.select(context)
        self._record_cortex_event(
            observation,
            "executor_selection",
            {
                **selection.model_dump(mode="json"),
                "selected_candidate_id": selection.candidate_id,
                "role": intent.source_role.value,
                "fallback": False,
            },
        )
        if selection.status is CandidateSelectionStatus.ABSTAINED:
            return None
        resolved_command_id = command_id or self._command_id(
            observation,
            intent.intent_id,
            selection.selection_id,
        )
        command = self._candidate_compiler.materialize(
            context,
            selection,
            command_id=resolved_command_id,
        )
        assert selection.candidate_id is not None
        lineage = CommandLineage(
            command_id=command.command_id,
            intent_id=intent.intent_id,
            candidate_id=selection.candidate_id,
            selection_id=selection.selection_id,
            source_role=intent.source_role,
            source_id=intent.source_id,
            source_version=intent.source_version,
            executor_id=selection.executor_id,
            executor_version=selection.executor_version,
            situation_assessment_id=intent.situation_assessment_id,
            macro_plan_id=(intent.macro_plan_id if isinstance(intent, MacroIntent) else None),
            selected_game_loop=observation.game_loop,
        )
        return _PreparedCommand(
            command=command,
            lineage=lineage,
            semantic_action=semantic_action,
            macro_step_ordinal=macro_step_ordinal,
        )

    def _record_command_lineage(
        self,
        observation: ObservationEnvelope,
        prepared: _PreparedCommand,
    ) -> None:
        existing = self._command_lineages.get(prepared.command.command_id)
        if existing is not None and existing != prepared.lineage:
            raise RuntimeError("command ID was reused with conflicting Cortex lineage")
        self._command_lineages[prepared.command.command_id] = prepared.lineage
        self._record_cortex_event(
            observation,
            "command_lineage",
            {
                "lineage": prepared.lineage.model_dump(mode="json"),
                "command_id": prepared.command.command_id,
                "macro_plan_id": prepared.lineage.macro_plan_id,
                "semantic_action": prepared.semantic_action,
                "macro_step_ordinal": prepared.macro_step_ordinal,
            },
        )

    def record_execution(self, report: ExecutionReport) -> None:
        metadata = self._macro_command_steps.get(report.command_id)
        existing = self._terminal_execution_fingerprints.get(report.command_id)
        super().record_execution(report)
        if existing is not None:
            return
        if metadata is None:
            return
        self._macro_outcome_revision += 1
        if self._macro_inflight_command_id == report.command_id:
            self._macro_inflight_command_id = None
        succeeded = report.status is ExecutionStatus.SUCCEEDED
        if self._macro_plan is not None and metadata[0] == self._macro_plan.plan_id:
            self._advance_macro_step(
                metadata[2],
                succeeded=succeeded,
                persist=True,
                report=report,
            )
        if succeeded and report.action_name is not None:
            token = hima_previous_action_for_runtime_action(report.action_name)
            if token is not None:
                self._previous_hima_actions.append((self._execution_game_loop(report), token))
        else:
            self._macro_plan_frozen = True
            self._urgent_replan_requested = True

    def _advance_macro_step(
        self,
        ordinal: int,
        *,
        succeeded: bool,
        persist: bool,
        report: ExecutionReport | None = None,
    ) -> None:
        step = self._macro_step(ordinal)
        if step is None or self._macro_plan is None:
            return
        completed = step.completed_repeats + (1 if succeeded else 0)
        status = (
            MacroStepStatus.CONFIRMED
            if succeeded and completed >= step.repeat
            else MacroStepStatus.PENDING
            if succeeded
            else MacroStepStatus.BLOCKED
        )
        updated = step.model_copy(
            update={
                "completed_repeats": min(completed, step.repeat),
                "status": status,
                "reason": None if succeeded else "execution_failed",
            }
        )
        self._replace_macro_step(updated)
        if persist and report is not None:
            self.store.append_event(
                run_id=report.run_id,
                episode_id=report.episode_id,
                step_id=report.step_id,
                event_type="macro_step_updated",
                payload={
                    "plan_id": self._macro_plan.plan_id,
                    "step": updated.model_dump(mode="json"),
                    "command_id": report.command_id,
                    "execution_status": report.status.value,
                },
            )
        if status is MacroStepStatus.CONFIRMED and all(
            candidate.status in {MacroStepStatus.CONFIRMED, MacroStepStatus.OBSOLETE}
            for candidate in self._macro_plan.steps
        ):
            self._urgent_replan_requested = True

    def _set_macro_step_status(
        self,
        ordinal: int,
        status: MacroStepStatus,
        reason: str | None,
    ) -> None:
        step = self._macro_step(ordinal)
        if step is None or step.status is status and step.reason == reason:
            return
        self._replace_macro_step(step.model_copy(update={"status": status, "reason": reason}))

    def _replace_macro_step(self, replacement: MacroStep) -> None:
        assert self._macro_plan is not None
        self._macro_plan = self._macro_plan.model_copy(
            update={
                "steps": [
                    replacement if step.ordinal == replacement.ordinal else step
                    for step in self._macro_plan.steps
                ]
            }
        )

    def _macro_step(self, ordinal: int) -> MacroStep | None:
        if self._macro_plan is None:
            return None
        return next(
            (step for step in self._macro_plan.steps if step.ordinal == ordinal),
            None,
        )

    def _macro_step_is_complete(self, ordinal: int) -> bool:
        step = self._macro_step(ordinal)
        if step is None:
            return False
        return step.status in {MacroStepStatus.CONFIRMED, MacroStepStatus.OBSOLETE}

    def _request_macro_if_exhausted(self) -> None:
        if self._macro_plan is None:
            return
        if all(
            step.status in {MacroStepStatus.CONFIRMED, MacroStepStatus.OBSOLETE}
            for step in self._macro_plan.steps
        ):
            self._urgent_replan_requested = True

    def _recent_hima_actions(self, game_loop: int) -> list[str]:
        earliest = max(0, game_loop - _HIMA_PREVIOUS_ACTION_WINDOW_GAME_LOOPS)
        self._previous_hima_actions = [
            item for item in self._previous_hima_actions if item[0] >= earliest
        ]
        return [
            action
            for confirmed_loop, action in sorted(self._previous_hima_actions)
            if confirmed_loop <= game_loop
        ]

    @staticmethod
    def _execution_game_loop(report: ExecutionReport) -> int:
        loops = [entry.game_loop for entry in report.primitive_trace if entry.game_loop is not None]
        if report.effect_evidence is not None:
            loops.extend(
                loop
                for loop in (
                    report.effect_evidence.dispatch_game_loop,
                    report.effect_evidence.accepted_game_loop,
                    report.effect_evidence.confirmed_game_loop,
                )
                if loop is not None
            )
        return max(loops, default=report.step_id)

    @staticmethod
    def _is_timeout_error(error: Exception) -> bool:
        return isinstance(error, TimeoutError) or type(error).__name__.endswith("Timeout")

    def _cortex_idle_reason(self) -> IdleReason:
        if self._macro_task is not None:
            return IdleReason.WAITING_FOR_PLANNER
        if self._last_planner_failure is IdleReason.PLANNER_TIMEOUT:
            return IdleReason.PLANNER_TIMEOUT
        if self._macro_inflight_command_id is not None:
            return IdleReason.PLAN_COMMANDS_DEFERRED
        if self._macro_plan is not None:
            if any(
                step.status in {MacroStepStatus.PENDING, MacroStepStatus.DEFERRED}
                for step in self._macro_plan.steps
            ):
                return IdleReason.PLAN_COMMANDS_DEFERRED
            return IdleReason.PLAN_EXHAUSTED
        return IdleReason.NO_LEGAL_ACTION

    def _decision_summary(self, progress: GoalProgressReport | None) -> str:
        if progress is None:
            return "Waiting for a specialist macro plan; deterministic reflex remains active."
        if progress.unique_next_action is not None:
            return f"Next verified macro action: {progress.unique_next_action}."
        return (
            f"Macro goal is {progress.status.value}; fast executor uses current legal candidates."
        )

    def _record_cortex_event(
        self,
        observation: ObservationEnvelope,
        event_type: str,
        payload: BaseModel | dict[str, Any],
    ) -> None:
        self.store.append_event(
            run_id=observation.run_id,
            episode_id=observation.episode_id,
            step_id=observation.step_id,
            event_type=event_type,
            payload=payload,
        )

    def _refresh_playbook(
        self,
        observation: ObservationEnvelope,
        assessment: SituationAssessment,
    ) -> None:
        if self._playbook_store is None:
            self._playbook_selection = None
            return
        query = PlaybookQuery(
            context=PlaybookContext(
                agent_race=self.config.environment.agent_race,
                opponent_race=self.config.environment.opponent_race,
                phase=assessment.phase,
                map_name=self.config.environment.scenario,
                tags=tuple(assessment.threats),
            ),
            top_k=self.config.cortex.playbook.top_k,
            min_confidence=self.config.cortex.playbook.min_confidence,
            include_candidates=self.config.cortex.playbook.include_candidates,
        )
        selection = self._playbook_store.retrieve(query)
        fingerprint = (assessment.phase.value, *selection.lesson_ids)
        self._playbook_selection = selection
        if fingerprint == self._playbook_selection_fingerprint:
            return
        self._playbook_selection_fingerprint = fingerprint
        self._record_cortex_event(
            observation,
            "playbook_retrieved",
            {
                "phase": assessment.phase.value,
                "lesson_ids": list(selection.lesson_ids),
                "hit_count": len(selection.hits),
                "hits": [hit.model_dump(mode="json") for hit in selection.hits],
            },
        )

    def end_episode(self, result: EpisodeResult) -> None:
        already_recorded = self._episode_result_fingerprint is not None
        super().end_episode(result)
        if already_recorded or self._playbook_reviewer is None:
            return
        events = self.store.events_after(
            result.run_id,
            0,
            100_000,
            episode_id=result.episode_id,
        )
        cases, lessons = self._playbook_reviewer.review_episode(
            events,
            result,
            agent_race=self.config.environment.agent_race,
            opponent_race=self.config.environment.opponent_race,
        )
        for case in cases:
            self.store.append_event(
                run_id=result.run_id,
                episode_id=result.episode_id,
                step_id=result.steps,
                event_type="playbook_case_recorded",
                payload=case,
            )
        for lesson in lessons:
            event_type = (
                "playbook_lesson_promoted"
                if lesson.status is LessonStatus.PROMOTED
                else "playbook_lesson_candidate"
            )
            self.store.append_event(
                run_id=result.run_id,
                episode_id=result.episode_id,
                step_id=result.steps,
                event_type=event_type,
                payload=lesson,
            )
        self.store.append_event(
            run_id=result.run_id,
            episode_id=result.episode_id,
            step_id=result.steps,
            event_type="postgame_review_completed",
            payload={
                "case_count": len(cases),
                "lesson_update_count": len(lessons),
                "playbook_path": str(self._playbook_reviewer.store.database_path),
            },
        )

    @staticmethod
    def _intent_id(
        observation: ObservationEnvelope,
        role: CortexRole,
        identity: str,
    ) -> str:
        digest = hashlib.sha256(
            (
                f"{observation.run_id}|{observation.episode_id}|"
                f"{observation.step_id}|{observation.game_loop}|{role.value}|{identity}"
            ).encode()
        ).hexdigest()
        return f"intent:{digest}"

    @staticmethod
    def _command_id(
        observation: ObservationEnvelope,
        intent_id: str,
        selection_id: str,
    ) -> str:
        digest = hashlib.sha256(
            (
                f"{observation.run_id}|{observation.episode_id}|"
                f"{observation.step_id}|{observation.game_loop}|"
                f"{intent_id}|{selection_id}"
            ).encode()
        ).hexdigest()
        return f"cortex:{digest}"

    async def close(self) -> None:
        try:
            await self._cancel_planner()
        finally:
            try:
                if self._macro_sidecar is not None:
                    await self._macro_sidecar.close()
                elif self._macro_client is not None:
                    await self._macro_client.close()
            finally:
                try:
                    close = getattr(self.provider, "close", None)
                    if close is not None:
                        await close()
                finally:
                    try:
                        if self._playbook_store is not None:
                            self._playbook_store.close()
                    finally:
                        self.store.close()

    async def _cancel_planner(self) -> None:
        if self._macro_task is not None:
            self._macro_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._macro_task
            self._macro_task = None
        self._macro_source_observation = None
        self._macro_task_started_at = None
        if self._macro_recovery_task is not None:
            self._macro_recovery_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._macro_recovery_task
            self._macro_recovery_task = None
        await super()._cancel_planner()
