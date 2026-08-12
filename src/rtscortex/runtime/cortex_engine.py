"""SC2-native specialist runtime with a bounded, low-latency executor."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import math
import os
import re
import time
from dataclasses import asdict, dataclass
from typing import Any, Literal, Protocol

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
    PlacementLedgerEvent,
)
from rtscortex.contracts.interfaces import LLMProvider
from rtscortex.cortex import (
    AttemptKey,
    AuthoritativeBuildCircuitState,
    CandidateCompiler,
    CandidateSelectionStatus,
    CommandLineage,
    CortexRole,
    DeterministicCandidateExecutor,
    DeterministicSituationAnalyzer,
    DeterministicTacticalAgent,
    ExecutionAwareTacticalPolicyProvider,
    ExpansionGoalKey,
    ExpansionGoalState,
    FastExecutorContext,
    IntentArbiter,
    MacroIntent,
    MacroPlan,
    MacroStep,
    MacroStepStatus,
    ReflexIntent,
    ResourceClaim,
    RoleAgentContext,
    RoleAgentCoordinator,
    RoleId,
    SituationAssessment,
    SituationProvider,
    StrategicAgenda,
    StrategicArbitration,
    StrategicIntent,
    StrategicIntentAdapter,
    TacticalIntent,
    TacticalPolicyProvider,
    TerminalCollapseReason,
    counterfactual_observation_fingerprint,
    hima_previous_action_for_runtime_action,
    is_terminal_collapse,
    macro_goal_spec,
    macro_plan_from_hima,
    runtime_frontier,
    townhall_recovery_runtime_actions,
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
from rtscortex.memory import EventStore, StoredEvent
from rtscortex.placement import CANONICAL_PLACEMENT_SPECS, canonical_footprint_cells
from rtscortex.playbook import (
    CortexPlaybookReviewer,
    LessonStatus,
    PlaybookCandidateGuard,
    PlaybookContext,
    PlaybookIntentGuard,
    PlaybookPromotionSweep,
    PlaybookQuery,
    PlaybookRule,
    PlaybookRuleApplication,
    PlaybookRuleEvaluation,
    PlaybookRuleKind,
    PlaybookSelection,
    PlaybookStore,
    RecentTerminalFeedback,
    candidate_signature,
    evaluation_kind,
)
from rtscortex.policy.hima import (
    HIMAInputContext,
    HIMALiveProposalResponse,
)
from rtscortex.policy.models import (
    MacroActionStep,
    MacroPolicyProposal,
    PolicyActionAssessment,
    PolicyActionClassification,
)
from rtscortex.progress import (
    GoalBlockerKind,
    GoalProgressReport,
    GoalProgressVerifier,
    GoalSpec,
)
from rtscortex.races import RaceProfile, race_profile
from rtscortex.reflex import ReflexEngine
from rtscortex.runtime.engine import (
    _ACTIONABLE_COMMAND_STATUSES,
    CommandStatus,
    RuntimeEngine,
)
from rtscortex.runtime.validation import (
    ValidationDisposition,
    ValidationFailure,
    ValidationOutcome,
)

_HIMA_PREVIOUS_ACTION_WINDOW_GAME_LOOPS = int(60 * 22.4)
_MACRO_REJECTION_RETRY_GAME_LOOPS = 16
_CORTEX_SNAPSHOT_TYPE = "cortex-engine-v1"
_OPERATION_IDENTITY = re.compile(r"^operation:[0-9a-f]{64}$")
_BUILD_LEGALITY_IDENTITY = re.compile(r"^build-legality:[0-9a-f]{64}$")
_ATTEMPT_IDENTITY = re.compile(r"^attempt:[0-9a-f]{64}$")
_EXPERIMENT_IDENTITY_VALUE = re.compile(r"^[a-z][a-z0-9_-]*$")


def _experiment_run_identity_from_environment(seed: int) -> dict[str, Any] | None:
    """Return the runner-bound experiment identity or reject a partial identity.

    The formal and production-canary wrappers set these values on the actual
    RTSCortex process.  Persisting them from the runtime, rather than copying a
    status-row value during analysis, gives the analyzer an independent identity
    boundary for each SC2 artifact.
    """

    names = {
        "mode": "RTSCORTEX_EXPERIMENT_MODE",
        "experiment_kind": "RTSCORTEX_EXPERIMENT_KIND",
        "arm": "RTSCORTEX_EXPERIMENT_ARM",
        "subject_arm": "RTSCORTEX_EXPERIMENT_SUBJECT_ARM",
    }
    raw = {field: os.environ.get(name) for field, name in names.items()}
    if not any(value is not None for value in raw.values()):
        return None
    mode = (raw["mode"] or "").strip()
    kind = (raw["experiment_kind"] or "").strip()
    arm = (raw["arm"] or "").strip()
    subject_arm = (raw["subject_arm"] or "").strip() or None
    if not mode or not kind or not arm:
        raise RuntimeError("experiment run identity is partial")
    if any(_EXPERIMENT_IDENTITY_VALUE.fullmatch(value) is None for value in (mode, kind, arm)):
        raise RuntimeError("experiment run identity contains an invalid typed value")
    if subject_arm is not None and _EXPERIMENT_IDENTITY_VALUE.fullmatch(subject_arm) is None:
        raise RuntimeError("experiment subject arm contains an invalid typed value")
    if kind == "behavior":
        if arm not in {"frozen", "evolving", "active"}:
            raise RuntimeError("behavior experiment arm is invalid")
        if arm in {"frozen", "evolving"} and subject_arm != arm:
            raise RuntimeError("formal behavior subject arm must equal its behavior arm")
        if arm == "active" and subject_arm is not None:
            raise RuntimeError("active canary behavior cannot claim a subject arm")
    elif kind == "calibration":
        if arm != "shadow" or subject_arm not in {"frozen", "evolving", "active"}:
            raise RuntimeError("calibration identity requires shadow and a typed subject arm")
    else:
        raise RuntimeError("experiment kind must be behavior or calibration")
    if mode == "causal_canary" and not (
        kind == "behavior"
        and arm == "active"
        and subject_arm is None
        or kind == "calibration"
        and arm == "shadow"
        and subject_arm == "active"
    ):
        raise RuntimeError("causal canary identity has an invalid role binding")
    return {
        "schema_version": "1.0",
        "source": "runner_environment",
        "seed": int(seed),
        "mode": mode,
        "experiment_kind": kind,
        "arm": arm,
        "subject_arm": subject_arm,
    }


_PROTOSS_POWERED_STRUCTURE_TYPES = frozenset(
    {
        "cyberneticscore",
        "forge",
        "gateway",
        "shieldbattery",
        "stargate",
    }
)
_RECOVERABLE_EXPANSION_FAILURE_CODES = frozenset(
    {
        "invalid_expansion_anchor",
        "no_legal_placement",
        "target_not_created",
    }
)


def _macro_frontier_is_usable(frontier: PolicyActionAssessment | None) -> bool:
    if frontier is None:
        return False
    return frontier.classification in {
        PolicyActionClassification.MAPPED_LEGAL_NOW,
        PolicyActionClassification.MAPPED_DEFERRED,
    }


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


@dataclass(frozen=True)
class _TerminalCollapseMacroHold:
    plan_id: str
    townhall_recovery_availability_signature: tuple[str, ...]


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
        approved_hard_rule_ids: tuple[str, ...] | None = None,
        situation_provider: SituationProvider | None = None,
        shadow_situation_provider: SituationProvider | None = None,
        tactical_provider: TacticalPolicyProvider | None = None,
        shadow_tactical_provider: TacticalPolicyProvider | None = None,
    ) -> None:
        if config.agent.variant != "cortex":
            raise ValueError("CortexRuntimeEngine requires agent.variant=cortex")
        if macro_sidecar is not None and macro_client is None:
            raise ValueError("a macro sidecar requires its matching client")
        if (
            config.cortex.macro.kind in {"hima", "hima_ensemble", "scripted"}
            and macro_client is None
            and macro_startup_failure is None
        ):
            raise ValueError("enabled cortex macro policy requires a live macro client")
        if macro_startup_failure is not None and config.cortex.macro.required:
            raise ValueError("required macro specialists cannot start in degraded mode")
        if config.cortex.macro.kind == "disabled" and macro_client is not None:
            raise ValueError("a macro client cannot be attached when the specialist is disabled")
        super().__init__(config=config, store=store, provider=provider)
        self._race_profile: RaceProfile = race_profile(config.environment.agent_race)
        self.goal_progress_verifier = GoalProgressVerifier(
            self._race_profile.data.progress_action_specs
        )
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
        self._approved_hard_rule_ids = approved_hard_rule_ids
        self._playbook_selection: PlaybookSelection | None = None
        self._playbook_selection_fingerprint: tuple[str, ...] | None = None
        self._playbook_rules: tuple[PlaybookRule, ...] = ()
        self._consumed_canary_fixture_rule_ids: set[str] = set()
        self._pending_playbook_rule_evaluations: dict[str, PlaybookRuleEvaluation] = {}
        self._terminal_strategy_rule_evaluations: dict[str, PlaybookRuleEvaluation] = {}
        self._playbook_promotion_sweep_done = False
        self._playbook_intent_guard = PlaybookIntentGuard()
        self._playbook_candidate_guard = PlaybookCandidateGuard()
        self._recent_terminal_feedback: dict[str, RecentTerminalFeedback] = {}
        self._current_situation: SituationAssessment | None = None
        self._macro_goal: GoalSpec | None = None
        self._macro_plan_frozen = False
        self._terminal_collapse_macro_hold: _TerminalCollapseMacroHold | None = None
        self._macro_inflight_command_id: str | None = None
        self._macro_command_steps: dict[str, tuple[str, str, int | None]] = {}
        self._command_lineages: dict[str, CommandLineage] = {}
        self._attempt_ordinals: dict[str, int] = {}
        self._previous_hima_actions: list[tuple[int, str]] = []
        self._expansion_commitment_id: str | None = None
        self._expansion_commitment_started_game_loop: int | None = None
        self._expansion_anchor_evaluations: list[dict[str, Any]] = []
        self._expansion_candidates_exhausted = False
        self._expansion_scout_state = "not_discovered_yet"
        self._expansion_scout_generation = 0
        self._expansion_commitment_generation: int | None = None
        self._expansion_commitment_dispatched = False
        self._expansion_exhausted_generation: int | None = None
        self._expansion_scout_visited_waypoints = 0
        self._expansion_scout_total_waypoints = 0
        self._expansion_goal: ExpansionGoalState | None = None
        if config.cortex.situation.kind == "model_active" and situation_provider is None:
            raise ValueError("model_active Situation requires a SituationProvider")
        if config.cortex.situation.kind == "model_shadow" and shadow_situation_provider is None:
            raise ValueError("model_shadow Situation requires a shadow SituationProvider")
        if config.cortex.tactical.kind == "model_active" and tactical_provider is None:
            raise ValueError("model_active tactical policy requires a TacticalPolicyProvider")
        if config.cortex.tactical.kind == "model_shadow" and shadow_tactical_provider is None:
            raise ValueError("model_shadow tactical policy requires a shadow provider")
        self._situation = situation_provider or DeterministicSituationAnalyzer(
            valid_for_game_loops=1
        )
        self._shadow_situation = shadow_situation_provider
        deterministic_tactical = DeterministicTacticalAgent(
            retreat_health_threshold=(config.cortex.tactical.retreat_health_threshold),
            minimum_advance_army_supply=(config.cortex.tactical.minimum_advance_army_supply),
            reacquire_cooldown_game_loops=(config.cortex.tactical.reacquire_cooldown_game_loops),
        )
        self._tactical: TacticalPolicyProvider = tactical_provider or deterministic_tactical
        self._shadow_tactical = shadow_tactical_provider
        self._candidate_compiler = CandidateCompiler()
        self._executor = DeterministicCandidateExecutor()
        self._strategic_adapter = StrategicIntentAdapter(self._race_profile)
        self._role_agents = RoleAgentCoordinator(
            self._race_profile,
            self._strategic_adapter,
        )
        self._last_defense_inventory_signatures: dict[str, str] = {}
        self._strategic_arbiter = IntentArbiter(
            switch_margin=config.cortex.arbiter.switch_margin,
            max_intents=config.cortex.arbiter.max_intents,
        )
        self._strategic_agenda: StrategicAgenda | None = None
        self._pending_strategic_arbitration: StrategicArbitration | None = None
        self._strategic_by_legacy_intent: dict[str, StrategicIntent] = {}
        # Raw bridge no-start identities are durable semantic tombstones. They
        # prevent Cortex from preparing the same operation/material again after
        # a placement was accepted but never started.
        self._raw_build_material_tombstones: dict[str, set[str]] = {}
        self._raw_build_material_tombstone_circuits: set[str] = set()
        self._authoritative_build_pre_dispatch_circuits: dict[
            str,
            AuthoritativeBuildCircuitState,
        ] = {}
        self._authoritative_build_pre_dispatch_defer_signatures: set[tuple[str, str | None]] = set()

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
        self._strategic_by_legacy_intent = {}
        self._pending_strategic_arbitration = None
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
        self._update_expansion_candidate_state(observation)

        assessment = self._situation.assess(observation)
        self._current_situation = assessment
        self._record_cortex_event(observation, "situation_assessed", assessment)
        self._update_terminal_collapse_macro_hold(observation, assessment)
        if self._shadow_situation is not None:
            shadow_assessment = self._shadow_situation.assess(observation)
            self._record_cortex_event(
                observation,
                "situation_shadow_assessed",
                {
                    "active_assessment_id": assessment.assessment_id,
                    "assessment": shadow_assessment.model_dump(mode="json"),
                },
            )
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

        reflex_started = time.perf_counter()
        raw_reflex = [
            command
            for command in self.reflex.evaluate(observation)
            if command.command_id not in self._command_states
        ]
        claimed_reflex_actors = frozenset(command.actor for command in raw_reflex)

        raw_tactical_intents = self._tactical.evaluate(observation, assessment)
        tactical_intents = self._role_agents.own_tactical_intents(tuple(raw_tactical_intents))
        defense_intents = self._role_agents.propose_defense_intents(
            RoleAgentContext(
                observation=observation,
                situation=assessment,
                source_intents=tuple(tactical_intents),
            ),
            claimed_actor_scopes=claimed_reflex_actors,
        )
        for diagnostic in self._role_agents.drain_defense_diagnostics():
            self._record_cortex_event(
                observation,
                "defense_actor_state",
                diagnostic,
            )
        if self._shadow_tactical is not None:
            shadow_started = time.perf_counter()
            shadow_intents = self._shadow_tactical.evaluate(observation, assessment)
            self._record_cortex_event(
                observation,
                "tactical_policy_shadow",
                {
                    "provider_id": self._shadow_tactical.provider_id,
                    "provider_version": self._shadow_tactical.provider_version,
                    "latency_ms": (time.perf_counter() - shadow_started) * 1_000,
                    "active_intent_ids": [intent.intent_id for intent in tactical_intents],
                    "shadow_intents": [intent.model_dump(mode="json") for intent in shadow_intents],
                },
            )
        tactical_prepared = [
            item
            for intent in (*defense_intents, *tactical_intents)
            if (item := self._compile_intent(observation, intent)) is not None
        ]
        prepared.extend(tactical_prepared)

        reflex_prepared = [
            item
            for command in raw_reflex
            if (item := self._prepare_reflex_command(observation, assessment, command)) is not None
        ]
        prepared.extend(reflex_prepared)
        reflex_latency_ms = (time.perf_counter() - reflex_started) * 1_000

        prepared = self._apply_strategic_arbitration(observation, prepared)

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
        structure_budget = self._apply_structure_budget_guard(
            [*guarded_macro.accepted, *guarded_reflex.accepted],
            observation,
        )
        rejected_commands.extend(
            self._apply_validation_failures(structure_budget.failures, observation)
        )
        budgeted_macro = [
            command
            for command in structure_budget.accepted
            if command.source is ActionSource.PLANNER
        ]
        budgeted_reflex = [
            command
            for command in structure_budget.accepted
            if command.source is ActionSource.REFLEX
        ]
        (
            available_macro,
            available_reflex,
            busy_actor_rejections,
            busy_actor_candidates,
        ) = self._defer_busy_actor_commands(
            budgeted_macro,
            budgeted_reflex,
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
        defense_outcome, defense_inventory = self._apply_defense_inventory_guard(
            outcome.accepted,
            observation,
        )
        rejected_commands.extend(
            self._apply_validation_failures(defense_outcome.failures, observation)
        )
        self._record_defense_inventory(
            observation,
            unit_evaluations=defense_inventory,
        )
        accepted_commands: list[ActionCommand] = []
        for command in defense_outcome.accepted:
            prepared_command = prepared_by_id[command.command_id]
            terminal_failure = self._guard_terminal_collapse_macro_dispatch(
                observation,
                prepared_command,
            )
            if terminal_failure is not None:
                rejected_commands.extend(
                    self._apply_validation_failures([terminal_failure], observation)
                )
                continue
            prepared_command = self._bind_dispatch_attempt(prepared_command)
            prepared_by_id[command.command_id] = prepared_command
            accepted_commands.append(prepared_command.command)
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
            self._mark_playbook_counterfactual_observable(
                observation,
                prepared_command.lineage,
            )
            self._record_command_lineage(observation, prepared_command)
            self._role_agents.record_dispatch(
                command,
                responsibility=prepared_command.lineage.responsibility,
                game_loop=observation.game_loop,
            )
            responsibility = prepared_command.lineage.responsibility
            if responsibility in {
                RoleId.OFFENSE.value,
                RoleId.FOCUS_FIRE.value,
                RoleId.RETREAT.value,
            } and isinstance(self._tactical, ExecutionAwareTacticalPolicyProvider):
                self._tactical.record_dispatch(
                    command,
                    responsibility=responsibility,
                    observation=observation,
                    situation=assessment,
                )
            self._transition_command(command, CommandStatus.DISPATCHED, observation)
            if prepared_command.lineage.source_role is CortexRole.MACRO:
                plan_id = prepared_command.lineage.macro_plan_id
                if plan_id is None:
                    raise RuntimeError("macro command lineage is missing its plan ID")
                self._macro_inflight_command_id = command.command_id
                assert prepared_command.semantic_action is not None
                self._macro_command_steps[command.command_id] = (
                    plan_id,
                    prepared_command.semantic_action,
                    prepared_command.macro_step_ordinal,
                )
                if (
                    self._expansion_commitment_id is not None
                    and prepared_command.semantic_action == self._townhall_semantic_action()
                    and not self._expansion_commitment_dispatched
                ):
                    self._expansion_commitment_dispatched = True
                    self._record_cortex_event(
                        observation,
                        "expansion_commitment_dispatched",
                        {
                            "commitment_id": self._expansion_commitment_id,
                            "command_id": command.command_id,
                            "operation_id": command.operation_id,
                            "macro_plan_id": plan_id,
                            "macro_step_ordinal": prepared_command.macro_step_ordinal,
                        },
                    )
                if prepared_command.macro_step_ordinal is not None:
                    self._set_macro_step_status(
                        prepared_command.macro_step_ordinal,
                        MacroStepStatus.DISPATCHED,
                        None,
                    )
        self._commit_dispatched_strategic_agenda(
            observation,
            accepted_commands,
            prepared_by_id,
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
                "planner_candidates": [
                    command.model_dump(mode="json")
                    for command in (*macro_candidates, *tactical_candidates)
                ],
                "reflex_candidates": [
                    command.model_dump(mode="json") for command in reflex_candidates
                ],
                "candidate_counts": {
                    "macro": len(macro_candidates),
                    "tactical": len(tactical_candidates),
                    "reflex": len(reflex_candidates),
                    "busy_actor": len(busy_actor_candidates),
                    "validated": len(candidate_outcome.accepted),
                    "selected": len(accepted_commands),
                },
                "goal_progress": (
                    None if goal_progress is None else goal_progress.model_dump(mode="json")
                ),
            },
        )
        self._last_decision = batch
        for command in batch.commands:
            self._decision_by_command_id[command.command_id] = batch
        self._request_macro_if_exhausted()
        if self._record_runtime_checkpoint_if_due(observation):
            self._record_cortex_checkpoint(observation)
        return batch

    def _apply_structure_budget_guard(
        self,
        commands: list[ActionCommand],
        observation: ObservationEnvelope,
    ) -> ValidationOutcome:
        accepted: list[ActionCommand] = []
        failures: list[ValidationFailure] = []
        current_ids = {command.command_id for command in commands}
        reserved_by_structure: dict[str, int] = {}
        dispatched_by_structure: dict[str, int] = {}
        for lifecycle in self._command_states.values():
            if lifecycle.command.command_id in current_ids or lifecycle.status not in {
                CommandStatus.PENDING,
                CommandStatus.DEFERRED,
                CommandStatus.DISPATCHED,
            }:
                continue
            target = self._structure_target_for_action(lifecycle.command.name)
            if target is not None:
                counts = (
                    dispatched_by_structure
                    if lifecycle.status is CommandStatus.DISPATCHED
                    else reserved_by_structure
                )
                counts[target] = counts.get(target, 0) + 1
        completed_or_constructing: dict[str, int] = {}
        constructing_by_structure: dict[str, int] = {}
        for structure in observation.state.own_structures:
            key = structure.unit_type.casefold()
            completed_or_constructing[key] = completed_or_constructing.get(key, 0) + 1
            if structure.status == "constructing":
                constructing_by_structure[key] = constructing_by_structure.get(key, 0) + 1
        selected_by_structure: dict[str, int] = {}
        for command in sorted(commands, key=lambda item: (-item.priority, item.command_id)):
            target = self._structure_target_for_action(command.name)
            limit = (
                None
                if target is None
                else self._race_profile.data.structure_saturation_limits.get(target)
            )
            if target is None or limit is None:
                accepted.append(command)
                continue
            key = target.casefold()
            unobserved_dispatches = max(
                0,
                dispatched_by_structure.get(target, 0) - constructing_by_structure.get(key, 0),
            )
            effective_count = (
                completed_or_constructing.get(key, 0)
                + reserved_by_structure.get(target, 0)
                + unobserved_dispatches
                + selected_by_structure.get(target, 0)
            )
            if effective_count >= limit:
                failures.append(
                    ValidationFailure(
                        command=command,
                        reason=(
                            f"global_structure_saturation_limit:{target}:{effective_count}/{limit}"
                        ),
                        disposition=ValidationDisposition.OBSOLETE,
                    )
                )
                continue
            selected_by_structure[target] = selected_by_structure.get(target, 0) + 1
            accepted.append(command)
        return ValidationOutcome(
            accepted=accepted,
            rejected=[f"{failure.command.command_id}: {failure.reason}" for failure in failures],
            failures=failures,
        )

    def _apply_defense_inventory_guard(
        self,
        commands: list[ActionCommand],
        observation: ObservationEnvelope,
    ) -> tuple[ValidationOutcome, tuple[dict[str, object], ...]]:
        """Apply race-wide defense unit caps at the final dispatch boundary."""

        current_ids = {command.command_id for command in commands}
        lifecycle_counts: dict[tuple[str, CommandStatus], int] = {}
        for lifecycle in self._command_states.values():
            if (
                lifecycle.command.command_id in current_ids
                or lifecycle.status is not CommandStatus.DISPATCHED
            ):
                continue
            key = (lifecycle.command.name, lifecycle.status)
            lifecycle_counts[key] = lifecycle_counts.get(key, 0) + 1

        accepted: list[ActionCommand] = []
        failures: list[ValidationFailure] = []
        selected: dict[str, int] = {}
        inventory: dict[str, dict[str, int]] = {}
        for item_type, cap in self._race_profile.data.defense_unit_saturation_limits.items():
            action_name = f"Train_{item_type}"
            queue = [
                item
                for item in observation.state.production_queue
                if item.name in {item_type, action_name}
            ]
            constructing = sum(item.progress > 0.0 for item in queue)
            queued = len(queue) - constructing
            dispatched = lifecycle_counts.get((action_name, CommandStatus.DISPATCHED), 0)
            inventory[item_type] = {
                "completed": sum(
                    unit.unit_type == item_type and unit.health_fraction > 0.0
                    for unit in observation.state.own_units
                ),
                "constructing_or_training": constructing,
                "queued": queued,
                "reserved": sum(
                    lifecycle_counts.get((action_name, status), 0)
                    for status in (CommandStatus.PENDING, CommandStatus.DEFERRED)
                ),
                "dispatched_not_terminal": dispatched,
                "dispatches_not_already_observed": max(0, dispatched - len(queue)),
                "hard_cap": cap,
            }

        for command in sorted(commands, key=lambda item: (-item.priority, item.command_id)):
            item_type = command.name.removeprefix("Train_")
            counts = inventory.get(item_type) if command.name.startswith("Train_") else None
            if counts is None:
                accepted.append(command)
                continue
            effective = (
                counts["completed"]
                + counts["constructing_or_training"]
                + counts["queued"]
                + counts["reserved"]
                + counts["dispatches_not_already_observed"]
                + selected.get(item_type, 0)
            )
            cap = counts["hard_cap"]
            if effective >= cap:
                reason = f"global_defense_inventory_cap:{item_type}:{effective}/{cap}"
                failures.append(
                    ValidationFailure(
                        command=command,
                        reason=reason,
                        disposition=ValidationDisposition.OBSOLETE,
                    )
                )
                self._record_cortex_event(
                    observation,
                    "defense_inventory_cap_blocked",
                    {
                        "command_id": command.command_id,
                        "action_name": command.name,
                        "source": command.source.value,
                        "item_type": item_type,
                        "effective_count": effective,
                        "hard_cap": cap,
                        "reason": reason,
                    },
                )
                continue
            selected[item_type] = selected.get(item_type, 0) + 1
            accepted.append(command)

        evaluations: list[dict[str, object]] = []
        for item_type, counts in inventory.items():
            current_batch_selected = selected.get(item_type, 0)
            effective = (
                counts["completed"]
                + counts["constructing_or_training"]
                + counts["queued"]
                + counts["reserved"]
                + counts["dispatches_not_already_observed"]
                + current_batch_selected
            )
            cap = counts["hard_cap"]
            evaluations.append(
                {
                    "item_type": item_type,
                    "action_name": f"Train_{item_type}",
                    **counts,
                    "current_batch_selected": current_batch_selected,
                    "effective_count": effective,
                    "decision": (
                        "over_cap"
                        if effective > cap
                        else "at_cap"
                        if effective == cap
                        else "within_cap"
                    ),
                }
            )
        return (
            ValidationOutcome(
                accepted=accepted,
                rejected=[
                    f"{failure.command.command_id}: {failure.reason}" for failure in failures
                ],
                failures=failures,
            ),
            tuple(evaluations),
        )

    def _structure_target_for_action(self, action_name: str) -> str | None:
        return next(
            (
                spec.effect_target
                for spec in self._race_profile.data.progress_action_specs
                if spec.name == action_name and spec.effect_kind.value == "structure"
            ),
            None,
        )

    async def _activate_episode(self, observation: ObservationEnvelope) -> None:
        episode_key = (observation.run_id, observation.episode_id)
        changed = self._episode_key != episode_key
        if changed:
            self._last_defense_inventory_signatures.clear()
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
        existing_experiment_identity = self.store.last_event(
            observation.run_id,
            observation.episode_id,
            "experiment_run_identity",
        )
        experiment_identity = _experiment_run_identity_from_environment(self.config.run.seed)
        if existing_experiment_identity is not None and experiment_identity is None:
            raise RuntimeError(
                "experiment identity environment is missing while recovering an identified run"
            )
        if experiment_identity is not None:
            expected_identity = {
                **experiment_identity,
                "run_id": observation.run_id,
                "episode_id": observation.episode_id,
            }
            if existing_experiment_identity is None:
                self.store.append_event(
                    run_id=observation.run_id,
                    episode_id=observation.episode_id,
                    step_id=observation.step_id,
                    event_type="experiment_run_identity",
                    payload=expected_identity,
                )
            elif existing_experiment_identity.payload != expected_identity:
                raise RuntimeError("experiment run identity changed during runtime recovery")
        self._macro_health_announced_for = None
        self._macro_plan = None
        self._macro_proposal = None
        self._playbook_selection = None
        self._playbook_selection_fingerprint = None
        self._playbook_rules = ()
        self._consumed_canary_fixture_rule_ids = set()
        self._pending_playbook_rule_evaluations = {}
        self._terminal_strategy_rule_evaluations = {}
        self._recent_terminal_feedback = {}
        self._current_situation = None
        self._macro_goal = None
        self._macro_plan_frozen = False
        self._terminal_collapse_macro_hold = None
        self._macro_inflight_command_id = None
        self._macro_command_steps = {}
        self._command_lineages = {}
        self._attempt_ordinals = {}
        self._previous_hima_actions = []
        self._expansion_commitment_id = None
        self._expansion_commitment_started_game_loop = None
        self._expansion_anchor_evaluations = []
        self._expansion_candidates_exhausted = False
        self._expansion_scout_state = "not_discovered_yet"
        self._expansion_scout_generation = 0
        self._expansion_commitment_generation = None
        self._expansion_commitment_dispatched = False
        self._expansion_exhausted_generation = None
        self._expansion_scout_visited_waypoints = 0
        self._expansion_scout_total_waypoints = 0
        self._expansion_goal = None
        self._strategic_agenda = None
        self._strategic_by_legacy_intent = {}
        self._macro_source_observation = None
        self._macro_task_started_at = None
        self._macro_task_outcome_revision = None
        self._macro_outcome_revision = 0
        self._next_macro_retry_game_loop = None
        self._raw_build_material_tombstones = {}
        self._raw_build_material_tombstone_circuits = set()
        self._authoritative_build_pre_dispatch_circuits = {}
        self._authoritative_build_pre_dispatch_defer_signatures = set()
        self._restore_consumed_canary_fixture_rules(observation)
        self._recover_cortex_episode(observation)
        if (
            self.store.last_event(
                observation.run_id,
                observation.episode_id,
                "race_profile_activated",
            )
            is None
        ):
            self._record_cortex_event(
                observation,
                "race_profile_activated",
                self._race_profile.data.capability_snapshot(),
            )

    def _recover_cortex_episode(self, observation: ObservationEnvelope) -> None:
        checkpoint = self.store.latest_snapshot(
            observation.run_id,
            observation.episode_id,
            _CORTEX_SNAPSHOT_TYPE,
        )
        checkpoint_event_id = 0
        if checkpoint is not None:
            self._restore_cortex_checkpoint(checkpoint.payload)
            checkpoint_event_id = checkpoint.through_event_id

        build_feedback_events = sorted(
            (
                event
                for event_type in ("placement_ledger_transition", "execution")
                for event in self.store.events_of_type(
                    observation.run_id,
                    observation.episode_id,
                    event_type,
                    after_event_id=checkpoint_event_id,
                )
            ),
            key=lambda item: item.event_id,
        )
        for event in build_feedback_events:
            self._remember_raw_build_placement_transition(event.payload)
            if event.event_type != "execution":
                continue
            report = ExecutionReport.model_validate(event.payload)
            if self._execution_confirms_authoritative_build_reset(report):
                assert report.operation_id is not None
                self._authoritative_build_pre_dispatch_circuits.pop(report.operation_id, None)
                self._authoritative_build_pre_dispatch_defer_signatures = {
                    key
                    for key in self._authoritative_build_pre_dispatch_defer_signatures
                    if key[0] != report.operation_id
                }

        plan_event = self.store.last_event(
            observation.run_id,
            observation.episode_id,
            "macro_plan_accepted",
            after_event_id=checkpoint_event_id,
        )
        active_plan_accept_event_id = checkpoint_event_id
        if plan_event is not None:
            active_plan_accept_event_id = plan_event.event_id
            plan_payload = plan_event.payload.get("plan", plan_event.payload)
            self._macro_plan = MacroPlan.model_validate(plan_payload)
            raw_response = self._macro_plan.raw_proposal
            if raw_response:
                if isinstance(raw_response.get("proposal"), dict):
                    self._macro_proposal = MacroPolicyProposal.model_validate(
                        raw_response["proposal"]
                    )
                elif "selected" in raw_response:
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

        obsolete_event = self.store.last_event(
            observation.run_id,
            observation.episode_id,
            "macro_frontier_obsolete",
            after_event_id=checkpoint_event_id,
        )
        if (
            obsolete_event is not None
            and obsolete_event.event_id > active_plan_accept_event_id
            and self._macro_plan is not None
            and obsolete_event.payload.get("plan_id") == self._macro_plan.plan_id
            and obsolete_event.payload.get("reason")
            == TerminalCollapseReason.MACRO_FRONTIER_OBSOLETE.value
        ):
            for ordinal in obsolete_event.payload.get("obsolete_ordinals", ()):
                if isinstance(ordinal, int):
                    self._set_macro_step_status(
                        ordinal,
                        MacroStepStatus.OBSOLETE,
                        TerminalCollapseReason.MACRO_FRONTIER_OBSOLETE.value,
                    )
            if obsolete_event.payload.get("plan_frozen") is True:
                self._macro_plan_frozen = True
                self._urgent_replan_requested = False
                self._terminal_collapse_macro_hold = _TerminalCollapseMacroHold(
                    plan_id=self._macro_plan.plan_id,
                    townhall_recovery_availability_signature=tuple(
                        str(item)
                        for item in obsolete_event.payload.get(
                            "townhall_recovery_availability_signature",
                            (),
                        )
                    ),
                )

        agenda_event = self.store.last_event(
            observation.run_id,
            observation.episode_id,
            "strategic_agenda_committed",
            after_event_id=checkpoint_event_id,
        )
        if agenda_event is not None:
            agenda_payload = agenda_event.payload.get("agenda")
            if isinstance(agenda_payload, dict):
                agenda = StrategicAgenda.model_validate(agenda_payload)
                if agenda.commitment_until_game_loop > observation.game_loop:
                    self._strategic_agenda = agenda

        for event in self.store.events_of_type(
            observation.run_id,
            observation.episode_id,
            "command_lineage",
            after_event_id=checkpoint_event_id,
        ):
            payload = event.payload
            lineage = CommandLineage.model_validate(payload.get("lineage", payload))
            self._command_lineages[lineage.command_id] = lineage
            if lineage.operation_id is not None and lineage.attempt_ordinal is not None:
                self._attempt_ordinals[lineage.operation_id] = max(
                    self._attempt_ordinals.get(lineage.operation_id, 0),
                    lineage.attempt_ordinal + 1,
                )
            ordinal = payload.get("macro_step_ordinal")
            semantic = payload.get("semantic_action")
            if lineage.macro_plan_id is not None and isinstance(semantic, str):
                self._macro_command_steps[lineage.command_id] = (
                    lineage.macro_plan_id,
                    semantic,
                    ordinal if isinstance(ordinal, int) else None,
                )

        started_commitment = self.store.last_event(
            observation.run_id,
            observation.episode_id,
            "expansion_commitment_started",
            after_event_id=checkpoint_event_id,
        )
        terminal_commitment = self.store.last_event(
            observation.run_id,
            observation.episode_id,
            "expansion_commitment_terminal",
            after_event_id=checkpoint_event_id,
        )
        dispatched_commitment = self.store.last_event(
            observation.run_id,
            observation.episode_id,
            "expansion_commitment_dispatched",
            after_event_id=checkpoint_event_id,
        )
        goal_event_types = {
            "expansion_goal_started",
            "expansion_goal_reopened",
            "expansion_candidate_epoch_exhausted",
            "expansion_goal_terminal",
        }
        latest_goal = next(
            (
                event
                for event in sorted(
                    (
                        event
                        for event_type in goal_event_types
                        for event in self.store.events_of_type(
                            observation.run_id,
                            observation.episode_id,
                            event_type,
                            after_event_id=checkpoint_event_id,
                        )
                    ),
                    key=lambda item: item.event_id,
                    reverse=True,
                )
            ),
            None,
        )
        if latest_goal is not None:
            self._expansion_goal = ExpansionGoalState.model_validate(latest_goal.payload)
            exhausted_epochs = self._expansion_goal.exhausted_candidate_epochs
            if exhausted_epochs:
                self._expansion_exhausted_generation = max(exhausted_epochs)
            self._expansion_scout_generation = max(
                self._expansion_scout_generation,
                self._expansion_goal.current_candidate_epoch,
            )
            self._expansion_candidates_exhausted = (
                self._expansion_goal.phase == "waiting_for_candidates"
                and self._expansion_goal.current_candidate_epoch in exhausted_epochs
            )
        if started_commitment is not None and (
            terminal_commitment is None
            or terminal_commitment.event_id < started_commitment.event_id
        ):
            self._expansion_commitment_id = str(started_commitment.payload["commitment_id"])
            self._expansion_commitment_started_game_loop = int(
                started_commitment.payload["started_game_loop"]
            )
            generation = started_commitment.payload.get("generation_id")
            self._expansion_commitment_generation = (
                int(generation) if isinstance(generation, int) else None
            )
            self._expansion_commitment_dispatched = False
            if (
                dispatched_commitment is not None
                and dispatched_commitment.event_id > started_commitment.event_id
            ):
                self._expansion_commitment_dispatched = True
            self._expansion_anchor_evaluations = [
                dict(event.payload)
                for event in self.store.events_of_type(
                    observation.run_id,
                    observation.episode_id,
                    "expansion_anchor_rejected",
                    after_event_id=max(
                        checkpoint_event_id,
                        started_commitment.event_id,
                    ),
                )
            ]
        elif terminal_commitment is not None:
            self._expansion_commitment_id = None
            self._expansion_commitment_started_game_loop = None
            self._expansion_commitment_generation = None
            self._expansion_commitment_dispatched = False
            self._expansion_anchor_evaluations = []
        if terminal_commitment is not None and (
            terminal_commitment.payload.get("terminal_state") == "expansion_candidates_exhausted"
        ):
            generation = terminal_commitment.payload.get("generation_id")
            if isinstance(generation, int):
                self._expansion_exhausted_generation = generation
                self._expansion_candidates_exhausted = True

        recovered_macro_outcomes: list[ExecutionReport] = []
        for event in self.store.events_of_type(
            observation.run_id,
            observation.episode_id,
            "execution",
            after_event_id=checkpoint_event_id,
        ):
            report = ExecutionReport.model_validate(event.payload)
            recovered_lineage = self._command_lineages.get(report.command_id)
            if recovered_lineage is None or recovered_lineage.source_role is not CortexRole.MACRO:
                continue
            if report.status is ExecutionStatus.SUCCEEDED and report.action_name is not None:
                loop = self._execution_game_loop(report)
                token = hima_previous_action_for_runtime_action(
                    report.action_name,
                    self._race_profile.data,
                )
                if token is not None:
                    self._previous_hima_actions.append((loop, token))
            recovered_macro_outcomes.append(report)

        if self._macro_plan is not None:
            for report in recovered_macro_outcomes:
                metadata = self._macro_command_steps.get(report.command_id)
                if (
                    metadata is not None
                    and metadata[0] == self._macro_plan.plan_id
                    and metadata[2] is not None
                ):
                    succeeded = report.status is ExecutionStatus.SUCCEEDED
                    self._advance_macro_step(
                        metadata[2],
                        succeeded=succeeded,
                        persist=False,
                        report=report,
                    )
                    if not succeeded and not self._recoverable_expansion_failure(report):
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
                and metadata[2] is not None
            ):
                self._set_macro_step_status(
                    metadata[2],
                    MacroStepStatus.DISPATCHED,
                    None,
                )

    def _record_cortex_checkpoint(self, observation: ObservationEnvelope) -> None:
        self.store.record_snapshot(
            run_id=observation.run_id,
            episode_id=observation.episode_id,
            snapshot_type=_CORTEX_SNAPSHOT_TYPE,
            step_id=observation.step_id,
            payload=self._cortex_checkpoint_payload(observation.game_loop),
        )

    def _cortex_checkpoint_payload(self, game_loop: int) -> dict[str, Any]:
        return {
            "format_version": "1",
            "game_loop": game_loop,
            "macro_plan": (
                None if self._macro_plan is None else self._macro_plan.model_dump(mode="json")
            ),
            "macro_proposal": (
                None
                if self._macro_proposal is None
                else self._macro_proposal.model_dump(mode="json")
            ),
            "macro_goal": (
                None if self._macro_goal is None else self._macro_goal.model_dump(mode="json")
            ),
            "macro_plan_frozen": self._macro_plan_frozen,
            "terminal_collapse_macro_hold": (
                None
                if self._terminal_collapse_macro_hold is None
                else {
                    "plan_id": self._terminal_collapse_macro_hold.plan_id,
                    "townhall_recovery_availability_signature": list(
                        self._terminal_collapse_macro_hold.townhall_recovery_availability_signature
                    ),
                }
            ),
            "macro_inflight_command_id": self._macro_inflight_command_id,
            "macro_command_steps": {
                command_id: [plan_id, semantic_action, ordinal]
                for command_id, (
                    plan_id,
                    semantic_action,
                    ordinal,
                ) in self._macro_command_steps.items()
            },
            "command_lineages": {
                command_id: lineage.model_dump(mode="json")
                for command_id, lineage in self._command_lineages.items()
            },
            "attempt_ordinals": dict(self._attempt_ordinals),
            "previous_hima_actions": [
                [game_loop, token] for game_loop, token in self._previous_hima_actions
            ],
            "expansion": {
                "commitment_id": self._expansion_commitment_id,
                "commitment_started_game_loop": (self._expansion_commitment_started_game_loop),
                "anchor_evaluations": list(self._expansion_anchor_evaluations),
                "candidates_exhausted": self._expansion_candidates_exhausted,
                "scout_state": self._expansion_scout_state,
                "scout_generation": self._expansion_scout_generation,
                "commitment_generation": self._expansion_commitment_generation,
                "commitment_dispatched": self._expansion_commitment_dispatched,
                "exhausted_generation": self._expansion_exhausted_generation,
                "visited_waypoints": self._expansion_scout_visited_waypoints,
                "total_waypoints": self._expansion_scout_total_waypoints,
                "goal": (
                    None
                    if self._expansion_goal is None
                    else self._expansion_goal.model_dump(mode="json")
                ),
            },
            "strategic_agenda": (
                None
                if self._strategic_agenda is None
                else self._strategic_agenda.model_dump(mode="json")
            ),
            "last_plan_accepted_game_loop": self._last_plan_accepted_game_loop,
            "macro_outcome_revision": self._macro_outcome_revision,
            "next_macro_retry_game_loop": self._next_macro_retry_game_loop,
            "raw_build_material_tombstones": {
                operation_id: sorted(materials)
                for operation_id, materials in self._raw_build_material_tombstones.items()
                if materials
            },
            "raw_build_material_tombstone_circuits": sorted(
                self._raw_build_material_tombstone_circuits
            ),
            "authoritative_build_pre_dispatch_circuits": {
                operation_id: state.model_dump(mode="json")
                for operation_id, state in sorted(
                    self._authoritative_build_pre_dispatch_circuits.items()
                )
            },
        }

    def _restore_cortex_checkpoint(self, payload: dict[str, Any]) -> None:
        if payload.get("format_version") != "1":
            raise RuntimeError("unsupported Cortex recovery snapshot version")
        macro_plan = payload.get("macro_plan")
        self._macro_plan = None if macro_plan is None else MacroPlan.model_validate(macro_plan)
        macro_proposal = payload.get("macro_proposal")
        self._macro_proposal = (
            None if macro_proposal is None else MacroPolicyProposal.model_validate(macro_proposal)
        )
        macro_goal = payload.get("macro_goal")
        self._macro_goal = None if macro_goal is None else GoalSpec.model_validate(macro_goal)
        self._macro_plan_frozen = bool(payload.get("macro_plan_frozen"))
        terminal_hold = payload.get("terminal_collapse_macro_hold")
        self._terminal_collapse_macro_hold = (
            None
            if not isinstance(terminal_hold, dict)
            else _TerminalCollapseMacroHold(
                plan_id=str(terminal_hold["plan_id"]),
                townhall_recovery_availability_signature=tuple(
                    str(item)
                    for item in terminal_hold.get(
                        "townhall_recovery_availability_signature",
                        (),
                    )
                ),
            )
        )
        inflight = payload.get("macro_inflight_command_id")
        self._macro_inflight_command_id = None if inflight is None else str(inflight)
        self._macro_command_steps = {
            str(command_id): (
                str(values[0]),
                str(values[1]),
                None if values[2] is None else int(values[2]),
            )
            for command_id, values in dict(payload.get("macro_command_steps", {})).items()
        }
        self._command_lineages = {
            str(command_id): CommandLineage.model_validate(lineage)
            for command_id, lineage in dict(payload.get("command_lineages", {})).items()
        }
        self._attempt_ordinals = {
            str(operation_id): int(ordinal)
            for operation_id, ordinal in dict(payload.get("attempt_ordinals", {})).items()
        }
        self._previous_hima_actions = [
            (int(item[0]), str(item[1])) for item in payload.get("previous_hima_actions", ())
        ]
        expansion = dict(payload.get("expansion", {}))
        commitment_id = expansion.get("commitment_id")
        self._expansion_commitment_id = None if commitment_id is None else str(commitment_id)
        started_loop = expansion.get("commitment_started_game_loop")
        self._expansion_commitment_started_game_loop = (
            None if started_loop is None else int(started_loop)
        )
        self._expansion_anchor_evaluations = [
            dict(item) for item in expansion.get("anchor_evaluations", ())
        ]
        self._expansion_candidates_exhausted = bool(expansion.get("candidates_exhausted"))
        self._expansion_scout_state = str(expansion.get("scout_state", "not_discovered_yet"))
        self._expansion_scout_generation = int(expansion.get("scout_generation", 0))
        commitment_generation = expansion.get("commitment_generation")
        self._expansion_commitment_generation = (
            None if commitment_generation is None else int(commitment_generation)
        )
        self._expansion_commitment_dispatched = bool(expansion.get("commitment_dispatched", False))
        exhausted_generation = expansion.get("exhausted_generation")
        self._expansion_exhausted_generation = (
            None if exhausted_generation is None else int(exhausted_generation)
        )
        self._expansion_scout_visited_waypoints = int(expansion.get("visited_waypoints", 0))
        self._expansion_scout_total_waypoints = int(expansion.get("total_waypoints", 0))
        expansion_goal = expansion.get("goal")
        self._expansion_goal = (
            None if expansion_goal is None else ExpansionGoalState.model_validate(expansion_goal)
        )
        agenda = payload.get("strategic_agenda")
        self._strategic_agenda = None if agenda is None else StrategicAgenda.model_validate(agenda)
        accepted_loop = payload.get("last_plan_accepted_game_loop")
        self._last_plan_accepted_game_loop = None if accepted_loop is None else int(accepted_loop)
        self._macro_outcome_revision = int(payload.get("macro_outcome_revision", 0))
        retry_loop = payload.get("next_macro_retry_game_loop")
        self._next_macro_retry_game_loop = None if retry_loop is None else int(retry_loop)
        raw_tombstones = payload.get("raw_build_material_tombstones", {})
        self._raw_build_material_tombstones = {
            str(operation_id): {str(identity) for identity in identities}
            for operation_id, identities in (
                raw_tombstones.items() if isinstance(raw_tombstones, dict) else ()
            )
            if isinstance(identities, (list, tuple, set, frozenset)) and identities
        }
        circuits = payload.get("raw_build_material_tombstone_circuits", ())
        self._raw_build_material_tombstone_circuits = (
            {str(operation_id) for operation_id in circuits}
            if isinstance(circuits, (list, tuple, set, frozenset))
            else set()
        )
        authoritative_circuits = payload.get(
            "authoritative_build_pre_dispatch_circuits",
            {},
        )
        self._authoritative_build_pre_dispatch_circuits = {
            str(operation_id): AuthoritativeBuildCircuitState.model_validate(state)
            for operation_id, state in (
                authoritative_circuits.items() if isinstance(authoritative_circuits, dict) else ()
            )
            if isinstance(state, dict)
        }

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

    def _update_terminal_collapse_macro_hold(
        self,
        observation: ObservationEnvelope,
        assessment: SituationAssessment,
    ) -> None:
        hold = self._terminal_collapse_macro_hold
        if hold is None:
            return
        recovery_signature = self._townhall_recovery_availability_signature(observation)
        if (
            is_terminal_collapse(assessment)
            and recovery_signature == hold.townhall_recovery_availability_signature
        ):
            return
        reason = (
            "terminal_collapse_townhall_recovery_availability_changed"
            if is_terminal_collapse(assessment)
            else "terminal_collapse_material_state_changed"
        )
        replacement_plan_present = (
            self._macro_plan is not None and self._macro_plan.plan_id != hold.plan_id
        )
        self._terminal_collapse_macro_hold = None
        if not replacement_plan_present:
            self._macro_plan = None
            self._macro_proposal = None
            self._macro_goal = None
            self._macro_plan_frozen = False
            self._next_macro_retry_game_loop = None
            self._urgent_replan_requested = True
        self._record_cortex_event(
            observation,
            "terminal_collapse_macro_hold_released",
            {
                "plan_id": hold.plan_id,
                "current_game_loop": observation.game_loop,
                "current_situation_assessment_id": assessment.assessment_id,
                "army_readiness": assessment.army_readiness.value,
                "own_base_count": assessment.bases.own_base_count,
                "own_production_capacity": assessment.bases.own_production_capacity,
                "threat_level": assessment.threat_level.value,
                "replacement_plan_present": replacement_plan_present,
                "reason": reason,
            },
        )

    def _townhall_recovery_availability_signature(
        self,
        observation: ObservationEnvelope,
    ) -> tuple[str, ...]:
        recovery_actions = townhall_recovery_runtime_actions(self._race_profile.data)
        signatures = [
            json.dumps(
                action.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
            )
            for action in observation.available_actions
            if action.name in recovery_actions
        ]
        recovery_semantics = {
            self._semantic_action_for_runtime(runtime_action) for runtime_action in recovery_actions
        }
        if self._macro_proposal is not None and any(
            step.canonical_action in recovery_semantics for step in self._macro_proposal.steps
        ):
            signatures.extend(
                "resource-readiness:"
                + json.dumps(
                    {
                        "action": action_name,
                        "ready": self._runtime_action_resources_ready(
                            action_name,
                            observation,
                        ),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                for action_name in sorted(recovery_actions)
            )
        return tuple(sorted(signatures))

    def _runtime_action_resources_ready(
        self,
        action_name: str,
        observation: ObservationEnvelope,
    ) -> bool:
        spec = next(
            (
                candidate
                for candidate in self._race_profile.data.progress_action_specs
                if candidate.name == action_name
            ),
            None,
        )
        if spec is None:
            return False
        economy = observation.state.economy
        return (
            economy.minerals >= spec.minerals
            and economy.vespene >= spec.vespene
            and economy.supply_cap - economy.supply_used >= spec.supply
        )

    def _should_start_macro(self, observation: ObservationEnvelope) -> bool:
        if (
            self._macro_client is None
            or self._macro_task is not None
            or self._macro_requests_suspended
            or self._terminal_collapse_macro_hold is not None
        ):
            return False
        if (
            self._next_macro_retry_game_loop is not None
            and observation.game_loop < self._next_macro_retry_game_loop
        ):
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
                profile=self._race_profile.data,
            )
            plan = plan.model_copy(
                update={"proposal_source_game_loop": source_observation.game_loop}
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
                        "raw_response_hash": plan_digest,
                    }
                )
                self._record_cortex_event(
                    observation,
                    "race_brain_coordinated",
                    policy_response,
                )
            compact_proposal = MacroPolicyProposal.model_validate(plan.raw_proposal["proposal"])
            frontier = runtime_frontier(
                compact_proposal,
                observation,
                self._recent_hima_actions(observation.game_loop),
                self._race_profile.data,
            )
            fallback = self._fallback_frontier(
                compact_proposal,
                observation,
                frontier,
            )
            speculative_after_inflight = (
                self._macro_inflight_command_id is not None
                and frontier is not None
                and frontier.classification
                in {
                    PolicyActionClassification.MAPPED_FUTURE,
                    PolicyActionClassification.MAPPED_LEGAL_NOW,
                    PolicyActionClassification.MAPPED_DEFERRED,
                }
            )
            if not plan.steps or not (
                _macro_frontier_is_usable(frontier)
                or fallback is not None
                or self._prerequisite_resolution_is_active(frontier, observation)
                or speculative_after_inflight
            ):
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
            goal = macro_goal_spec(
                plan,
                observation,
                self.goal_progress_verifier,
                self._race_profile.data,
            )
            self._accept_macro_plan(
                plan,
                compact_proposal,
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
        self._next_macro_retry_game_loop = observation.game_loop + min(
            self.config.cortex.macro.interval_game_loops,
            _MACRO_REJECTION_RETRY_GAME_LOOPS,
        )
        self._urgent_replan_requested = False
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
        proposal_source_game_loop = (
            plan.proposal_source_game_loop
            if plan.proposal_source_game_loop is not None
            else plan.created_game_loop
        )
        plan = plan.model_copy(
            update={
                "created_game_loop": observation.game_loop,
                "proposal_source_game_loop": proposal_source_game_loop,
                "expires_game_loop": (
                    observation.game_loop + self.config.cortex.macro.plan_ttl_game_loops
                ),
            }
        )
        frontier_assessment = runtime_frontier(
            proposal,
            observation,
            self._recent_hima_actions(observation.game_loop),
            self._race_profile.data,
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
        self._ensure_expansion_commitment(observation)
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
                "accepted_while_command_inflight": (self._macro_inflight_command_id is not None),
                "inflight_command_id": self._macro_inflight_command_id,
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
        if (
            frontier_assessment is not None
            and frontier_assessment.classification is PolicyActionClassification.MAPPED_DEFERRED
        ):
            self._record_cortex_event(
                observation,
                "macro_frontier_deferred",
                {
                    "plan_id": plan.plan_id,
                    "ordinal": frontier_assessment.ordinal,
                    "semantic_action": frontier_assessment.source_action,
                    "runtime_action": frontier_assessment.runtime_action,
                    "reason": frontier_assessment.reason_code,
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
        executable_ordinals = {step.ordinal for step in self._macro_plan.steps}
        remaining_steps = [
            step
            for step in self._macro_proposal.steps
            if step.ordinal in executable_ordinals
            and not self._macro_step_is_complete(step.ordinal)
        ]
        if not remaining_steps:
            self._macro_plan_frozen = True
            self._urgent_replan_requested = True
            self._record_cortex_event(
                observation,
                "macro_executable_horizon_exhausted",
                {
                    "plan_id": self._macro_plan.plan_id,
                    "executable_ordinals": sorted(executable_ordinals),
                    "opaque_future_ordinals": [
                        step.ordinal
                        for step in self._macro_proposal.steps
                        if step.ordinal not in executable_ordinals
                    ],
                },
            )
            return None
        remaining_proposal = self._macro_proposal.model_copy(update={"steps": remaining_steps})
        remaining_proposal = self._proposal_with_expansion_commitment(remaining_proposal)
        frontier = runtime_frontier(
            remaining_proposal,
            observation,
            self._recent_hima_actions(observation.game_loop),
            self._race_profile.data,
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
            reason = self._fallback_reason(blocked_frontier, fallback, observation)
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
        frontier = self._apply_terminal_collapse_macro_boundary(
            observation,
            assessment,
            remaining_proposal,
            frontier,
        )
        if frontier is None:
            return None
        if frontier.classification is PolicyActionClassification.MAPPED_DEFERRED:
            step = self._macro_step(frontier.ordinal)
            status_changed = (
                step is None
                or step.status is not MacroStepStatus.DEFERRED
                or step.reason != frontier.reason_code
            )
            self._set_macro_step_status(
                frontier.ordinal,
                MacroStepStatus.DEFERRED,
                frontier.reason_code,
            )
            if status_changed:
                self._record_cortex_event(
                    observation,
                    "macro_frontier_deferred",
                    {
                        "plan_id": self._macro_plan.plan_id,
                        "ordinal": frontier.ordinal,
                        "semantic_action": frontier.source_action,
                        "runtime_action": frontier.runtime_action,
                        "reason": frontier.reason_code,
                    },
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
        saturated_structure = self._saturated_structure_target(frontier, observation)
        if saturated_structure is not None:
            self._set_macro_step_status(
                frontier.ordinal,
                MacroStepStatus.OBSOLETE,
                "global_structure_saturation_satisfied",
            )
            self._record_cortex_event(
                observation,
                "macro_step_deduplicated",
                {
                    "plan_id": self._macro_plan.plan_id,
                    "ordinal": frontier.ordinal,
                    "semantic_action": frontier.source_action,
                    "runtime_action": frontier.runtime_action,
                    "target_structure": saturated_structure,
                    "reason": "global_structure_saturation_satisfied",
                },
            )
            return self._prepare_macro_command(observation, assessment, goal_progress)
        pending_structure = self._pending_structure_target(frontier, observation)
        if pending_structure is not None:
            step = self._macro_step(frontier.ordinal)
            status_changed = (
                step is None
                or step.status is not MacroStepStatus.DEFERRED
                or step.reason != "same_structure_in_progress"
            )
            self._set_macro_step_status(
                frontier.ordinal,
                MacroStepStatus.DEFERRED,
                "same_structure_in_progress",
            )
            if status_changed:
                self._record_cortex_event(
                    observation,
                    "macro_structure_deferred",
                    {
                        "plan_id": self._macro_plan.plan_id,
                        "ordinal": frontier.ordinal,
                        "semantic_action": frontier.source_action,
                        "runtime_action": frontier.runtime_action,
                        "target_structure": pending_structure,
                        "reason": "same_structure_in_progress",
                    },
                )
            return None
        if (
            frontier.source_action == self._supply_macro_action()
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
        advances_plan_step = step is not None and step.semantic_action == frontier.source_action
        step_identity = (
            f"{frontier.ordinal}:{step.completed_repeats}"
            if step is not None
            else f"prerequisite:{frontier.source_action}"
        )
        intent = MacroIntent(
            intent_id=self._intent_id(
                observation,
                CortexRole.MACRO,
                f"{self._macro_plan.plan_id}:{step_identity}",
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
            macro_step_ordinal=(frontier.ordinal if advances_plan_step else None),
        )

    def _apply_terminal_collapse_macro_boundary(
        self,
        observation: ObservationEnvelope,
        assessment: SituationAssessment,
        proposal: MacroPolicyProposal,
        frontier: PolicyActionAssessment,
    ) -> PolicyActionAssessment | None:
        if not is_terminal_collapse(assessment):
            return frontier
        recovery = (
            frontier
            if self._frontier_is_townhall_recovery(frontier)
            else self._terminal_collapse_recovery_frontier(proposal, observation)
        )
        recovery_is_legal = (
            recovery is not None
            and recovery.classification is PolicyActionClassification.MAPPED_LEGAL_NOW
            and self._frontier_is_townhall_recovery(recovery)
            and recovery.runtime_action is not None
            and self._runtime_action_resources_ready(
                recovery.runtime_action,
                observation,
            )
        )
        recovery_ordinal = recovery.ordinal if recovery is not None and recovery_is_legal else None
        obsolete = self._obsolete_terminal_collapse_steps(
            recovery_ordinal=recovery_ordinal,
        )
        if obsolete:
            self._record_terminal_collapse_macro_obsolete(
                observation,
                assessment,
                obsolete,
                plan_frozen=not recovery_is_legal,
            )
        if recovery_is_legal:
            self._terminal_collapse_macro_hold = None
            return recovery
        assert self._macro_plan is not None
        self._macro_plan_frozen = True
        self._urgent_replan_requested = False
        self._next_macro_retry_game_loop = None
        self._terminal_collapse_macro_hold = _TerminalCollapseMacroHold(
            plan_id=self._macro_plan.plan_id,
            townhall_recovery_availability_signature=(
                self._townhall_recovery_availability_signature(observation)
            ),
        )
        return None

    def _terminal_collapse_recovery_frontier(
        self,
        proposal: MacroPolicyProposal,
        observation: ObservationEnvelope,
    ) -> PolicyActionAssessment | None:
        recovery_semantics = {
            self._semantic_action_for_runtime(runtime_action)
            for runtime_action in townhall_recovery_runtime_actions(self._race_profile.data)
        }
        recovery_steps = [
            step for step in proposal.steps if step.canonical_action in recovery_semantics
        ]
        if not recovery_steps:
            return None
        return runtime_frontier(
            proposal.model_copy(update={"steps": recovery_steps}),
            observation,
            self._recent_hima_actions(observation.game_loop),
            self._race_profile.data,
        )

    def _obsolete_terminal_collapse_steps(
        self,
        *,
        recovery_ordinal: int | None,
    ) -> list[MacroStep]:
        assert self._macro_plan is not None
        obsolete: list[MacroStep] = []
        for step in self._macro_plan.steps:
            if self._macro_step_is_complete(step.ordinal) or step.ordinal == recovery_ordinal:
                continue
            obsolete.append(step)
            self._set_macro_step_status(
                step.ordinal,
                MacroStepStatus.OBSOLETE,
                TerminalCollapseReason.MACRO_FRONTIER_OBSOLETE.value,
            )
        return obsolete

    def _record_terminal_collapse_macro_obsolete(
        self,
        observation: ObservationEnvelope,
        assessment: SituationAssessment,
        obsolete: list[MacroStep],
        *,
        plan_frozen: bool,
    ) -> None:
        assert self._macro_plan is not None
        first = min(obsolete, key=lambda step: step.ordinal)
        available_recovery = townhall_recovery_runtime_actions(self._race_profile.data)
        self._record_cortex_event(
            observation,
            "macro_frontier_obsolete",
            {
                "plan_id": self._macro_plan.plan_id,
                "semantic_action": first.semantic_action,
                "runtime_action": first.runtime_actions[0] if first.runtime_actions else None,
                "ordinal": first.ordinal,
                "obsolete_ordinals": sorted(step.ordinal for step in obsolete),
                "proposal_source_game_loop": (
                    self._macro_plan.proposal_source_game_loop
                    if self._macro_plan.proposal_source_game_loop is not None
                    else self._macro_plan.created_game_loop
                ),
                "current_game_loop": observation.game_loop,
                "current_situation_assessment_id": assessment.assessment_id,
                "army_readiness": assessment.army_readiness.value,
                "own_base_count": assessment.bases.own_base_count,
                "own_production_capacity": assessment.bases.own_production_capacity,
                "threat_level": assessment.threat_level.value,
                "source_model_id": self._macro_plan.source_model_id,
                "source_model_version": self._macro_plan.source_model_revision,
                "townhall_recovery_actions_available": sorted(
                    action.name
                    for action in observation.available_actions
                    if action.name in available_recovery
                ),
                "townhall_recovery_availability_signature": list(
                    self._townhall_recovery_availability_signature(observation)
                ),
                "plan_frozen": plan_frozen,
                "reason": TerminalCollapseReason.MACRO_FRONTIER_OBSOLETE.value,
            },
        )

    def _frontier_is_townhall_recovery(
        self,
        frontier: PolicyActionAssessment,
    ) -> bool:
        return self._macro_action_is_townhall_recovery(
            frontier.source_action,
            frontier.runtime_action,
        )

    def _macro_action_is_townhall_recovery(
        self,
        semantic_action: str | None,
        runtime_action: str | None,
    ) -> bool:
        if (
            semantic_action is None
            or runtime_action is None
            or runtime_action not in townhall_recovery_runtime_actions(self._race_profile.data)
        ):
            return False
        return semantic_action == self._semantic_action_for_runtime(runtime_action)

    def _proposal_with_expansion_commitment(
        self,
        proposal: MacroPolicyProposal,
    ) -> MacroPolicyProposal:
        if (
            self._expansion_commitment_id is None
            or not self._expansion_commitment_dispatched
            or self._expansion_candidates_exhausted
            or self._expansion_goal is None
            or self._expansion_goal.observed_base_count >= self._expansion_goal.desired_base_count
            or self._expansion_goal.terminal_state is not None
        ):
            return proposal
        townhall_action = self._townhall_semantic_action()
        if any(step.canonical_action == townhall_action for step in proposal.steps):
            return proposal
        ordinals = [step.ordinal for step in proposal.steps]
        if self._macro_plan is not None:
            ordinals.extend(step.ordinal for step in self._macro_plan.steps)
        commitment_step = MacroActionStep(
            ordinal=max(ordinals, default=-1) + 1,
            canonical_action=townhall_action,
            category="build",
            raw_token=townhall_action,
        )
        return proposal.model_copy(update={"steps": [*proposal.steps, commitment_step]})

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
        supply_action = self._supply_macro_action()
        gas_runtime_action = self._runtime_action_for_target(self._race_profile.data.gas_structure)
        gas_action = self._semantic_action_for_runtime(gas_runtime_action)
        townhall_action = self._townhall_semantic_action()
        fallback_unit_action = self._fallback_unit_macro_action()
        if (
            blocked_frontier.source_action != supply_action
            and free_supply <= self.config.cortex.executor.supply_emergency_free_supply
        ):
            emergency = self._legal_proposal_step(
                proposal,
                observation,
                supply_action,
            )
            if emergency is not None:
                return emergency
        if (
            blocked_frontier.reason_code == "insufficient_vespene"
            and self._needs_more_gas_infrastructure(observation)
        ):
            gas_closure = self._legal_synthetic_action(
                proposal,
                observation,
                gas_runtime_action,
                ordinal=blocked_frontier.ordinal,
            )
            if gas_closure is not None:
                return gas_closure
        prerequisite = self._prerequisite_resolution(blocked_frontier, observation)
        if prerequisite is not None and prerequisite.unique_next_action is not None:
            closure = self._legal_synthetic_action(
                proposal,
                observation,
                prerequisite.unique_next_action,
                ordinal=blocked_frontier.ordinal,
            )
            if closure is not None:
                return closure
        townhall_count = sum(
            structure.unit_type in self._race_profile.data.townhall_types
            for structure in observation.state.own_structures
        )
        gas_saturated_before_expansion = (
            blocked_frontier.source_action == gas_action
            and blocked_frontier.reason_code == "action_unavailable_now"
            and townhall_count < 2
            and observation.state.economy.minerals >= 400
        )
        if (
            blocked_frontier.reason_code != "insufficient_vespene"
            and not gas_saturated_before_expansion
        ):
            return self._legal_independent_frontier(
                proposal,
                observation,
                blocked_frontier,
            )
        fallback_actions = []
        if gas_saturated_before_expansion or (
            observation.state.economy.minerals >= 800 and townhall_count < 2
        ):
            fallback_actions.append(townhall_action)
        fallback_actions.append(fallback_unit_action)
        if free_supply <= self.config.cortex.executor.resource_fallback_pylon_free_supply:
            fallback_actions.append(supply_action)
        if townhall_action not in fallback_actions:
            fallback_actions.append(townhall_action)
        for action_name in fallback_actions:
            fallback = self._legal_proposal_step(
                proposal,
                observation,
                action_name,
            )
            if fallback is not None:
                return fallback
        return self._legal_independent_frontier(
            proposal,
            observation,
            blocked_frontier,
        )

    def _legal_independent_frontier(
        self,
        proposal: MacroPolicyProposal,
        observation: ObservationEnvelope,
        blocked_frontier: PolicyActionAssessment,
    ) -> PolicyActionAssessment | None:
        """Find the earliest later step that is legal from current global state."""

        blocked_domain = (
            None
            if blocked_frontier.runtime_action is None
            else self._race_profile.domain_for_action(blocked_frontier.runtime_action)
        )
        previous_actions = self._recent_hima_actions(observation.game_loop)
        for step in sorted(proposal.steps, key=lambda item: item.ordinal):
            if step.ordinal <= blocked_frontier.ordinal:
                continue
            isolated = proposal.model_copy(update={"steps": [step], "diagnostics": []})
            assessment = runtime_frontier(
                isolated,
                observation,
                previous_actions,
                self._race_profile.data,
            )
            if (
                assessment is None
                or assessment.classification is not PolicyActionClassification.MAPPED_LEGAL_NOW
                or assessment.runtime_action is None
                or self._race_profile.domain_for_action(assessment.runtime_action) is blocked_domain
                or self._saturated_structure_target(assessment, observation) is not None
            ):
                continue
            return assessment
        return None

    def _prerequisite_resolution(
        self,
        frontier: PolicyActionAssessment | None,
        observation: ObservationEnvelope,
    ) -> GoalProgressReport | None:
        if (
            frontier is None
            or frontier.runtime_action is None
            or not (frontier.reason_code or "").startswith("missing_prerequisite_")
        ):
            return None
        try:
            goal = self.goal_progress_verifier.goal_from_action_names(
                strategic_goal=f"Resolve prerequisites for {frontier.runtime_action}",
                action_names=[frontier.runtime_action],
                observation=observation,
                goal_id=f"prerequisite:{frontier.runtime_action}",
            )
        except ValueError:
            return None
        report = self.goal_progress_verifier.verify(observation, goal)
        prerequisite_blockers = {
            GoalBlockerKind.MISSING_PREREQUISITE,
            GoalBlockerKind.PREREQUISITE_IN_PROGRESS,
        }
        if not any(blocker.kind in prerequisite_blockers for blocker in report.blockers):
            return None
        return report

    def _prerequisite_resolution_is_active(
        self,
        frontier: PolicyActionAssessment | None,
        observation: ObservationEnvelope,
    ) -> bool:
        report = self._prerequisite_resolution(frontier, observation)
        if report is None:
            return False
        return report.unique_next_action is not None or any(
            blocker.kind is GoalBlockerKind.PREREQUISITE_IN_PROGRESS for blocker in report.blockers
        )

    def _legal_synthetic_action(
        self,
        proposal: MacroPolicyProposal,
        observation: ObservationEnvelope,
        runtime_action: str,
        *,
        ordinal: int,
    ) -> PolicyActionAssessment | None:
        semantic_action = self._semantic_action_for_runtime(runtime_action)
        prefix = runtime_action.partition("_")[0]
        category: Literal["build", "train", "research"]
        if prefix == "Build":
            category = "build"
        elif prefix == "Train":
            category = "train"
        elif prefix == "Research":
            category = "research"
        else:
            return None
        step = MacroActionStep(
            ordinal=ordinal,
            canonical_action=semantic_action,
            category=category,
            raw_token=semantic_action,
        )
        isolated = proposal.model_copy(update={"steps": [step], "diagnostics": []})
        assessment = runtime_frontier(
            isolated,
            observation,
            self._recent_hima_actions(observation.game_loop),
            self._race_profile.data,
        )
        if (
            assessment is None
            or assessment.classification is not PolicyActionClassification.MAPPED_LEGAL_NOW
        ):
            return None
        return assessment

    def _fallback_reason(
        self,
        blocked: PolicyActionAssessment,
        fallback: PolicyActionAssessment,
        observation: ObservationEnvelope,
    ) -> str:
        if (
            fallback.source_action == self._supply_macro_action()
            and self._free_supply(observation)
            <= self.config.cortex.executor.supply_emergency_free_supply
        ):
            return "supply_emergency"
        if (blocked.reason_code or "").startswith("missing_prerequisite_"):
            return "prerequisite_closure"
        if (
            blocked.reason_code == "insufficient_vespene"
            and fallback.source_action
            == self._semantic_action_for_target(self._race_profile.data.gas_structure)
        ):
            return "prerequisite_closure"
        if blocked.reason_code == "insufficient_vespene":
            return "resource_fallback"
        if fallback.ordinal > blocked.ordinal:
            return "independent_frontier"
        return "resource_fallback"

    def _has_gas_infrastructure(self, observation: ObservationEnvelope) -> bool:
        gas_structure = self._race_profile.data.gas_structure.casefold()
        return any(
            structure.unit_type.casefold() == gas_structure
            for structure in observation.state.own_structures
        )

    def _needs_more_gas_infrastructure(self, observation: ObservationEnvelope) -> bool:
        """Return whether gas-starved macro should close gas infrastructure first."""

        gas_structure = self._race_profile.data.gas_structure.casefold()
        completed_townhalls = sum(
            structure.unit_type in self._race_profile.data.townhall_types
            and structure.status != "constructing"
            for structure in observation.state.own_structures
        )
        if completed_townhalls <= 0:
            return False
        gas_structures = sum(
            structure.unit_type.casefold() == gas_structure
            for structure in observation.state.own_structures
        )
        return gas_structures < completed_townhalls * 2

    def _supply_macro_action(self) -> str:
        return self._semantic_action_for_target(self._race_profile.data.supply_provider)

    def _pending_structure_target(
        self,
        frontier: PolicyActionAssessment,
        observation: ObservationEnvelope,
    ) -> str | None:
        if frontier.runtime_action is None:
            return None
        spec = next(
            (
                item
                for item in self._race_profile.data.progress_action_specs
                if item.name == frontier.runtime_action and item.effect_kind.value == "structure"
            ),
            None,
        )
        if spec is None:
            return None
        constructing = sum(
            structure.unit_type.casefold() == spec.effect_target.casefold()
            and structure.status == "constructing"
            for structure in observation.state.own_structures
        )
        step = self._macro_step(frontier.ordinal)
        completed_in_plan = 0 if step is None else step.completed_repeats
        return spec.effect_target if constructing > completed_in_plan else None

    def _saturated_structure_target(
        self,
        frontier: PolicyActionAssessment,
        observation: ObservationEnvelope,
    ) -> str | None:
        if frontier.runtime_action is None:
            return None
        spec = next(
            (
                item
                for item in self._race_profile.data.progress_action_specs
                if item.name == frontier.runtime_action and item.effect_kind.value == "structure"
            ),
            None,
        )
        if spec is None:
            return None
        limit = self._race_profile.data.structure_saturation_limits.get(spec.effect_target)
        if limit is None:
            return None
        current = sum(
            structure.unit_type.casefold() == spec.effect_target.casefold()
            for structure in observation.state.own_structures
        )
        return spec.effect_target if current >= limit else None

    def _fallback_unit_macro_action(self) -> str:
        worker = self._race_profile.data.worker_type.casefold()
        candidates = [
            spec
            for spec in self._race_profile.data.progress_action_specs
            if spec.effect_kind.value == "unit"
            and spec.effect_target.casefold() != worker
            and spec.vespene == 0
        ]
        if not candidates:
            raise RuntimeError(f"{self._race_profile.race.value} profile has no fallback unit")
        selected = min(candidates, key=lambda spec: (spec.minerals, spec.name))
        return self._semantic_action_for_runtime(selected.name)

    def _semantic_action_for_target(self, effect_target: str) -> str:
        return self._semantic_action_for_runtime(self._runtime_action_for_target(effect_target))

    def _runtime_action_for_target(self, effect_target: str) -> str:
        matching = [
            spec.name
            for spec in self._race_profile.data.progress_action_specs
            if spec.effect_target.casefold() == effect_target.casefold()
        ]
        if not matching:
            raise RuntimeError(
                f"{self._race_profile.race.value} profile has no action for {effect_target}"
            )
        return matching[0]

    def _semantic_action_for_runtime(self, runtime_action: str) -> str:
        for mapping in self._race_profile.data.macro_action_mappings:
            if runtime_action in mapping.runtime_actions:
                return mapping.semantic_action
        raise RuntimeError(
            f"{self._race_profile.race.value} profile has no HIMA mapping for {runtime_action}"
        )

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
                self._race_profile.data,
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
            objective=self._reflex_objective(command.name),
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

    @staticmethod
    def _reflex_objective(action_name: str) -> str:
        if action_name == "Retreat":
            return "Protect threatened units immediately"
        if action_name == "Effect_InjectLarva":
            return "Maintain deterministic Zerg larva production"
        if action_name == "Build_CreepTumor_Queen_Screen":
            return "Extend deterministic Zerg creep coverage"
        if action_name == "Build_CreepTumor_Tumor_Screen":
            return "Continue deterministic Zerg creep expansion from a mature tumor"
        if action_name in {"Train_Probe", "Train_SCV"}:
            race = "Protoss" if action_name == "Train_Probe" else "Terran"
            return f"Maintain deterministic {race} worker production"
        if action_name == "Morph_OrbitalCommand":
            return "Upgrade the Terran economy to Orbital Command"
        if action_name == "Effect_CalldownMULE_Screen":
            return "Convert available Orbital energy into mineral income"
        return "Respond to the visible enemy threat"

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
        assessment = self._current_situation
        if assessment is None:
            raise RuntimeError("role agents require a current Situation assessment")
        strategic_intent = self._role_agents.evaluate(
            RoleAgentContext(
                observation=observation,
                situation=assessment,
                source_intents=(intent,),
            )
        )[intent.intent_id]
        self._strategic_by_legacy_intent[intent.intent_id] = strategic_intent
        if self._authoritative_build_pre_dispatch_blocks(
            observation,
            strategic_intent,
        ):
            operation_id = strategic_intent.operation_id
            assert operation_id is not None
            state = self._authoritative_build_pre_dispatch_circuits[operation_id]
            self._strategic_by_legacy_intent[intent.intent_id] = strategic_intent.model_copy(
                update={
                    "hard_blockers": tuple(
                        dict.fromkeys(
                            (
                                *strategic_intent.hard_blockers,
                                "authoritative_build_pre_dispatch_circuit_open",
                            )
                        )
                    )
                }
            )
            defer_key = (operation_id, state.blocked_semantic_material_identity)
            if defer_key not in self._authoritative_build_pre_dispatch_defer_signatures:
                self._authoritative_build_pre_dispatch_defer_signatures.add(defer_key)
                self._record_cortex_event(
                    observation,
                    "authoritative_build_pre_dispatch_circuit_defer",
                    {
                        "operation_id": operation_id,
                        "action_name": state.action_name,
                        "streak": state.streak,
                        "threshold": state.threshold,
                        "failure_count": state.failure_count,
                        "last_failure_code": state.last_failure_code,
                        "opened_command_id": state.opened_command_id,
                        "opened_attempt_id": (
                            None
                            if state.opened_command_id is None
                            or state.opened_attempt_ordinal is None
                            else AttemptKey(
                                operation_id=operation_id,
                                command_id=state.opened_command_id,
                                attempt_ordinal=state.opened_attempt_ordinal,
                            ).attempt_id
                        ),
                        "opened_attempt_ordinal": state.opened_attempt_ordinal,
                        "opened_game_loop": state.opened_game_loop,
                        "material_legality_identity": state.material_legality_identity,
                        "blocked_semantic_material_identity": (
                            state.blocked_semantic_material_identity
                        ),
                        "material_evidence_valid": state.material_evidence_valid,
                        "invalid_evidence_reasons": list(state.invalid_evidence_reasons),
                        "operation_epoch_changed": state.operation_epoch_changed,
                        "next_action": "wait_for_material_legality_change_or_new_operation",
                    },
                )
            return None
        if self._raw_build_tombstone_blocks(strategic_intent):
            operation_id = strategic_intent.operation_id
            assert operation_id is not None
            self._strategic_by_legacy_intent[intent.intent_id] = strategic_intent.model_copy(
                update={
                    "hard_blockers": tuple(
                        dict.fromkeys(
                            (*strategic_intent.hard_blockers, "raw_build_material_tombstone")
                        )
                    )
                }
            )
            self._record_cortex_event(
                observation,
                "raw_build_tombstone_defer",
                {
                    "intent_id": intent.intent_id,
                    "operation_id": operation_id,
                    "action_names": list(strategic_intent.action_names),
                    "material_legality_identities": sorted(
                        self._raw_build_material_tombstones[operation_id]
                    ),
                    "circuit_open": operation_id in self._raw_build_material_tombstone_circuits,
                    "next_action": "replan_with_new_builder_target_or_ability",
                },
            )
            return None
        self._record_cortex_event(
            observation,
            "role_intent_emitted",
            {
                "legacy_intent_id": intent.intent_id,
                "intent": strategic_intent.model_dump(mode="json"),
            },
        )
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
        context = self._guard_candidate_context(
            observation,
            strategic_intent,
            context,
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
            self._strategic_by_legacy_intent[intent.intent_id] = strategic_intent.model_copy(
                update={"hard_blockers": ("no_executable_candidate",)}
            )
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
        command = command.model_copy(
            update={
                "semantic_source_role": intent.source_role.value,
                "semantic_action": semantic_action,
                "townhall_recovery": (
                    self._macro_action_is_townhall_recovery(
                        semantic_action,
                        command.name,
                    )
                    if intent.source_role is CortexRole.MACRO
                    else None
                ),
            }
        )
        if strategic_intent.operation_id is not None:
            command = command.model_copy(
                update={
                    "operation_id": strategic_intent.operation_id,
                }
            )
        assert selection.candidate_id is not None
        lineage = CommandLineage(
            command_id=command.command_id,
            operation_id=strategic_intent.operation_id,
            attempt_id=None,
            attempt_ordinal=None,
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
            strategic_intent_id=strategic_intent.intent_id,
            responsibility=strategic_intent.role.value,
            playbook_rule_ids=strategic_intent.playbook_rule_ids,
            selected_game_loop=observation.game_loop,
        )
        return _PreparedCommand(
            command=command,
            lineage=lineage,
            semantic_action=semantic_action,
            macro_step_ordinal=macro_step_ordinal,
        )

    def _bind_dispatch_attempt(self, prepared: _PreparedCommand) -> _PreparedCommand:
        operation_id = prepared.lineage.operation_id
        if operation_id is None:
            return prepared
        if prepared.lineage.attempt_id is not None or prepared.command.attempt_id is not None:
            raise RuntimeError("dispatch attempt was bound before accepted dispatch")
        attempt_ordinal = self._attempt_ordinals.get(operation_id, 0)
        attempt_id = AttemptKey(
            operation_id=operation_id,
            command_id=prepared.command.command_id,
            attempt_ordinal=attempt_ordinal,
        ).attempt_id
        self._attempt_ordinals[operation_id] = attempt_ordinal + 1
        return _PreparedCommand(
            command=prepared.command.model_copy(
                update={
                    "operation_id": operation_id,
                    "attempt_id": attempt_id,
                    "attempt_ordinal": attempt_ordinal,
                }
            ),
            lineage=prepared.lineage.model_copy(
                update={
                    "attempt_id": attempt_id,
                    "attempt_ordinal": attempt_ordinal,
                }
            ),
            semantic_action=prepared.semantic_action,
            macro_step_ordinal=prepared.macro_step_ordinal,
        )

    def _apply_strategic_arbitration(
        self,
        observation: ObservationEnvelope,
        prepared: list[_PreparedCommand],
    ) -> list[_PreparedCommand]:
        mode = self.config.cortex.arbiter.mode
        if mode == "disabled" or not self._strategic_by_legacy_intent:
            return prepared
        intents, playbook_deltas, playbook_rule_ids = self._guard_strategic_intents(observation)
        result = self._strategic_arbiter.arbitrate(
            intents,
            observation,
            previous_agenda=self._strategic_agenda,
            playbook_deltas=playbook_deltas,
            playbook_rule_ids=playbook_rule_ids,
        )
        self._pending_strategic_arbitration = result
        selected = set(result.selected_intent_ids)
        actual = tuple(
            sorted(
                self._strategic_by_legacy_intent[item.lineage.intent_id].intent_id
                for item in prepared
            )
        )
        self._record_cortex_event(
            observation,
            "intent_arbitrated",
            {
                "mode": mode,
                "arbitration": result.model_dump(mode="json"),
                "actual_prepared_intent_ids": list(actual),
            },
        )
        decisions = {decision.intent_id: decision for decision in result.decisions}
        annotated = [
            _PreparedCommand(
                command=item.command,
                lineage=item.lineage.model_copy(
                    update={
                        "arbiter_mode": mode,
                        "intent_decision": decisions[
                            self._strategic_by_legacy_intent[item.lineage.intent_id].intent_id
                        ].status.value,
                        "playbook_rule_ids": self._strategic_by_legacy_intent[
                            item.lineage.intent_id
                        ].playbook_rule_ids,
                    }
                ),
                semantic_action=item.semantic_action,
                macro_step_ordinal=item.macro_step_ordinal,
            )
            for item in prepared
        ]
        if mode == "shadow":
            self._record_cortex_event(
                observation,
                "intent_arbiter_shadow_diff",
                {
                    "actual_intent_ids": list(actual),
                    "shadow_selected_intent_ids": list(result.selected_intent_ids),
                    "only_actual": sorted(set(actual) - selected),
                    "only_shadow": sorted(selected - set(actual)),
                },
            )
            return annotated
        return [
            item
            for item in annotated
            if self._strategic_by_legacy_intent[item.lineage.intent_id].intent_id in selected
        ]

    def _commit_dispatched_strategic_agenda(
        self,
        observation: ObservationEnvelope,
        accepted_commands: list[ActionCommand],
        prepared_by_id: dict[str, _PreparedCommand],
    ) -> None:
        result = self._pending_strategic_arbitration
        if result is None:
            return
        dispatched_intent_ids: set[str] = set()
        for command in accepted_commands:
            prepared = prepared_by_id.get(command.command_id)
            if prepared is None or prepared.lineage.strategic_intent_id is None:
                continue
            dispatched_intent_ids.add(prepared.lineage.strategic_intent_id)
        if not dispatched_intent_ids:
            return
        decisions = {
            decision.intent_id: decision
            for decision in result.decisions
            if decision.intent_id in dispatched_intent_ids
        }
        intents = {
            intent.intent_id: intent
            for intent in self._strategic_by_legacy_intent.values()
            if intent.intent_id in dispatched_intent_ids
        }
        if set(decisions) != set(intents):
            raise RuntimeError("dispatched strategic agenda lost intent decision provenance")
        claims = [intent.resource_claim for intent in intents.values()]
        self._strategic_agenda = StrategicAgenda(
            run_id=observation.run_id,
            episode_id=observation.episode_id,
            game_loop=observation.game_loop,
            active_intent_ids=tuple(sorted(dispatched_intent_ids)),
            active_continuity_keys=tuple(
                sorted(intent.continuity_key for intent in intents.values())
            ),
            reserved_resources=ResourceClaim(
                minerals=sum(claim.minerals for claim in claims),
                vespene=sum(claim.vespene for claim in claims),
                supply=sum(claim.supply for claim in claims),
                reservation_game_loops=max(
                    (claim.reservation_game_loops for claim in claims),
                    default=1,
                ),
            ),
            commitment_until_game_loop=max(
                (
                    observation.game_loop + intent.resource_claim.reservation_game_loops
                    for intent in intents.values()
                ),
                default=observation.game_loop,
            ),
            total_score=sum(decision.score.total for decision in decisions.values()),
        )
        self._record_cortex_event(
            observation,
            "strategic_agenda_committed",
            {
                "agenda": self._strategic_agenda.model_dump(mode="json"),
                "command_ids": sorted(command.command_id for command in accepted_commands),
            },
        )

    def _guard_strategic_intents(
        self,
        observation: ObservationEnvelope,
    ) -> tuple[tuple[StrategicIntent, ...], dict[str, float], dict[str, tuple[str, ...]]]:
        mode = self.config.cortex.playbook.rule_mode
        assessment = self._current_situation
        if mode == "disabled" or assessment is None or not self._available_playbook_rules():
            return tuple(self._strategic_by_legacy_intent.values()), {}, {}
        context = self._playbook_context(assessment)
        guarded: list[StrategicIntent] = []
        deltas: dict[str, float] = {}
        rule_ids: dict[str, tuple[str, ...]] = {}
        for legacy_id, intent in tuple(self._strategic_by_legacy_intent.items()):
            rules = self._available_playbook_rules()
            result = self._playbook_intent_guard.evaluate(
                intent,
                context=context,
                situation=assessment,
                rules=rules,
                game_loop=observation.game_loop,
                behavior_before_hash=counterfactual_observation_fingerprint(observation),
                mode=mode,
            )
            updated = intent
            if result.blocked:
                updated = intent.model_copy(
                    update={
                        "hard_blockers": tuple(
                            dict.fromkeys((*intent.hard_blockers, "playbook_hard_rule"))
                        ),
                        "playbook_rule_ids": result.rule_ids,
                    }
                )
            elif result.rule_ids:
                updated = intent.model_copy(update={"playbook_rule_ids": result.rule_ids})
            self._strategic_by_legacy_intent[legacy_id] = updated
            guarded.append(updated)
            deltas[updated.intent_id] = result.score_delta
            rule_ids[updated.intent_id] = result.rule_ids
            self._record_playbook_applications(observation, result.applications)
        return tuple(guarded), deltas, rule_ids

    def _guard_candidate_context(
        self,
        observation: ObservationEnvelope,
        intent: StrategicIntent,
        context: FastExecutorContext,
    ) -> FastExecutorContext:
        mode = self.config.cortex.playbook.rule_mode
        assessment = self._current_situation
        for signature, feedback in tuple(self._recent_terminal_feedback.items()):
            if feedback.expires_game_loop < observation.game_loop:
                del self._recent_terminal_feedback[signature]
        recent_feedback = tuple(self._recent_terminal_feedback.values())
        if (
            mode == "disabled"
            or assessment is None
            or (not self._available_playbook_rules() and not recent_feedback)
        ):
            return context
        playbook_context = self._playbook_context(assessment)
        candidates = []
        for candidate in context.candidates:
            rules = self._available_playbook_rules()
            result = self._playbook_candidate_guard.evaluate(
                candidate,
                role=intent.role.value,
                context=playbook_context,
                situation=assessment,
                rules=rules,
                run_id=observation.run_id,
                episode_id=observation.episode_id,
                step_id=observation.step_id,
                game_loop=observation.game_loop,
                behavior_before_hash=counterfactual_observation_fingerprint(observation),
                mode=mode,
                recent_feedback=recent_feedback,
                operation_id=intent.operation_id,
                attempt_ordinal=(
                    None
                    if intent.operation_id is None
                    else self._attempt_ordinals.get(intent.operation_id, 0)
                ),
            )
            self._record_playbook_applications(observation, result.applications)
            if result.blocked:
                continue
            if mode == "active" and result.score_delta:
                candidate = candidate.model_copy(
                    update={
                        "features": candidate.features.model_copy(
                            update={"playbook_score": result.score_delta}
                        )
                    }
                )
            candidates.append(candidate)
        return context.model_copy(update={"candidates": candidates})

    def _record_playbook_applications(
        self,
        observation: ObservationEnvelope,
        applications: tuple[PlaybookRuleApplication, ...],
    ) -> None:
        rules_by_id = {rule.rule_id: rule for rule in self._playbook_rules}
        for application in applications:
            if self._playbook_store is not None:
                self._playbook_store.record_rule_application(application)
            self._record_cortex_event(
                observation,
                "playbook_rule_applied",
                application,
            )
            rule = rules_by_id.get(application.rule_id)
            if rule is None:
                continue
            if rule.evidence.get("canary_fixture") is True and application.reason in {
                "shadow_would_block",
                "rule_blocked",
            }:
                self._consumed_canary_fixture_rule_ids.add(rule.rule_id)
            would_block = application.reason in {"shadow_would_block", "rule_blocked"}
            actual_outcome: Literal["pending", "blocked", "allowed"]
            if would_block and self.config.cortex.playbook.rule_mode == "shadow":
                actual_outcome = "pending"
                false_block = None
            elif application.blocked:
                actual_outcome = "blocked"
                false_block = None
            else:
                actual_outcome = "allowed"
                false_block = False
            digest = hashlib.sha256(
                (f"{application.application_id}|{rule.strength.value}|{rule.status.value}").encode()
            ).hexdigest()
            evaluation = PlaybookRuleEvaluation(
                evaluation_id=f"rule-evaluation:{digest}",
                application_id=application.application_id,
                rule_id=rule.rule_id,
                run_id=application.run_id,
                episode_id=application.episode_id,
                step_id=application.step_id,
                game_loop=application.game_loop,
                target_kind=application.target_kind,
                target_id=application.target_id,
                rule_kind=application.rule_kind or evaluation_kind(rule.category),
                action_name=application.action_name,
                role=application.role,
                counterfactual_key=(
                    application.counterfactual_key
                    or self._playbook_counterfactual_key(application, rule)
                ),
                counterfactual_signature=application.counterfactual_signature,
                behavior_before_hash=application.behavior_before_hash,
                decision_epoch=application.decision_epoch,
                rule_fingerprint=application.rule_fingerprint,
                predicate_fingerprint=application.predicate_fingerprint,
                counterfactual_observable=False,
                strategic_outcome_window_end_game_loop=(
                    application.game_loop + 448
                    if (application.rule_kind or evaluation_kind(rule.category))
                    is PlaybookRuleKind.STRATEGY
                    else None
                ),
                strength_at_evaluation=rule.strength,
                status_at_evaluation=rule.status,
                shadow_decision="would_block" if would_block else "would_allow",
                actual_outcome=actual_outcome,
                execution_false_block=false_block,
                false_block=false_block,
            )
            self._record_cortex_event(
                observation,
                "playbook_rule_evaluated",
                evaluation,
            )
            if actual_outcome == "pending":
                self._pending_playbook_rule_evaluations[evaluation.evaluation_id] = evaluation

    def _available_playbook_rules(self) -> tuple[PlaybookRule, ...]:
        return tuple(
            rule
            for rule in self._playbook_rules
            if rule.rule_id not in self._consumed_canary_fixture_rule_ids
        )

    def _restore_consumed_canary_fixture_rules(
        self,
        observation: ObservationEnvelope,
    ) -> None:
        if self._playbook_store is None:
            return
        fixture_rule_ids = {
            rule.rule_id
            for rule in self._playbook_store.rules()
            if rule.evidence.get("canary_fixture") is True
        }
        if not fixture_rule_ids:
            return
        for event in self.store.events_of_type(
            observation.run_id,
            observation.episode_id,
            "playbook_rule_applied",
        ):
            rule_id = event.payload.get("rule_id")
            reason = event.payload.get("reason")
            if rule_id in fixture_rule_ids and reason in {"shadow_would_block", "rule_blocked"}:
                self._consumed_canary_fixture_rule_ids.add(str(rule_id))

    @staticmethod
    def _playbook_counterfactual_key(
        application: PlaybookRuleApplication,
        rule: PlaybookRule,
    ) -> str:
        digest = hashlib.sha256(
            (
                f"{rule.rule_id}|{application.target_kind}|"
                f"{application.action_name or ''}|{application.role or ''}"
            ).encode()
        ).hexdigest()
        return f"counterfactual:{digest}"

    def _mark_playbook_counterfactual_observable(
        self,
        observation: ObservationEnvelope,
        lineage: CommandLineage,
    ) -> None:
        target_ids = {lineage.candidate_id}
        if lineage.strategic_intent_id is not None:
            target_ids.add(lineage.strategic_intent_id)
        for evaluation_id, evaluation in tuple(self._pending_playbook_rule_evaluations.items()):
            if evaluation.target_id not in target_ids:
                continue
            observable = evaluation.model_copy(update={"counterfactual_observable": True})
            self._pending_playbook_rule_evaluations[evaluation_id] = observable
            self._record_cortex_event(
                observation,
                "playbook_rule_evaluated",
                observable,
            )

    def _resolve_playbook_rule_evaluations(
        self,
        report: ExecutionReport,
        lineage: CommandLineage | None,
    ) -> None:
        if lineage is None:
            return
        target_ids = {lineage.candidate_id}
        if lineage.strategic_intent_id is not None:
            target_ids.add(lineage.strategic_intent_id)
        matching = [
            evaluation
            for evaluation in self._pending_playbook_rule_evaluations.values()
            if evaluation.target_id in target_ids and evaluation.counterfactual_observable
        ]
        for evaluation in matching:
            actual_outcome = (
                "satisfied_by_peer"
                if report.failure_code == "engagement_target_eliminated"
                else report.status.value
            )
            execution_false_block = (
                (
                    True
                    if report.status is ExecutionStatus.SUCCEEDED
                    else False
                    if report.status is ExecutionStatus.FAILED
                    else None
                )
                if evaluation.rule_kind is PlaybookRuleKind.EXECUTION_GUARD
                else None
            )
            resolved = evaluation.model_copy(
                update={
                    "step_id": report.step_id,
                    "game_loop": self._execution_game_loop(report),
                    "actual_outcome": actual_outcome,
                    "execution_false_block": execution_false_block,
                    "false_block": execution_false_block,
                }
            )
            self.store.append_event(
                run_id=report.run_id,
                episode_id=report.episode_id,
                step_id=report.step_id,
                event_type="playbook_rule_evaluated",
                payload=resolved,
            )
            del self._pending_playbook_rule_evaluations[evaluation.evaluation_id]
            if evaluation.rule_kind is PlaybookRuleKind.STRATEGY:
                self._terminal_strategy_rule_evaluations[evaluation.evaluation_id] = resolved

    def _finalize_unselected_playbook_evaluations(self, result: EpisodeResult) -> None:
        for evaluation in tuple(self._pending_playbook_rule_evaluations.values()):
            terminal = evaluation.model_copy(
                update={
                    "step_id": result.steps,
                    "actual_outcome": "not_selected",
                    "execution_false_block": None,
                    "false_block": None,
                }
            )
            self.store.append_event(
                run_id=result.run_id,
                episode_id=result.episode_id,
                step_id=result.steps,
                event_type="playbook_rule_evaluated",
                payload=terminal,
            )
        self._pending_playbook_rule_evaluations.clear()

    def _resolve_strategic_rule_evaluations(
        self,
        result: EpisodeResult,
        consequences: tuple[Any, ...],
    ) -> None:
        negative_types = {
            "threat_unanswered",
            "expansion_delayed",
            "production_imbalance",
            "timing_attack_failed",
            "unnecessary_retreat",
            "advantage_not_converted",
        }
        for evaluation in self._terminal_strategy_rule_evaluations.values():
            window_end = (
                evaluation.game_loop + 448
                if evaluation.strategic_outcome_window_end_game_loop is None
                else evaluation.strategic_outcome_window_end_game_loop
            )
            relevant = [
                consequence
                for consequence in consequences
                if not consequence.censored
                and consequence.end_game_loop >= evaluation.game_loop
                and consequence.start_game_loop <= window_end
                if (
                    evaluation.action_name is None
                    or consequence.semantic_action == evaluation.action_name
                )
                and (
                    evaluation.role is None
                    or consequence.role is None
                    or consequence.role == evaluation.role
                )
            ]
            regret = (
                True
                if any(item.consequence_type.value in negative_types for item in relevant)
                else False
                if any(
                    item.consequence_type.value == "successful_key_decision" for item in relevant
                )
                else None
            )
            resolved = evaluation.model_copy(update={"strategic_regret": regret})
            self.store.append_event(
                run_id=result.run_id,
                episode_id=result.episode_id,
                step_id=result.steps,
                event_type="playbook_rule_evaluated",
                payload=resolved,
            )
        self._terminal_strategy_rule_evaluations.clear()

    def _guard_terminal_collapse_macro_dispatch(
        self,
        observation: ObservationEnvelope,
        prepared: _PreparedCommand,
    ) -> ValidationFailure | None:
        assessment = self._current_situation
        if (
            prepared.lineage.source_role is not CortexRole.MACRO
            or assessment is None
            or not is_terminal_collapse(assessment)
            or self._macro_action_is_townhall_recovery(
                prepared.semantic_action,
                prepared.command.name,
            )
        ):
            return None
        self._record_cortex_event(
            observation,
            "terminal_collapse_non_recovery_macro_dispatch",
            {
                "command_id": prepared.command.command_id,
                "plan_id": prepared.lineage.macro_plan_id,
                "semantic_action": prepared.semantic_action,
                "runtime_action": prepared.command.name,
                "current_game_loop": observation.game_loop,
                "current_situation_assessment_id": assessment.assessment_id,
                "army_readiness": assessment.army_readiness.value,
                "own_base_count": assessment.bases.own_base_count,
                "own_production_capacity": assessment.bases.own_production_capacity,
                "threat_level": assessment.threat_level.value,
                "reason": TerminalCollapseReason.NON_RECOVERY_MACRO_DISPATCH.value,
            },
        )
        return ValidationFailure(
            command=prepared.command,
            reason=TerminalCollapseReason.NON_RECOVERY_MACRO_DISPATCH.value,
            disposition=ValidationDisposition.OBSOLETE,
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
        assessment = self._current_situation
        terminal_collapse = assessment is not None and is_terminal_collapse(assessment)
        townhall_recovery = (
            self._macro_action_is_townhall_recovery(
                prepared.semantic_action,
                prepared.command.name,
            )
            if prepared.lineage.source_role is CortexRole.MACRO
            else None
        )
        self._record_cortex_event(
            observation,
            "command_lineage",
            {
                "lineage": prepared.lineage.model_dump(mode="json"),
                "command_id": prepared.command.command_id,
                "macro_plan_id": prepared.lineage.macro_plan_id,
                "semantic_action": prepared.semantic_action,
                "macro_step_ordinal": prepared.macro_step_ordinal,
                "terminal_collapse": terminal_collapse,
                "townhall_recovery": townhall_recovery,
            },
        )

    def _semantic_build_material_identity(
        self,
        observation: ObservationEnvelope,
        intent: StrategicIntent | str,
        *,
        world_target: tuple[float, float] | None,
        builder_tag: int | None,
        ability_id: int | None,
    ) -> str | None:
        """Hash validated material Build legality without revision churn."""

        identity, _, _ = self._semantic_build_material_evidence(
            observation,
            intent,
            world_target=world_target,
            builder_tag=builder_tag,
            ability_id=ability_id,
        )
        return identity

    def _semantic_build_material_evidence(
        self,
        observation: ObservationEnvelope,
        intent: StrategicIntent | str,
        *,
        world_target: tuple[float, float] | None,
        builder_tag: int | None,
        ability_id: int | None,
    ) -> tuple[str | None, tuple[str, ...], int | None]:
        """Return a typed identity only after binding exact material evidence."""

        action_names = (intent,) if isinstance(intent, str) else tuple(intent.action_names)
        matching_actions = [
            action for action in observation.available_actions if action.name in action_names
        ]
        actor_scopes = sorted(
            {scope for action in matching_actions for scope in action.actor_scopes}
        )

        def unit_tag(unit: Any) -> int | None:
            try:
                parsed = int(unit.unit_id, 0)
            except (TypeError, ValueError):
                return None
            return parsed if parsed > 0 else None

        invalid_reasons: list[str] = []
        if not matching_actions:
            invalid_reasons.append("build_action_unavailable")
        if builder_tag is None or builder_tag <= 0:
            invalid_reasons.append("builder_tag_missing")
        if ability_id is None or ability_id <= 0:
            invalid_reasons.append("ability_id_missing")
        if (
            world_target is None
            or len(world_target) != 2
            or not all(math.isfinite(float(value)) for value in world_target)
        ):
            invalid_reasons.append("world_target_missing")

        action_spec = next(
            (
                CANONICAL_PLACEMENT_SPECS[action_name]
                for action_name in action_names
                if action_name in CANONICAL_PLACEMENT_SPECS
            ),
            None,
        )
        if action_spec is None:
            invalid_reasons.append("placement_spec_unavailable")

        exact_builder = next(
            (
                unit
                for unit in observation.state.own_units
                if builder_tag is not None and unit_tag(unit) == int(builder_tag)
            ),
            None,
        )
        scope_bound_builders = [
            unit
            for unit in observation.state.own_units
            if set(unit.actor_scopes) & set(actor_scopes)
        ]
        if exact_builder is not None and (
            not actor_scopes or exact_builder in scope_bound_builders
        ):
            bound_builder = exact_builder
        elif len(scope_bound_builders) == 1:
            bound_builder = scope_bound_builders[0]
        elif exact_builder is not None and not scope_bound_builders:
            bound_builder = exact_builder
        else:
            bound_builder = None
            invalid_reasons.append(
                "builder_binding_ambiguous" if scope_bound_builders else "builder_not_observed"
            )
        bound_builder_tag = None if bound_builder is None else unit_tag(bound_builder)
        if bound_builder is not None and bound_builder_tag is None:
            invalid_reasons.append("builder_tag_invalid")

        if invalid_reasons:
            return None, tuple(dict.fromkeys(invalid_reasons)), bound_builder_tag

        assert world_target is not None
        assert action_spec is not None
        assert bound_builder is not None
        assert bound_builder_tag is not None
        target_cells = canonical_footprint_cells(world_target, action_spec)
        structure_specs = {
            spec.structure_type.casefold(): spec for spec in CANONICAL_PLACEMENT_SPECS.values()
        }

        def occupied_cells(unit: Any) -> frozenset[tuple[int, int]]:
            if unit.position is None:
                return frozenset()
            spec = structure_specs.get(unit.unit_type.casefold())
            if spec is not None:
                return canonical_footprint_cells(unit.position, spec)
            return frozenset({(math.floor(unit.position[0]), math.floor(unit.position[1]))})

        def occupancy_identity(unit: Any) -> list[Any]:
            return [
                unit.unit_id,
                unit.unit_type,
                unit.alliance,
                None
                if unit.position is None
                else [round(unit.position[0], 2), round(unit.position[1], 2)],
            ]

        state = observation.state
        target_occupants = [
            unit
            for unit in (*state.own_units, *state.visible_enemies)
            if unit_tag(unit) != bound_builder_tag and target_cells & occupied_cells(unit)
        ]
        target_occupants.extend(
            unit for unit in state.own_structures if target_cells & occupied_cells(unit)
        )
        powered_structure = bool(
            action_spec is not None
            and action_spec.structure_type.casefold() in _PROTOSS_POWERED_STRUCTURE_TYPES
        )
        nearby_power_sources = [
            unit
            for unit in state.own_structures
            if powered_structure
            and world_target is not None
            and unit.position is not None
            and unit.unit_type.casefold() == "pylon"
            and math.dist(unit.position, world_target) <= 7.0
        ]
        payload = {
            "action_names": sorted(action_names),
            "builder_tag": bound_builder_tag,
            "ability_id": ability_id,
            "builder": [
                bound_builder.unit_id,
                bound_builder.unit_type,
                bound_builder.alliance,
            ],
            "target_occupancy": sorted(
                (occupancy_identity(unit) for unit in target_occupants),
                key=lambda item: item[0],
            ),
            "nearby_power_sources": sorted(
                (occupancy_identity(unit) for unit in nearby_power_sources),
                key=lambda item: item[0],
            ),
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return f"semantic-build-material:{digest}", (), bound_builder_tag

    def _authoritative_build_pre_dispatch_blocks(
        self,
        observation: ObservationEnvelope,
        intent: StrategicIntent,
    ) -> bool:
        operation_id = intent.operation_id
        if operation_id is None or not any(
            action.startswith("Build_") for action in intent.action_names
        ):
            return False
        state = self._authoritative_build_pre_dispatch_circuits.get(operation_id)
        if state is None or not state.circuit_open:
            return False
        if not state.material_evidence_valid:
            return True
        current_identity, invalid_reasons, bound_builder_tag = (
            self._semantic_build_material_evidence(
                observation,
                intent,
                world_target=state.world_target,
                builder_tag=state.builder_tag,
                ability_id=state.ability_id,
            )
        )
        if current_identity is None:
            signature = (operation_id, None)
            if signature not in self._authoritative_build_pre_dispatch_defer_signatures:
                self._authoritative_build_pre_dispatch_defer_signatures.add(signature)
                self._record_cortex_event(
                    observation,
                    "authoritative_build_pre_dispatch_invalid_evidence",
                    {
                        "operation_id": operation_id,
                        "action_name": state.action_name,
                        "reason": "current_material_evidence_invalid",
                        "invalid_evidence_reasons": list(invalid_reasons),
                        "builder_tag": state.builder_tag,
                        "bound_builder_tag": bound_builder_tag,
                        "ability_id": state.ability_id,
                        "world_target": state.world_target,
                        "circuit_open": True,
                    },
                )
            return True
        if current_identity == state.blocked_semantic_material_identity:
            return True
        self._authoritative_build_pre_dispatch_circuits.pop(operation_id, None)
        self._authoritative_build_pre_dispatch_defer_signatures = {
            key
            for key in self._authoritative_build_pre_dispatch_defer_signatures
            if key[0] != operation_id
        }
        self._record_cortex_event(
            observation,
            "authoritative_build_pre_dispatch_circuit_reset",
            {
                "operation_id": operation_id,
                "action_name": state.action_name,
                "reason": "semantic_legality_material_change",
                "previous_semantic_material_identity": (state.blocked_semantic_material_identity),
                "current_semantic_material_identity": current_identity,
                "raw_material_legality_identity": state.material_legality_identity,
                "previous_builder_tag": state.builder_tag,
                "bound_builder_tag": bound_builder_tag,
                "ability_id": state.ability_id,
                "world_target": state.world_target,
                "game_loop": observation.game_loop,
                "operation_epoch_changed": False,
                "reset_from_streak": state.streak,
            },
        )
        return False

    @staticmethod
    def _optional_builder_tag(value: Any) -> int | None:
        if value is None:
            return None
        try:
            parsed = int(value, 0) if isinstance(value, str) else int(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None

    @staticmethod
    def _optional_world_target(value: Any) -> tuple[float, float] | None:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            return None
        try:
            point = float(value[0]), float(value[1])
        except (TypeError, ValueError):
            return None
        return point if all(math.isfinite(item) for item in point) else None

    @staticmethod
    def _optional_nonnegative_int(value: Any) -> int | None:
        if isinstance(value, bool):
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed >= 0 else None

    def _authoritative_material_reset_is_valid(
        self,
        *,
        state: AuthoritativeBuildCircuitState,
        authoritative: dict[str, Any],
        latest: ObservationEnvelope | None,
    ) -> bool:
        if not state.material_evidence_valid or latest is None:
            return False
        action_name = authoritative.get("action_name")
        material_identity = authoritative.get("material_legality_identity")
        builder_tag = self._optional_builder_tag(authoritative.get("builder_tag"))
        ability_id = self._optional_nonnegative_int(authoritative.get("ability_id"))
        world_target = self._optional_world_target(authoritative.get("world_target"))
        if (
            not isinstance(action_name, str)
            or not action_name.startswith("Build_")
            or not isinstance(material_identity, str)
            or _BUILD_LEGALITY_IDENTITY.fullmatch(material_identity) is None
            or builder_tag is None
            or ability_id is None
            or ability_id <= 0
            or world_target is None
        ):
            return False
        current_identity, invalid_reasons, bound_builder_tag = (
            self._semantic_build_material_evidence(
                latest,
                action_name,
                world_target=world_target,
                builder_tag=builder_tag,
                ability_id=ability_id,
            )
        )
        if (
            current_identity is None
            or invalid_reasons
            or current_identity == state.blocked_semantic_material_identity
        ):
            return False
        reason = authoritative.get("material_change_reason")
        if reason == "builder_changed":
            return (
                bound_builder_tag is not None
                and state.builder_tag is not None
                and bound_builder_tag != state.builder_tag
                and builder_tag == bound_builder_tag
            )
        if reason == "ability_changed":
            return state.ability_id is not None and ability_id != state.ability_id
        if reason == "target_state_changed":
            return state.world_target == world_target
        return False

    @staticmethod
    def _authoritative_success_reset_is_valid(
        *,
        authoritative: dict[str, Any],
        transition: dict[str, Any],
    ) -> bool:
        reset_reason = authoritative.get("reset_reason")
        next_state = transition.get("next_state")
        return bool(
            reset_reason == "build_started"
            and next_state == "build_started"
            or reset_reason == "effect_confirmed"
            and next_state == "occupied"
        )

    @staticmethod
    def _execution_confirms_authoritative_build_reset(report: ExecutionReport) -> bool:
        evidence = report.effect_evidence
        if (
            report.status is not ExecutionStatus.SUCCEEDED
            or report.operation_id is None
            or report.action_name is None
            or not report.action_name.startswith("Build_")
            or evidence is None
            or evidence.effect_kind != "build"
        ):
            return False
        return bool(
            evidence.build_started
            or evidence.new_structure_tag is not None
            or evidence.confirmation_kind in {"builder_order", "supporting_quorum", "new_structure"}
            or any(
                transition.next_state in {"build_started", "occupied"}
                for transition in evidence.placement_ledger_transitions
            )
        )

    def _remember_raw_build_placement_transition(self, payload: Any) -> None:
        """Replay raw no-start and authoritative pre-dispatch circuit state."""

        if not isinstance(payload, dict):
            return
        transition = payload.get("transition", payload)
        if not isinstance(transition, dict):
            return
        authoritative = transition.get("authoritative_pre_dispatch")
        if not isinstance(authoritative, dict):
            authoritative = payload.get("authoritative_pre_dispatch")
        if isinstance(authoritative, dict):
            nested_operation = authoritative.get("operation_id")
            parent_operation = payload.get("operation_id")
            operation_id = next(
                (
                    value
                    for value in (parent_operation, nested_operation)
                    if isinstance(value, str) and _OPERATION_IDENTITY.fullmatch(value)
                ),
                None,
            )
            if operation_id is not None:
                invalid_reasons: list[str] = []
                if not isinstance(nested_operation, str) or (
                    _OPERATION_IDENTITY.fullmatch(nested_operation) is None
                ):
                    invalid_reasons.append("operation_id_invalid")
                if parent_operation != nested_operation:
                    invalid_reasons.append("parent_operation_mismatch")
                nested_action = authoritative.get("action_name")
                parent_action = payload.get("action_name")
                if not isinstance(nested_action, str) or not nested_action.startswith("Build_"):
                    invalid_reasons.append("build_action_invalid")
                if parent_action != nested_action:
                    invalid_reasons.append("parent_action_mismatch")
                nested_command = authoritative.get("command_id")
                parent_command = payload.get("command_id")
                if not isinstance(nested_command, str) or not nested_command:
                    invalid_reasons.append("command_id_missing")
                if parent_command != nested_command:
                    invalid_reasons.append("parent_command_mismatch")
                nested_attempt = self._optional_nonnegative_int(
                    authoritative.get("attempt_ordinal")
                )
                nested_attempt_id = authoritative.get("attempt_id")
                if (
                    not isinstance(nested_attempt_id, str)
                    or _ATTEMPT_IDENTITY.fullmatch(nested_attempt_id) is None
                    or not isinstance(nested_operation, str)
                    or _OPERATION_IDENTITY.fullmatch(nested_operation) is None
                    or not isinstance(nested_command, str)
                    or not nested_command
                    or nested_attempt is None
                    or nested_attempt_id
                    != AttemptKey(
                        operation_id=nested_operation,
                        command_id=nested_command,
                        attempt_ordinal=nested_attempt,
                    ).attempt_id
                ):
                    invalid_reasons.append("attempt_id_invalid")
                parent_attempt = self._optional_nonnegative_int(payload.get("attempt_ordinal"))
                if parent_attempt != nested_attempt:
                    invalid_reasons.append("parent_attempt_mismatch")
                parent_attempt_id = payload.get("attempt_id")
                if parent_attempt_id != nested_attempt_id:
                    invalid_reasons.append("parent_attempt_id_mismatch")
                if self._optional_nonnegative_int(authoritative.get("threshold")) != 3:
                    invalid_reasons.append("authoritative_threshold_invalid")
                status = str(authoritative.get("status") or "")
                state_transition = str(
                    authoritative.get("state_transition") or authoritative.get("transition") or ""
                )
                latest = (
                    self._latest_observation(
                        str(payload.get("run_id") or self._episode_key[0]),
                        str(payload.get("episode_id") or self._episode_key[1]),
                    )
                    if self._episode_key is not None
                    else None
                )
                if status == "reset" or state_transition == "open_to_reset":
                    existing = self._authoritative_build_pre_dispatch_circuits.get(operation_id)
                    reset_reason = authoritative.get("reset_reason")
                    reset_valid = existing is not None and (
                        self._authoritative_success_reset_is_valid(
                            authoritative=authoritative,
                            transition=transition,
                        )
                        or reset_reason == "material_state_changed"
                        and self._authoritative_material_reset_is_valid(
                            state=existing,
                            authoritative=authoritative,
                            latest=latest,
                        )
                    )
                    if reset_valid and not invalid_reasons:
                        self._authoritative_build_pre_dispatch_circuits.pop(operation_id, None)
                        self._authoritative_build_pre_dispatch_defer_signatures = {
                            key
                            for key in self._authoritative_build_pre_dispatch_defer_signatures
                            if key[0] != operation_id
                        }
                    elif existing is not None:
                        retained_reasons = tuple(
                            dict.fromkeys(
                                (
                                    *existing.invalid_evidence_reasons,
                                    *invalid_reasons,
                                    "authoritative_reset_invalid",
                                )
                            )
                        )
                        self._authoritative_build_pre_dispatch_circuits[operation_id] = (
                            existing.model_copy(
                                update={
                                    "material_evidence_valid": False,
                                    "invalid_evidence_reasons": retained_reasons,
                                }
                            )
                        )
                elif authoritative.get("circuit_open") is True:
                    existing = self._authoritative_build_pre_dispatch_circuits.get(operation_id)
                    if existing is not None and existing.circuit_open:
                        # Post-open Raw feedback is a boundary violation, never a new
                        # authoritative failure or replacement baseline.
                        pass
                    else:
                        action_value = authoritative.get("action_name")
                        action_name = (
                            action_value
                            if isinstance(action_value, str) and action_value.startswith("Build_")
                            else None
                        )
                        world_target = self._optional_world_target(
                            authoritative.get("world_target")
                        )
                        builder_tag = self._optional_builder_tag(authoritative.get("builder_tag"))
                        ability_id = self._optional_nonnegative_int(authoritative.get("ability_id"))
                        if world_target is None:
                            invalid_reasons.append("world_target_missing")
                        if builder_tag is None:
                            invalid_reasons.append("builder_tag_missing")
                        if ability_id is None or ability_id <= 0:
                            invalid_reasons.append("ability_id_missing")
                        material_value = authoritative.get("material_legality_identity")
                        material_identity = (
                            material_value
                            if isinstance(material_value, str)
                            and _BUILD_LEGALITY_IDENTITY.fullmatch(material_value)
                            else None
                        )
                        if material_identity is None:
                            invalid_reasons.append("material_legality_identity_invalid")
                        if latest is None:
                            blocked_semantic_identity = None
                            invalid_reasons.append("observation_missing")
                        elif action_name is None:
                            blocked_semantic_identity = None
                        else:
                            (
                                blocked_semantic_identity,
                                semantic_invalid_reasons,
                                _,
                            ) = self._semantic_build_material_evidence(
                                latest,
                                action_name,
                                world_target=world_target,
                                builder_tag=builder_tag,
                                ability_id=ability_id,
                            )
                            invalid_reasons.extend(semantic_invalid_reasons)
                        streak_value = self._optional_nonnegative_int(authoritative.get("streak"))
                        if streak_value is None or streak_value < 3:
                            invalid_reasons.append("authoritative_open_streak_invalid")
                        if state_transition != "closed_to_open":
                            invalid_reasons.append("closed_to_open_transition_missing")
                        opened_command_id = (
                            nested_command
                            if isinstance(nested_command, str) and nested_command
                            else None
                        )
                        attempt_value = nested_attempt
                        failure_value = authoritative.get("failure_code")
                        last_failure_code = (
                            failure_value
                            if isinstance(failure_value, str) and failure_value
                            else None
                        )
                        if last_failure_code is None:
                            invalid_reasons.append("failure_code_missing")
                        opened_game_loop = self._optional_nonnegative_int(
                            authoritative.get("observation_game_loop")
                        )
                        if opened_game_loop is None:
                            opened_game_loop = self._optional_nonnegative_int(
                                transition.get("game_loop")
                            )
                        if opened_game_loop is None:
                            opened_game_loop = 0
                            invalid_reasons.append("observation_game_loop_missing")
                        normalized_reasons = tuple(dict.fromkeys(invalid_reasons))
                        material_evidence_valid = not normalized_reasons
                        streak = max(3, streak_value or 3)
                        failure_count = max(
                            streak,
                            self._optional_nonnegative_int(authoritative.get("failure_count"))
                            or streak,
                        )
                        self._authoritative_build_pre_dispatch_circuits[operation_id] = (
                            AuthoritativeBuildCircuitState(
                                operation_id=operation_id,
                                action_name=action_name,
                                streak=streak,
                                threshold=3,
                                circuit_open=True,
                                failure_count=failure_count,
                                last_failure_code=last_failure_code,
                                opened_command_id=opened_command_id,
                                opened_attempt_ordinal=attempt_value,
                                opened_game_loop=opened_game_loop,
                                builder_tag=builder_tag,
                                ability_id=ability_id,
                                world_target=world_target,
                                material_legality_identity=material_identity,
                                blocked_semantic_material_identity=(blocked_semantic_identity),
                                material_evidence_valid=material_evidence_valid,
                                invalid_evidence_reasons=normalized_reasons,
                                operation_epoch_changed=bool(
                                    authoritative.get("operation_epoch_changed", False)
                                ),
                            )
                        )
        no_start = transition.get("placement_no_start")
        if not isinstance(no_start, dict):
            no_start = payload.get("placement_no_start")
        if not isinstance(no_start, dict):
            return
        operation_id = no_start.get("operation_id") or payload.get("operation_id")
        if not isinstance(operation_id, str) or not operation_id:
            return
        evidence = no_start.get("evidence")
        if not isinstance(evidence, dict):
            evidence = {}
        material_identity = (
            transition.get("material_legality_identity")
            or no_start.get("material_legality_identity")
            or evidence.get("material_legality_identity")
        )
        status = str(no_start.get("status") or "")
        if status == "reset" or transition.get("next_state") == "occupied":
            self._raw_build_material_tombstones.pop(operation_id, None)
            self._raw_build_material_tombstone_circuits.discard(operation_id)
            return
        if isinstance(material_identity, str) and material_identity:
            self._raw_build_material_tombstones.setdefault(operation_id, set()).add(
                material_identity
            )
        if no_start.get("circuit_open") is True:
            self._raw_build_material_tombstone_circuits.add(operation_id)

    def record_placement_transition(
        self,
        event: PlacementLedgerEvent,
    ) -> Literal["recorded", "already_recorded"]:
        result = super().record_placement_transition(event)
        self._remember_raw_build_placement_transition(event.model_dump(mode="json"))
        return result

    def _raw_build_tombstone_blocks(self, intent: StrategicIntent) -> bool:
        return bool(
            intent.operation_id
            and intent.operation_id in self._raw_build_material_tombstone_circuits
            and self._raw_build_material_tombstones.get(intent.operation_id)
            and any(action_name.startswith("Build_") for action_name in intent.action_names)
        )

    def record_execution(self, report: ExecutionReport) -> None:
        metadata = self._macro_command_steps.get(report.command_id)
        existing = self._terminal_execution_fingerprints.get(report.command_id)
        super().record_execution(report)
        if existing is not None:
            return
        if report.authoritative_pre_dispatch is not None:
            self._remember_raw_build_placement_transition(report.model_dump(mode="json"))
        if report.effect_evidence is not None:
            for placement_transition in report.effect_evidence.placement_ledger_transitions:
                self._remember_raw_build_placement_transition(
                    {
                        "operation_id": report.operation_id,
                        "transition": placement_transition.model_dump(mode="json"),
                    }
                )
        if report.status is ExecutionStatus.SUCCEEDED and report.operation_id is not None:
            self._raw_build_material_tombstones.pop(report.operation_id, None)
            self._raw_build_material_tombstone_circuits.discard(report.operation_id)
        if self._execution_confirms_authoritative_build_reset(report):
            assert report.operation_id is not None
            self._authoritative_build_pre_dispatch_circuits.pop(report.operation_id, None)
            self._authoritative_build_pre_dispatch_defer_signatures = {
                key
                for key in self._authoritative_build_pre_dispatch_defer_signatures
                if key[0] != report.operation_id
            }
        self._remember_terminal_feedback(report)
        lineage = self._command_lineages.get(report.command_id)
        self._resolve_playbook_rule_evaluations(report, lineage)
        responsibility = None if lineage is None else lineage.responsibility
        tactical_responsibilities = {
            RoleId.OFFENSE.value,
            RoleId.FOCUS_FIRE.value,
            RoleId.RETREAT.value,
        }
        if responsibility in tactical_responsibilities and isinstance(
            self._tactical, ExecutionAwareTacticalPolicyProvider
        ):
            tactical_transition = self._tactical.record_execution(
                report,
                game_loop=self._execution_game_loop(report),
            )
            if tactical_transition is not None:
                self.store.append_event(
                    run_id=report.run_id,
                    episode_id=report.episode_id,
                    step_id=report.step_id,
                    event_type=(
                        "tactical_actor_state"
                        if "target_tag" not in tactical_transition
                        else "tactical_target_state"
                    ),
                    payload=tactical_transition,
                )
        defense_transition = self._role_agents.record_execution(
            report,
            responsibility=responsibility,
            game_loop=self._execution_game_loop(report),
        )
        if defense_transition is not None:
            self.store.append_event(
                run_id=report.run_id,
                episode_id=report.episode_id,
                step_id=report.step_id,
                event_type="defense_actor_state",
                payload=defense_transition,
            )
        if metadata is None:
            return
        self._macro_outcome_revision += 1
        if self._macro_inflight_command_id == report.command_id:
            self._macro_inflight_command_id = None
        succeeded = report.status is ExecutionStatus.SUCCEEDED
        if (
            self._macro_plan is not None
            and metadata[0] == self._macro_plan.plan_id
            and metadata[2] is not None
        ):
            self._advance_macro_step(
                metadata[2],
                succeeded=succeeded,
                persist=True,
                report=report,
            )
        if succeeded and report.action_name is not None:
            token = hima_previous_action_for_runtime_action(
                report.action_name,
                self._race_profile.data,
            )
            if token is not None:
                self._previous_hima_actions.append((self._execution_game_loop(report), token))
            if report.action_name == self._townhall_runtime_action():
                self._terminate_expansion_commitment_from_report(
                    report,
                    terminal_state=(
                        "nexus_effect_confirmed"
                        if report.action_name == "Build_Nexus_Near"
                        else "townhall_effect_confirmed"
                    ),
                )
        else:
            if self._recoverable_expansion_failure(report):
                self._record_expansion_anchor_rejection(report)
                self._urgent_replan_requested = False
            else:
                self._macro_plan_frozen = True
                plan = self._macro_plan
                terminal_townhall_failure = (
                    plan is not None
                    and self._current_situation is not None
                    and is_terminal_collapse(self._current_situation)
                    and report.action_name == self._townhall_runtime_action()
                )
                if terminal_townhall_failure:
                    assert plan is not None
                    latest_observation = self._latest_observation(
                        report.run_id,
                        report.episode_id,
                    )
                    self._terminal_collapse_macro_hold = _TerminalCollapseMacroHold(
                        plan_id=plan.plan_id,
                        townhall_recovery_availability_signature=(
                            ("observation-unavailable",)
                            if latest_observation is None
                            else self._townhall_recovery_availability_signature(latest_observation)
                        ),
                    )
                    self._urgent_replan_requested = False
                    self._next_macro_retry_game_loop = None
                else:
                    self._urgent_replan_requested = True

    def _latest_observation(
        self,
        run_id: str,
        episode_id: str,
    ) -> ObservationEnvelope | None:
        event = self.store.last_event(run_id, episode_id, "observation")
        if event is None:
            return None
        return ObservationEnvelope.model_validate(event.payload)

    def _remember_terminal_feedback(self, report: ExecutionReport) -> None:
        if (
            report.action_name is None
            or report.actor is None
            or report.operation_id is None
            or report.attempt_ordinal is None
        ):
            return
        arguments = report.requested_arguments or report.resolved_arguments
        signature = candidate_signature(
            report.action_name,
            report.actor,
            arguments,
        )
        feedback_key = f"{report.operation_id}|{signature}"
        if report.status is ExecutionStatus.SUCCEEDED:
            self._recent_terminal_feedback.pop(feedback_key, None)
            return
        if report.status is not ExecutionStatus.FAILED:
            return
        hard_failure_codes = {
            "bridge_integrity_error",
            "candidate_outside_dispatch",
            "friendly_target",
            "primitive_argument_out_of_range",
            "target_not_attackable_by_actor",
        }
        hard_suppression = (report.failure_code or "") in hard_failure_codes
        cooldown = 336 if hard_suppression else 112
        terminal_game_loop = self._execution_game_loop(report)
        self._recent_terminal_feedback[feedback_key] = RecentTerminalFeedback(
            signature=signature,
            action_name=report.action_name,
            actor=report.actor,
            failure_code=report.failure_code or "unknown_failure",
            command_id=report.command_id,
            operation_id=report.operation_id,
            attempt_ordinal=report.attempt_ordinal,
            terminal_game_loop=terminal_game_loop,
            expires_game_loop=terminal_game_loop + cooldown,
            hard_suppression=hard_suppression,
        )

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
        recoverable_expansion = report is not None and self._recoverable_expansion_failure(report)
        status = (
            MacroStepStatus.CONFIRMED
            if succeeded and completed >= step.repeat
            else MacroStepStatus.PENDING
            if succeeded
            else MacroStepStatus.DEFERRED
            if recoverable_expansion
            else MacroStepStatus.BLOCKED
        )
        updated = step.model_copy(
            update={
                "completed_repeats": min(completed, step.repeat),
                "status": status,
                "reason": (
                    None
                    if succeeded
                    else report.failure_code
                    if recoverable_expansion and report is not None
                    else "execution_failed"
                ),
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

    def _ensure_expansion_commitment(self, observation: ObservationEnvelope) -> None:
        plan = self._macro_plan
        proposal = self._macro_proposal
        if plan is None or proposal is None:
            return
        townhall_action = self._townhall_semantic_action()
        executable_ordinals = {step.ordinal for step in plan.steps}
        expansion_steps = [
            step
            for step in proposal.steps
            if step.ordinal in executable_ordinals and step.canonical_action == townhall_action
        ]
        if not expansion_steps:
            if (
                self._expansion_commitment_id is not None
                and not self._expansion_commitment_dispatched
            ):
                self._terminate_expansion_commitment(
                    observation,
                    terminal_state="strategic_cancellation",
                )
            return
        observed_base_count = self._observed_townhall_count(observation)
        desired_base_count = max(2, max(step.repeat for step in expansion_steps))
        self._ensure_expansion_goal(
            observation,
            desired_base_count=desired_base_count,
            observed_base_count=observed_base_count,
        )
        goal = self._expansion_goal
        if (
            goal is None
            or goal.terminal_state is not None
            or goal.observed_base_count >= goal.desired_base_count
            or observation.game_loop < goal.cooldown_until_game_loop
        ):
            return
        if self._expansion_commitment_id is not None or (
            self._expansion_exhausted_generation is not None
            and self._expansion_exhausted_generation == self._expansion_scout_generation
        ):
            return
        commitment_id = f"{goal.goal_id}:candidate-epoch:{self._expansion_scout_generation}"
        self._expansion_commitment_id = commitment_id
        self._expansion_commitment_generation = self._expansion_scout_generation
        self._expansion_commitment_dispatched = False
        self._expansion_commitment_started_game_loop = observation.game_loop
        self._expansion_anchor_evaluations = []
        self._record_cortex_event(
            observation,
            "expansion_commitment_started",
            {
                "commitment_id": commitment_id,
                "started_game_loop": observation.game_loop,
                "semantic_action": townhall_action,
                "generation_id": self._expansion_commitment_generation,
                "goal": goal.model_dump(mode="json"),
                "terminal_states": [
                    (
                        "nexus_effect_confirmed"
                        if self._race_profile.race.value == "protoss"
                        else "townhall_effect_confirmed"
                    ),
                    "expansion_candidates_exhausted",
                    "strategic_cancellation",
                ],
            },
        )

    def _ensure_expansion_goal(
        self,
        observation: ObservationEnvelope,
        *,
        desired_base_count: int,
        observed_base_count: int,
    ) -> None:
        current = self._expansion_goal
        if current is not None:
            current = current.model_copy(update={"observed_base_count": observed_base_count})
            self._expansion_goal = current
            if observed_base_count >= current.desired_base_count:
                self._terminalize_expansion_goal(
                    observation,
                    terminal_state="desired_base_count_satisfied",
                )
                return
            if desired_base_count <= current.desired_base_count:
                return
            strategic_revision = current.strategic_revision + 1
        else:
            strategic_revision = 0
        key = ExpansionGoalKey(
            run_id=observation.run_id,
            episode_id=observation.episode_id,
            race=self._race_profile.race.value,
            desired_base_count=desired_base_count,
            strategic_revision=strategic_revision,
        )
        self._expansion_goal = ExpansionGoalState(
            goal_id=key.goal_id,
            baseline_base_count=observed_base_count,
            desired_base_count=desired_base_count,
            observed_base_count=observed_base_count,
            strategic_revision=strategic_revision,
            current_candidate_epoch=self._expansion_scout_generation,
        )
        self._record_cortex_event(
            observation,
            "expansion_goal_started",
            self._expansion_goal,
        )

    def _observed_townhall_count(self, observation: ObservationEnvelope) -> int:
        townhall_types = set(self._race_profile.data.townhall_types)
        return sum(
            structure.unit_type in townhall_types and structure.health_fraction > 0.0
            for structure in observation.state.own_structures
        )

    def _terminalize_expansion_goal(
        self,
        observation: ObservationEnvelope,
        *,
        terminal_state: str,
    ) -> None:
        goal = self._expansion_goal
        if goal is None or goal.terminal_state is not None:
            return
        self._expansion_goal = goal.model_copy(update={"terminal_state": terminal_state})
        self._record_cortex_event(
            observation,
            "expansion_goal_terminal",
            self._expansion_goal,
        )

    def _update_expansion_candidate_state(
        self,
        observation: ObservationEnvelope,
    ) -> None:
        if self._expansion_goal is not None:
            observed_base_count = self._observed_townhall_count(observation)
            self._expansion_goal = self._expansion_goal.model_copy(
                update={
                    "observed_base_count": observed_base_count,
                    "current_candidate_epoch": self._expansion_scout_generation,
                }
            )
            if (
                observed_base_count >= self._expansion_goal.desired_base_count
                and self._expansion_goal.terminal_state is None
            ):
                if self._expansion_commitment_id is not None:
                    self._terminate_expansion_commitment(
                        observation,
                        terminal_state="desired_base_count_satisfied",
                    )
                self._terminalize_expansion_goal(
                    observation,
                    terminal_state="desired_base_count_satisfied",
                )
        previous_state = self._expansion_scout_state
        structured_state = next(
            (
                alert.split("=", 1)[1]
                for alert in observation.alerts
                if alert.casefold().startswith("expansion_scout_state=")
            ),
            None,
        )
        generation_text = next(
            (
                alert.split("=", 1)[1]
                for alert in observation.alerts
                if alert.casefold().startswith("expansion_scout_generation=")
            ),
            None,
        )
        progress = next(
            (
                alert.split("=", 1)[1]
                for alert in observation.alerts
                if alert.casefold().startswith("expansion_scout_waypoints=")
            ),
            None,
        )
        if structured_state is not None:
            self._expansion_scout_state = structured_state.casefold()
        if generation_text is not None:
            try:
                generation = max(1, int(generation_text))
            except ValueError:
                generation = self._expansion_scout_generation
            if generation > self._expansion_scout_generation:
                self._expansion_scout_generation = generation
                if (
                    self._expansion_commitment_id is not None
                    and self._expansion_commitment_generation in {None, 0}
                ):
                    self._expansion_commitment_generation = generation
                self._expansion_candidates_exhausted = False
                if (
                    self._expansion_goal is not None
                    and self._expansion_goal.terminal_state is None
                    and self._expansion_goal.phase == "waiting_for_candidates"
                ):
                    self._expansion_goal = self._expansion_goal.model_copy(
                        update={
                            "phase": "active",
                            "current_candidate_epoch": generation,
                            "cooldown_until_game_loop": 0,
                        }
                    )
                    self._record_cortex_event(
                        observation,
                        "expansion_goal_reopened",
                        self._expansion_goal,
                    )
        if progress is not None:
            try:
                visited, total = progress.split("/", 1)
                self._expansion_scout_visited_waypoints = max(0, int(visited))
                self._expansion_scout_total_waypoints = max(0, int(total))
            except ValueError:
                self._expansion_scout_visited_waypoints = 0
                self._expansion_scout_total_waypoints = 0
        if self._expansion_scout_state != previous_state:
            self._record_cortex_event(
                observation,
                "expansion_scout_state_changed",
                {
                    "previous_state": previous_state,
                    "state": self._expansion_scout_state,
                    "generation_id": self._expansion_scout_generation,
                    "visited_waypoints": self._expansion_scout_visited_waypoints,
                    "total_waypoints": self._expansion_scout_total_waypoints,
                    "commitment_id": self._expansion_commitment_id,
                },
            )
        townhall_action = self._townhall_runtime_action()
        candidate_available = any(
            action.name == townhall_action and bool(action.argument_candidates)
            for action in observation.available_actions
        )
        legacy_exhausted = "expansion_candidates_exhausted" in {
            alert.casefold() for alert in observation.alerts
        }
        reported_exhausted = (
            self._expansion_scout_state == "all_candidates_exhausted"
            and self._expansion_scout_total_waypoints > 0
            and self._expansion_scout_visited_waypoints >= self._expansion_scout_total_waypoints
        )
        if structured_state is None:
            reported_exhausted = legacy_exhausted
        if candidate_available and (
            self._expansion_exhausted_generation != self._expansion_scout_generation
        ):
            self._expansion_candidates_exhausted = False
            return
        if self._expansion_commitment_id is None:
            # Scouting before HIMA requests an expansion cannot terminalize a
            # commitment that does not exist yet.
            self._expansion_candidates_exhausted = False
            return
        self._expansion_candidates_exhausted = reported_exhausted
        if reported_exhausted:
            self._expansion_exhausted_generation = self._expansion_scout_generation
            self._terminate_expansion_commitment(
                observation,
                terminal_state="expansion_candidates_exhausted",
            )
            if self._expansion_goal is not None:
                exhausted = tuple(
                    sorted(
                        {
                            *self._expansion_goal.exhausted_candidate_epochs,
                            self._expansion_scout_generation,
                        }
                    )
                )
                self._expansion_goal = self._expansion_goal.model_copy(
                    update={
                        "exhausted_candidate_epochs": exhausted,
                        "phase": "waiting_for_candidates",
                        "retry_budget": max(0, self._expansion_goal.retry_budget - 1),
                        "cooldown_until_game_loop": observation.game_loop + 32,
                    }
                )
                self._record_cortex_event(
                    observation,
                    "expansion_candidate_epoch_exhausted",
                    self._expansion_goal,
                )
                if self._expansion_goal.retry_budget == 0:
                    self._terminalize_expansion_goal(
                        observation,
                        terminal_state="global_retry_budget_exhausted",
                    )

    def _record_expansion_anchor_rejection(self, report: ExecutionReport) -> None:
        anchor = next(
            (
                value
                for value in (*report.resolved_arguments, *report.requested_arguments)
                if isinstance(value, (int, str))
            ),
            None,
        )
        evaluation = {
            "commitment_id": self._expansion_commitment_id,
            "command_id": report.command_id,
            "anchor": anchor,
            "failure_code": report.failure_code,
            "game_loop": self._execution_game_loop(report),
        }
        self._expansion_anchor_evaluations.append(evaluation)
        self.store.append_event(
            run_id=report.run_id,
            episode_id=report.episode_id,
            step_id=report.step_id,
            event_type="expansion_anchor_rejected",
            payload=evaluation,
        )

    def _terminate_expansion_commitment(
        self,
        observation: ObservationEnvelope,
        *,
        terminal_state: str,
    ) -> None:
        commitment_id = self._expansion_commitment_id
        if commitment_id is None:
            return
        self._record_cortex_event(
            observation,
            "expansion_commitment_terminal",
            {
                "commitment_id": commitment_id,
                "terminal_state": terminal_state,
                "started_game_loop": self._expansion_commitment_started_game_loop,
                "terminal_game_loop": observation.game_loop,
                "evaluated_anchors": list(self._expansion_anchor_evaluations),
                "generation_id": self._expansion_commitment_generation,
                "scout_state": self._expansion_scout_state,
                "visited_waypoints": self._expansion_scout_visited_waypoints,
                "total_waypoints": self._expansion_scout_total_waypoints,
            },
        )
        self._expansion_commitment_id = None
        self._expansion_commitment_started_game_loop = None
        self._expansion_commitment_generation = None
        self._expansion_commitment_dispatched = False
        self._expansion_anchor_evaluations = []

    def _terminate_expansion_commitment_from_report(
        self,
        report: ExecutionReport,
        *,
        terminal_state: str,
    ) -> None:
        commitment_id = self._expansion_commitment_id
        if commitment_id is None:
            return
        self.store.append_event(
            run_id=report.run_id,
            episode_id=report.episode_id,
            step_id=report.step_id,
            event_type="expansion_commitment_terminal",
            payload={
                "commitment_id": commitment_id,
                "terminal_state": terminal_state,
                "started_game_loop": self._expansion_commitment_started_game_loop,
                "terminal_game_loop": self._execution_game_loop(report),
                "evaluated_anchors": list(self._expansion_anchor_evaluations),
                "generation_id": self._expansion_commitment_generation,
                "scout_state": self._expansion_scout_state,
                "visited_waypoints": self._expansion_scout_visited_waypoints,
                "total_waypoints": self._expansion_scout_total_waypoints,
                "command_id": report.command_id,
            },
        )
        self._expansion_commitment_id = None
        self._expansion_commitment_started_game_loop = None
        self._expansion_commitment_generation = None
        self._expansion_commitment_dispatched = False
        self._expansion_anchor_evaluations = []
        if self._expansion_goal is not None:
            observed = min(
                self._expansion_goal.desired_base_count,
                self._expansion_goal.observed_base_count + 1,
            )
            self._expansion_goal = self._expansion_goal.model_copy(
                update={"observed_base_count": observed}
            )
            if observed >= self._expansion_goal.desired_base_count:
                self._expansion_goal = self._expansion_goal.model_copy(
                    update={"terminal_state": "desired_base_count_satisfied"}
                )
                self.store.append_event(
                    run_id=report.run_id,
                    episode_id=report.episode_id,
                    step_id=report.step_id,
                    event_type="expansion_goal_terminal",
                    payload=self._expansion_goal,
                )

    def _recoverable_expansion_failure(self, report: ExecutionReport) -> bool:
        return (
            report.action_name == self._townhall_runtime_action()
            and report.failure_code in _RECOVERABLE_EXPANSION_FAILURE_CODES
        )

    def _townhall_runtime_action(self) -> str:
        actions = townhall_recovery_runtime_actions(self._race_profile.data)
        if len(actions) != 1:
            raise RuntimeError(
                f"{self._race_profile.race.value} profile must define exactly one "
                "direct townhall recovery action"
            )
        return next(iter(actions))

    def _townhall_semantic_action(self) -> str:
        return self._semantic_action_for_runtime(self._townhall_runtime_action())

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
        if self._macro_plan is None or self._terminal_collapse_macro_hold is not None:
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

    def _record_defense_inventory(
        self,
        observation: ObservationEnvelope,
        *,
        unit_evaluations: tuple[dict[str, object], ...] = (),
    ) -> None:
        active_commands = tuple(
            (lifecycle.command.name, lifecycle.status.value)
            for lifecycle in self._command_states.values()
            if lifecycle.status is CommandStatus.DISPATCHED
        )
        role_evaluations = self._role_agents.defense_inventory_evaluations(
            observation,
            active_commands=active_commands,
        )
        capped_unit_types = {str(payload["item_type"]) for payload in unit_evaluations}
        evaluations = (
            *(
                payload
                for payload in role_evaluations
                if payload["item_type"] not in capped_unit_types
            ),
            *unit_evaluations,
        )
        for payload in evaluations:
            item_type = str(payload["item_type"])
            signature = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            if self._last_defense_inventory_signatures.get(item_type) == signature:
                continue
            self._last_defense_inventory_signatures[item_type] = signature
            self._record_cortex_event(
                observation,
                "defense_inventory_evaluated",
                payload,
            )

    def _refresh_playbook(
        self,
        observation: ObservationEnvelope,
        assessment: SituationAssessment,
    ) -> None:
        if self._playbook_store is None:
            self._playbook_selection = None
            self._playbook_rules = ()
            return
        if (
            not self._playbook_promotion_sweep_done
            and self.config.cortex.playbook.rule_mode == "active"
            and self.config.cortex.playbook.learning_mode == "evolving"
        ):
            self._playbook_promotion_sweep_done = True
            sweep = PlaybookPromotionSweep(self._playbook_store).run()
            self._record_cortex_event(
                observation,
                "playbook_promotion_sweep",
                asdict(sweep),
            )
        context = self._playbook_context(assessment)
        self._playbook_rules = self._playbook_store.rules_for_guard(
            context=context,
            max_hard=self.config.cortex.playbook.max_hard_rules,
            max_soft=self.config.cortex.playbook.max_soft_rules,
            approved_hard_rule_ids=self._approved_hard_rule_ids,
        )
        query = PlaybookQuery(
            context=context,
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
                "executable_rule_count": len(self._playbook_rules),
                "approved_hard_rule_ids": list(self._approved_hard_rule_ids or ()),
            },
        )

    def _playbook_context(self, assessment: SituationAssessment) -> PlaybookContext:
        return PlaybookContext(
            agent_race=self.config.environment.agent_race,
            opponent_race=self.config.environment.opponent_race,
            phase=assessment.phase,
            map_name=self.config.environment.scenario,
            tags=tuple(assessment.threats),
        )

    def end_episode(self, result: EpisodeResult) -> None:
        already_recorded = self._episode_result_fingerprint is not None
        if not already_recorded and self._expansion_commitment_id is not None:
            if not self._expansion_anchor_evaluations:
                self._expansion_anchor_evaluations.append(
                    {
                        "commitment_id": self._expansion_commitment_id,
                        "command_id": None,
                        "anchor": None,
                        "failure_code": "no_expansion_anchor_evaluated",
                        "game_loop": result.steps,
                    }
                )
            self.store.append_event(
                run_id=result.run_id,
                episode_id=result.episode_id,
                step_id=result.steps,
                event_type="expansion_commitment_terminal",
                payload={
                    "commitment_id": self._expansion_commitment_id,
                    "terminal_state": "strategic_cancellation",
                    "started_game_loop": self._expansion_commitment_started_game_loop,
                    "terminal_game_loop": result.steps,
                    "evaluated_anchors": list(self._expansion_anchor_evaluations),
                    "generation_id": self._expansion_commitment_generation,
                    "reason": "episode_end",
                },
            )
            self._expansion_commitment_id = None
            self._expansion_commitment_started_game_loop = None
            self._expansion_commitment_generation = None
            self._expansion_commitment_dispatched = False
            self._expansion_anchor_evaluations = []
        super().end_episode(result)
        if not already_recorded:
            for event in self.store.events_of_type(
                result.run_id,
                result.episode_id,
                "execution",
            ):
                report = ExecutionReport.model_validate(event.payload)
                self._resolve_playbook_rule_evaluations(
                    report,
                    self._command_lineages.get(report.command_id),
                )
            self._finalize_unselected_playbook_evaluations(result)
        self.store.record_snapshot(
            run_id=result.run_id,
            episode_id=result.episode_id,
            snapshot_type=_CORTEX_SNAPSHOT_TYPE,
            step_id=result.steps,
            payload=self._cortex_checkpoint_payload(result.steps),
        )
        if already_recorded or self._playbook_reviewer is None:
            self.store.flush()
            return
        events = self._all_episode_events(result.run_id, result.episode_id)
        cases, lessons = self._playbook_reviewer.review_episode(
            events,
            result,
            agent_race=self.config.environment.agent_race,
            opponent_race=self.config.environment.opponent_race,
        )
        consequence_counts: dict[str, int] = {}
        for consequence in self._playbook_reviewer.last_consequences:
            consequence_type = consequence.consequence_type.value
            consequence_counts[consequence_type] = consequence_counts.get(consequence_type, 0) + 1
            self.store.append_event(
                run_id=result.run_id,
                episode_id=result.episode_id,
                step_id=result.steps,
                event_type="strategic_consequence_attributed",
                payload=consequence,
            )
        self._resolve_strategic_rule_evaluations(
            result,
            self._playbook_reviewer.last_consequences,
        )
        for case in cases:
            self.store.append_event(
                run_id=result.run_id,
                episode_id=result.episode_id,
                step_id=result.steps,
                event_type="playbook_case_recorded",
                payload=case,
            )
        emitted_lesson_signatures: set[str] = set()
        for lesson in lessons:
            if lesson.signature in emitted_lesson_signatures:
                continue
            emitted_lesson_signatures.add(lesson.signature)
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
        for rule in self._playbook_reviewer.last_rule_updates:
            self.store.append_event(
                run_id=result.run_id,
                episode_id=result.episode_id,
                step_id=result.steps,
                event_type="playbook_rule_updated",
                payload=rule,
            )
        self.store.append_event(
            run_id=result.run_id,
            episode_id=result.episode_id,
            step_id=result.steps,
            event_type="postgame_review_completed",
            payload={
                "case_count": len(cases),
                "lesson_update_count": len(lessons),
                "strategic_consequence_count": len(self._playbook_reviewer.last_consequences),
                "strategic_consequence_counts": dict(sorted(consequence_counts.items())),
                "playbook_path": str(self._playbook_reviewer.store.database_path),
            },
        )
        self.store.flush()

    def _all_episode_events(
        self,
        run_id: str,
        episode_id: str,
        *,
        page_size: int = 10_000,
    ) -> list[StoredEvent]:
        events: list[StoredEvent] = []
        after_event_id = 0
        while True:
            page = self.store.events_after(
                run_id,
                after_event_id,
                page_size,
                episode_id=episode_id,
            )
            if not page:
                return events
            events.extend(page)
            after_event_id = page[-1].event_id
            if len(page) < page_size:
                return events

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
