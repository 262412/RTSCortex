"""Two-timescale runtime coordinating deliberation and reflexes."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

from rtscortex.agents import (
    ActionModule,
    ContextBudget,
    MemoryModule,
    PlanningModule,
    ReflectionModule,
)
from rtscortex.config import ExperimentConfig
from rtscortex.contracts import (
    CURRENT_PROTOCOL_VERSION,
    ActionBatch,
    ActionCommand,
    ActionSource,
    EpisodeResult,
    EpisodeSummary,
    ExecutionReport,
    ExecutionStage,
    ExecutionStatus,
    IdleReason,
    ObservationEnvelope,
)
from rtscortex.contracts.interfaces import (
    ActivePlanSnapshot,
    AgentContext,
    AgentModule,
    CommandLifecycleSnapshot,
    LLMProvider,
    ModuleResult,
)
from rtscortex.memory import EventStore
from rtscortex.reflex import ReflexEngine
from rtscortex.runtime.validation import (
    ActionArbiter,
    ActionValidator,
    ValidationDisposition,
    ValidationFailure,
)


@dataclass(frozen=True)
class PlanState:
    strategic_goal: str
    summary: str
    commands: list[ActionCommand]
    source_step_id: int
    created_game_loop: int


class CommandStatus(StrEnum):
    PENDING = "pending"
    DEFERRED = "deferred"
    DISPATCHED = "dispatched"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNCONFIRMED = "unconfirmed"
    EXPIRED = "expired"
    REJECTED = "rejected"
    OBSOLETE = "obsolete"
    SUPERSEDED = "superseded"


@dataclass(frozen=True)
class CommandLifecycle:
    command: ActionCommand
    status: CommandStatus
    reason: str | None = None


_ACTIONABLE_COMMAND_STATUSES = {CommandStatus.PENDING, CommandStatus.DEFERRED}
_TERMINAL_COMMAND_STATUSES = {
    CommandStatus.SUCCEEDED,
    CommandStatus.FAILED,
    CommandStatus.CANCELLED,
    CommandStatus.UNCONFIRMED,
}
_COMMAND_STATUS_BY_EXECUTION_STATUS = {
    ExecutionStatus.SUCCEEDED: CommandStatus.SUCCEEDED,
    ExecutionStatus.FAILED: CommandStatus.FAILED,
    ExecutionStatus.CANCELLED: CommandStatus.CANCELLED,
    ExecutionStatus.UNCONFIRMED: CommandStatus.UNCONFIRMED,
}
_ALLOWED_COMMAND_TRANSITIONS = {
    CommandStatus.PENDING: {
        CommandStatus.DEFERRED,
        CommandStatus.DISPATCHED,
        CommandStatus.CANCELLED,
        CommandStatus.EXPIRED,
        CommandStatus.REJECTED,
        CommandStatus.OBSOLETE,
        CommandStatus.SUPERSEDED,
    },
    CommandStatus.DEFERRED: {
        CommandStatus.PENDING,
        CommandStatus.CANCELLED,
        CommandStatus.EXPIRED,
        CommandStatus.REJECTED,
        CommandStatus.OBSOLETE,
        CommandStatus.SUPERSEDED,
    },
    CommandStatus.DISPATCHED: {
        CommandStatus.SUCCEEDED,
        CommandStatus.FAILED,
        CommandStatus.CANCELLED,
        CommandStatus.UNCONFIRMED,
    },
}


class RuntimeEngine:
    """Run a non-blocking strategic planner alongside a synchronous reflex path."""

    def __init__(
        self,
        *,
        config: ExperimentConfig,
        store: EventStore,
        provider: LLMProvider,
    ) -> None:
        self.config = config
        self.store = store
        self.provider = provider
        context_budget = ContextBudget(**config.context.model_dump())
        self.memory_module = MemoryModule(store, config.memory.short_term_window, context_budget)
        self.reflection_module = ReflectionModule(provider, context_budget)
        self.planning_module = PlanningModule(provider, context_budget)
        self.action_module = ActionModule(
            config.runtime.max_actions,
            config.runtime.planner_command_ttl_game_loops,
        )
        reflex_enabled = config.reflex.enabled and config.agent.variant in {
            "reflex_only",
            "planner_reflection_memory_reflex",
        }
        self.reflex = ReflexEngine(
            enabled=reflex_enabled,
            low_health_threshold=config.reflex.low_health_threshold,
        )
        self.arbiter = ActionArbiter()
        self.validator = ActionValidator(config.runtime.max_actions)
        self._planner_enabled = config.agent.variant in {
            "planner_only",
            "planner_reflection_memory_reflex",
        }
        self._reflection_enabled = config.agent.variant == "planner_reflection_memory_reflex"
        self._planner_task: asyncio.Task[PlanState] | None = None
        self._cached_plan: PlanState | None = None
        self._last_planner_started_game_loop: int | None = None
        self._last_plan_accepted_game_loop: int | None = None
        self._urgent_replan_requested = False
        self._last_alerts: tuple[str, ...] = ()
        self._last_planner_failure: IdleReason | None = None
        self._exhaustion_requested_for: tuple[int, int] | None = None
        self._command_states: dict[str, CommandLifecycle] = {}
        self._reported_command_reasons: set[tuple[str, str]] = set()
        self._terminal_execution_fingerprints: dict[str, str] = {}
        self._episode_result_fingerprint: str | None = None
        self._last_decision: ActionBatch | None = None
        self._decision_by_command_id: dict[str, ActionBatch] = {}
        self._last_execution: ExecutionReport | None = None
        self._episode_key: tuple[str, str] | None = None

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
        await self._collect_finished_planner(observation)
        self._note_alerts(observation)
        planner_due = self._planner_enabled and self._should_plan(observation)
        self._expire_planner_commands(observation)

        context = self._agent_context(observation)
        memory_result = await self._run_module(self.memory_module, context)
        context = self._agent_context(observation, memory=memory_result.updates)

        if planner_due:
            await self._begin_planner_cycle(context)

        reflex_started = time.perf_counter()
        reflex_candidates = [
            command
            for command in self.reflex.evaluate(observation)
            if command.command_id not in self._command_states
        ]
        reflex_latency_ms = (time.perf_counter() - reflex_started) * 1000

        planner_candidates = self._active_planner_commands(observation)
        (
            planner_commands,
            reflex_commands,
            rejected_commands,
            busy_actor_candidates,
        ) = self._defer_busy_actor_commands(
            planner_candidates,
            reflex_candidates,
            observation,
        )
        candidate_outcome = self.validator.validate_candidates(
            [*planner_commands, *reflex_commands],
            observation,
        )
        rejected_commands.extend(
            self._apply_validation_failures(
                candidate_outcome.failures,
                observation,
            )
        )
        for command in candidate_outcome.accepted:
            lifecycle = self._command_states.get(command.command_id)
            if (
                command.source is ActionSource.PLANNER
                and lifecycle is not None
                and lifecycle.status is CommandStatus.DEFERRED
            ):
                self._transition_command(command, CommandStatus.PENDING, observation)
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
        for command in accepted_commands:
            self._transition_command(command, CommandStatus.DISPATCHED, observation)
        plan = self._cached_plan or PlanState(
            "", "", [], observation.step_id, observation.game_loop
        )
        idle_reason = None if accepted_commands else self._idle_reason()
        batch = ActionBatch(
            run_id=observation.run_id,
            episode_id=observation.episode_id,
            step_id=observation.step_id,
            decision_id=(
                f"{observation.run_id}:{observation.episode_id}:{observation.step_id}:decision"
            ),
            strategic_goal=plan.strategic_goal,
            summary=plan.summary,
            planner_pending=self._planner_task is not None,
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
                "reflex_latency_ms": reflex_latency_ms,
                "reflex_latency_target_ms": self.config.reflex.target_latency_ms,
                "tick_latency_ms": (time.perf_counter() - tick_started) * 1000,
                "preemptions": [asdict(record) for record in arbitration.preemptions],
                "planner_candidates": [
                    command.model_dump(mode="json") for command in planner_candidates
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
            },
        )
        self._last_decision = batch
        for command in batch.commands:
            self._decision_by_command_id[command.command_id] = batch
        self._request_replan_if_exhausted()
        return batch

    def _agent_context(
        self,
        observation: ObservationEnvelope,
        *,
        memory: dict[str, Any] | None = None,
    ) -> AgentContext:
        execution = self._last_execution
        matching_decision = (
            None if execution is None else self._decision_by_command_id.get(execution.command_id)
        )
        return AgentContext(
            observation=observation,
            memory={} if memory is None else memory,
            last_execution=execution,
            last_decision=matching_decision,
            active_plan=self._active_plan_snapshot(),
        )

    def _active_plan_snapshot(self) -> ActivePlanSnapshot | None:
        active = [
            lifecycle
            for lifecycle in self._command_states.values()
            if lifecycle.status
            in {
                CommandStatus.PENDING,
                CommandStatus.DEFERRED,
                CommandStatus.DISPATCHED,
            }
        ]
        if self._cached_plan is None and not active:
            return None
        active.sort(key=lambda item: (item.command.created_game_loop, item.command.command_id))
        return ActivePlanSnapshot(
            strategic_goal=("" if self._cached_plan is None else self._cached_plan.strategic_goal),
            summary="" if self._cached_plan is None else self._cached_plan.summary,
            commands=tuple(
                CommandLifecycleSnapshot(
                    command_id=lifecycle.command.command_id,
                    actor=lifecycle.command.actor,
                    name=lifecycle.command.name,
                    arguments=tuple(lifecycle.command.arguments),
                    source=lifecycle.command.source.value,
                    status=lifecycle.status.value,
                    reason=lifecycle.reason,
                    created_game_loop=lifecycle.command.created_game_loop,
                    ttl_game_loops=lifecycle.command.ttl_game_loops,
                )
                for lifecycle in active
            ),
        )

    def _should_plan(self, observation: ObservationEnvelope) -> bool:
        if self._planner_task is not None:
            return False
        if self._urgent_replan_requested:
            return True
        if self._last_planner_started_game_loop is None:
            return True
        elapsed = observation.game_loop - self._last_planner_started_game_loop
        return elapsed >= self.config.runtime.planning_interval_game_loops

    async def _begin_planner_cycle(self, context: AgentContext) -> None:
        observation = context.observation
        self._last_planner_started_game_loop = observation.game_loop
        self._urgent_replan_requested = False
        self._last_planner_failure = None
        self._record_planner_event(
            observation,
            "planner_started",
            {"started_game_loop": observation.game_loop},
        )
        if self.config.runtime.deterministic:
            await self._run_deterministic_planner(context)
        elif self.config.environment.pause_until_first_plan and self._cached_plan is None:
            await self._run_initial_plan_barrier(context)
        else:
            self._planner_task = asyncio.create_task(self._deliberate_with_timeout(context))

    def _note_alerts(self, observation: ObservationEnvelope) -> None:
        alerts = tuple(sorted(observation.alerts))
        if alerts and alerts != self._last_alerts:
            self._urgent_replan_requested = True
        self._last_alerts = alerts

    async def _run_deterministic_planner(self, context: AgentContext) -> None:
        try:
            plan = await self._deliberate_with_timeout(context)
            self._accept_plan(plan, context.observation)
        except TimeoutError:
            self._last_planner_failure = IdleReason.PLANNER_TIMEOUT
            self._record_planner_event(context.observation, "planner_timeout", {})
        except Exception as error:
            self._record_planner_event(
                context.observation,
                "planner_error",
                {"error_type": type(error).__name__, "message": str(error)},
            )

    async def _run_initial_plan_barrier(self, context: AgentContext) -> None:
        """Require one valid plan before allowing the live game to advance."""

        try:
            plan = await self._deliberate_with_timeout(context)
            self._accept_plan(plan, context.observation)
        except TimeoutError as error:
            self._last_planner_failure = IdleReason.PLANNER_TIMEOUT
            self._record_planner_event(context.observation, "planner_timeout", {})
            raise RuntimeError("initial planner timed out before the game could start") from error
        except Exception as error:
            self._record_planner_event(
                context.observation,
                "planner_error",
                {"error_type": type(error).__name__, "message": str(error)},
            )
            raise RuntimeError("initial planner failed before the game could start") from error

    async def _collect_finished_planner(self, observation: ObservationEnvelope) -> None:
        if self._planner_task is None or not self._planner_task.done():
            return
        try:
            plan = self._planner_task.result()
            self._accept_plan(plan, observation)
        except TimeoutError:
            self._last_planner_failure = IdleReason.PLANNER_TIMEOUT
            self._record_planner_event(observation, "planner_timeout", {})
        except Exception as error:
            self._record_planner_event(
                observation,
                "planner_error",
                {"error_type": type(error).__name__, "message": str(error)},
            )
        finally:
            self._planner_task = None

    async def _deliberate_with_timeout(self, context: AgentContext) -> PlanState:
        started = time.perf_counter()
        try:
            plan = await asyncio.wait_for(
                self._deliberate(context),
                timeout=self.config.runtime.planner_timeout_seconds,
            )
        except TimeoutError:
            self._record_planner_event(
                context.observation,
                "planner_cycle",
                {
                    "status": "timeout",
                    "latency_ms": (time.perf_counter() - started) * 1000,
                },
            )
            raise
        except Exception as error:
            self._record_planner_event(
                context.observation,
                "planner_cycle",
                {
                    "status": "error",
                    "latency_ms": (time.perf_counter() - started) * 1000,
                    "error_type": type(error).__name__,
                },
            )
            raise
        self._record_planner_event(
            context.observation,
            "planner_cycle",
            {
                "status": "success",
                "latency_ms": (time.perf_counter() - started) * 1000,
            },
        )
        return plan

    async def _deliberate(self, context: AgentContext) -> PlanState:
        working_memory: dict[str, Any] = dict(context.memory)
        if self._reflection_enabled:
            reflection = await self._run_module(self.reflection_module, context)
            working_memory.update(reflection.updates)
            for lesson in reflection.updates.get("lessons", []):
                self.store.add_lesson(
                    run_id=context.observation.run_id,
                    episode_id=context.observation.episode_id,
                    source_step_id=context.observation.step_id,
                    content=str(lesson),
                )
        planning_context = AgentContext(
            observation=context.observation,
            memory=working_memory,
            last_execution=context.last_execution,
            last_decision=context.last_decision,
            active_plan=context.active_plan,
        )
        planning = await self._run_module(self.planning_module, planning_context)
        working_memory.update(planning.updates)
        action_context = AgentContext(
            observation=context.observation,
            memory=working_memory,
            last_execution=context.last_execution,
            last_decision=context.last_decision,
            active_plan=context.active_plan,
        )
        action = await self._run_module(self.action_module, action_context)
        return PlanState(
            strategic_goal=str(action.updates["strategic_goal"]),
            summary=str(action.updates["plan_summary"]),
            commands=action.commands,
            source_step_id=context.observation.step_id,
            created_game_loop=context.observation.game_loop,
        )

    async def _activate_episode(self, observation: ObservationEnvelope) -> None:
        episode_key = (observation.run_id, observation.episode_id)
        if self._episode_key == episode_key:
            return
        await self._cancel_planner()
        self._episode_key = episode_key
        self._cached_plan = None
        self._last_planner_started_game_loop = None
        self._last_plan_accepted_game_loop = None
        self._urgent_replan_requested = False
        self._last_alerts = ()
        self._last_planner_failure = None
        self._exhaustion_requested_for = None
        self._command_states = {}
        self._reported_command_reasons = set()
        self._terminal_execution_fingerprints = {}
        self._episode_result_fingerprint = None
        self._last_decision = None
        self._decision_by_command_id = {}
        self._last_execution = None

        decision_events = self.store.events_of_type(
            observation.run_id,
            observation.episode_id,
            "decision",
        )
        for decision_event in decision_events:
            decision = ActionBatch.model_validate(decision_event.payload["batch"])
            self._last_decision = decision
            for command in decision.commands:
                self._decision_by_command_id[command.command_id] = decision
        execution_event = self.store.last_event(
            observation.run_id,
            observation.episode_id,
            "execution",
        )
        if execution_event is not None:
            self._last_execution = ExecutionReport.model_validate(execution_event.payload)
        episode_result_event = self.store.last_event(
            observation.run_id,
            observation.episode_id,
            "episode_result",
        )
        if episode_result_event is not None:
            recovered_result = EpisodeResult.model_validate(episode_result_event.payload)
            self._episode_result_fingerprint = self._episode_fingerprint(recovered_result)

        planner_started_event = self.store.last_event(
            observation.run_id,
            observation.episode_id,
            "planner_started",
        )
        if planner_started_event is not None:
            self._last_planner_started_game_loop = int(
                planner_started_event.payload["started_game_loop"]
            )

        plan_event = self.store.last_event(
            observation.run_id,
            observation.episode_id,
            "plan_accepted",
        )
        plan_uses_lifecycle_protocol = False
        if plan_event is not None:
            plan_uses_lifecycle_protocol = (
                plan_event.payload.get("lifecycle_protocol") == CURRENT_PROTOCOL_VERSION
            )
            self._cached_plan = PlanState(
                strategic_goal=str(plan_event.payload["strategic_goal"]),
                summary=str(plan_event.payload["summary"]),
                commands=[
                    ActionCommand.model_validate(command)
                    for command in plan_event.payload["commands"]
                ],
                source_step_id=int(plan_event.payload["source_step_id"]),
                created_game_loop=int(plan_event.payload["created_game_loop"]),
            )
        if self._cached_plan is not None:
            self._last_plan_accepted_game_loop = self._cached_plan.created_game_loop

        lifecycle_events = self.store.events_of_type(
            observation.run_id,
            observation.episode_id,
            "command_lifecycle",
        )
        for event in lifecycle_events:
            command = ActionCommand.model_validate(event.payload["command"])
            reason = None if event.payload.get("reason") is None else str(event.payload["reason"])
            self._command_states[command.command_id] = CommandLifecycle(
                command=command,
                status=CommandStatus(str(event.payload["status"])),
                reason=reason,
            )
            if reason is not None:
                self._reported_command_reasons.add((command.command_id, reason))
        execution_reports: list[ExecutionReport] = []
        for event in self.store.events_of_type(
            observation.run_id,
            observation.episode_id,
            "execution",
        ):
            report = ExecutionReport.model_validate(event.payload)
            fingerprint = self._execution_fingerprint(report)
            previous = self._terminal_execution_fingerprints.get(report.command_id)
            if previous is not None and previous != fingerprint:
                raise RuntimeError(
                    "conflicting terminal execution reports recovered for command "
                    f"{report.command_id!r}"
                )
            self._terminal_execution_fingerprints[report.command_id] = fingerprint
            execution_reports.append(report)

        plan_commands = {
            command.command_id: command
            for command in (() if self._cached_plan is None else self._cached_plan.commands)
        }
        for report in execution_reports:
            lifecycle = self._command_states.get(report.command_id)
            recovered_command = (
                lifecycle.command if lifecycle is not None else plan_commands.get(report.command_id)
            )
            if recovered_command is None:
                if report.protocol_version == CURRENT_PROTOCOL_VERSION:
                    raise RuntimeError(
                        "terminal execution report has no recoverable command lifecycle: "
                        f"{report.command_id!r}"
                    )
                continue
            self._validate_execution_identity(report, recovered_command)
            recovered_status = _COMMAND_STATUS_BY_EXECUTION_STATUS[report.status]
            if lifecycle is not None and lifecycle.status in _TERMINAL_COMMAND_STATUSES:
                if lifecycle.status is not recovered_status:
                    raise RuntimeError(
                        "terminal command lifecycle conflicts with recovered execution "
                        f"for {report.command_id!r}"
                    )
                continue
            self._set_command_lifecycle(
                recovered_command,
                recovered_status,
                run_id=observation.run_id,
                episode_id=observation.episode_id,
                step_id=observation.step_id,
                game_loop=observation.game_loop,
                reason=report.failure_code or report.failure_reason,
            )

        if self._cached_plan is not None:
            for command in self._cached_plan.commands:
                if command.command_id in self._command_states:
                    continue
                if lifecycle_events or plan_uses_lifecycle_protocol:
                    self._set_command_lifecycle(
                        command,
                        CommandStatus.PENDING,
                        run_id=observation.run_id,
                        episode_id=observation.episode_id,
                        step_id=observation.step_id,
                        game_loop=observation.game_loop,
                        reason="recovered accepted command after partial lifecycle write",
                    )
                    continue
                self._command_states[command.command_id] = CommandLifecycle(
                    command=command,
                    status=CommandStatus.SUPERSEDED,
                    reason="legacy runtime state cannot prove command was not dispatched",
                )
                self._urgent_replan_requested = True

    def _accept_plan(self, plan: PlanState, observation: ObservationEnvelope) -> None:
        command_ids = [command.command_id for command in plan.commands]
        duplicate_ids = sorted(
            command_id for command_id in set(command_ids) if command_ids.count(command_id) > 1
        )
        reused_ids = sorted(set(command_ids) & self._command_states.keys())
        if duplicate_ids or reused_ids:
            rendered = ", ".join(duplicate_ids or reused_ids)
            detail = (
                "duplicates another plan command"
                if duplicate_ids
                else "already has lifecycle state"
            )
            raise RuntimeError(f"planner command ID {rendered} {detail}")

        normalized_commands = [
            command.model_copy(
                update={
                    "created_game_loop": observation.game_loop,
                    "ttl_game_loops": self.config.runtime.planner_command_ttl_game_loops,
                }
            )
            for command in plan.commands
        ]
        active_lifecycles = sorted(
            (
                lifecycle
                for lifecycle in self._command_states.values()
                if lifecycle.status
                in {
                    CommandStatus.PENDING,
                    CommandStatus.DEFERRED,
                    CommandStatus.DISPATCHED,
                }
            ),
            key=lambda item: (item.command.created_game_loop, item.command.command_id),
        )
        planner_commands_by_semantic: dict[str, list[ActionCommand]] = {}
        for lifecycle in active_lifecycles:
            semantic = self._command_semantic_key(lifecycle.command)
            if lifecycle.command.source is ActionSource.PLANNER:
                planner_commands_by_semantic.setdefault(semantic, []).append(lifecycle.command)

        retained_command_ids: set[str] = set()
        accepted_commands: list[ActionCommand] = []
        for command in normalized_commands:
            semantic = self._command_semantic_key(command)
            existing = planner_commands_by_semantic.get(semantic, [])
            if existing:
                retained = existing.pop(0)
                retained_command_ids.add(retained.command_id)
                accepted_commands.append(retained)
            else:
                accepted_commands.append(command)

        for lifecycle in list(self._command_states.values()):
            if (
                lifecycle.status in _ACTIONABLE_COMMAND_STATUSES
                and lifecycle.command.command_id not in retained_command_ids
            ):
                self._transition_command(
                    lifecycle.command,
                    CommandStatus.SUPERSEDED,
                    observation,
                    reason="replaced by a newer accepted plan",
                )
        accepted_plan = PlanState(
            strategic_goal=plan.strategic_goal,
            summary=plan.summary,
            commands=accepted_commands,
            source_step_id=plan.source_step_id,
            created_game_loop=observation.game_loop,
        )
        fingerprint = self._plan_fingerprint(accepted_plan)
        previous_fingerprint = (
            None if self._cached_plan is None else self._plan_fingerprint(self._cached_plan)
        )
        self._cached_plan = accepted_plan
        self._last_plan_accepted_game_loop = accepted_plan.created_game_loop
        self._last_planner_failure = None
        self._exhaustion_requested_for = None
        self.store.append_event(
            run_id=observation.run_id,
            episode_id=observation.episode_id,
            step_id=observation.step_id,
            event_type="plan_accepted",
            payload={
                "strategic_goal": accepted_plan.strategic_goal,
                "summary": accepted_plan.summary,
                "commands": [command.model_dump(mode="json") for command in accepted_plan.commands],
                "source_step_id": accepted_plan.source_step_id,
                "created_game_loop": accepted_plan.created_game_loop,
                "source_game_loop": plan.created_game_loop,
                "accepted_game_loop": observation.game_loop,
                "plan_age_game_loops": max(0, observation.game_loop - plan.created_game_loop),
                "fingerprint": fingerprint,
                "is_revision": (
                    previous_fingerprint is not None and previous_fingerprint != fingerprint
                ),
                "lifecycle_protocol": CURRENT_PROTOCOL_VERSION,
                "retained_command_ids": sorted(retained_command_ids),
            },
        )
        for command in accepted_plan.commands:
            if command.command_id not in self._command_states:
                self._transition_command(command, CommandStatus.PENDING, observation)

    @staticmethod
    def _plan_fingerprint(plan: PlanState) -> str:
        semantic_commands = [
            {
                "actor": command.actor,
                "name": command.name,
                "arguments": command.arguments,
                "priority": command.priority,
                "ttl_game_loops": command.ttl_game_loops,
                "preconditions": command.preconditions,
            }
            for command in plan.commands
        ]
        encoded = json.dumps(
            {
                "strategic_goal": plan.strategic_goal,
                "summary": plan.summary,
                "commands": semantic_commands,
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _command_semantic_key(command: ActionCommand) -> str:
        return json.dumps(
            {
                "actor": command.actor,
                "name": command.name,
                "arguments": command.arguments,
            },
            ensure_ascii=False,
            sort_keys=True,
        )

    def _active_planner_commands(
        self,
        observation: ObservationEnvelope,
    ) -> list[ActionCommand]:
        del observation
        if self._cached_plan is None:
            return []
        return [
            command
            for command in self._cached_plan.commands
            if (lifecycle := self._command_states.get(command.command_id)) is not None
            and lifecycle.status in _ACTIONABLE_COMMAND_STATUSES
        ]

    def _expire_planner_commands(self, observation: ObservationEnvelope) -> None:
        if self._cached_plan is None:
            return
        for command in self._cached_plan.commands:
            lifecycle = self._command_states.get(command.command_id)
            if lifecycle is None or lifecycle.status not in _ACTIONABLE_COMMAND_STATUSES:
                continue
            if observation.game_loop - command.created_game_loop >= command.ttl_game_loops:
                self._transition_command(
                    command,
                    CommandStatus.EXPIRED,
                    observation,
                    reason="planner command TTL expired before dispatch",
                )
                self._urgent_replan_requested = True

    def _defer_busy_actor_commands(
        self,
        planner_commands: list[ActionCommand],
        reflex_commands: list[ActionCommand],
        observation: ObservationEnvelope,
    ) -> tuple[list[ActionCommand], list[ActionCommand], list[str], list[ActionCommand]]:
        """Keep one in-flight semantic command per actor until its terminal report."""

        busy_actors = {
            lifecycle.command.actor
            for lifecycle in self._command_states.values()
            if lifecycle.status is CommandStatus.DISPATCHED
        }
        if not busy_actors:
            return planner_commands, reflex_commands, [], []

        available_planner: list[ActionCommand] = []
        available_reflex: list[ActionCommand] = []
        rejected: list[str] = []
        blocked: list[ActionCommand] = []
        reason = "actor has an in-flight dispatched command"
        for command in planner_commands:
            if command.actor not in busy_actors:
                available_planner.append(command)
                continue
            blocked.append(command)
            self._transition_command(
                command,
                CommandStatus.DEFERRED,
                observation,
                reason=reason,
            )
            if self._report_command_reason_once(command, reason):
                rejected.append(f"{command.command_id}: {reason}")
        for command in reflex_commands:
            if command.actor in busy_actors:
                blocked.append(command)
            else:
                available_reflex.append(command)
        return available_planner, available_reflex, rejected, blocked

    def _apply_validation_failures(
        self,
        failures: list[ValidationFailure],
        observation: ObservationEnvelope,
    ) -> list[str]:
        recorded: list[str] = []
        status_by_disposition = {
            ValidationDisposition.DEFERRED: CommandStatus.DEFERRED,
            ValidationDisposition.REJECTED: CommandStatus.REJECTED,
            ValidationDisposition.OBSOLETE: CommandStatus.OBSOLETE,
        }
        for failure in failures:
            rendered = f"{failure.command.command_id}: {failure.reason}"
            self._transition_command(
                failure.command,
                status_by_disposition[failure.disposition],
                observation,
                reason=failure.reason,
            )
            if self._report_command_reason_once(failure.command, failure.reason):
                recorded.append(rendered)
            if (
                failure.command.source is ActionSource.PLANNER
                and failure.disposition is not ValidationDisposition.DEFERRED
            ):
                self._urgent_replan_requested = True
        return recorded

    def _report_command_reason_once(self, command: ActionCommand, reason: str) -> bool:
        """Keep rejection metrics stable without hiding real lifecycle transitions."""

        key = (command.command_id, reason)
        if key in self._reported_command_reasons:
            return False
        self._reported_command_reasons.add(key)
        return True

    def _transition_command(
        self,
        command: ActionCommand,
        status: CommandStatus,
        observation: ObservationEnvelope,
        *,
        reason: str | None = None,
    ) -> bool:
        return self._set_command_lifecycle(
            command,
            status,
            run_id=observation.run_id,
            episode_id=observation.episode_id,
            step_id=observation.step_id,
            game_loop=observation.game_loop,
            reason=reason,
        )

    def _set_command_lifecycle(
        self,
        command: ActionCommand,
        status: CommandStatus,
        *,
        run_id: str,
        episode_id: str,
        step_id: int,
        game_loop: int | None,
        reason: str | None = None,
    ) -> bool:
        lifecycle = CommandLifecycle(command=command, status=status, reason=reason)
        current = self._command_states.get(command.command_id)
        if current is not None and current.command != command:
            raise RuntimeError(
                f"command ID {command.command_id!r} was reused with different semantics"
            )
        if current is not None and current.status is status and current.reason == reason:
            return False
        if (
            current is not None
            and current.status is not status
            and status not in _ALLOWED_COMMAND_TRANSITIONS.get(current.status, set())
        ):
            raise RuntimeError(
                f"illegal command lifecycle transition for {command.command_id!r}: "
                f"{current.status.value} -> {status.value}"
            )
        self._command_states[command.command_id] = lifecycle
        self.store.append_event(
            run_id=run_id,
            episode_id=episode_id,
            step_id=step_id,
            event_type="command_lifecycle",
            payload={
                "command": command.model_dump(mode="json"),
                "status": status.value,
                "reason": reason,
                "game_loop": game_loop,
            },
        )
        return True

    def _idle_reason(self) -> IdleReason:
        if self.config.agent.variant == "noop":
            return IdleReason.NOOP_BASELINE
        if self._planner_task is not None:
            return IdleReason.WAITING_FOR_PLANNER
        if self._last_planner_failure is IdleReason.PLANNER_TIMEOUT:
            return IdleReason.PLANNER_TIMEOUT
        if self._cached_plan is not None and any(
            self._command_states.get(command.command_id) is not None
            and self._command_states[command.command_id].status in _ACTIONABLE_COMMAND_STATUSES
            for command in self._cached_plan.commands
        ):
            return IdleReason.PLAN_COMMANDS_DEFERRED
        if self._cached_plan is not None and self._cached_plan.commands:
            return IdleReason.PLAN_EXHAUSTED
        return IdleReason.NO_LEGAL_ACTION

    def _request_replan_if_exhausted(self) -> None:
        if not self._planner_enabled or self._cached_plan is None:
            return
        if not self._cached_plan.commands:
            return
        if any(
            self._command_states.get(command.command_id) is not None
            and self._command_states[command.command_id].status in _ACTIONABLE_COMMAND_STATUSES
            for command in self._cached_plan.commands
        ):
            return
        plan_key = (
            self._cached_plan.source_step_id,
            self._cached_plan.created_game_loop,
        )
        if self._exhaustion_requested_for == plan_key:
            return
        self._exhaustion_requested_for = plan_key
        self._urgent_replan_requested = True

    async def _run_module(
        self,
        module: AgentModule,
        context: AgentContext,
    ) -> ModuleResult:
        started = time.perf_counter()
        try:
            result = await module.run(context)
        except Exception as error:
            self._record_planner_event(
                context.observation,
                "module_error",
                {
                    "module": module.name,
                    "latency_ms": (time.perf_counter() - started) * 1000,
                    "error_type": type(error).__name__,
                    "message": str(error),
                },
            )
            raise
        payload: dict[str, Any] = {
            "module": module.name,
            "latency_ms": (time.perf_counter() - started) * 1000,
            "command_count": len(result.commands),
            "model_call": result.model_call,
        }
        if result.model_call:
            payload.update(
                {
                    "provider": self.config.provider.kind,
                    "model": self.config.provider.model,
                    "usage": getattr(self.provider, "last_usage", None),
                }
            )
        if module.name in {"reflection", "planning", "action"}:
            payload["output"] = result.updates
        self._record_planner_event(context.observation, "module_result", payload)
        return result

    def _record_planner_event(
        self, observation: ObservationEnvelope, event_type: str, payload: dict[str, Any]
    ) -> None:
        self.store.append_event(
            run_id=observation.run_id,
            episode_id=observation.episode_id,
            step_id=observation.step_id,
            event_type=event_type,
            payload=payload,
        )

    def record_execution(self, report: ExecutionReport) -> None:
        self._record_execution_from(
            report,
            allowed_from={CommandStatus.DISPATCHED},
        )

    def _record_execution_from(
        self,
        report: ExecutionReport,
        *,
        allowed_from: set[CommandStatus],
    ) -> None:
        fingerprint = self._execution_fingerprint(report)
        existing = self._terminal_execution_fingerprints.get(report.command_id)
        if existing is not None:
            if existing == fingerprint:
                return
            raise RuntimeError(
                f"conflicting terminal execution report for command {report.command_id!r}"
            )
        lifecycle = self._command_states.get(report.command_id)
        if lifecycle is None:
            raise RuntimeError(f"execution report references unknown command {report.command_id!r}")
        if lifecycle.status not in allowed_from:
            raise RuntimeError(
                f"command {report.command_id!r} cannot complete from "
                f"state {lifecycle.status.value!r}"
            )
        self._validate_execution_identity(report, lifecycle.command)
        self.store.append_event(
            run_id=report.run_id,
            episode_id=report.episode_id,
            step_id=report.step_id,
            event_type="execution",
            payload=report,
        )
        self._terminal_execution_fingerprints[report.command_id] = fingerprint
        self._last_execution = report
        status = _COMMAND_STATUS_BY_EXECUTION_STATUS[report.status]
        self._set_command_lifecycle(
            lifecycle.command,
            status,
            run_id=report.run_id,
            episode_id=report.episode_id,
            step_id=report.step_id,
            game_loop=None,
            reason=report.failure_code or report.failure_reason,
        )
        if report.status is not ExecutionStatus.SUCCEEDED:
            self._urgent_replan_requested = True

    def _validate_execution_identity(
        self,
        report: ExecutionReport,
        command: ActionCommand,
    ) -> None:
        mismatches: list[str] = []
        if self._episode_key != (report.run_id, report.episode_id):
            mismatches.append("run_id/episode_id")
        if report.action_name is not None and report.action_name != command.name:
            mismatches.append("action_name")
        if report.actor is not None and report.actor != command.actor:
            mismatches.append("actor")
        if report.source is not None and report.source is not command.source:
            mismatches.append("source")
        has_complete_provenance = all(
            value is not None for value in (report.action_name, report.actor, report.source)
        )
        if (
            report.protocol_version == CURRENT_PROTOCOL_VERSION
            or has_complete_provenance
            or report.requested_arguments
        ) and report.requested_arguments != command.arguments:
            mismatches.append("requested_arguments")
        if mismatches:
            raise RuntimeError(
                f"execution report identity mismatch for {report.command_id!r}: "
                + ", ".join(mismatches)
            )

    @staticmethod
    def _execution_fingerprint(report: ExecutionReport) -> str:
        encoded = json.dumps(
            report.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    def end_episode(self, result: EpisodeResult) -> None:
        if self._episode_key is not None and self._episode_key != (
            result.run_id,
            result.episode_id,
        ):
            raise RuntimeError(
                "episode result does not match the active runtime episode: "
                f"{result.run_id!r}/{result.episode_id!r}"
            )
        fingerprint = self._episode_fingerprint(result)
        if self._episode_result_fingerprint is not None:
            if self._episode_result_fingerprint == fingerprint:
                return
            raise RuntimeError(
                f"conflicting episode result for {result.run_id!r}/{result.episode_id!r}"
            )
        failure_code, failure_reason = self._missing_execution_details(result)
        for lifecycle in list(self._command_states.values()):
            if lifecycle.status not in {
                CommandStatus.PENDING,
                CommandStatus.DEFERRED,
                CommandStatus.DISPATCHED,
            }:
                continue
            command = lifecycle.command
            command_failure_code = failure_code
            command_failure_reason = failure_reason
            if lifecycle.status is not CommandStatus.DISPATCHED:
                command_failure_code = "episode_ended_before_dispatch"
                command_failure_reason = (
                    "episode ended before the command was selected for dispatch"
                )
            self._record_execution_from(
                ExecutionReport(
                    run_id=result.run_id,
                    episode_id=result.episode_id,
                    step_id=result.steps,
                    command_id=command.command_id,
                    success=False,
                    action_name=command.name,
                    actor=command.actor,
                    source=command.source,
                    requested_arguments=command.arguments,
                    status=ExecutionStatus.CANCELLED,
                    execution_stage=ExecutionStage.EPISODE_END,
                    failure_code=command_failure_code,
                    failure_reason=command_failure_reason,
                    game_result=result.outcome.value,
                ),
                allowed_from={
                    CommandStatus.PENDING,
                    CommandStatus.DEFERRED,
                    CommandStatus.DISPATCHED,
                },
            )
        self.store.record_episode(result)
        lessons = self.store.lessons(result.run_id, result.episode_id)
        self.store.record_episode_summary(
            EpisodeSummary(
                run_id=result.run_id,
                episode_id=result.episode_id,
                scenario=result.scenario,
                seed=result.seed,
                outcome=result.outcome,
                summary=(
                    f"{result.scenario} ended with {result.outcome.value} after "
                    f"{result.steps} steps and score {result.score:.2f}."
                ),
                lessons=lessons,
                source_step_id=result.steps,
                metrics=result.metrics,
            )
        )
        self._episode_result_fingerprint = fingerprint

    @staticmethod
    def _episode_fingerprint(result: EpisodeResult) -> str:
        encoded = json.dumps(
            result.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _missing_execution_details(result: EpisodeResult) -> tuple[str, str]:
        failure_reason = result.failure_reason or ""
        synthetic_prefixes = (
            "worker exited with status ",
            "live run was cancelled ",
            "live run failed: ",
        )
        if failure_reason.startswith(synthetic_prefixes):
            return (
                "worker_terminated_before_execution_report",
                "worker terminated before reporting command completion: " + failure_reason,
            )
        return (
            "bridge_execution_report_missing",
            "episode ended before the Bridge reported command completion",
        )

    async def close(self) -> None:
        await self._cancel_planner()
        close = getattr(self.provider, "close", None)
        if close is not None:
            await close()
        self.store.close()

    async def _cancel_planner(self) -> None:
        if self._planner_task is not None:
            self._planner_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._planner_task
            self._planner_task = None
