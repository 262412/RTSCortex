"""Two-timescale runtime coordinating deliberation and reflexes."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import time
from dataclasses import asdict, dataclass
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
    ActionBatch,
    ActionCommand,
    ActionSource,
    EpisodeResult,
    EpisodeSummary,
    ExecutionReport,
    ObservationEnvelope,
)
from rtscortex.contracts.interfaces import AgentContext, AgentModule, LLMProvider, ModuleResult
from rtscortex.memory import EventStore
from rtscortex.reflex import ReflexEngine
from rtscortex.runtime.validation import ActionArbiter, ActionValidator


@dataclass(frozen=True)
class PlanState:
    strategic_goal: str
    summary: str
    commands: list[ActionCommand]
    source_step_id: int
    created_game_loop: int


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
        self.action_module = ActionModule(config.runtime.max_actions)
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
        self._last_plan_game_loop: int | None = None
        self._last_decision: ActionBatch | None = None
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

        context = AgentContext(
            observation=observation,
            last_execution=self._last_execution,
            last_decision=self._last_decision,
        )
        memory_result = await self._run_module(self.memory_module, context)
        context = AgentContext(
            observation=observation,
            memory=memory_result.updates,
            last_execution=self._last_execution,
            last_decision=self._last_decision,
        )

        if self._planner_enabled and self._should_plan(observation):
            self._last_plan_game_loop = observation.game_loop
            if self.config.runtime.deterministic:
                await self._run_deterministic_planner(context)
            elif self.config.environment.pause_until_first_plan and self._cached_plan is None:
                await self._run_initial_plan_barrier(context)
            elif self._planner_task is None:
                self._planner_task = asyncio.create_task(self._deliberate_with_timeout(context))

        reflex_started = time.perf_counter()
        reflex_commands = self.reflex.evaluate(observation)
        reflex_latency_ms = (time.perf_counter() - reflex_started) * 1000

        planner_commands = [] if self._cached_plan is None else self._cached_plan.commands
        candidate_outcome = self.validator.validate_candidates(
            [*planner_commands, *reflex_commands],
            observation,
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
        merged = arbitration.selected
        if not merged:
            fallback = self._fallback(observation)
            if fallback is not None:
                merged = [fallback]
        outcome = self.validator.validate(merged, observation)
        accepted_commands = outcome.accepted
        rejected_commands = [*candidate_outcome.rejected, *outcome.rejected]
        if not accepted_commands:
            fallback = self._fallback(observation)
            if fallback is not None:
                fallback_outcome = self.validator.validate([fallback], observation)
                accepted_commands = fallback_outcome.accepted
                rejected_commands = [*rejected_commands, *fallback_outcome.rejected]
        plan = self._cached_plan or PlanState(
            "", "", [], observation.step_id, observation.game_loop
        )
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
            },
        )
        self._last_decision = batch
        return batch

    def _should_plan(self, observation: ObservationEnvelope) -> bool:
        if self._planner_task is not None:
            return False
        if self._cached_plan is None or self._last_plan_game_loop is None:
            return True
        elapsed = observation.game_loop - self._last_plan_game_loop
        if elapsed >= self.config.runtime.planning_interval_game_loops:
            return True
        alert_interval = max(1, self.config.runtime.planning_interval_game_loops // 4)
        return bool(observation.alerts) and elapsed >= alert_interval

    async def _run_deterministic_planner(self, context: AgentContext) -> None:
        try:
            plan = await self._deliberate_with_timeout(context)
            self._accept_plan(plan, context.observation)
        except TimeoutError:
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
        )
        planning = await self._run_module(self.planning_module, planning_context)
        working_memory.update(planning.updates)
        action_context = AgentContext(
            observation=context.observation,
            memory=working_memory,
            last_execution=context.last_execution,
            last_decision=context.last_decision,
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
        self._last_plan_game_loop = None
        self._last_decision = None
        self._last_execution = None

        decision_event = self.store.last_event(
            observation.run_id,
            observation.episode_id,
            "decision",
        )
        if decision_event is not None:
            self._last_decision = ActionBatch.model_validate(decision_event.payload["batch"])
        execution_event = self.store.last_event(
            observation.run_id,
            observation.episode_id,
            "execution",
        )
        if execution_event is not None:
            self._last_execution = ExecutionReport.model_validate(execution_event.payload)

        plan_event = self.store.last_event(
            observation.run_id,
            observation.episode_id,
            "plan_accepted",
        )
        if plan_event is not None:
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
        elif self._last_decision is not None:
            planner_commands = [
                command
                for command in self._last_decision.commands
                if command.source is ActionSource.PLANNER
            ]
            if planner_commands:
                self._cached_plan = PlanState(
                    strategic_goal=self._last_decision.strategic_goal,
                    summary=self._last_decision.summary,
                    commands=planner_commands,
                    source_step_id=self._last_decision.step_id,
                    created_game_loop=max(
                        command.created_game_loop for command in planner_commands
                    ),
                )
        if self._cached_plan is not None:
            self._last_plan_game_loop = self._cached_plan.created_game_loop

    def _accept_plan(self, plan: PlanState, observation: ObservationEnvelope) -> None:
        accepted_plan = PlanState(
            strategic_goal=plan.strategic_goal,
            summary=plan.summary,
            commands=[
                command.model_copy(update={"created_game_loop": observation.game_loop})
                for command in plan.commands
            ],
            source_step_id=plan.source_step_id,
            created_game_loop=observation.game_loop,
        )
        fingerprint = self._plan_fingerprint(accepted_plan)
        previous_fingerprint = (
            None if self._cached_plan is None else self._plan_fingerprint(self._cached_plan)
        )
        self._cached_plan = accepted_plan
        self._last_plan_game_loop = accepted_plan.created_game_loop
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
            },
        )

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

    @staticmethod
    def _fallback(observation: ObservationEnvelope) -> ActionCommand | None:
        action = next(
            (
                candidate
                for candidate in observation.available_actions
                if candidate.name == "No_Operation"
            ),
            None,
        )
        if action is None:
            return None
        return ActionCommand(
            command_id=(
                f"{observation.run_id}:{observation.episode_id}:{observation.step_id}:fallback:0"
            ),
            actor=action.actor_scopes[0] if action.actor_scopes else "global",
            name="No_Operation",
            arguments=[],
            priority=0,
            ttl_game_loops=1,
            created_game_loop=observation.game_loop,
            source=ActionSource.FALLBACK,
        )

    def record_execution(self, report: ExecutionReport) -> None:
        self.store.append_event(
            run_id=report.run_id,
            episode_id=report.episode_id,
            step_id=report.step_id,
            event_type="execution",
            payload=report,
        )
        self._last_execution = report

    def end_episode(self, result: EpisodeResult) -> None:
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
