"""Two-timescale runtime coordinating deliberation and reflexes."""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass
from typing import Any

from rtscortex.agents import ActionModule, MemoryModule, PlanningModule, ReflectionModule
from rtscortex.config import ExperimentConfig
from rtscortex.contracts import (
    ActionBatch,
    ActionCommand,
    ActionSource,
    EpisodeResult,
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
        self.memory_module = MemoryModule(store, config.memory.short_term_window)
        self.reflection_module = ReflectionModule(provider)
        self.planning_module = PlanningModule(provider)
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

    async def tick(self, observation: ObservationEnvelope) -> ActionBatch:
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
            elif self._planner_task is None:
                self._planner_task = asyncio.create_task(self._deliberate_with_timeout(context))

        reflex_started = time.perf_counter()
        reflex_commands = self.reflex.evaluate(observation)
        reflex_latency_ms = (time.perf_counter() - reflex_started) * 1000

        planner_commands = [] if self._cached_plan is None else self._cached_plan.commands
        merged = self.arbiter.merge(
            planner_commands,
            reflex_commands,
            game_loop=observation.game_loop,
        )
        if not merged:
            fallback = self._fallback(observation)
            if fallback is not None:
                merged = [fallback]
        outcome = self.validator.validate(merged, observation)
        plan = self._cached_plan or PlanState("", "", [])
        batch = ActionBatch(
            run_id=observation.run_id,
            episode_id=observation.episode_id,
            step_id=observation.step_id,
            decision_id=(
                f"{observation.run_id}:{observation.episode_id}:{observation.step_id}:decision"
            ),
            strategic_goal=plan.strategic_goal,
            summary=plan.summary,
            commands=outcome.accepted,
            rejected_commands=outcome.rejected,
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
            self._cached_plan = await self._deliberate_with_timeout(context)
        except TimeoutError:
            self._record_planner_event(context.observation, "planner_timeout", {})
        except Exception as error:
            self._record_planner_event(
                context.observation,
                "planner_error",
                {"error_type": type(error).__name__, "message": str(error)},
            )

    async def _collect_finished_planner(self, observation: ObservationEnvelope) -> None:
        if self._planner_task is None or not self._planner_task.done():
            return
        try:
            self._cached_plan = self._planner_task.result()
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
        return await asyncio.wait_for(
            self._deliberate(context),
            timeout=self.config.runtime.planner_timeout_seconds,
        )

    async def _deliberate(self, context: AgentContext) -> PlanState:
        working_memory: dict[str, Any] = dict(context.memory)
        if self._reflection_enabled:
            reflection = await self._run_module(
                self.reflection_module,
                context,
                model_call=True,
            )
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
        planning = await self._run_module(
            self.planning_module,
            planning_context,
            model_call=True,
        )
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
        )

    async def _run_module(
        self,
        module: AgentModule,
        context: AgentContext,
        *,
        model_call: bool = False,
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
            "model_call": model_call,
        }
        if model_call:
            payload.update(
                {
                    "provider": self.config.provider.kind,
                    "model": self.config.provider.model,
                    "usage": getattr(self.provider, "last_usage", None),
                }
            )
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
        if not any(action.name == "No_Operation" for action in observation.available_actions):
            return None
        return ActionCommand(
            command_id=(
                f"{observation.run_id}:{observation.episode_id}:{observation.step_id}:fallback:0"
            ),
            actor="global",
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

    async def close(self) -> None:
        if self._planner_task is not None:
            self._planner_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._planner_task
        close = getattr(self.provider, "close", None)
        if close is not None:
            await close()
        self.store.close()
