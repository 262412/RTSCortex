"""Memory, reflection, planning, and action modules inspired by Orak's module chain."""

from __future__ import annotations

import json
from typing import Any

from rtscortex.agents.models import PlanningOutput, ReflectionOutput
from rtscortex.contracts import ActionCommand, ActionSource
from rtscortex.contracts.interfaces import AgentContext, LLMProvider, ModuleResult
from rtscortex.memory import EventStore


class MemoryModule:
    name = "memory"

    def __init__(self, store: EventStore, short_term_window: int) -> None:
        self.store = store
        self.short_term_window = short_term_window

    async def run(self, context: AgentContext) -> ModuleResult:
        observation = context.observation
        recent = self.store.recent_events(
            observation.run_id, observation.episode_id, self.short_term_window
        )
        return ModuleResult(
            module=self.name,
            updates={
                "recent_events": [event.__dict__ for event in recent],
                "lessons": [
                    lesson.__dict__
                    for lesson in self.store.lesson_records(
                        observation.run_id,
                        observation.episode_id,
                    )
                ],
                "episode_summaries": [
                    summary.model_dump(mode="json")
                    for summary in self.store.recent_episode_summaries(observation.run_id)
                ],
            },
        )


class ReflectionModule:
    name = "reflection"

    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider

    async def run(self, context: AgentContext) -> ModuleResult:
        if context.last_decision is None:
            return ModuleResult(module=self.name, updates={"reflection": None, "lessons": []})
        payload = {
            "observation": context.observation.model_dump(mode="json"),
            "last_decision": context.last_decision.model_dump(mode="json"),
            "last_execution": (
                None
                if context.last_execution is None
                else context.last_execution.model_dump(mode="json")
            ),
        }
        output = await self.provider.generate(
            ReflectionOutput,
            system_prompt=(
                "Evaluate the previous StarCraft II decision. Return a concise summary and "
                "only reusable, evidence-backed lessons."
            ),
            user_prompt=json.dumps(payload, ensure_ascii=False, sort_keys=True),
        )
        return ModuleResult(
            module=self.name,
            updates={"reflection": output.summary, "lessons": output.lessons},
            model_call=True,
        )


class PlanningModule:
    name = "planning"

    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider

    async def run(self, context: AgentContext) -> ModuleResult:
        payload = {
            "observation": context.observation.model_dump(mode="json"),
            "memory": context.memory,
        }
        output = await self.provider.generate(
            PlanningOutput,
            system_prompt=(
                "Create a short-horizon StarCraft II plan using only the available actions. "
                "Return typed action proposals; never emit raw PySC2 calls."
            ),
            user_prompt=json.dumps(payload, ensure_ascii=False, sort_keys=True),
        )
        return ModuleResult(
            module=self.name,
            updates={"plan": output.model_dump(mode="json")},
            model_call=True,
        )


class ActionModule:
    name = "action"

    def __init__(self, max_actions: int) -> None:
        self.max_actions = max_actions

    async def run(self, context: AgentContext) -> ModuleResult:
        raw_plan: dict[str, Any] = context.memory.get("plan", {})
        plan = PlanningOutput.model_validate(raw_plan)
        commands = [
            ActionCommand(
                command_id=(
                    f"{context.observation.run_id}:{context.observation.episode_id}:"
                    f"{context.observation.step_id}:planner:{index}"
                ),
                actor=proposal.actor,
                name=proposal.name,
                arguments=proposal.arguments,
                priority=proposal.priority,
                ttl_game_loops=proposal.ttl_game_loops,
                created_game_loop=context.observation.game_loop,
                source=ActionSource.PLANNER,
            )
            for index, proposal in enumerate(plan.proposed_actions[: self.max_actions])
        ]
        return ModuleResult(
            module=self.name,
            updates={
                "strategic_goal": plan.strategic_goal,
                "plan_summary": " | ".join(plan.steps),
            },
            commands=commands,
        )
