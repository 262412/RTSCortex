"""Memory, reflection, planning, and action modules inspired by Orak's module chain."""

from __future__ import annotations

import json
from typing import Any

from rtscortex.agents.models import PlanningOutput, ReflectionOutput
from rtscortex.contracts import ActionCommand, ActionSource, ObservationEnvelope
from rtscortex.contracts.interfaces import AgentContext, LLMProvider, ModuleResult
from rtscortex.memory import EventStore, StoredEvent


def _model_observation(observation: ObservationEnvelope) -> dict[str, Any]:
    """Project an observation into the compact, structured context used by an LLM."""

    payload = observation.model_dump(mode="json")
    payload.pop("observed_at")
    payload.pop("text_observation")
    payload.pop("image_uri")
    return payload


def _compact_event(event: StoredEvent) -> dict[str, Any] | None:
    if event.event_type == "decision":
        batch = event.payload["batch"]
        return {
            "event_type": event.event_type,
            "step_id": event.step_id,
            "strategic_goal": batch.get("strategic_goal", ""),
            "summary": batch.get("summary", ""),
            "commands": [
                {
                    "actor": command["actor"],
                    "name": command["name"],
                    "arguments": command.get("arguments", []),
                    "source": command["source"],
                }
                for command in batch.get("commands", [])
            ],
            "rejected_commands": batch.get("rejected_commands", []),
        }
    if event.event_type == "execution":
        return {
            "event_type": event.event_type,
            "step_id": event.step_id,
            "command_id": event.payload.get("command_id"),
            "success": event.payload.get("success"),
            "failure_reason": event.payload.get("failure_reason"),
            "pysc2_function": event.payload.get("pysc2_function"),
        }
    return None


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
        compact_events = [
            compact for event in recent if (compact := _compact_event(event)) is not None
        ]
        return ModuleResult(
            module=self.name,
            updates={
                "recent_events": compact_events,
                "lessons": [
                    {
                        "source_step_id": lesson.source_step_id,
                        "content": lesson.content,
                    }
                    for lesson in self.store.lesson_records(
                        observation.run_id,
                        observation.episode_id,
                    )
                ],
                "episode_summaries": [
                    summary.model_dump(mode="json")
                    for summary in self.store.recent_episode_summaries(observation.run_id, limit=3)
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
            "observation": _model_observation(context.observation),
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
            "observation": _model_observation(context.observation),
            "memory": context.memory,
        }
        output = await self.provider.generate(
            PlanningOutput,
            system_prompt=(
                "Create a short-horizon StarCraft II plan using only the available actions. "
                "Use an actor exactly as listed in the chosen action's actor_scopes. Return "
                "typed action proposals; never emit raw PySC2 calls. The arguments field is "
                "a positional JSON array in the exact argument_names order, and every value "
                "must match the corresponding argument_types entry. For example, an action "
                'with argument_names=["target"] receives arguments=["0x1001c0001"], '
                "never a named {name, value} object. Return at most three concise plan steps "
                "and three proposed actions, with no more than one action per actor."
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
                preconditions=(
                    {"unit_exists": str(proposal.arguments[0])}
                    if proposal.name == "Attack_Unit" and proposal.arguments
                    else {}
                ),
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
