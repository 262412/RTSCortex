"""Memory, reflection, planning, and action modules inspired by Orak's module chain."""

from __future__ import annotations

from typing import Any

from rtscortex.agents.context import (
    ContextBudget,
    build_planning_context,
    build_reflection_context,
    compact_memory_events,
    compact_spatial_context,
    model_observation,
)
from rtscortex.agents.models import (
    PlanningOutput,
    ReflectionOutput,
    planning_output_model,
    project_planning_observation,
)
from rtscortex.contracts import ActionCommand, ActionSource, ObservationEnvelope
from rtscortex.contracts.interfaces import AgentContext, LLMProvider, ModuleResult
from rtscortex.memory import EventStore, StoredEvent


def _model_observation(observation: ObservationEnvelope) -> dict[str, Any]:
    """Project an observation into the compact, structured context used by an LLM."""

    return model_observation(observation)[0]


def _compact_spatial_context(text_observation: str) -> list[str]:
    """Retain actionable screen/minimap coordinates without upstream prompt bulk."""

    return compact_spatial_context(text_observation)


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
    if event.event_type in {"planner_error", "planner_timeout", "module_error"}:
        return {
            "event_type": event.event_type,
            "step_id": event.step_id,
            "module": event.payload.get("module"),
            "error_type": event.payload.get("error_type"),
            "message": event.payload.get("message"),
        }
    return None


REFLECTION_SYSTEM_PROMPT = (
    "Evaluate the previous decision from matching execution and current state. A no_op never "
    "proves a plan action ran; never infer execution from summaries. Return concise "
    "evidence-backed lessons."
)


PLANNING_SYSTEM_PROMPT = (
    "Plan only the actions needed in the current StarCraft II decision cycle. Use only "
    "the exact, complete action name from observation.available_actions and an actor from "
    "that action's actor_scopes; never shorten or invent a name. Return "
    "at most two short steps and three typed actions, with one action per actor and no "
    "raw PySC2 calls. arguments is a positional JSON array in argument_names order; a "
    "position uses two integer coordinates in a nested array such as [[80,80]]. Use "
    "coordinates only from observation.spatial_context for that exact action. A Build_ "
    "action already moves its worker into range and performs placement. Never pair "
    "Move_Screen or Move_Minimap with a Build_ action for the same actor; emit only the "
    "Build_ action. Do not repeat a successfully "
    "executed action or an active_plan command that remains valid. In a Protoss opening "
    "without enemies, prioritize legal economy and production: Pylon, then Gateway. Emit "
    "Build_Pylon_Screen only when supply_free <= 4 and no Pylon in own_structures has "
    "status='constructing'; otherwise do not propose it. After a completed Gateway exists, "
    "prioritize Train_Zealot, especially while army_supply is zero, and keep producing "
    "Zealots when minerals and supply allow. Do not choose Train_Stalker without a completed "
    "CyberneticsCore and enough vespene."
)


def _planner_preconditions(name: str, arguments: list[Any]) -> dict[str, Any]:
    if name == "Attack_Unit" and arguments:
        return {"unit_exists": str(arguments[0])}
    if name == "Build_Pylon_Screen":
        return {
            "max_supply_free": 4,
            "no_pending_structure": "Pylon",
        }
    if name == "Build_Gateway_Screen":
        return {"structure_absent": "Gateway"}
    return {}


class MemoryModule:
    name = "memory"

    def __init__(
        self,
        store: EventStore,
        short_term_window: int,
        context_budget: ContextBudget | None = None,
    ) -> None:
        self.store = store
        self.short_term_window = short_term_window
        self.context_budget = context_budget or ContextBudget()

    async def run(self, context: AgentContext) -> ModuleResult:
        observation = context.observation
        recent = self.store.recent_events(
            observation.run_id,
            observation.episode_id,
            max(self.short_term_window * 8, self.context_budget.max_recent_events * 8),
        )
        compact_events = compact_memory_events(
            [compact for event in recent if (compact := _compact_event(event)) is not None]
        )[-self.short_term_window :]
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
                        limit=max(10, self.context_budget.max_lessons * 2),
                    )
                ],
                "episode_summaries": [
                    summary.model_dump(mode="json")
                    for summary in self.store.recent_episode_summaries(
                        observation.run_id,
                        limit=max(3, self.context_budget.max_episode_summaries * 2),
                    )
                ],
            },
        )


class ReflectionModule:
    name = "reflection"

    def __init__(self, provider: LLMProvider, context_budget: ContextBudget | None = None) -> None:
        self.provider = provider
        self.context_budget = context_budget or ContextBudget()

    async def run(self, context: AgentContext) -> ModuleResult:
        if context.last_decision is None:
            return ModuleResult(module=self.name, updates={"reflection": None, "lessons": []})
        prompt_context = build_reflection_context(
            observation=context.observation,
            last_decision=context.last_decision,
            last_execution=context.last_execution,
            budget=self.context_budget,
            system_prompt=REFLECTION_SYSTEM_PROMPT,
        )
        output = await self.provider.generate(
            ReflectionOutput,
            system_prompt=REFLECTION_SYSTEM_PROMPT,
            user_prompt=prompt_context.user_prompt,
        )
        return ModuleResult(
            module=self.name,
            updates={
                "reflection": output.summary,
                "lessons": output.lessons,
                "context_compaction": prompt_context.statistics,
            },
            model_call=True,
        )


class PlanningModule:
    name = "planning"

    def __init__(self, provider: LLMProvider, context_budget: ContextBudget | None = None) -> None:
        self.provider = provider
        self.context_budget = context_budget or ContextBudget()

    async def run(self, context: AgentContext) -> ModuleResult:
        planning_observation = project_planning_observation(context.observation)
        prompt_context = build_planning_context(
            observation=planning_observation,
            memory=context.memory,
            last_decision=context.last_decision,
            last_execution=context.last_execution,
            budget=self.context_budget,
            system_prompt=PLANNING_SYSTEM_PROMPT,
        )
        output_type = planning_output_model(planning_observation)
        output = await self.provider.generate(
            output_type,
            system_prompt=PLANNING_SYSTEM_PROMPT,
            user_prompt=prompt_context.user_prompt,
        )
        return ModuleResult(
            module=self.name,
            updates={
                "plan": output.model_dump(mode="json"),
                "context_compaction": prompt_context.statistics,
            },
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
                preconditions=_planner_preconditions(proposal.name, proposal.arguments),
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
