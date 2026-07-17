"""Memory, reflection, planning, and action modules inspired by Orak's module chain."""

from __future__ import annotations

from typing import Any

from rtscortex.agents.context import (
    ContextBudget,
    build_planning_context,
    build_reflection_context,
    compact_execution_payload,
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
            **compact_execution_payload(event.payload),
            "step_id": event.step_id,
        }
    if event.event_type in {
        "planner_error",
        "planner_timeout",
        "module_error",
        "module_failed",
    }:
        return {
            "event_type": event.event_type,
            "step_id": event.step_id,
            "module": event.payload.get("module"),
            "error_type": event.payload.get("error_type"),
            "message": event.payload.get("message"),
        }
    return None


REFLECTION_SYSTEM_PROMPT = (
    "Use matching execution and deterministic goal_progress to evaluate the previous decision. "
    "A no_op never proves a plan action ran. Report whether the goal advanced, then return "
    "concise "
    "evidence-backed lessons; never infer execution from summaries."
)


PLANNING_SYSTEM_PROMPT = (
    "Plan only the actions needed in the current StarCraft II decision cycle. Use only "
    "the exact, complete action name from observation.available_actions and an actor from "
    "that action's actor_scopes; never shorten or invent a name. Return "
    "at most two short steps and three typed actions, with one action per actor and no "
    "raw PySC2 calls. arguments is a positional JSON array in argument_names order; a "
    "position uses two integer coordinates in a nested array such as [[80,80]]. Use "
    "only a complete argument list from that action and actor's argument_candidates. "
    "Never emit No_Operation; return an empty proposed_actions list when no legal action "
    "exists. Stop and Hold_Position are forbidden when goal_progress reports an advancing "
    "action unless defensive_hold_required is true. Prefer goal_progress.unique_next_action "
    "when it is present. An Attack_Unit target must be an enemy tag from argument_candidates. "
    "A Build_ "
    "action already moves its worker into range and performs placement. Never pair "
    "Move_Screen or Move_Minimap with a Build_ action for the same actor; emit only the "
    "Build_ action. Do not move the Builder merely to wait for minerals: keep it near the "
    "current base and return no Builder action until the next opening build is legal. Use "
    "Move_Minimap for the Builder only when the opening Pylon and Gateway are complete and "
    "an expansion must be scouted; prefer an unseen resource-cluster candidate and keep that "
    "move as the actor's only proposal. Do not repeat a successfully "
    "executed action or any active_plan command whose status is pending, deferred, or "
    "dispatched. In a Protoss opening "
    "without enemies, prioritize legal economy and production: Pylon, then Gateway. Emit "
    "Build_Pylon_Screen only when supply_free <= 4 and no Pylon in own_structures has "
    "status='constructing'; otherwise do not propose it. After a completed Gateway exists, "
    "prioritize Train_Zealot, especially while army_supply is zero, and keep producing "
    "Zealots when minerals and supply allow. Do not choose Train_Stalker without a completed "
    "CyberneticsCore and enough vespene."
)


def _planner_preconditions(name: str, arguments: list[Any]) -> dict[str, Any]:
    if name == "Attack_Unit" and arguments:
        return {"enemy_target_exists": str(arguments[0])}
    if name == "Build_Pylon_Screen":
        return {
            "max_supply_free": 4,
            "no_pending_structure": "Pylon",
        }
    if name == "Build_Gateway_Screen":
        return {"structure_absent": "Gateway"}
    return {}


def _has_observation_bound_position(arguments: list[Any]) -> bool:
    return any(
        isinstance(argument, list)
        and len(argument) == 2
        and all(isinstance(value, int) and not isinstance(value, bool) for value in argument)
        for argument in arguments
    )


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
            return ModuleResult(
                module=self.name,
                updates={
                    "reflection": None,
                    "lessons": [],
                    "goal_progress": (
                        None
                        if context.goal_progress is None
                        else context.goal_progress.model_dump(mode="json")
                    ),
                },
            )
        prompt_context = build_reflection_context(
            observation=context.observation,
            last_decision=context.last_decision,
            last_execution=context.last_execution,
            goal_progress=context.goal_progress,
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
                "goal_progress": (
                    None
                    if context.goal_progress is None
                    else context.goal_progress.model_dump(mode="json")
                ),
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
            active_plan=context.active_plan,
            last_execution=context.last_execution,
            goal_progress=context.goal_progress,
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
                "goal_progress": (
                    None
                    if context.goal_progress is None
                    else context.goal_progress.model_dump(mode="json")
                ),
                "context_compaction": prompt_context.statistics,
            },
            model_call=True,
        )


class ActionModule:
    name = "action"

    def __init__(self, max_actions: int, planner_command_ttl_game_loops: int = 16) -> None:
        self.max_actions = max_actions
        self.planner_command_ttl_game_loops = planner_command_ttl_game_loops

    async def run(self, context: AgentContext) -> ModuleResult:
        raw_plan: dict[str, Any] = context.memory.get("plan", {})
        plan = PlanningOutput.model_validate(raw_plan)
        commands: list[ActionCommand] = []
        position_actors = {
            proposal.actor
            for proposal in plan.proposed_actions
            if _has_observation_bound_position(proposal.arguments)
        }
        selected_position_actors: set[str] = set()
        selected_proposals = []
        for index, proposal in enumerate(plan.proposed_actions):
            if proposal.actor in position_actors:
                if (
                    proposal.actor in selected_position_actors
                    or not _has_observation_bound_position(proposal.arguments)
                ):
                    continue
                selected_position_actors.add(proposal.actor)
            selected_proposals.append((index, proposal))
            if len(selected_proposals) >= self.max_actions:
                break

        for index, proposal in selected_proposals:
            commands.append(
                ActionCommand(
                    command_id=(
                        f"{context.observation.run_id}:{context.observation.episode_id}:"
                        f"{context.observation.step_id}:planner:{index}"
                    ),
                    actor=proposal.actor,
                    name=proposal.name,
                    arguments=proposal.arguments,
                    priority=proposal.priority,
                    ttl_game_loops=self.planner_command_ttl_game_loops,
                    created_game_loop=context.observation.game_loop,
                    source=ActionSource.PLANNER,
                    preconditions=_planner_preconditions(proposal.name, proposal.arguments),
                )
            )
        return ModuleResult(
            module=self.name,
            updates={
                "strategic_goal": plan.strategic_goal,
                "plan_summary": " | ".join(plan.steps),
            },
            commands=commands,
        )
