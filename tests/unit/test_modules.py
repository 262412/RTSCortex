from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, NoReturn

import pytest
from pydantic import BaseModel, ValidationError

from rtscortex.agents import (
    ActionModule,
    ContextBudget,
    ContextBudgetExceeded,
    MemoryModule,
    PlanningModule,
    PlanningOutput,
    ReflectionModule,
)
from rtscortex.agents.models import (
    ActionProposal,
    ReflectionOutput,
    planning_output_model,
)
from rtscortex.contracts import (
    ActionArgumentType,
    ActionBatch,
    AvailableAction,
    EconomyState,
    ExecutionReport,
    ProductionItem,
    UnitState,
)
from rtscortex.contracts.interfaces import AgentContext, ResponseT
from rtscortex.memory import EventStore
from tests.helpers import make_observation


class FailingProvider:
    async def generate(
        self,
        response_type: type[ResponseT],
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> NoReturn:
        del response_type, system_prompt, user_prompt
        raise AssertionError("provider should not be called on the first step")


def test_reflection_skips_first_decision() -> None:
    module = ReflectionModule(FailingProvider())
    result = asyncio.run(module.run(AgentContext(observation=make_observation())))
    assert result.updates == {"reflection": None, "lessons": []}


class CapturingProvider:
    def __init__(self) -> None:
        self.system_prompt = ""
        self.user_prompt = ""
        self.response_type: type[BaseModel] | None = None

    async def generate(
        self,
        response_type: type[ResponseT],
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> ResponseT:
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.response_type = response_type
        output: BaseModel
        if issubclass(response_type, PlanningOutput):
            output = PlanningOutput(strategic_goal="Hold", steps=[], proposed_actions=[])
        else:
            output = ReflectionOutput(summary="Review", lessons=[])
        return response_type.model_validate(output.model_dump())


def test_model_modules_omit_verbose_text_observation() -> None:
    observation = make_observation().model_copy(
        update={"text_observation": "raw-observation-marker " * 1000}
    )
    provider = CapturingProvider()
    planning = PlanningModule(provider)

    asyncio.run(planning.run(AgentContext(observation=observation)))

    prompt = json.loads(provider.user_prompt)
    assert "text_observation" not in prompt["observation"]
    assert "positional JSON array" in provider.system_prompt

    decision = ActionBatch(
        run_id=observation.run_id,
        episode_id=observation.episode_id,
        step_id=0,
        decision_id="decision-0",
    )
    asyncio.run(
        ReflectionModule(provider).run(
            AgentContext(observation=observation, last_decision=decision)
        )
    )
    reflection_prompt = json.loads(provider.user_prompt)
    assert "text_observation" not in reflection_prompt["observation"]


def test_planning_prompt_requires_exact_direct_actions() -> None:
    provider = CapturingProvider()

    asyncio.run(PlanningModule(provider).run(AgentContext(observation=make_observation())))

    assert "exact, complete action name" in provider.system_prompt
    assert "integer coordinates" in provider.system_prompt
    assert "Never pair Move_Screen or Move_Minimap with a Build_ action" in (provider.system_prompt)
    assert "supply_free <= 4" in provider.system_prompt
    assert "status='constructing'" in provider.system_prompt
    assert "After a completed Gateway exists" in provider.system_prompt
    assert "prioritize Train_Zealot" in provider.system_prompt
    assert "army_supply is zero" in provider.system_prompt
    assert "Do not choose Train_Stalker without a completed CyberneticsCore" in (
        provider.system_prompt
    )


def test_reflection_prompt_requires_execution_evidence() -> None:
    observation = make_observation()
    provider = CapturingProvider()
    decision = ActionBatch(
        run_id=observation.run_id,
        episode_id=observation.episode_id,
        step_id=0,
        decision_id="decision-0",
    )

    asyncio.run(
        ReflectionModule(provider).run(
            AgentContext(observation=observation, last_decision=decision)
        )
    )

    assert "matching execution" in provider.system_prompt
    assert "no_op never proves a plan action ran" in provider.system_prompt


def test_planner_keeps_only_compact_spatial_lines_from_upstream_text() -> None:
    base = make_observation()
    observation = base.model_copy(
        update={
            "state": base.state.model_copy(
                update={
                    "economy": base.state.economy.model_copy(
                        update={"supply_used": 11, "supply_cap": 15}
                    )
                }
            ),
            "available_actions": [
                *base.available_actions,
                AvailableAction(
                    name="Build_Pylon_Screen",
                    argument_names=["screen"],
                    argument_types=[ActionArgumentType.POSITION],
                    actor_scopes=["Builder/Builder-Probe-1"],
                ),
            ],
            "text_observation": "\n".join(
                [
                    "[Builder]",
                    "Team Builder-Probe-1 Info:",
                    "    Team minimap position: [14, 18]",
                    "    Unit: Probe Tag: 0x101 ScreenPos: [62, 70] Health: 20",
                    "Build_Pylon_Screen candidates: [[48, 56], [80, 72]]",
                    "Relevant Knowledge:",
                    "    verbose-knowledge-marker",
                    "Now, start generating your analysis and actions:",
                ]
            ),
        }
    )
    provider = CapturingProvider()

    asyncio.run(PlanningModule(provider).run(AgentContext(observation=observation)))

    prompt = json.loads(provider.user_prompt)
    assert prompt["observation"]["spatial_context"] == [
        "[Builder]",
        "Team Builder-Probe-1 Info:",
        "Team minimap position: [14, 18]",
        "Unit: Probe Tag: 0x101 ScreenPos: [62, 70] Health: 20",
        "Build_Pylon_Screen candidates: [[48, 56], [80, 72]]",
    ]
    assert "verbose-knowledge-marker" not in provider.user_prompt
    assert "prioritize legal economy and production" in provider.system_prompt


def test_planner_projects_completed_gateway_opening_to_first_zealot() -> None:
    observation = make_observation(include_enemy=False)
    observation = observation.model_copy(
        update={
            "state": observation.state.model_copy(
                update={
                    "economy": EconomyState(
                        minerals=170,
                        vespene=0,
                        supply_used=14,
                        supply_cap=23,
                        workers=14,
                        army_supply=0,
                    ),
                    "own_structures": [
                        UnitState(
                            unit_id="gateway-1",
                            unit_type="Gateway",
                            alliance="self",
                            status="idle",
                        )
                    ],
                }
            ),
            "available_actions": [
                AvailableAction(
                    name="Build_Pylon_Screen",
                    argument_names=["screen"],
                    argument_types=[ActionArgumentType.POSITION],
                    actor_scopes=["Builder/Builder-Probe-1"],
                ),
                AvailableAction(
                    name="Build_Gateway_Screen",
                    argument_names=["screen"],
                    argument_types=[ActionArgumentType.POSITION],
                    actor_scopes=["Builder/Builder-Probe-1"],
                ),
                AvailableAction(name="Train_Zealot", actor_scopes=["Developer/Empty"]),
                AvailableAction(name="Train_Stalker", actor_scopes=["Developer/Empty"]),
                AvailableAction(name="No_Operation", actor_scopes=["Developer/Empty"]),
            ],
            "text_observation": "\n".join(
                [
                    "Build_Pylon_Screen candidates: [[48, 56]]",
                    "Build_Gateway_Screen candidates: [[64, 64]]",
                ]
            ),
        }
    )
    provider = CapturingProvider()

    asyncio.run(PlanningModule(provider).run(AgentContext(observation=observation)))

    prompt = json.loads(provider.user_prompt)
    assert [action["name"] for action in prompt["observation"]["available_actions"]] == [
        "Train_Zealot",
        "No_Operation",
    ]
    assert "spatial_context" not in prompt["observation"]
    assert provider.response_type is not None
    provider.response_type.model_validate(
        {
            "strategic_goal": "Field the first combat unit",
            "steps": ["Train one Zealot"],
            "proposed_actions": [
                {
                    "actor": "Developer/Empty",
                    "name": "Train_Zealot",
                    "arguments": [],
                }
            ],
        }
    )
    with pytest.raises(ValidationError):
        provider.response_type.model_validate(
            {
                "strategic_goal": "Repeat production",
                "steps": ["Build another Gateway"],
                "proposed_actions": [
                    {
                        "actor": "Builder/Builder-Probe-1",
                        "name": "Build_Gateway_Screen",
                        "arguments": [[64, 64]],
                    }
                ],
            }
        )


def test_planner_exposes_pylon_only_at_tight_supply_without_pending_pylon() -> None:
    observation = make_observation(include_enemy=False)
    pylon = AvailableAction(
        name="Build_Pylon_Screen",
        argument_names=["screen"],
        argument_types=[ActionArgumentType.POSITION],
        actor_scopes=["Builder/Builder-Probe-1"],
    )
    observation = observation.model_copy(
        update={"available_actions": [pylon, AvailableAction(name="No_Operation")]}
    )

    high_supply_provider = CapturingProvider()
    asyncio.run(PlanningModule(high_supply_provider).run(AgentContext(observation=observation)))
    high_supply_prompt = json.loads(high_supply_provider.user_prompt)
    assert [
        action["name"] for action in high_supply_prompt["observation"]["available_actions"]
    ] == ["No_Operation"]

    tight_state = observation.state.model_copy(
        update={
            "economy": EconomyState(
                minerals=100,
                supply_used=11,
                supply_cap=15,
                army_supply=1,
            )
        }
    )
    tight_observation = observation.model_copy(update={"state": tight_state})
    tight_supply_provider = CapturingProvider()
    asyncio.run(
        PlanningModule(tight_supply_provider).run(AgentContext(observation=tight_observation))
    )
    tight_supply_prompt = json.loads(tight_supply_provider.user_prompt)
    assert [
        action["name"] for action in tight_supply_prompt["observation"]["available_actions"]
    ] == ["Build_Pylon_Screen", "No_Operation"]

    pending_state = tight_state.model_copy(
        update={
            "own_structures": [
                UnitState(
                    unit_id="pylon-1",
                    unit_type="Pylon",
                    alliance="self",
                    status="constructing",
                )
            ]
        }
    )
    pending_provider = CapturingProvider()
    asyncio.run(
        PlanningModule(pending_provider).run(
            AgentContext(observation=tight_observation.model_copy(update={"state": pending_state}))
        )
    )
    pending_prompt = json.loads(pending_provider.user_prompt)
    assert [action["name"] for action in pending_prompt["observation"]["available_actions"]] == [
        "No_Operation"
    ]


def test_planner_exposes_stalker_only_with_completed_core_and_gas() -> None:
    observation = make_observation(include_enemy=False)
    stalker = AvailableAction(name="Train_Stalker", actor_scopes=["Developer/Empty"])
    observation = observation.model_copy(
        update={
            "available_actions": [stalker, AvailableAction(name="No_Operation")],
            "state": observation.state.model_copy(
                update={
                    "economy": EconomyState(
                        minerals=200,
                        vespene=50,
                        supply_used=16,
                        supply_cap=23,
                        army_supply=2,
                    )
                }
            ),
        }
    )

    missing_core_provider = CapturingProvider()
    asyncio.run(PlanningModule(missing_core_provider).run(AgentContext(observation=observation)))
    missing_core_prompt = json.loads(missing_core_provider.user_prompt)
    assert [
        action["name"] for action in missing_core_prompt["observation"]["available_actions"]
    ] == ["No_Operation"]

    completed_core = UnitState(
        unit_id="core-1",
        unit_type="CyberneticsCore",
        alliance="self",
        status="idle",
    )
    ready_observation = observation.model_copy(
        update={"state": observation.state.model_copy(update={"own_structures": [completed_core]})}
    )
    ready_provider = CapturingProvider()
    asyncio.run(PlanningModule(ready_provider).run(AgentContext(observation=ready_observation)))
    ready_prompt = json.loads(ready_provider.user_prompt)
    assert [action["name"] for action in ready_prompt["observation"]["available_actions"]] == [
        "Train_Stalker",
        "No_Operation",
    ]

    low_gas_state = ready_observation.state.model_copy(
        update={"economy": ready_observation.state.economy.model_copy(update={"vespene": 49})}
    )
    low_gas_provider = CapturingProvider()
    asyncio.run(
        PlanningModule(low_gas_provider).run(
            AgentContext(observation=ready_observation.model_copy(update={"state": low_gas_state}))
        )
    )
    low_gas_prompt = json.loads(low_gas_provider.user_prompt)
    assert [action["name"] for action in low_gas_prompt["observation"]["available_actions"]] == [
        "No_Operation"
    ]


def test_memory_module_keeps_only_compact_decision_and_execution_events(
    tmp_path: Path,
) -> None:
    store = EventStore(tmp_path / "events.sqlite3", tmp_path / "events.jsonl")
    observation = make_observation()
    batch = ActionBatch(
        run_id=observation.run_id,
        episode_id=observation.episode_id,
        step_id=0,
        decision_id="decision-0",
        strategic_goal="Hold",
        summary="Wait",
    )
    store.append_event(
        run_id=observation.run_id,
        episode_id=observation.episode_id,
        step_id=0,
        event_type="observation",
        payload=observation.model_copy(update={"text_observation": "large " * 1000}),
    )
    store.append_event(
        run_id=observation.run_id,
        episode_id=observation.episode_id,
        step_id=0,
        event_type="module_result",
        payload={"module": "planning", "output": {"raw": "large " * 1000}},
    )
    store.append_event(
        run_id=observation.run_id,
        episode_id=observation.episode_id,
        step_id=0,
        event_type="decision",
        payload={"batch": batch.model_dump(mode="json")},
    )
    store.append_event(
        run_id=observation.run_id,
        episode_id=observation.episode_id,
        step_id=0,
        event_type="execution",
        payload=ExecutionReport(
            run_id=observation.run_id,
            episode_id=observation.episode_id,
            step_id=0,
            command_id="command-0",
            success=True,
            pysc2_function="no_op",
        ),
    )
    module = MemoryModule(store, short_term_window=20)

    result = asyncio.run(module.run(AgentContext(observation=observation)))
    store.close()

    assert [event["event_type"] for event in result.updates["recent_events"]] == [
        "decision",
        "execution",
    ]
    assert "large" not in json.dumps(result.updates)


def test_memory_module_preserves_recent_planner_errors(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events.sqlite3", tmp_path / "events.jsonl")
    observation = make_observation()
    store.append_event(
        run_id=observation.run_id,
        episode_id=observation.episode_id,
        step_id=7,
        event_type="planner_error",
        payload={"error_type": "ValueError", "message": "invalid action"},
    )

    result = asyncio.run(
        MemoryModule(store, short_term_window=20).run(AgentContext(observation=observation))
    )
    store.close()

    assert result.updates["recent_events"] == [
        {
            "event_type": "planner_error",
            "step_id": 7,
            "module": None,
            "error_type": "ValueError",
            "message": "invalid action",
        }
    ]


def test_planning_context_is_structurally_compacted_to_budget() -> None:
    observation = make_observation(include_enemy=False)
    decision = ActionBatch(
        run_id=observation.run_id,
        episode_id=observation.episode_id,
        step_id=19,
        decision_id="decision-19",
        strategic_goal="Build production",
        summary="Pylon then Gateway",
    )
    repeated_events = [
        {
            "event_type": "decision",
            "step_id": step_id,
            "strategic_goal": "Build production",
            "summary": "Pylon then Gateway",
            "commands": [],
            "rejected_commands": [],
        }
        for step_id in range(20)
    ]
    provider = CapturingProvider()
    module = PlanningModule(
        provider,
        ContextBudget(
            max_prompt_chars=4_000,
            max_recent_events=4,
            max_lessons=2,
            max_episode_summaries=1,
        ),
    )

    result = asyncio.run(
        module.run(
            AgentContext(
                observation=observation,
                last_decision=decision,
                memory={
                    "recent_events": repeated_events,
                    "reflection": "The opening remains legal.",
                    "lessons": [
                        {"source_step_id": 1, "content": "obsolete-lesson"},
                        {"source_step_id": 18, "content": "Build supply before cap."},
                        {"source_step_id": 19, "content": "Keep one Probe available."},
                    ],
                    "episode_summaries": [
                        {"episode_id": f"old-{index}", "summary": "old " * 500}
                        for index in range(3)
                    ],
                },
            )
        )
    )

    prompt = json.loads(provider.user_prompt)
    assert len(provider.system_prompt) + len(provider.user_prompt) <= 4_000
    assert prompt["context_compaction"]["final_chars"] == (
        len(provider.system_prompt) + len(provider.user_prompt)
    )
    assert (
        prompt["observation"]["available_actions"]
        == observation.model_dump(mode="json")["available_actions"]
    )
    assert prompt["active_plan"]["strategic_goal"] == "Build production"
    assert prompt["memory"]["reflection"] == "The opening remains legal."
    assert [lesson["content"] for lesson in prompt["memory"]["lessons"]] == [
        "Build supply before cap.",
        "Keep one Probe available.",
    ]
    assert len(prompt["memory"]["recent_events"]) == 1
    assert prompt["memory"]["recent_events"][0]["repeat_count"] == 20
    assert prompt["context_compaction"]["dropped_episode_summaries"] >= 2
    assert (
        prompt["context_compaction"]["original_chars"] > prompt["context_compaction"]["final_chars"]
    )
    assert "obsolete-lesson" not in provider.user_prompt
    assert result.updates["context_compaction"] == prompt["context_compaction"]
    assert "only the actions needed" in provider.system_prompt
    assert "Do not repeat" in provider.system_prompt


def test_reflection_context_obeys_budget_without_truncating_json() -> None:
    observation = make_observation(include_enemy=False)
    decision = ActionBatch(
        run_id=observation.run_id,
        episode_id=observation.episode_id,
        step_id=1,
        decision_id="decision-1",
        strategic_goal="Hold",
        summary="long-summary " * 500,
        rejected_commands=[f"rejected-{index}-" + "x" * 200 for index in range(20)],
    )
    provider = CapturingProvider()
    module = ReflectionModule(provider, ContextBudget(max_prompt_chars=2_500))

    asyncio.run(module.run(AgentContext(observation=observation, last_decision=decision)))

    prompt = json.loads(provider.user_prompt)
    assert len(provider.system_prompt) + len(provider.user_prompt) <= 2_500
    assert prompt["context_compaction"]["final_chars"] == (
        len(provider.system_prompt) + len(provider.user_prompt)
    )
    assert prompt["observation"]["available_actions"]
    assert prompt["last_decision"]["strategic_goal"] == "Hold"
    assert prompt["context_compaction"]["compacted"] is True


def test_planning_context_aggregates_hundreds_of_units() -> None:
    observation = make_observation(include_enemy=False)
    state = observation.state.model_copy(
        update={
            "own_units": [
                UnitState(
                    unit_id=f"probe-{index}",
                    unit_type="Probe",
                    alliance="self",
                    position=(float(index), 10.0),
                    health_fraction=0.1 if index == 199 else 1.0,
                )
                for index in range(200)
            ],
            "visible_enemies": [
                UnitState(
                    unit_id=f"zergling-{index}",
                    unit_type="Zergling",
                    alliance="enemy",
                    position=(80.0, float(index)),
                    health_fraction=0.2 if index == 149 else 1.0,
                )
                for index in range(150)
            ],
            "own_structures": [
                UnitState(
                    unit_id=f"pylon-{index}",
                    unit_type="Pylon",
                    alliance="self",
                    position=(20.0, float(index)),
                    health_fraction=0.3 if index == 99 else 1.0,
                    status="constructing" if index == 99 else None,
                )
                for index in range(100)
            ],
            "production_queue": [
                ProductionItem(name="Zealot", producer_id="gateway-1", progress=0.5)
            ],
        }
    )
    provider = CapturingProvider()
    module = PlanningModule(provider, ContextBudget(max_prompt_chars=6_000))

    asyncio.run(
        module.run(AgentContext(observation=observation.model_copy(update={"state": state})))
    )

    prompt = json.loads(provider.user_prompt)
    prompt_state = prompt["observation"]["state"]
    assert len(provider.system_prompt) + len(provider.user_prompt) <= 6_000
    assert prompt_state["own_unit_groups"] == [
        {
            "unit_type": "Probe",
            "count": 200,
            "min_health_fraction": 0.1,
            "average_health_fraction": 0.9955,
            "sample_positions": [[0.0, 10.0], [1.0, 10.0]],
        }
    ]
    assert prompt_state["visible_enemy_groups"][0]["count"] == 150
    assert prompt_state["own_structure_groups"][0]["count"] == 100
    assert "probe-199" in {unit["unit_id"] for unit in prompt_state["own_units"]}
    assert "zergling-149" in {unit["unit_id"] for unit in prompt_state["visible_enemies"]}
    assert len(prompt_state["own_units"]) <= 12
    assert len(prompt_state["visible_enemies"]) <= 16
    assert "pylon-99" in {structure["unit_id"] for structure in prompt_state["own_structures"]}
    assert len(prompt_state["own_structures"]) <= 16
    assert prompt_state["production_queue"] == [
        {"name": "Zealot", "producer_id": "gateway-1", "progress": 0.5}
    ]
    assert prompt_state["economy"] == state.model_dump(mode="json")["economy"]
    assert prompt["context_compaction"]["aggregated_own_units"] >= 188
    assert prompt["context_compaction"]["aggregated_own_structures"] >= 84
    assert prompt["context_compaction"]["aggregated_visible_enemies"] >= 134


def test_prompt_budget_fails_clearly_when_mandatory_schema_cannot_fit() -> None:
    observation = make_observation().model_copy(
        update={
            "available_actions": [
                action.model_copy(update={"name": "x" * 3_000})
                for action in make_observation().available_actions
            ]
        }
    )

    with pytest.raises(ContextBudgetExceeded, match="mandatory observation and action schema"):
        asyncio.run(
            PlanningModule(CapturingProvider(), ContextBudget(max_prompt_chars=2_000)).run(
                AgentContext(observation=observation)
            )
        )


def test_planning_output_limits_candidate_count() -> None:
    with pytest.raises(ValidationError):
        PlanningOutput(
            strategic_goal="Attack",
            proposed_actions=[
                ActionProposal(actor=f"army-{index}", name="Attack_Unit") for index in range(4)
            ],
        )


def test_action_proposal_rejects_string_coordinates() -> None:
    with pytest.raises(ValidationError):
        ActionProposal.model_validate(
            {
                "actor": "Builder/Builder-Probe-1",
                "name": "Build_Pylon_Screen",
                "arguments": [["60", "40"]],
            }
        )


def test_planning_output_model_restricts_names_and_multiple_available_actors() -> None:
    base = make_observation()
    observation = base.model_copy(
        update={
            "state": base.state.model_copy(
                update={
                    "economy": base.state.economy.model_copy(
                        update={"supply_used": 11, "supply_cap": 15}
                    )
                }
            ),
            "available_actions": [
                AvailableAction(
                    name="Build_Pylon_Screen",
                    argument_names=["screen"],
                    argument_types=[ActionArgumentType.POSITION],
                    actor_scopes=[
                        "Builder/Builder-Probe-1",
                        "Builder/Builder-Probe-2",
                    ],
                )
            ],
        }
    )
    output_type = planning_output_model(observation)
    valid_payload: dict[str, Any] = {
        "strategic_goal": "Build supply",
        "steps": ["Build one Pylon"],
        "proposed_actions": [
            {
                "actor": "Builder/Builder-Probe-2",
                "name": "Build_Pylon_Screen",
                "arguments": [[60, 40]],
            }
        ],
    }

    output = output_type.model_validate(valid_payload)

    assert output.model_dump(mode="json")["proposed_actions"][0]["arguments"] == [[60, 40]]
    for field, invalid_value in (
        ("name", "Build_Pylon"),
        ("actor", "Builder/Builder-Probe-3"),
    ):
        invalid_payload = {
            **valid_payload,
            "proposed_actions": [{**valid_payload["proposed_actions"][0], field: invalid_value}],
        }
        with pytest.raises(ValidationError):
            output_type.model_validate(invalid_payload)


def test_planning_output_model_binds_each_action_to_its_actor_and_arguments() -> None:
    base = make_observation()
    observation = base.model_copy(
        update={
            "state": base.state.model_copy(
                update={
                    "economy": base.state.economy.model_copy(
                        update={"supply_used": 11, "supply_cap": 15}
                    )
                }
            ),
            "available_actions": [
                AvailableAction(
                    name="Build_Pylon_Screen",
                    argument_names=["screen"],
                    argument_types=[ActionArgumentType.POSITION],
                    actor_scopes=["Builder/Builder-Probe-1"],
                ),
                AvailableAction(
                    name="Train_Zealot",
                    actor_scopes=["Developer/Empty"],
                ),
            ],
        }
    )
    output_type = planning_output_model(observation)
    base_payload = {"strategic_goal": "Continue the opening", "steps": []}

    for proposal in (
        {
            "actor": "Builder/Builder-Probe-1",
            "name": "Build_Pylon_Screen",
            "arguments": [[60, 40]],
        },
        {
            "actor": "Developer/Empty",
            "name": "Train_Zealot",
            "arguments": [],
        },
    ):
        output_type.model_validate({**base_payload, "proposed_actions": [proposal]})

    for proposal in (
        {
            "actor": "Builder/Builder-Probe-1",
            "name": "Train_Zealot",
            "arguments": [],
        },
        {
            "actor": "Developer/Empty",
            "name": "Build_Pylon_Screen",
            "arguments": [[60, 40]],
        },
        {
            "actor": "Builder/Builder-Probe-1",
            "name": "Build_Pylon_Screen",
            "arguments": [],
        },
        {
            "actor": "Builder/Builder-Probe-1",
            "name": "Build_Pylon_Screen",
            "arguments": [["60", "40"]],
        },
        {
            "actor": "Developer/Empty",
            "name": "Train_Zealot",
            "arguments": [1],
        },
        {
            "actor": "Developer/Empty",
            "name": "Train_Zealot",
        },
    ):
        with pytest.raises(ValidationError):
            output_type.model_validate({**base_payload, "proposed_actions": [proposal]})


def test_planning_output_model_uses_strict_declared_argument_types() -> None:
    action = AvailableAction(
        name="Typed_Action",
        argument_names=[
            "text",
            "count",
            "ratio",
            "enabled",
            "screen",
            "tag",
            "value",
        ],
        argument_types=[
            ActionArgumentType.STRING,
            ActionArgumentType.INTEGER,
            ActionArgumentType.NUMBER,
            ActionArgumentType.BOOLEAN,
            ActionArgumentType.POSITION,
            ActionArgumentType.TAG,
            ActionArgumentType.ANY,
        ],
        actor_scopes=["Typed/Actor"],
    )
    observation = make_observation().model_copy(update={"available_actions": [action]})
    output_type = planning_output_model(observation)
    valid_arguments: list[Any] = ["value", 2, 0.5, True, [60, 40], "0x10", "free"]
    payload = {
        "strategic_goal": "Exercise the action schema",
        "steps": [],
        "proposed_actions": [
            {
                "actor": "Typed/Actor",
                "name": "Typed_Action",
                "arguments": valid_arguments,
            }
        ],
    }

    output_type.model_validate(payload)
    output_type.model_validate(
        {
            **payload,
            "proposed_actions": [
                {**payload["proposed_actions"][0], "arguments": [*valid_arguments[:5], 16, "free"]}
            ],
        }
    )

    for index, invalid_value in enumerate((1, "2", "0.5", 1, ["60", 40], 0.5)):
        invalid_arguments = valid_arguments.copy()
        invalid_arguments[index] = invalid_value
        with pytest.raises(ValidationError):
            output_type.model_validate(
                {
                    **payload,
                    "proposed_actions": [
                        {
                            **payload["proposed_actions"][0],
                            "arguments": invalid_arguments,
                        }
                    ],
                }
            )

    for invalid_arguments in (valid_arguments[:-1], [*valid_arguments, "extra"]):
        with pytest.raises(ValidationError):
            output_type.model_validate(
                {
                    **payload,
                    "proposed_actions": [
                        {
                            **payload["proposed_actions"][0],
                            "arguments": invalid_arguments,
                        }
                    ],
                }
            )


def test_planning_output_model_disallows_proposals_without_available_actions() -> None:
    observation = make_observation().model_copy(update={"available_actions": []})
    output_type = planning_output_model(observation)
    empty_payload = {
        "strategic_goal": "Wait",
        "steps": [],
        "proposed_actions": [],
    }

    output_type.model_validate(empty_payload)

    with pytest.raises(ValidationError):
        output_type.model_validate(
            {
                **empty_payload,
                "proposed_actions": [
                    {
                        "actor": "Builder/Builder-Probe-1",
                        "name": "Build_Pylon_Screen",
                        "arguments": [[60, 40]],
                    }
                ],
            }
        )


def test_action_module_requires_attack_target_to_still_exist() -> None:
    plan = PlanningOutput(
        strategic_goal="Attack",
        proposed_actions=[
            ActionProposal(
                actor="army",
                name="Attack_Unit",
                arguments=["enemy-1"],
            )
        ],
    )

    result = asyncio.run(
        ActionModule(max_actions=3).run(
            AgentContext(
                observation=make_observation(),
                memory={"plan": plan.model_dump(mode="json")},
            )
        )
    )

    assert result.commands[0].preconditions == {"unit_exists": "enemy-1"}


def test_action_module_guards_planner_pylon_by_supply_and_pending_construction() -> None:
    plan = PlanningOutput(
        strategic_goal="Build supply",
        proposed_actions=[
            ActionProposal(
                actor="Builder/Builder-Probe-1",
                name="Build_Pylon_Screen",
                arguments=[[60, 40]],
            )
        ],
    )

    result = asyncio.run(
        ActionModule(max_actions=3).run(
            AgentContext(
                observation=make_observation(),
                memory={"plan": plan.model_dump(mode="json")},
            )
        )
    )

    assert result.commands[0].preconditions == {
        "max_supply_free": 4,
        "no_pending_structure": "Pylon",
    }


def test_action_module_guards_planner_gateway_when_structure_exists() -> None:
    plan = PlanningOutput(
        strategic_goal="Add production",
        proposed_actions=[
            ActionProposal(
                actor="Builder/Builder-Probe-1",
                name="Build_Gateway_Screen",
                arguments=[[60, 40]],
            )
        ],
    )

    result = asyncio.run(
        ActionModule(max_actions=3).run(
            AgentContext(
                observation=make_observation(),
                memory={"plan": plan.model_dump(mode="json")},
            )
        )
    )

    assert result.commands[0].preconditions == {"structure_absent": "Gateway"}


def test_action_module_preserves_candidates_when_actor_is_duplicated() -> None:
    plan = PlanningOutput(
        strategic_goal="Produce army",
        proposed_actions=[
            ActionProposal(actor="Developer/Empty", name="Train_Zealot"),
            ActionProposal(actor="Developer/Empty", name="Research_WarpGate"),
        ],
    )

    result = asyncio.run(
        ActionModule(max_actions=3).run(
            AgentContext(
                observation=make_observation(),
                memory={"plan": plan.model_dump(mode="json")},
            )
        )
    )

    assert [command.name for command in result.commands] == [
        "Train_Zealot",
        "Research_WarpGate",
    ]
