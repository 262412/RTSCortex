from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import NoReturn

import pytest
from pydantic import BaseModel, ValidationError

from rtscortex.agents import (
    ActionModule,
    MemoryModule,
    PlanningModule,
    PlanningOutput,
    ReflectionModule,
)
from rtscortex.agents.models import ActionProposal, ReflectionOutput
from rtscortex.contracts import ActionBatch, ExecutionReport
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

    async def generate(
        self,
        response_type: type[ResponseT],
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> ResponseT:
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        output: BaseModel
        if response_type is PlanningOutput:
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


def test_planning_output_limits_candidate_count() -> None:
    with pytest.raises(ValidationError):
        PlanningOutput(
            strategic_goal="Attack",
            proposed_actions=[
                ActionProposal(actor=f"army-{index}", name="Attack_Unit") for index in range(4)
            ],
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
