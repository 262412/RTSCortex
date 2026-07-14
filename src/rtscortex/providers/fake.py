"""Deterministic provider used by tests and offline development."""

from __future__ import annotations

import json

from pydantic import BaseModel

from rtscortex.agents.models import ActionProposal, PlanningOutput, ReflectionOutput
from rtscortex.contracts.interfaces import ResponseT


class FakeProvider:
    """Generate predictable typed outputs without network or model calls."""

    last_usage: dict[str, int] | None = None

    async def generate(
        self,
        response_type: type[ResponseT],
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> ResponseT:
        del system_prompt
        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        payload = json.loads(user_prompt)
        if response_type is ReflectionOutput:
            execution = payload.get("last_execution")
            success = execution is None or execution.get("success", False)
            output: BaseModel = ReflectionOutput(
                summary="Previous action succeeded." if success else "Previous action failed.",
                lessons=[] if success else ["Revalidate action availability before execution."],
            )
        elif issubclass(response_type, PlanningOutput):
            observation = payload["observation"]
            available = {item["name"]: item for item in observation["available_actions"]}
            enemies = observation["state"]["visible_enemies"]
            if enemies and "Attack_Unit" in available:
                actor_scopes = available["Attack_Unit"].get("actor_scopes", [])
                proposal = ActionProposal(
                    actor=actor_scopes[0] if actor_scopes else "army",
                    name="Attack_Unit",
                    arguments=[enemies[0]["unit_id"]],
                    priority=60,
                )
                output = PlanningOutput(
                    strategic_goal="Remove the visible threat",
                    steps=["Focus fire the nearest visible enemy"],
                    proposed_actions=[proposal],
                )
            else:
                output = PlanningOutput(
                    strategic_goal="Maintain a safe state",
                    steps=["Wait for actionable information"],
                    proposed_actions=[],
                )
        else:
            raise TypeError(f"FakeProvider does not support {response_type.__name__}")
        return response_type.model_validate(output.model_dump())
