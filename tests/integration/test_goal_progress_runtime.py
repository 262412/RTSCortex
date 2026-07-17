from __future__ import annotations

import asyncio
import json
from pathlib import Path

from rtscortex.agents.models import PlanningOutput
from rtscortex.contracts import (
    ActionArgumentType,
    AvailableAction,
    EconomyState,
    ObservationEnvelope,
    UnitState,
)
from rtscortex.contracts.interfaces import ResponseT
from rtscortex.memory import EventStore
from rtscortex.runtime.engine import RuntimeEngine
from tests.helpers import make_config, make_observation


class GatewayPlanningProvider:
    def __init__(self) -> None:
        self.schemas: list[str] = []
        self.last_usage: dict[str, int] | None = None

    async def generate(
        self,
        response_type: type[ResponseT],
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> ResponseT:
        del system_prompt, user_prompt
        assert issubclass(response_type, PlanningOutput)
        self.schemas.append(json.dumps(response_type.model_json_schema()))
        if len(self.schemas) > 1:
            return response_type.model_validate(
                {
                    "strategic_goal": "Wait for the current Gateway objective",
                    "steps": ["Do not issue a duplicate build"],
                    "proposed_actions": [],
                }
            )
        return response_type.model_validate(
            {
                "strategic_goal": "Build the first Gateway",
                "steps": ["Build the legal Gateway now"],
                "proposed_actions": [
                    {
                        "actor": "Builder/Probe-1",
                        "name": "Build_Gateway_Screen",
                        "arguments": [[60, 60]],
                        "priority": 60,
                    }
                ],
            }
        )


def _gateway_observation(
    *,
    step_id: int = 0,
    game_loop: int = 100,
) -> ObservationEnvelope:
    base = make_observation(
        step_id=step_id,
        game_loop=game_loop,
        include_enemy=False,
    )
    return base.model_copy(
        update={
            "state": base.state.model_copy(
                update={
                    "economy": EconomyState(
                        minerals=200,
                        supply_used=12,
                        supply_cap=15,
                        workers=12,
                    ),
                    "own_structures": [
                        UnitState(
                            unit_id="pylon-1",
                            unit_type="Pylon",
                            alliance="self",
                            status="idle",
                        )
                    ],
                }
            ),
            "available_actions": [
                AvailableAction(
                    name="Stop",
                    actor_scopes=["Builder/Probe-1"],
                ),
                AvailableAction(
                    name="Hold_Position",
                    actor_scopes=["Builder/Probe-1"],
                ),
                AvailableAction(
                    name="Build_Gateway_Screen",
                    argument_names=["screen"],
                    argument_types=[ActionArgumentType.POSITION],
                    actor_scopes=["Builder/Probe-1"],
                    argument_candidates=[[[60, 60]]],
                ),
            ],
        }
    )


def test_runtime_feeds_progress_to_planner_and_console_events(tmp_path: Path) -> None:
    provider = GatewayPlanningProvider()
    store = EventStore(tmp_path / "events.sqlite3", tmp_path / "events.jsonl")
    runtime = RuntimeEngine(
        config=make_config(tmp_path, planning_interval_game_loops=1_000),
        store=store,
        provider=provider,
    )
    first = _gateway_observation()

    try:
        batch = asyncio.run(runtime.tick(first))

        assert [command.name for command in batch.commands] == ["Build_Gateway_Screen"]
        assert len(provider.schemas) == 1
        assert '"Stop"' in provider.schemas[0]
        assert '"Hold_Position"' in provider.schemas[0]

        progress_events = store.events_of_type(
            first.run_id,
            first.episode_id,
            "goal_progress",
        )
        assert progress_events
        latest_progress = progress_events[-1].payload
        assert latest_progress["strategic_goal"] == "Build the first Gateway"
        assert latest_progress["advancing_actions"] == ["Build_Gateway_Screen"]
        assert latest_progress["unique_next_action"] == "Build_Gateway_Screen"

        before = len(progress_events)
        asyncio.run(runtime.tick(_gateway_observation(step_id=1, game_loop=101)))

        assert len(provider.schemas) == 2
        assert '"Stop"' not in provider.schemas[1]
        assert '"Hold_Position"' not in provider.schemas[1]
        planning_events = [
            event
            for event in store.events_of_type(
                first.run_id,
                first.episode_id,
                "module_result",
            )
            if event.payload.get("module") == "planning"
        ]
        assert planning_events[-1].payload["output"]["goal_progress"] is not None
        assert runtime._cached_plan is not None
        assert runtime._cached_plan.goal_spec is not None
        assert len(store.events_of_type(first.run_id, first.episode_id, "goal_progress")) == before
    finally:
        asyncio.run(runtime.close())
