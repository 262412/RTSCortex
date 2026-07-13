from __future__ import annotations

import asyncio
import json
from pathlib import Path

from rtscortex.agents import ActionProposal, PlanningOutput
from rtscortex.contracts.interfaces import ResponseT
from rtscortex.evaluation import run_mock_episode
from rtscortex.memory import EventStore
from rtscortex.providers import FakeProvider
from rtscortex.runtime import RuntimeEngine
from tests.helpers import make_config


class SlowSecondPlanProvider:
    calls = 0

    async def generate(
        self,
        response_type: type[ResponseT],
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> ResponseT:
        del system_prompt
        self.calls += 1
        if self.calls == 2:
            await asyncio.sleep(1)
        observation = json.loads(user_prompt)["observation"]
        output = PlanningOutput(
            strategic_goal="Attack once",
            steps=["Reuse this plan if the next call times out"],
            proposed_actions=[
                ActionProposal(
                    actor="army",
                    name="Attack_Unit",
                    arguments=[observation["state"]["visible_enemies"][0]["unit_id"]],
                )
            ],
        )
        return response_type.model_validate(output.model_dump())


class AlwaysSlowProvider:
    async def generate(
        self,
        response_type: type[ResponseT],
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> ResponseT:
        del response_type, system_prompt, user_prompt
        await asyncio.sleep(1)
        raise AssertionError("planner timeout should cancel this call")


def test_mock_episode_runs_full_module_chain(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    store = EventStore(tmp_path / "events.sqlite3", tmp_path / "events.jsonl")
    runtime = RuntimeEngine(config=config, store=store, provider=FakeProvider())

    async def execute() -> None:
        try:
            result = await run_mock_episode(
                config=config,
                runtime=runtime,
                run_id="integration-run",
                episode_id="episode-0",
                seed=0,
            )
            assert result.score == 100.0
            assert result.metrics["action_success_rate"] == 1.0
            decision = store.last_event("integration-run", "episode-0", "decision")
            assert decision is not None
            module_events = [
                event.payload["module"]
                for event in store.recent_events("integration-run", "episode-0", 50)
                if event.event_type == "module_result" and event.step_id == 0
            ]
            assert module_events == ["memory", "reflection", "planning", "action"]
        finally:
            await runtime.close()

    asyncio.run(execute())


def test_planner_timeout_reuses_previous_valid_plan(tmp_path: Path) -> None:
    config = make_config(
        tmp_path,
        variant="planner_only",
        planner_timeout_seconds=0.01,
    )
    config.runtime.planning_interval_game_loops = 1
    store = EventStore(tmp_path / "events.sqlite3", tmp_path / "events.jsonl")
    runtime = RuntimeEngine(config=config, store=store, provider=SlowSecondPlanProvider())

    async def execute() -> None:
        from tests.helpers import make_observation

        try:
            first = await runtime.tick(make_observation(step_id=0, game_loop=0))
            second = await runtime.tick(make_observation(step_id=1, game_loop=1))
            assert first.commands[0].command_id.endswith(":0:planner:0")
            assert second.commands[0] == first.commands[0]
            timeout = store.last_event("run-1", "episode-1", "planner_timeout")
            assert timeout is not None
        finally:
            await runtime.close()

    asyncio.run(execute())


def test_background_planner_is_timeout_bounded(tmp_path: Path) -> None:
    config = make_config(
        tmp_path,
        variant="planner_only",
        deterministic=False,
        planner_timeout_seconds=0.01,
    )
    store = EventStore(tmp_path / "events.sqlite3", tmp_path / "events.jsonl")
    runtime = RuntimeEngine(config=config, store=store, provider=AlwaysSlowProvider())

    async def execute() -> None:
        from tests.helpers import make_observation

        try:
            first = await runtime.tick(make_observation(step_id=0, game_loop=0))
            assert first.commands[0].name == "No_Operation"
            await asyncio.sleep(0.02)
            await runtime.tick(make_observation(step_id=1, game_loop=1))
            timeout = store.last_event("run-1", "episode-1", "planner_timeout")
            assert timeout is not None
        finally:
            await runtime.close()

    asyncio.run(execute())
