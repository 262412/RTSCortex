from __future__ import annotations

import asyncio
import json
from pathlib import Path

from rtscortex.agents import ActionProposal, PlanningOutput
from rtscortex.contracts import AvailableAction, EpisodeOutcome, EpisodeResult, ExecutionReport
from rtscortex.contracts.interfaces import ResponseT
from rtscortex.evaluation import run_mock_episode
from rtscortex.memory import EventStore
from rtscortex.providers import FakeProvider
from rtscortex.runtime import RuntimeEngine
from tests.helpers import make_config, make_observation


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


class UnexpectedProvider:
    async def generate(
        self,
        response_type: type[ResponseT],
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> ResponseT:
        del response_type, system_prompt, user_prompt
        raise AssertionError("a recovered plan should avoid a new model call")


class ChangingPlanProvider:
    def __init__(self) -> None:
        self.calls = 0

    async def generate(
        self,
        response_type: type[ResponseT],
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> ResponseT:
        del system_prompt, user_prompt
        self.calls += 1
        output = PlanningOutput(
            strategic_goal=f"Goal {self.calls}",
            steps=[f"Plan revision {self.calls}"],
            proposed_actions=[
                ActionProposal(
                    actor="army",
                    name="Attack_Unit",
                    arguments=["enemy-1"],
                    priority=60,
                )
            ],
        )
        return response_type.model_validate(output.model_dump())


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
                (event.payload["module"], event.payload["model_call"])
                for event in store.recent_events("integration-run", "episode-0", 50)
                if event.event_type == "module_result" and event.step_id == 0
            ]
            assert module_events == [
                ("memory", False),
                ("reflection", False),
                ("planning", True),
                ("action", False),
            ]
            planning_event = next(
                event
                for event in store.recent_events("integration-run", "episode-0", 50)
                if event.event_type == "module_result" and event.payload["module"] == "planning"
            )
            assert planning_event.payload["output"]["plan"]["strategic_goal"] == (
                "Remove the visible threat"
            )
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
        try:
            first = await runtime.tick(make_observation(step_id=0, game_loop=0))
            assert first.commands[0].name == "No_Operation"
            assert first.planner_pending is True
            await asyncio.sleep(0.02)
            await runtime.tick(make_observation(step_id=1, game_loop=1))
            timeout = store.last_event("run-1", "episode-1", "planner_timeout")
            assert timeout is not None
        finally:
            await runtime.close()

    asyncio.run(execute())


def test_background_plan_ttl_starts_when_plan_is_accepted(tmp_path: Path) -> None:
    config = make_config(
        tmp_path,
        variant="planner_only",
        deterministic=False,
        planning_interval_game_loops=1000,
    )
    store = EventStore(tmp_path / "events.sqlite3", tmp_path / "events.jsonl")
    runtime = RuntimeEngine(config=config, store=store, provider=FakeProvider())

    async def execute() -> None:
        try:
            first = await runtime.tick(make_observation(step_id=0, game_loop=0))
            assert first.planner_pending is True
            await asyncio.sleep(0.01)

            second = await runtime.tick(make_observation(step_id=1, game_loop=100))

            assert second.commands[0].source.value == "planner"
            assert second.commands[0].created_game_loop == 100
            assert second.planner_pending is False
            plan = store.last_event("run-1", "episode-1", "plan_accepted")
            assert plan is not None
            assert plan.payload["source_step_id"] == 0
            assert plan.payload["created_game_loop"] == 100
            assert plan.payload["source_game_loop"] == 0
            assert plan.payload["accepted_game_loop"] == 100
            assert plan.payload["plan_age_game_loops"] == 100
        finally:
            await runtime.close()

    asyncio.run(execute())


def test_background_plan_rejects_a_target_that_disappeared_while_planning(
    tmp_path: Path,
) -> None:
    config = make_config(
        tmp_path,
        variant="planner_only",
        deterministic=False,
        planning_interval_game_loops=1000,
    )
    store = EventStore(tmp_path / "events.sqlite3", tmp_path / "events.jsonl")
    runtime = RuntimeEngine(config=config, store=store, provider=FakeProvider())

    async def execute() -> None:
        try:
            await runtime.tick(make_observation(step_id=0, game_loop=0))
            await asyncio.sleep(0.01)

            batch = await runtime.tick(
                make_observation(step_id=1, game_loop=100, include_enemy=False)
            )

            assert batch.commands[0].source.value == "fallback"
            assert any("unit_exists" in reason for reason in batch.rejected_commands)
        finally:
            await runtime.close()

    asyncio.run(execute())


def test_runtime_restart_recovers_plan_execution_and_episode_summary(tmp_path: Path) -> None:
    config = make_config(tmp_path, variant="planner_only")
    database = tmp_path / "events.sqlite3"
    journal = tmp_path / "events.jsonl"

    async def execute() -> None:
        first_runtime = RuntimeEngine(
            config=config,
            store=EventStore(database, journal),
            provider=FakeProvider(),
        )
        first = await first_runtime.tick(make_observation(step_id=0, game_loop=0))
        first_runtime.record_execution(
            ExecutionReport(
                run_id="run-1",
                episode_id="episode-1",
                step_id=0,
                command_id=first.commands[0].command_id,
                success=True,
            )
        )
        await first_runtime.close()

        recovered_store = EventStore(database, journal)
        recovered_runtime = RuntimeEngine(
            config=config,
            store=recovered_store,
            provider=UnexpectedProvider(),
        )
        second = await recovered_runtime.tick(make_observation(step_id=1, game_loop=1))
        assert second.commands[0] == first.commands[0]
        result = EpisodeResult(
            run_id="run-1",
            episode_id="episode-1",
            scenario="pvz_task1_level1",
            seed=0,
            outcome=EpisodeOutcome.VICTORY,
            score=100,
            steps=2,
        )
        recovered_runtime.end_episode(result)
        summary = recovered_store.episode_summary("run-1", "episode-1")
        assert summary is not None
        assert summary.outcome is EpisodeOutcome.VICTORY
        await recovered_runtime.close()

    asyncio.run(execute())


def test_runtime_records_reflex_preemption(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    store = EventStore(tmp_path / "events.sqlite3", tmp_path / "events.jsonl")
    runtime = RuntimeEngine(config=config, store=store, provider=FakeProvider())

    async def execute() -> None:
        try:
            batch = await runtime.tick(
                make_observation(step_id=0, game_loop=0, alerts=["under_attack"])
            )
            assert batch.commands[0].source.value == "reflex"
            decision = store.last_event("run-1", "episode-1", "decision")
            assert decision is not None
            assert decision.payload["preemptions"] == [
                {
                    "actor": "army",
                    "winner_command_id": "run-1:episode-1:0:reflex:0",
                    "loser_command_id": "run-1:episode-1:0:planner:0",
                }
            ]
        finally:
            await runtime.close()

    asyncio.run(execute())


def test_runtime_marks_semantic_plan_revisions(tmp_path: Path) -> None:
    config = make_config(
        tmp_path,
        variant="planner_only",
        planning_interval_game_loops=1,
    )
    store = EventStore(tmp_path / "events.sqlite3", tmp_path / "events.jsonl")
    runtime = RuntimeEngine(config=config, store=store, provider=ChangingPlanProvider())

    async def execute() -> None:
        try:
            await runtime.tick(make_observation(step_id=0, game_loop=0))
            await runtime.tick(make_observation(step_id=1, game_loop=1))
            revisions = [
                event.payload["is_revision"]
                for event in store.recent_events("run-1", "episode-1", 100)
                if event.event_type == "plan_accepted"
            ]
            assert revisions == [False, True]
        finally:
            await runtime.close()

    asyncio.run(execute())


def test_fallback_uses_a_routable_actor_scope(tmp_path: Path) -> None:
    config = make_config(tmp_path, variant="noop")
    store = EventStore(tmp_path / "events.sqlite3", tmp_path / "events.jsonl")
    runtime = RuntimeEngine(config=config, store=store, provider=UnexpectedProvider())
    observation = make_observation().model_copy(
        update={
            "available_actions": [
                AvailableAction(
                    name="No_Operation",
                    actor_scopes=["CombatGroup7/Adept-1"],
                )
            ]
        }
    )

    async def execute() -> None:
        try:
            batch = await runtime.tick(observation)
            assert batch.commands[0].actor == "CombatGroup7/Adept-1"
        finally:
            await runtime.close()

    asyncio.run(execute())
