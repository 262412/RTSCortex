from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from rtscortex.agents import ActionProposal, PlanningOutput, ReflectionOutput
from rtscortex.contracts import (
    ActionArgumentType,
    ActionCommand,
    ActionSource,
    AvailableAction,
    EconomyState,
    EpisodeOutcome,
    EpisodeResult,
    ExecutionReport,
    ExecutionStage,
    ExecutionStatus,
    IdleReason,
)
from rtscortex.contracts.interfaces import ResponseT
from rtscortex.evaluation import run_mock_episode
from rtscortex.memory import EventStore
from rtscortex.providers import FakeProvider
from rtscortex.runtime import RuntimeEngine
from rtscortex.runtime.engine import CommandStatus, PlanState
from rtscortex.runtime.validation import ValidationDisposition, ValidationFailure
from tests.helpers import make_config, make_observation


def test_console_setting_does_not_change_deterministic_action_batch(tmp_path: Path) -> None:
    async def decide(*, console_enabled: bool, suffix: str) -> dict[str, Any]:
        config = make_config(tmp_path / suffix)
        config = config.model_copy(
            update={"console": config.console.model_copy(update={"enabled": console_enabled})}
        )
        runtime = RuntimeEngine(
            config=config,
            store=EventStore(
                tmp_path / suffix / "events.sqlite3",
                tmp_path / suffix / "events.jsonl",
            ),
            provider=FakeProvider(),
        )
        try:
            batch = await runtime.tick(make_observation())
            return batch.model_dump(mode="json")
        finally:
            await runtime.close()

    disabled = asyncio.run(decide(console_enabled=False, suffix="disabled"))
    enabled = asyncio.run(decide(console_enabled=True, suffix="enabled"))

    assert enabled == disabled


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


class ControlledBackgroundProvider:
    def __init__(self) -> None:
        self.calls = 0
        self.release_first = asyncio.Event()
        self.release_second = asyncio.Event()

    async def generate(
        self,
        response_type: type[ResponseT],
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> ResponseT:
        del system_prompt
        self.calls += 1
        if self.calls == 1:
            await self.release_first.wait()
        else:
            await self.release_second.wait()
        observation = json.loads(user_prompt)["observation"]
        output = PlanningOutput(
            strategic_goal="Defend",
            proposed_actions=[
                ActionProposal(
                    actor="army",
                    name="Attack_Unit",
                    arguments=[observation["state"]["visible_enemies"][0]["unit_id"]],
                )
            ],
        )
        return response_type.model_validate(output.model_dump())


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
                    arguments=["0x1"],
                    priority=60,
                )
            ],
        )
        return response_type.model_validate(output.model_dump())


class RepeatingPlanCapturingProvider:
    def __init__(self) -> None:
        self.prompts: list[dict[str, Any]] = []

    async def generate(
        self,
        response_type: type[ResponseT],
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> ResponseT:
        del system_prompt
        prompt = json.loads(user_prompt)
        self.prompts.append(prompt)
        enemy_id = prompt["observation"]["state"]["visible_enemies"][0]["unit_id"]
        output = PlanningOutput(
            strategic_goal="Keep pressure on the visible threat",
            proposed_actions=[
                ActionProposal(
                    actor="army",
                    name="Attack_Unit",
                    arguments=[enemy_id],
                    priority=60,
                )
            ],
        )
        return response_type.model_validate(output.model_dump())


class InvalidAttackBeforeBuildProvider:
    async def generate(
        self,
        response_type: type[ResponseT],
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> ResponseT:
        del system_prompt, user_prompt
        output = PlanningOutput(
            strategic_goal="Build the first Pylon",
            steps=["Build a Pylon at a validated candidate"],
            proposed_actions=[
                ActionProposal(
                    actor="Builder/Builder-Probe-1",
                    name="Build_Pylon_Screen",
                    arguments=[[60, 40]],
                    priority=50,
                ),
            ],
        )
        return response_type.model_validate(output.model_dump())


class EmptyPlanProvider:
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
            strategic_goal="Wait for a legal action",
            proposed_actions=[],
        )
        return response_type.model_validate(output.model_dump())


class TwoCommandsForOneActorProvider:
    async def generate(
        self,
        response_type: type[ResponseT],
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> ResponseT:
        del system_prompt
        enemy_id = json.loads(user_prompt)["observation"]["state"]["visible_enemies"][0]["unit_id"]
        output = PlanningOutput(
            strategic_goal="Execute one army command at a time",
            proposed_actions=[
                ActionProposal(
                    actor="army",
                    name="Attack_Unit",
                    arguments=[enemy_id],
                    priority=60,
                ),
                ActionProposal(
                    actor="army",
                    name="Attack_Unit",
                    arguments=[enemy_id],
                    priority=50,
                ),
            ],
        )
        return response_type.model_validate(output.model_dump())


class CountingFakeProvider(FakeProvider):
    def __init__(self) -> None:
        self.calls = 0

    async def generate(
        self,
        response_type: type[ResponseT],
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> ResponseT:
        self.calls += 1
        return await super().generate(
            response_type,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )


class PromptCapturingFakeProvider(FakeProvider):
    def __init__(self) -> None:
        self.reflection_prompts: list[dict[str, Any]] = []

    async def generate(
        self,
        response_type: type[ResponseT],
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> ResponseT:
        if response_type is ReflectionOutput:
            self.reflection_prompts.append(json.loads(user_prompt))
        return await super().generate(
            response_type,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )


def successful_execution(command: ActionCommand, *, step_id: int = 0) -> ExecutionReport:
    return ExecutionReport(
        run_id="run-1",
        episode_id="episode-1",
        step_id=step_id,
        command_id=command.command_id,
        success=True,
        action_name=command.name,
        actor=command.actor,
        source=command.source,
        requested_arguments=command.arguments,
        resolved_arguments=command.arguments,
        status=ExecutionStatus.SUCCEEDED,
        execution_stage=ExecutionStage.PYSC2_ACCEPTANCE,
    )


def episode_result(*, failure_reason: str | None = None) -> EpisodeResult:
    return EpisodeResult(
        run_id="run-1",
        episode_id="episode-1",
        scenario="Simple64",
        seed=0,
        outcome=EpisodeOutcome.TRUNCATED,
        steps=1,
        failure_reason=failure_reason,
    )


def append_plan(
    store: EventStore,
    commands: list[ActionCommand],
    *,
    lifecycle_protocol: str | None = None,
) -> None:
    payload: dict[str, object] = {
        "strategic_goal": "Recover the accepted plan",
        "summary": "Two commands",
        "commands": [command.model_dump(mode="json") for command in commands],
        "source_step_id": 0,
        "created_game_loop": 0,
    }
    if lifecycle_protocol is not None:
        payload["lifecycle_protocol"] = lifecycle_protocol
    store.append_event(
        run_id="run-1",
        episode_id="episode-1",
        step_id=0,
        event_type="plan_accepted",
        payload=payload,
    )


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
            module_started = [
                event.payload["module"]
                for event in store.recent_events("integration-run", "episode-0", 80)
                if event.event_type == "module_started" and event.step_id == 0
            ]
            assert module_started == ["memory", "reflection", "planning", "action"]
            context_events = [
                event
                for event in store.recent_events("integration-run", "episode-0", 80)
                if event.event_type == "context_prepared" and event.step_id == 0
            ]
            assert len(context_events) == 1
            assert context_events[0].payload["module"] == "planning"
            assert context_events[0].payload["final_chars"] > 0
            assert context_events[0].payload["estimated_tokens"] > 0
            assert 0 < context_events[0].payload["compression_ratio"] <= 1
            planning_event = next(
                event
                for event in store.recent_events("integration-run", "episode-0", 50)
                if event.event_type == "module_result" and event.payload["module"] == "planning"
            )
            assert planning_event.payload["output"]["plan"]["strategic_goal"] == (
                "Remove the visible threat"
            )
            context_stats = planning_event.payload["output"]["context_compaction"]
            assert context_stats["final_chars"] <= context_stats["budget_chars"]
            assert context_stats["original_chars"] > 0
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
            assert second.commands == []
            assert second.idle_reason is IdleReason.PLANNER_TIMEOUT
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
            assert first.commands == []
            assert first.idle_reason is IdleReason.WAITING_FOR_PLANNER
            assert first.planner_pending is True
            await asyncio.sleep(0.02)
            await runtime.tick(make_observation(step_id=1, game_loop=1))
            timeout = store.last_event("run-1", "episode-1", "planner_timeout")
            assert timeout is not None
        finally:
            await runtime.close()

    asyncio.run(execute())


def test_background_planner_uses_fixed_start_cadence_and_remains_single_flight(
    tmp_path: Path,
) -> None:
    config = make_config(
        tmp_path,
        variant="planner_only",
        deterministic=False,
        planning_interval_game_loops=10,
    )
    provider = ControlledBackgroundProvider()
    store = EventStore(tmp_path / "events.sqlite3", tmp_path / "events.jsonl")
    runtime = RuntimeEngine(config=config, store=store, provider=provider)

    async def execute() -> None:
        try:
            first = await runtime.tick(make_observation(step_id=0, game_loop=0))
            await asyncio.sleep(0.01)
            assert first.idle_reason is IdleReason.WAITING_FOR_PLANNER
            assert provider.calls == 1

            await runtime.tick(make_observation(step_id=1, game_loop=10))
            await asyncio.sleep(0.01)
            assert provider.calls == 1

            provider.release_first.set()
            await asyncio.sleep(0.01)
            accepted = await runtime.tick(make_observation(step_id=2, game_loop=12))
            await asyncio.sleep(0.01)

            assert provider.calls == 2
            assert accepted.commands[0].command_id.endswith(":0:planner:0")
            assert accepted.planner_pending is True
            started = [
                event.payload["started_game_loop"]
                for event in store.recent_events("run-1", "episode-1", 100)
                if event.event_type == "planner_started"
            ]
            assert started == [0, 12]
        finally:
            await runtime.close()

    asyncio.run(execute())


def test_empty_plan_obeys_fixed_planner_start_interval(tmp_path: Path) -> None:
    config = make_config(
        tmp_path,
        variant="planner_only",
        planning_interval_game_loops=10,
    )
    provider = EmptyPlanProvider()
    store = EventStore(tmp_path / "events.sqlite3", tmp_path / "events.jsonl")
    runtime = RuntimeEngine(config=config, store=store, provider=provider)

    async def execute() -> None:
        try:
            first = await runtime.tick(
                make_observation(step_id=0, game_loop=0, include_enemy=False)
            )
            before_interval = await runtime.tick(
                make_observation(step_id=1, game_loop=9, include_enemy=False)
            )
            at_interval = await runtime.tick(
                make_observation(step_id=2, game_loop=10, include_enemy=False)
            )

            assert provider.calls == 2
            assert first.idle_reason is IdleReason.NO_LEGAL_ACTION
            assert before_interval.idle_reason is IdleReason.NO_LEGAL_ACTION
            assert at_interval.idle_reason is IdleReason.NO_LEGAL_ACTION
            assert [
                event.payload["started_game_loop"]
                for event in store.events_of_type("run-1", "episode-1", "planner_started")
            ] == [0, 10]
        finally:
            await runtime.close()

    asyncio.run(execute())


def test_first_plan_barrier_waits_for_initial_background_plan(tmp_path: Path) -> None:
    config = make_config(
        tmp_path,
        variant="planner_only",
        deterministic=False,
    )
    config.environment.pause_until_first_plan = True
    store = EventStore(tmp_path / "events.sqlite3", tmp_path / "events.jsonl")
    runtime = RuntimeEngine(config=config, store=store, provider=FakeProvider())

    async def execute() -> None:
        try:
            batch = await runtime.tick(make_observation(step_id=0, game_loop=0))

            assert batch.planner_pending is False
            assert batch.strategic_goal == "Remove the visible threat"
            assert batch.commands[0].source.value == "planner"
        finally:
            await runtime.close()

    asyncio.run(execute())


def test_invalid_candidate_cannot_block_valid_action_for_same_actor(tmp_path: Path) -> None:
    config = make_config(tmp_path, variant="planner_only")
    store = EventStore(tmp_path / "events.sqlite3", tmp_path / "events.jsonl")
    runtime = RuntimeEngine(
        config=config,
        store=store,
        provider=InvalidAttackBeforeBuildProvider(),
    )
    actor = "Builder/Builder-Probe-1"
    base_observation = make_observation(include_enemy=False)
    observation = base_observation.model_copy(
        update={
            "state": base_observation.state.model_copy(
                update={
                    "economy": EconomyState(
                        minerals=50,
                        supply_used=11,
                        supply_cap=15,
                    )
                }
            ),
            "available_actions": [
                AvailableAction(
                    name="Attack_Unit",
                    argument_names=["tag"],
                    argument_types=[ActionArgumentType.TAG],
                    actor_scopes=[actor],
                ),
                AvailableAction(
                    name="Build_Pylon_Screen",
                    argument_names=["screen"],
                    argument_types=[ActionArgumentType.POSITION],
                    actor_scopes=[actor],
                    argument_candidates=[[[60, 40]]],
                ),
                AvailableAction(
                    name="No_Operation",
                    actor_scopes=[actor],
                ),
            ],
        }
    )

    async def execute() -> None:
        try:
            batch = await runtime.tick(observation)

            assert [command.name for command in batch.commands] == ["Build_Pylon_Screen"]
            assert batch.commands[0].arguments == [[60, 40]]
            assert batch.rejected_commands == []
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


def test_runtime_serializes_commands_for_actor_with_unfinished_dispatch(
    tmp_path: Path,
) -> None:
    config = make_config(
        tmp_path,
        variant="planner_only",
        planning_interval_game_loops=1000,
    )
    store = EventStore(tmp_path / "events.sqlite3", tmp_path / "events.jsonl")
    runtime = RuntimeEngine(
        config=config,
        store=store,
        provider=TwoCommandsForOneActorProvider(),
    )

    async def execute() -> None:
        try:
            first = await runtime.tick(make_observation(step_id=0, game_loop=0))
            assert len(first.commands) == 1
            assert first.commands[0].command_id.endswith(":planner:0")

            blocked = await runtime.tick(make_observation(step_id=1, game_loop=1))
            assert blocked.commands == []
            assert blocked.idle_reason is IdleReason.PLAN_COMMANDS_DEFERRED
            assert blocked.rejected_commands == [
                "run-1:episode-1:0:planner:1: actor has an in-flight dispatched command"
            ]

            runtime.record_execution(successful_execution(first.commands[0], step_id=1))
            released = await runtime.tick(make_observation(step_id=2, game_loop=2))

            assert len(released.commands) == 1
            assert released.commands[0].command_id.endswith(":planner:1")
            lifecycle = [
                event.payload["status"]
                for event in store.events_of_type("run-1", "episode-1", "command_lifecycle")
                if event.payload["command"]["command_id"] == released.commands[0].command_id
            ]
            assert lifecycle == ["pending", "deferred", "pending", "dispatched"]
        finally:
            await runtime.close()

    asyncio.run(execute())


def test_planner_receives_lifecycle_snapshot_and_retains_inflight_semantics(
    tmp_path: Path,
) -> None:
    config = make_config(
        tmp_path,
        variant="planner_only",
        planning_interval_game_loops=1,
    )
    store = EventStore(tmp_path / "events.sqlite3", tmp_path / "events.jsonl")
    provider = RepeatingPlanCapturingProvider()
    runtime = RuntimeEngine(config=config, store=store, provider=provider)

    async def execute() -> None:
        try:
            first = await runtime.tick(make_observation(step_id=0, game_loop=0))
            command = first.commands[0]

            repeated = await runtime.tick(make_observation(step_id=1, game_loop=1))

            assert repeated.commands == []
            assert len(provider.prompts) == 2
            assert provider.prompts[1]["active_plan"] == {
                "strategic_goal": "Keep pressure on the visible threat",
                "summary": "",
                "commands": [
                    {
                        "command_id": command.command_id,
                        "actor": command.actor,
                        "name": command.name,
                        "arguments": command.arguments,
                        "source": "planner",
                        "status": "dispatched",
                        "reason": None,
                        "created_game_loop": 0,
                        "ttl_game_loops": config.runtime.planner_command_ttl_game_loops,
                        "expires_at_game_loop": (config.runtime.planner_command_ttl_game_loops),
                    }
                ],
            }
            lifecycle = [
                event.payload["status"]
                for event in store.events_of_type("run-1", "episode-1", "command_lifecycle")
                if event.payload["command"]["command_id"] == command.command_id
            ]
            assert lifecycle == ["pending", "dispatched"]
            plans = store.events_of_type("run-1", "episode-1", "plan_accepted")
            assert plans[-1].payload["retained_command_ids"] == [command.command_id]
            assert plans[-1].payload["commands"][0]["command_id"] == command.command_id
        finally:
            await runtime.close()

    asyncio.run(execute())


def test_lifecycle_deduplicates_status_and_reason_together(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events.sqlite3", tmp_path / "events.jsonl")
    runtime = RuntimeEngine(
        config=make_config(tmp_path, variant="planner_only"),
        store=store,
        provider=FakeProvider(),
    )
    command = ActionCommand(
        command_id="command-deferred",
        actor="Builder/Builder-Probe-1",
        name="Build_Pylon_Screen",
        arguments=[[48, 56]],
        source=ActionSource.PLANNER,
        ttl_game_loops=112,
        created_game_loop=0,
    )

    runtime._set_command_lifecycle(
        command,
        CommandStatus.PENDING,
        run_id="run-1",
        episode_id="episode-1",
        step_id=0,
        game_loop=0,
    )
    runtime._set_command_lifecycle(
        command,
        CommandStatus.DEFERRED,
        run_id="run-1",
        episode_id="episode-1",
        step_id=1,
        game_loop=1,
        reason="insufficient minerals",
    )
    runtime._set_command_lifecycle(
        command,
        CommandStatus.DEFERRED,
        run_id="run-1",
        episode_id="episode-1",
        step_id=2,
        game_loop=2,
        reason="actor has an in-flight dispatched command",
    )
    changed = runtime._set_command_lifecycle(
        command,
        CommandStatus.DEFERRED,
        run_id="run-1",
        episode_id="episode-1",
        step_id=3,
        game_loop=3,
        reason="actor has an in-flight dispatched command",
    )

    assert changed is False
    events = store.events_of_type("run-1", "episode-1", "command_lifecycle")
    assert [(event.payload["status"], event.payload["reason"]) for event in events] == [
        ("pending", None),
        ("deferred", "insufficient minerals"),
        ("deferred", "actor has an in-flight dispatched command"),
    ]


def test_rejection_reason_is_reported_once_across_lifecycle_round_trip(
    tmp_path: Path,
) -> None:
    store = EventStore(tmp_path / "events.sqlite3", tmp_path / "events.jsonl")
    runtime = RuntimeEngine(
        config=make_config(tmp_path, variant="planner_only"),
        store=store,
        provider=FakeProvider(),
    )
    observation = make_observation(step_id=0, game_loop=0)
    command = ActionCommand(
        command_id="command-round-trip",
        actor="Builder/Builder-Probe-1",
        name="Build_Pylon_Screen",
        arguments=[[48, 56]],
        source=ActionSource.PLANNER,
        ttl_game_loops=112,
        created_game_loop=0,
    )
    failure = ValidationFailure(
        command=command,
        reason="insufficient minerals",
        disposition=ValidationDisposition.DEFERRED,
    )
    runtime._set_command_lifecycle(
        command,
        CommandStatus.PENDING,
        run_id=observation.run_id,
        episode_id=observation.episode_id,
        step_id=observation.step_id,
        game_loop=observation.game_loop,
    )

    assert runtime._apply_validation_failures([failure], observation) == [
        "command-round-trip: insufficient minerals"
    ]
    runtime._transition_command(command, CommandStatus.PENDING, observation)
    assert runtime._apply_validation_failures([failure], observation) == []

    lifecycle = store.events_of_type(
        observation.run_id,
        observation.episode_id,
        "command_lifecycle",
    )
    assert [event.payload["status"] for event in lifecycle] == [
        "pending",
        "deferred",
        "pending",
        "deferred",
    ]

    restored = RuntimeEngine(
        config=make_config(tmp_path, variant="planner_only"),
        store=store,
        provider=FakeProvider(),
    )
    asyncio.run(restored._activate_episode(observation))
    restored._transition_command(command, CommandStatus.PENDING, observation)
    assert restored._apply_validation_failures([failure], observation) == []
    asyncio.run(runtime.close())
    asyncio.run(restored.close())


def test_obsolete_and_superseded_commands_emit_one_terminal_lifecycle_event(
    tmp_path: Path,
) -> None:
    store = EventStore(tmp_path / "events.sqlite3", tmp_path / "events.jsonl")
    runtime = RuntimeEngine(
        config=make_config(tmp_path, variant="planner_only"),
        store=store,
        provider=FakeProvider(),
    )
    observation = make_observation(step_id=0, game_loop=0)
    obsolete = ActionCommand(
        command_id="command-obsolete",
        actor="Builder/Builder-Probe-1",
        name="Build_Pylon_Screen",
        arguments=[[48, 56]],
        source=ActionSource.PLANNER,
        ttl_game_loops=112,
        created_game_loop=0,
    )
    superseded = obsolete.model_copy(update={"command_id": "command-superseded"})
    for command in (obsolete, superseded):
        runtime._set_command_lifecycle(
            command,
            CommandStatus.PENDING,
            run_id=observation.run_id,
            episode_id=observation.episode_id,
            step_id=observation.step_id,
            game_loop=observation.game_loop,
        )

    runtime._apply_validation_failures(
        [
            ValidationFailure(
                command=obsolete,
                reason="target structure already exists",
                disposition=ValidationDisposition.OBSOLETE,
            )
        ],
        observation,
    )
    runtime._cached_plan = PlanState(
        strategic_goal="old",
        summary="old",
        commands=[superseded],
        source_step_id=0,
        created_game_loop=0,
    )
    runtime._accept_plan(
        PlanState(
            strategic_goal="new",
            summary="new",
            commands=[],
            source_step_id=1,
            created_game_loop=1,
        ),
        observation,
    )

    lifecycle = store.events_of_type(
        observation.run_id,
        observation.episode_id,
        "command_lifecycle",
    )
    status_by_command = {
        command_id: [
            event.payload["status"]
            for event in lifecycle
            if event.payload["command"]["command_id"] == command_id
        ]
        for command_id in (obsolete.command_id, superseded.command_id)
    }
    assert status_by_command == {
        "command-obsolete": ["pending", "obsolete"],
        "command-superseded": ["pending", "superseded"],
    }
    asyncio.run(runtime.close())


def test_reflection_pairs_async_execution_with_its_dispatch_decision(tmp_path: Path) -> None:
    config = make_config(
        tmp_path,
        variant="planner_reflection_memory_reflex",
        planning_interval_game_loops=100,
    )
    store = EventStore(tmp_path / "events.sqlite3", tmp_path / "events.jsonl")
    provider = PromptCapturingFakeProvider()
    runtime = RuntimeEngine(config=config, store=store, provider=provider)

    async def execute() -> None:
        try:
            dispatched = await runtime.tick(make_observation(step_id=0, game_loop=0))
            command = dispatched.commands[0]
            empty_tick = await runtime.tick(make_observation(step_id=1, game_loop=1))
            assert empty_tick.commands == []

            runtime.record_execution(successful_execution(command, step_id=1))
            runtime._urgent_replan_requested = True
            await runtime.tick(make_observation(step_id=2, game_loop=2))

            assert len(provider.reflection_prompts) == 1
            matching = provider.reflection_prompts[0]["last_decision"]
            assert matching["step_id"] == 0
            assert matching["commands"][0]["command_id"] == command.command_id
            assert provider.reflection_prompts[0]["last_execution"]["command_id"] == (
                command.command_id
            )
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
    provider = CountingFakeProvider()
    runtime = RuntimeEngine(config=config, store=store, provider=provider)

    async def execute() -> None:
        try:
            await runtime.tick(make_observation(step_id=0, game_loop=0))
            await asyncio.sleep(0.01)

            batch = await runtime.tick(
                make_observation(step_id=1, game_loop=100, include_enemy=False)
            )

            assert batch.commands == []
            assert batch.idle_reason is IdleReason.PLAN_EXHAUSTED
            assert any("target_not_visible" in reason for reason in batch.rejected_commands)
            stale_id = "run-1:episode-1:0:planner:0"
            stale_events = [
                event
                for event in store.events_of_type("run-1", "episode-1", "command_lifecycle")
                if event.payload["command"]["command_id"] == stale_id
            ]
            assert [event.payload["status"] for event in stale_events] == [
                "pending",
                "rejected",
            ]
            assert stale_events[-1].payload["reason"] == "target_not_visible"

            await runtime.tick(make_observation(step_id=2, game_loop=101, include_enemy=False))
            await asyncio.sleep(0.01)
            assert provider.calls == 2
            assert [
                event.payload["started_game_loop"]
                for event in store.events_of_type("run-1", "episode-1", "planner_started")
            ] == [0, 101]
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
                action_name=first.commands[0].name,
                actor=first.commands[0].actor,
                source=first.commands[0].source,
                requested_arguments=first.commands[0].arguments,
                resolved_arguments=first.commands[0].arguments,
                status=ExecutionStatus.SUCCEEDED,
                execution_stage=ExecutionStage.PYSC2_ACCEPTANCE,
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
        assert second.commands == []
        assert second.idle_reason is IdleReason.PLAN_EXHAUSTED
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


def test_runtime_recovers_dispatched_transition_without_redispatch(tmp_path: Path) -> None:
    config = make_config(
        tmp_path,
        variant="planner_only",
        planning_interval_game_loops=100,
    )
    database = tmp_path / "events.sqlite3"
    journal = tmp_path / "events.jsonl"

    async def execute() -> None:
        first_runtime = RuntimeEngine(
            config=config,
            store=EventStore(database, journal),
            provider=FakeProvider(),
        )
        first = await first_runtime.tick(make_observation(step_id=0, game_loop=0))
        command_id = first.commands[0].command_id
        await first_runtime.close()

        recovered_store = EventStore(database, journal)
        recovered_runtime = RuntimeEngine(
            config=config,
            store=recovered_store,
            provider=UnexpectedProvider(),
        )
        try:
            recovered = await recovered_runtime.tick(make_observation(step_id=1, game_loop=1))

            assert recovered.commands == []
            assert recovered.idle_reason is IdleReason.PLAN_EXHAUSTED
            transitions = [
                event.payload["status"]
                for event in recovered_store.events_of_type(
                    "run-1", "episode-1", "command_lifecycle"
                )
                if event.payload["command"]["command_id"] == command_id
            ]
            assert transitions == ["pending", "dispatched"]
        finally:
            await recovered_runtime.close()

    asyncio.run(execute())


def test_execution_report_is_idempotent_and_conflicting_terminal_fails(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path, variant="planner_only")
    store = EventStore(tmp_path / "events.sqlite3", tmp_path / "events.jsonl")
    runtime = RuntimeEngine(config=config, store=store, provider=FakeProvider())

    async def execute() -> None:
        try:
            batch = await runtime.tick(make_observation(step_id=0, game_loop=0))
            report = successful_execution(batch.commands[0])

            runtime.record_execution(report)
            runtime.record_execution(report)

            assert len(store.events_of_type("run-1", "episode-1", "execution")) == 1
            conflicting = report.model_copy(
                update={
                    "success": False,
                    "status": ExecutionStatus.FAILED,
                    "execution_stage": ExecutionStage.EFFECT_VERIFICATION,
                    "failure_code": "effect_timeout",
                }
            )
            with pytest.raises(RuntimeError, match="conflicting terminal execution report"):
                runtime.record_execution(conflicting)
            runtime.end_episode(episode_result())
            assert len(store.events_of_type("run-1", "episode-1", "execution")) == 1
        finally:
            await runtime.close()

    asyncio.run(execute())


def test_execution_report_must_match_dispatched_command_identity(tmp_path: Path) -> None:
    config = make_config(tmp_path, variant="planner_only")
    store = EventStore(tmp_path / "events.sqlite3", tmp_path / "events.jsonl")
    runtime = RuntimeEngine(config=config, store=store, provider=FakeProvider())

    async def execute() -> None:
        try:
            batch = await runtime.tick(make_observation(step_id=0, game_loop=0))
            report = successful_execution(batch.commands[0])
            mismatches: list[tuple[dict[str, object], str]] = [
                ({"run_id": "other-run"}, "run_id/episode_id"),
                ({"episode_id": "other-episode"}, "run_id/episode_id"),
                ({"action_name": "Retreat"}, "action_name"),
                ({"actor": "other-actor"}, "actor"),
                ({"source": ActionSource.REFLEX}, "source"),
                ({"requested_arguments": ["0xdead"]}, "requested_arguments"),
            ]
            for update, field in mismatches:
                with pytest.raises(RuntimeError, match=field):
                    runtime.record_execution(report.model_copy(update=update))

            assert store.events_of_type("run-1", "episode-1", "execution") == []
            runtime.record_execution(report)
            runtime.record_execution(report)
            assert len(store.events_of_type("run-1", "episode-1", "execution")) == 1
        finally:
            await runtime.close()

    asyncio.run(execute())


def test_backfilled_legacy_execution_keeps_semantic_identity_checks(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path, variant="planner_only")
    store = EventStore(tmp_path / "events.sqlite3", tmp_path / "events.jsonl")
    runtime = RuntimeEngine(config=config, store=store, provider=FakeProvider())

    async def execute() -> None:
        try:
            batch = await runtime.tick(make_observation(step_id=0, game_loop=0))
            report = successful_execution(batch.commands[0]).model_copy(
                update={"protocol_version": "1.0"}
            )
            with pytest.raises(RuntimeError, match="action_name"):
                runtime.record_execution(report.model_copy(update={"action_name": "Retreat"}))

            runtime.record_execution(report)
            runtime.record_execution(report)
            events = store.events_of_type("run-1", "episode-1", "execution")
            assert len(events) == 1
            assert events[0].payload["protocol_version"] == "1.0"
        finally:
            await runtime.close()

    asyncio.run(execute())


def test_episode_end_cancels_missing_bridge_report_exactly_once(tmp_path: Path) -> None:
    config = make_config(tmp_path, variant="planner_only")
    store = EventStore(tmp_path / "events.sqlite3", tmp_path / "events.jsonl")
    runtime = RuntimeEngine(config=config, store=store, provider=FakeProvider())

    async def execute() -> None:
        try:
            batch = await runtime.tick(make_observation(step_id=0, game_loop=0))
            command = batch.commands[0]
            result = episode_result()

            runtime.end_episode(result)
            runtime.end_episode(result)

            execution_events = store.events_of_type("run-1", "episode-1", "execution")
            assert len(execution_events) == 1
            report = ExecutionReport.model_validate(execution_events[0].payload)
            assert report.command_id == command.command_id
            assert report.status is ExecutionStatus.CANCELLED
            assert report.execution_stage is ExecutionStage.EPISODE_END
            assert report.failure_code == "bridge_execution_report_missing"
            assert report.failure_reason == (
                "episode ended before the Bridge reported command completion"
            )
            assert report.action_name == command.name
            assert report.actor == command.actor
            assert report.source is command.source
            assert report.requested_arguments == command.arguments
            terminal = [
                event.payload["status"]
                for event in store.events_of_type("run-1", "episode-1", "command_lifecycle")
                if event.payload["command"]["command_id"] == command.command_id
            ]
            assert terminal == ["pending", "dispatched", "cancelled"]
            assert len(store.events_of_type("run-1", "episode-1", "episode_result")) == 1
            assert len(store.events_of_type("run-1", "episode-1", "episode_summary")) == 1
            with pytest.raises(RuntimeError, match="conflicting episode result"):
                runtime.end_episode(result.model_copy(update={"score": 1.0}))
        finally:
            await runtime.close()

    asyncio.run(execute())


def test_episode_end_cancels_planner_command_preempted_before_dispatch(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    store = EventStore(tmp_path / "events.sqlite3", tmp_path / "events.jsonl")
    runtime = RuntimeEngine(config=config, store=store, provider=FakeProvider())

    async def execute() -> None:
        try:
            batch = await runtime.tick(
                make_observation(step_id=0, game_loop=0, alerts=["under_attack"])
            )
            assert len(batch.commands) == 1
            assert batch.commands[0].source is ActionSource.REFLEX

            runtime.end_episode(episode_result())

            reports = [
                ExecutionReport.model_validate(event.payload)
                for event in store.events_of_type("run-1", "episode-1", "execution")
            ]
            assert len(reports) == 2
            planner_report = next(
                report for report in reports if report.source is ActionSource.PLANNER
            )
            assert planner_report.status is ExecutionStatus.CANCELLED
            assert planner_report.execution_stage is ExecutionStage.EPISODE_END
            assert planner_report.failure_code == "episode_ended_before_dispatch"
            transitions = [
                event.payload["status"]
                for event in store.events_of_type("run-1", "episode-1", "command_lifecycle")
                if event.payload["command"]["command_id"] == planner_report.command_id
            ]
            assert transitions == ["pending", "cancelled"]
        finally:
            await runtime.close()

    asyncio.run(execute())


def test_synthetic_worker_end_cancels_dispatched_command_with_worker_reason(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path, variant="planner_only")
    store = EventStore(tmp_path / "events.sqlite3", tmp_path / "events.jsonl")
    runtime = RuntimeEngine(config=config, store=store, provider=FakeProvider())

    async def execute() -> None:
        try:
            await runtime.tick(make_observation(step_id=0, game_loop=0))
            failure_reason = "worker exited with status 7 before reporting an episode result"
            runtime.end_episode(episode_result(failure_reason=failure_reason))

            event = store.last_event("run-1", "episode-1", "execution")
            assert event is not None
            report = ExecutionReport.model_validate(event.payload)
            assert report.status is ExecutionStatus.CANCELLED
            assert report.execution_stage is ExecutionStage.EPISODE_END
            assert report.failure_code == "worker_terminated_before_execution_report"
            assert report.failure_reason == (
                "worker terminated before reporting command completion: " + failure_reason
            )
        finally:
            await runtime.close()

    asyncio.run(execute())


def test_partial_lifecycle_recovers_missing_plan_command_as_pending(tmp_path: Path) -> None:
    config = make_config(tmp_path, variant="planner_only")
    store = EventStore(tmp_path / "events.sqlite3", tmp_path / "events.jsonl")
    commands = [
        ActionCommand(
            command_id=f"command-{index}",
            actor="army",
            name="Attack_Unit",
            arguments=["enemy-1"],
            ttl_game_loops=112,
            created_game_loop=0,
            source=ActionSource.PLANNER,
        )
        for index in range(2)
    ]
    append_plan(store, commands)
    store.append_event(
        run_id="run-1",
        episode_id="episode-1",
        step_id=0,
        event_type="command_lifecycle",
        payload={
            "command": commands[0].model_dump(mode="json"),
            "status": "pending",
            "reason": None,
            "game_loop": 0,
        },
    )
    runtime = RuntimeEngine(config=config, store=store, provider=UnexpectedProvider())

    async def execute() -> None:
        try:
            observation = make_observation(step_id=1, game_loop=1)
            await runtime._activate_episode(observation)

            recovered = [
                event
                for event in store.events_of_type("run-1", "episode-1", "command_lifecycle")
                if event.payload["command"]["command_id"] == "command-1"
            ]
            assert [event.payload["status"] for event in recovered] == ["pending"]
            assert recovered[0].payload["reason"] == (
                "recovered accepted command after partial lifecycle write"
            )
        finally:
            await runtime.close()

    asyncio.run(execute())


def test_current_plan_accept_crash_before_lifecycle_recovers_pending(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path, variant="planner_only")
    store = EventStore(tmp_path / "events.sqlite3", tmp_path / "events.jsonl")
    command = ActionCommand(
        command_id="command-0",
        actor="army",
        name="Attack_Unit",
        arguments=["0x1"],
        ttl_game_loops=112,
        created_game_loop=0,
        source=ActionSource.PLANNER,
    )
    append_plan(store, [command], lifecycle_protocol="1.1")
    runtime = RuntimeEngine(config=config, store=store, provider=UnexpectedProvider())

    async def execute() -> None:
        try:
            await runtime._activate_episode(make_observation(step_id=1, game_loop=1))

            transitions = [
                event.payload["status"]
                for event in store.events_of_type("run-1", "episode-1", "command_lifecycle")
                if event.payload["command"]["command_id"] == command.command_id
            ]
            assert transitions == ["pending"]
        finally:
            await runtime.close()

    asyncio.run(execute())


def test_execution_event_recovers_missing_terminal_lifecycle(tmp_path: Path) -> None:
    config = make_config(tmp_path, variant="planner_only")
    store = EventStore(tmp_path / "events.sqlite3", tmp_path / "events.jsonl")
    command = ActionCommand(
        command_id="command-0",
        actor="army",
        name="Attack_Unit",
        arguments=["enemy-1"],
        ttl_game_loops=112,
        created_game_loop=0,
        source=ActionSource.PLANNER,
    )
    append_plan(store, [command])
    for status in ("pending", "dispatched"):
        store.append_event(
            run_id="run-1",
            episode_id="episode-1",
            step_id=0,
            event_type="command_lifecycle",
            payload={
                "command": command.model_dump(mode="json"),
                "status": status,
                "reason": None,
                "game_loop": 0,
            },
        )
    report = successful_execution(command)
    store.append_event(
        run_id="run-1",
        episode_id="episode-1",
        step_id=0,
        event_type="execution",
        payload=report,
    )
    runtime = RuntimeEngine(config=config, store=store, provider=UnexpectedProvider())

    async def execute() -> None:
        try:
            await runtime._activate_episode(make_observation(step_id=1, game_loop=1))

            transitions = [
                event.payload["status"]
                for event in store.events_of_type("run-1", "episode-1", "command_lifecycle")
                if event.payload["command"]["command_id"] == command.command_id
            ]
            assert transitions == ["pending", "dispatched", "succeeded"]
            assert len(store.events_of_type("run-1", "episode-1", "execution")) == 1

            runtime.record_execution(report)
            assert len(store.events_of_type("run-1", "episode-1", "execution")) == 1
        finally:
            await runtime.close()

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
            runtime.record_execution(successful_execution(batch.commands[0]))
            follow_up = await runtime.tick(make_observation(step_id=1, game_loop=1, alerts=[]))
            assert follow_up.commands[0].command_id == ("run-1:episode-1:0:planner:0")
        finally:
            await runtime.close()

    asyncio.run(execute())


def test_same_reflex_command_id_is_dispatched_only_once(tmp_path: Path) -> None:
    config = make_config(tmp_path, variant="reflex_only")
    store = EventStore(tmp_path / "events.sqlite3", tmp_path / "events.jsonl")
    runtime = RuntimeEngine(config=config, store=store, provider=UnexpectedProvider())
    observation = make_observation(step_id=7, game_loop=20, alerts=["under_attack"])

    async def execute() -> None:
        try:
            first = await runtime.tick(observation)
            repeated = await runtime.tick(observation)

            assert len(first.commands) == 1
            assert first.commands[0].command_id == "run-1:episode-1:7:reflex:0"
            assert repeated.commands == []
            assert repeated.idle_reason is IdleReason.NO_LEGAL_ACTION
            lifecycle = [
                event
                for event in store.events_of_type("run-1", "episode-1", "command_lifecycle")
                if event.payload["command"]["command_id"] == "run-1:episode-1:7:reflex:0"
            ]
            assert [event.payload["status"] for event in lifecycle] == ["dispatched"]
        finally:
            await runtime.close()

    asyncio.run(execute())


def test_reaccepted_planner_command_id_cannot_return_to_pending(tmp_path: Path) -> None:
    config = make_config(
        tmp_path,
        variant="planner_only",
        planning_interval_game_loops=100,
    )
    store = EventStore(tmp_path / "events.sqlite3", tmp_path / "events.jsonl")
    runtime = RuntimeEngine(config=config, store=store, provider=FakeProvider())
    observation = make_observation(step_id=7, game_loop=20)

    async def execute() -> None:
        try:
            first = await runtime.tick(observation)
            assert len(first.commands) == 1
            command_id = first.commands[0].command_id

            runtime._urgent_replan_requested = True
            repeated = await runtime.tick(observation)

            assert repeated.commands == []
            lifecycle = [
                event.payload["status"]
                for event in store.events_of_type("run-1", "episode-1", "command_lifecycle")
                if event.payload["command"]["command_id"] == command_id
            ]
            assert lifecycle == ["pending", "dispatched"]
            planner_error = store.last_event("run-1", "episode-1", "planner_error")
            assert planner_error is not None
            assert "already has lifecycle state" in planner_error.payload["message"]
        finally:
            await runtime.close()

    asyncio.run(execute())


def test_pending_planner_command_expires_once_at_ttl_boundary(tmp_path: Path) -> None:
    config = make_config(
        tmp_path,
        planning_interval_game_loops=100,
    )
    config.runtime.planner_command_ttl_game_loops = 4
    store = EventStore(tmp_path / "events.sqlite3", tmp_path / "events.jsonl")
    runtime = RuntimeEngine(config=config, store=store, provider=FakeProvider())

    async def execute() -> None:
        try:
            first = await runtime.tick(
                make_observation(step_id=0, game_loop=0, alerts=["under_attack"])
            )
            assert first.commands[0].source.value == "reflex"

            expired = await runtime.tick(make_observation(step_id=1, game_loop=4))
            assert expired.commands == []
            assert expired.idle_reason is IdleReason.PLAN_EXHAUSTED

            await runtime.tick(make_observation(step_id=2, game_loop=5))
            old_command_events = [
                event
                for event in store.recent_events("run-1", "episode-1", 200)
                if event.event_type == "command_lifecycle"
                and event.payload["command"]["command_id"] == "run-1:episode-1:0:planner:0"
                and event.payload["status"] == "expired"
            ]
            assert len(old_command_events) == 1
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


def test_noop_baseline_returns_an_empty_semantic_batch(tmp_path: Path) -> None:
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
            assert batch.commands == []
            assert batch.idle_reason is IdleReason.NOOP_BASELINE
        finally:
            await runtime.close()

    asyncio.run(execute())
