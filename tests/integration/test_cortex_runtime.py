from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from rtscortex.config import (
    AgentSettings,
    CortexMacroSettings,
    CortexSettings,
    ExperimentConfig,
    ReflexSettings,
    RunSettings,
    RuntimeSettings,
)
from rtscortex.contracts import (
    ActionArgumentType,
    ActionBatch,
    ActionSource,
    AvailableAction,
    EconomyState,
    EpisodeOutcome,
    EpisodeResult,
    ExecutionReport,
    ExecutionStage,
    ExecutionStatus,
    ObservationEnvelope,
    SC2State,
    UnitState,
)
from rtscortex.evaluation import compute_cortex_observability
from rtscortex.memory import EventStore
from rtscortex.policy.hima import (
    HIMA_ADAPTER_VERSION,
    HIMA_PARSER_VERSION,
    HIMA_PINNED_REVISIONS,
    HIMA_VOCABULARY_VERSION,
    HIMAInputContext,
    HIMALiveHealth,
    HIMALiveProposalResponse,
    HIMAObservationAdapter,
    HIMAProposalParser,
)
from rtscortex.providers import FakeProvider
from rtscortex.runtime import CortexRuntimeEngine


class _FakeMacroClient:
    def __init__(self, output: str | list[str] = "Actions: ['Pylon']") -> None:
        self.outputs = [output] if isinstance(output, str) else output
        if not self.outputs:
            raise ValueError("fake macro client requires at least one output")
        self.contexts: list[HIMAInputContext] = []
        self.closed = False

    async def health(self) -> HIMALiveHealth:
        return HIMALiveHealth(
            model_id="SNUMPR/Protoss-a",
            model_revision=HIMA_PINNED_REVISIONS["SNUMPR/Protoss-a"],
            adapter_version=HIMA_ADAPTER_VERSION,
            parser_version=HIMA_PARSER_VERSION,
            vocabulary_version=HIMA_VOCABULARY_VERSION,
        )

    async def propose(
        self,
        context: HIMAInputContext,
        *,
        request_id: str | None = None,
    ) -> HIMALiveProposalResponse:
        output = self.outputs[min(len(self.contexts), len(self.outputs) - 1)]
        self.contexts.append(context)
        await asyncio.sleep(0)
        snapshot = HIMAObservationAdapter().adapt_context(context)
        return HIMALiveProposalResponse(
            request_id=request_id or "fake-request",
            run_id=context.observation.run_id,
            episode_id=context.observation.episode_id,
            step_id=context.observation.step_id,
            game_loop=context.observation.game_loop,
            projection_hash=snapshot.projection_hash,
            proposal=HIMAProposalParser().parse(output),
        )

    async def close(self) -> None:
        self.closed = True


class _BlockingFirstMacroClient(_FakeMacroClient):
    def __init__(self) -> None:
        super().__init__(["Actions: ['Pylon']", "Actions: ['Pylon']"])
        self.release_first = asyncio.Event()

    async def propose(
        self,
        context: HIMAInputContext,
        *,
        request_id: str | None = None,
    ) -> HIMALiveProposalResponse:
        if not self.contexts:
            await self.release_first.wait()
        return await super().propose(context, request_id=request_id)


class _TimeoutMacroClient(_FakeMacroClient):
    async def propose(
        self,
        context: HIMAInputContext,
        *,
        request_id: str | None = None,
    ) -> HIMALiveProposalResponse:
        self.contexts.append(context)
        await asyncio.sleep(0)
        raise TimeoutError("macro request timed out")


def _config(
    tmp_path: Path,
    *,
    macro: bool = True,
    macro_required: bool = True,
) -> ExperimentConfig:
    return ExperimentConfig(
        run=RunSettings(output_root=tmp_path, runtime_root=tmp_path / "runtime"),
        runtime=RuntimeSettings(
            deterministic=True,
            planning_interval_game_loops=112,
            planner_command_ttl_game_loops=112,
        ),
        agent=AgentSettings(variant="cortex"),
        cortex=CortexSettings(
            macro=CortexMacroSettings(
                kind="hima" if macro else "disabled",
                model_path=Path("/tmp/fake-hima") if macro else None,
                allow_unlicensed_weights=macro,
                required=macro_required,
                interval_game_loops=112,
                plan_ttl_game_loops=448,
            )
        ),
        reflex=ReflexSettings(enabled=True),
    )


def _macro_observation(*, step_id: int, game_loop: int, pylon: bool = False) -> ObservationEnvelope:
    structures = (
        [
            UnitState(
                unit_id="0xpylon",
                unit_type="Pylon",
                alliance="self",
            )
        ]
        if pylon
        else []
    )
    return ObservationEnvelope(
        run_id="cortex-run",
        episode_id="episode-1",
        step_id=step_id,
        game_loop=game_loop,
        state=SC2State(
            economy=EconomyState(
                minerals=200,
                supply_used=12,
                supply_cap=15,
                workers=12,
            ),
            own_structures=structures,
        ),
        available_actions=[
            AvailableAction(
                name="Build_Pylon_Screen",
                argument_names=["screen"],
                argument_types=[ActionArgumentType.POSITION],
                actor_scopes=["Builder/Probe-1"],
                argument_candidates=[[[65, 90]]],
            )
        ],
    )


def _store(tmp_path: Path) -> EventStore:
    return EventStore(tmp_path / "events.sqlite3", tmp_path / "events.jsonl")


def test_hima_macro_plan_dispatches_only_through_current_candidate_domain(
    tmp_path: Path,
) -> None:
    client = _FakeMacroClient()
    store = _store(tmp_path)
    runtime = CortexRuntimeEngine(
        config=_config(tmp_path),
        store=store,
        provider=FakeProvider(),
        macro_client=client,
    )

    async def exercise() -> tuple[ExecutionReport, list[str]]:
        await runtime.start()
        first = await runtime.tick(_macro_observation(step_id=0, game_loop=0))
        assert first.commands == []
        assert first.planner_pending is True
        for _ in range(5):
            await asyncio.sleep(0)

        second = await runtime.tick(_macro_observation(step_id=1, game_loop=1))
        assert len(second.commands) == 1
        command = second.commands[0]
        assert command.name == "Build_Pylon_Screen"
        assert command.actor == "Builder/Probe-1"
        assert command.arguments == [[65, 90]]
        assert command.source is ActionSource.PLANNER

        report = ExecutionReport(
            run_id=second.run_id,
            episode_id=second.episode_id,
            step_id=second.step_id,
            command_id=command.command_id,
            success=True,
            action_name=command.name,
            actor=command.actor,
            source=command.source,
            requested_arguments=command.arguments,
            resolved_arguments=command.arguments,
            status=ExecutionStatus.SUCCEEDED,
            execution_stage=ExecutionStage.EFFECT_VERIFICATION,
        )
        runtime.record_execution(report)
        await runtime.tick(_macro_observation(step_id=2, game_loop=2, pylon=True))
        for _ in range(5):
            await asyncio.sleep(0)
        previous = list(client.contexts[-1].previous_actions)
        await runtime.close()
        return report, previous

    report, previous_actions = asyncio.run(exercise())

    assert previous_actions == ["Pylon"]
    recovered = EventStore(tmp_path / "events.sqlite3", tmp_path / "events.jsonl")
    event_types = [
        event.event_type
        for event in recovered.events_of_type(
            report.run_id,
            report.episode_id,
            "command_lineage",
        )
    ]
    metrics = compute_cortex_observability(
        recovered.events_after(report.run_id, 0, 1_000, episode_id=report.episode_id)
    )
    assert event_types == ["command_lineage"]
    assert metrics.executor_candidate_violations == 0
    assert metrics.command_lineage_coverage == 1.0
    assert client.closed is True
    recovered.close()


def test_failed_macro_command_is_not_retried_while_replacement_plan_is_pending(
    tmp_path: Path,
) -> None:
    client = _FakeMacroClient(["Actions: ['Pylon']", "Actions: ['Pylon']"])
    runtime = CortexRuntimeEngine(
        config=_config(tmp_path),
        store=_store(tmp_path),
        provider=FakeProvider(),
        macro_client=client,
    )

    async def exercise() -> None:
        await runtime.start()
        await runtime.tick(_macro_observation(step_id=0, game_loop=0))
        for _ in range(5):
            await asyncio.sleep(0)
        dispatched = await runtime.tick(_macro_observation(step_id=1, game_loop=1))
        command = dispatched.commands[0]
        runtime.record_execution(
            ExecutionReport(
                run_id=dispatched.run_id,
                episode_id=dispatched.episode_id,
                step_id=dispatched.step_id,
                command_id=command.command_id,
                success=False,
                action_name=command.name,
                actor=command.actor,
                source=command.source,
                requested_arguments=command.arguments,
                resolved_arguments=command.arguments,
                status=ExecutionStatus.FAILED,
                execution_stage=ExecutionStage.EFFECT_VERIFICATION,
                failure_code="effect_timeout",
                failure_reason="effect was not observed",
            )
        )

        waiting = await runtime.tick(_macro_observation(step_id=2, game_loop=2))

        assert waiting.commands == []
        assert waiting.planner_pending is True
        await runtime.close()

    asyncio.run(exercise())


def test_unusable_initial_macro_plan_fails_required_startup_barrier(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    config = config.model_copy(
        update={
            "environment": config.environment.model_copy(
                update={"pause_until_first_plan": True}
            )
        }
    )
    runtime = CortexRuntimeEngine(
        config=config,
        store=_store(tmp_path),
        provider=FakeProvider(),
        macro_client=_FakeMacroClient("Actions: ['Sentry']"),
    )

    async def exercise() -> None:
        await runtime.start()
        with pytest.raises(RuntimeError, match="required HIMA macro specialist"):
            await runtime.tick(_macro_observation(step_id=0, game_loop=0))
        assert runtime._macro_plan is None
        await runtime.close()

    asyncio.run(exercise())


def test_missing_prerequisite_plan_is_rejected_without_hot_loop(
    tmp_path: Path,
) -> None:
    client = _FakeMacroClient("Actions: ['Gateway']")
    store = _store(tmp_path)
    runtime = CortexRuntimeEngine(
        config=_config(tmp_path),
        store=store,
        provider=FakeProvider(),
        macro_client=client,
    )
    blocked = _macro_observation(step_id=0, game_loop=0).model_copy(
        update={"available_actions": []}
    )

    async def exercise() -> None:
        await runtime.start()
        await runtime.tick(blocked)
        for _ in range(5):
            await asyncio.sleep(0)
        rejected = await runtime.tick(
            blocked.model_copy(update={"step_id": 1, "game_loop": 1})
        )
        assert rejected.commands == []
        assert rejected.planner_pending is False
        assert runtime._macro_plan is None
        await runtime.tick(
            blocked.model_copy(update={"step_id": 2, "game_loop": 2})
        )
        assert len(client.contexts) == 1
        failures = store.events_of_type(
            "cortex-run", "episode-1", "macro_plan_rejected"
        )
        assert failures[-1].payload["reason"] == "missing_prerequisite_pylon"
        assert not store.events_of_type(
            "cortex-run", "episode-1", "specialist_failed"
        )
        await runtime.close()

    asyncio.run(exercise())


def test_resource_deferred_frontier_satisfies_required_startup_barrier(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    config = config.model_copy(
        update={
            "environment": config.environment.model_copy(
                update={"pause_until_first_plan": True}
            )
        }
    )
    observation = _macro_observation(step_id=0, game_loop=0)
    observation = observation.model_copy(
        update={
            "state": observation.state.model_copy(
                update={
                    "economy": observation.state.economy.model_copy(
                        update={"minerals": 0}
                    )
                }
            ),
            "available_actions": [],
        }
    )
    runtime = CortexRuntimeEngine(
        config=config,
        store=_store(tmp_path),
        provider=FakeProvider(),
        macro_client=_FakeMacroClient("Actions: ['Pylon']"),
    )

    async def exercise() -> None:
        await runtime.start()
        batch = await runtime.tick(observation)
        assert batch.commands == []
        assert batch.planner_pending is False
        assert runtime._macro_plan is not None
        assert runtime._macro_plan.steps[0].status.value == "deferred"
        await runtime.close()

    asyncio.run(exercise())


def test_episode_transition_drains_and_discards_previous_macro_request(
    tmp_path: Path,
) -> None:
    client = _BlockingFirstMacroClient()
    runtime = CortexRuntimeEngine(
        config=_config(tmp_path),
        store=_store(tmp_path),
        provider=FakeProvider(),
        macro_client=client,
    )
    first = _macro_observation(step_id=0, game_loop=0)
    second = first.model_copy(update={"episode_id": "episode-2"})

    async def exercise() -> None:
        await runtime.start()
        first_batch = await runtime.tick(first)
        assert first_batch.planner_pending is True
        await asyncio.sleep(0)

        transition = asyncio.create_task(runtime.tick(second))
        await asyncio.sleep(0)
        assert transition.done() is False
        client.release_first.set()
        second_batch = await transition

        assert second_batch.episode_id == "episode-2"
        assert second_batch.commands == []
        assert second_batch.planner_pending is True
        assert runtime._macro_plan is None
        assert [context.observation.episode_id for context in client.contexts] == [
            "episode-1"
        ]
        for _ in range(5):
            await asyncio.sleep(0)
        await runtime.tick(
            second.model_copy(update={"step_id": 1, "game_loop": 1})
        )
        assert runtime._macro_plan is not None
        assert runtime._macro_plan.episode_id == "episode-2"
        await runtime.close()

    asyncio.run(exercise())


def test_episode_transition_requires_terminalizing_a_dispatched_command(
    tmp_path: Path,
) -> None:
    runtime = CortexRuntimeEngine(
        config=_config(tmp_path),
        store=_store(tmp_path),
        provider=FakeProvider(),
        macro_client=_FakeMacroClient(),
    )
    first = _macro_observation(step_id=0, game_loop=0)
    second_episode = first.model_copy(
        update={"episode_id": "episode-2", "step_id": 0, "game_loop": 0}
    )

    async def exercise() -> None:
        await runtime.start()
        await runtime.tick(first)
        for _ in range(5):
            await asyncio.sleep(0)
        dispatched = await runtime.tick(
            _macro_observation(step_id=1, game_loop=1)
        )
        assert len(dispatched.commands) == 1
        command_id = dispatched.commands[0].command_id

        with pytest.raises(RuntimeError, match="before end_episode terminalizes"):
            await runtime.tick(second_episode)

        assert command_id in runtime._command_states
        runtime.end_episode(
            EpisodeResult(
                run_id=first.run_id,
                episode_id=first.episode_id,
                scenario="Simple64",
                seed=0,
                outcome=EpisodeOutcome.TRUNCATED,
                steps=2,
                failure_reason="test transition",
            )
        )
        next_batch = await runtime.tick(second_episode)
        assert next_batch.episode_id == "episode-2"
        old_reports = runtime.store.events_of_type(
            first.run_id,
            first.episode_id,
            "execution",
        )
        assert len(old_reports) == 1
        assert old_reports[0].payload["command_id"] == command_id
        assert old_reports[0].payload["status"] == "cancelled"
        await runtime.close()

    asyncio.run(exercise())


def test_required_timed_out_macro_specialist_fails_closed_on_next_episode(
    tmp_path: Path,
) -> None:
    runtime = CortexRuntimeEngine(
        config=_config(tmp_path),
        store=_store(tmp_path),
        provider=FakeProvider(),
        macro_client=_TimeoutMacroClient(),
    )
    first = _macro_observation(step_id=0, game_loop=0)
    second_episode = first.model_copy(
        update={"episode_id": "episode-2", "step_id": 0, "game_loop": 0}
    )

    async def exercise() -> None:
        await runtime.start()
        await runtime.tick(first)
        for _ in range(5):
            await asyncio.sleep(0)
        timeout_batch = await runtime.tick(
            _macro_observation(step_id=1, game_loop=1)
        )
        assert timeout_batch.idle_reason is not None
        assert timeout_batch.idle_reason.value == "planner_timeout"
        assert runtime._macro_requests_suspended is True

        with pytest.raises(RuntimeError, match="required HIMA macro specialist is suspended"):
            await runtime.tick(second_episode)

        await runtime.close()

    asyncio.run(exercise())


def test_reflex_dispatch_also_has_typed_candidate_and_lineage(tmp_path: Path) -> None:
    store = _store(tmp_path)
    runtime = CortexRuntimeEngine(
        config=_config(tmp_path, macro=False),
        store=store,
        provider=FakeProvider(),
    )
    observation = ObservationEnvelope(
        run_id="reflex-run",
        episode_id="episode-1",
        step_id=0,
        game_loop=0,
        state=SC2State(
            economy=EconomyState(supply_used=4, supply_cap=15, army_supply=2),
            own_units=[
                UnitState(unit_id="0x10", unit_type="Adept", alliance="self")
            ],
            visible_enemies=[
                UnitState(unit_id="0x20", unit_type="Zergling", alliance="enemy")
            ],
        ),
        available_actions=[
            AvailableAction(
                name="Attack_Unit",
                argument_names=["tag"],
                argument_types=[ActionArgumentType.TAG],
                actor_scopes=["CombatGroup/Army-1"],
                argument_candidates=[["0x20"]],
            )
        ],
        alerts=["under_attack"],
    )

    async def exercise() -> None:
        await runtime.start()
        batch = await runtime.tick(observation)
        assert len(batch.commands) == 1
        assert batch.commands[0].source is ActionSource.REFLEX
        await runtime.close()

    asyncio.run(exercise())

    recovered = EventStore(tmp_path / "events.sqlite3", tmp_path / "events.jsonl")
    lineages = recovered.events_of_type("reflex-run", "episode-1", "command_lineage")
    candidate_sets = recovered.events_of_type(
        "reflex-run", "episode-1", "candidate_set_built"
    )
    assert len(lineages) == 1
    assert lineages[0].payload["lineage"]["source_role"] == "reflex"
    assert candidate_sets[0].payload["candidate_count"] == 1
    recovered.close()


def test_runtime_restart_does_not_redispatch_inflight_macro_command(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    first_client = _FakeMacroClient()
    first = CortexRuntimeEngine(
        config=config,
        store=_store(tmp_path),
        provider=FakeProvider(),
        macro_client=first_client,
    )

    async def dispatch_once() -> str:
        await first.start()
        await first.tick(_macro_observation(step_id=0, game_loop=0))
        for _ in range(5):
            await asyncio.sleep(0)
        batch = await first.tick(_macro_observation(step_id=1, game_loop=1))
        command_id = batch.commands[0].command_id
        await first.close()
        return command_id

    command_id = asyncio.run(dispatch_once())
    second_client = _FakeMacroClient()
    recovered = CortexRuntimeEngine(
        config=config,
        store=_store(tmp_path),
        provider=FakeProvider(),
        macro_client=second_client,
    )

    async def recover() -> None:
        await recovered.start()
        batch = await recovered.tick(_macro_observation(step_id=2, game_loop=2))
        assert batch.commands == []
        await recovered.close()

    asyncio.run(recover())

    store = _store(tmp_path)
    dispatched = [
        event
        for event in store.events_of_type("cortex-run", "episode-1", "command_lifecycle")
        if event.payload["status"] == "dispatched"
        and event.payload["command"]["command_id"] == command_id
    ]
    assert len(dispatched) == 1
    store.close()


def test_optional_macro_startup_failure_falls_back_to_reflex(tmp_path: Path) -> None:
    class FailingClient(_FakeMacroClient):
        async def health(self) -> HIMALiveHealth:
            raise RuntimeError("checkpoint unavailable")

    client = FailingClient()
    runtime = CortexRuntimeEngine(
        config=_config(tmp_path, macro_required=False),
        store=_store(tmp_path),
        provider=FakeProvider(),
        macro_client=client,
    )

    async def exercise() -> None:
        await runtime.start()
        batch = await runtime.tick(_macro_observation(step_id=0, game_loop=0))
        assert batch.commands == []
        assert batch.planner_pending is False
        await runtime.close()

    asyncio.run(exercise())

    store = _store(tmp_path)
    failures = store.events_of_type("cortex-run", "episode-1", "specialist_failed")
    assert len(failures) == 1
    assert failures[0].payload["stage"] == "startup"
    assert failures[0].payload["fallback"] == "deterministic_reflex"
    assert client.closed is True
    store.close()


def test_duplicate_terminal_report_advances_repeated_step_once(tmp_path: Path) -> None:
    client = _FakeMacroClient("Final Actions Summary: <Pylon> x 2")
    store = _store(tmp_path)
    runtime = CortexRuntimeEngine(
        config=_config(tmp_path),
        store=store,
        provider=FakeProvider(),
        macro_client=client,
    )

    async def exercise() -> None:
        await runtime.start()
        await runtime.tick(_macro_observation(step_id=0, game_loop=0))
        for _ in range(5):
            await asyncio.sleep(0)
        batch = await runtime.tick(_macro_observation(step_id=1, game_loop=1))
        command = batch.commands[0]
        report = ExecutionReport(
            run_id=batch.run_id,
            episode_id=batch.episode_id,
            step_id=batch.step_id,
            command_id=command.command_id,
            success=True,
            action_name=command.name,
            actor=command.actor,
            source=command.source,
            requested_arguments=command.arguments,
            resolved_arguments=command.arguments,
            status=ExecutionStatus.SUCCEEDED,
            execution_stage=ExecutionStage.EFFECT_VERIFICATION,
        )

        runtime.record_execution(report)
        runtime.record_execution(report)

        assert runtime._macro_plan is not None
        assert runtime._macro_plan.steps[0].completed_repeats == 1
        assert runtime._recent_hima_actions(1) == ["Pylon"]
        await runtime.close()

    asyncio.run(exercise())

    recovered = _store(tmp_path)
    updates = recovered.events_of_type(
        "cortex-run", "episode-1", "macro_step_updated"
    )
    executions = recovered.events_of_type("cortex-run", "episode-1", "execution")
    assert len(updates) == 1
    assert len(executions) == 1
    recovered.close()


def test_refresh_started_before_old_plan_outcome_is_discarded_and_reissued(
    tmp_path: Path,
) -> None:
    client = _FakeMacroClient(
        ["Actions: ['Pylon', 'Pylon']", "Actions: ['Gateway']"]
    )
    store = _store(tmp_path)
    runtime = CortexRuntimeEngine(
        config=_config(tmp_path),
        store=store,
        provider=FakeProvider(),
        macro_client=client,
    )

    def successful_report(batch: ActionBatch) -> ExecutionReport:
        command = batch.commands[0]
        return ExecutionReport(
            run_id=batch.run_id,
            episode_id=batch.episode_id,
            step_id=batch.step_id,
            command_id=command.command_id,
            success=True,
            action_name=command.name,
            actor=command.actor,
            source=command.source,
            requested_arguments=command.arguments,
            resolved_arguments=command.arguments,
            status=ExecutionStatus.SUCCEEDED,
            execution_stage=ExecutionStage.EFFECT_VERIFICATION,
        )

    async def exercise() -> None:
        await runtime.start()
        await runtime.tick(_macro_observation(step_id=0, game_loop=0))
        for _ in range(5):
            await asyncio.sleep(0)
        first = await runtime.tick(_macro_observation(step_id=1, game_loop=1))
        first_plan_id = runtime._macro_plan.plan_id if runtime._macro_plan else None
        runtime.record_execution(successful_report(first))

        old_plan_batch = await runtime.tick(
            _macro_observation(step_id=2, game_loop=112)
        )
        assert len(old_plan_batch.commands) == 1
        for _ in range(5):
            await asyncio.sleep(0)
        await runtime.tick(_macro_observation(step_id=3, game_loop=113))
        assert runtime._macro_plan is not None
        assert runtime._macro_plan.plan_id == first_plan_id
        assert len(store.events_of_type("cortex-run", "episode-1", "macro_plan_accepted")) == 1

        runtime.record_execution(successful_report(old_plan_batch))
        await runtime.tick(_macro_observation(step_id=4, game_loop=114))
        assert runtime._macro_plan is not None
        assert runtime._macro_plan.plan_id == first_plan_id
        assert len(store.events_of_type("cortex-run", "episode-1", "macro_plan_accepted")) == 1
        rejected = store.events_of_type(
            "cortex-run", "episode-1", "macro_plan_rejected"
        )
        assert rejected[-1].payload["reason"] == "stale_after_macro_outcome"

        for _ in range(5):
            await asyncio.sleep(0)
        await runtime.tick(
            _macro_observation(step_id=5, game_loop=115, pylon=True)
        )
        assert runtime._macro_plan is not None
        assert runtime._macro_plan.plan_id != first_plan_id
        assert len(store.events_of_type("cortex-run", "episode-1", "macro_plan_accepted")) == 2
        await runtime.close()

    asyncio.run(exercise())


def test_restart_does_not_apply_old_plan_execution_to_latest_plan(tmp_path: Path) -> None:
    config = _config(tmp_path)
    client = _FakeMacroClient(["Actions: ['Pylon']", "Actions: ['Pylon']"])
    first_runtime = CortexRuntimeEngine(
        config=config,
        store=_store(tmp_path),
        provider=FakeProvider(),
        macro_client=client,
    )

    async def create_two_plans() -> str:
        await first_runtime.start()
        await first_runtime.tick(_macro_observation(step_id=0, game_loop=0))
        for _ in range(5):
            await asyncio.sleep(0)
        first_batch = await first_runtime.tick(
            _macro_observation(step_id=1, game_loop=1)
        )
        command = first_batch.commands[0]
        first_runtime.record_execution(
            ExecutionReport(
                run_id=first_batch.run_id,
                episode_id=first_batch.episode_id,
                step_id=first_batch.step_id,
                command_id=command.command_id,
                success=True,
                action_name=command.name,
                actor=command.actor,
                source=command.source,
                requested_arguments=command.arguments,
                resolved_arguments=command.arguments,
                status=ExecutionStatus.SUCCEEDED,
                execution_stage=ExecutionStage.EFFECT_VERIFICATION,
            )
        )
        await first_runtime.tick(_macro_observation(step_id=2, game_loop=112))
        for _ in range(5):
            await asyncio.sleep(0)
        no_actions = _macro_observation(step_id=3, game_loop=113).model_copy(
            update={"available_actions": []}
        )
        await first_runtime.tick(no_actions)
        assert first_runtime._macro_plan is not None
        latest_plan_id = first_runtime._macro_plan.plan_id
        await first_runtime.close()
        return latest_plan_id

    latest_plan_id = asyncio.run(create_two_plans())
    recovered_runtime = CortexRuntimeEngine(
        config=config,
        store=_store(tmp_path),
        provider=FakeProvider(),
        macro_client=_FakeMacroClient(),
    )

    async def recover() -> None:
        await recovered_runtime.start()
        batch = await recovered_runtime.tick(
            _macro_observation(step_id=4, game_loop=114)
        )
        assert recovered_runtime._macro_plan is not None
        assert recovered_runtime._macro_plan.plan_id == latest_plan_id
        assert recovered_runtime._macro_plan.steps[0].completed_repeats == 0
        assert len(batch.commands) == 1
        await recovered_runtime.close()

    asyncio.run(recover())


def test_close_cleans_up_after_an_already_failed_macro_task(tmp_path: Path) -> None:
    client = _FakeMacroClient()
    runtime = CortexRuntimeEngine(
        config=_config(tmp_path),
        store=_store(tmp_path),
        provider=FakeProvider(),
        macro_client=client,
    )

    async def exercise() -> None:
        async def fail() -> HIMALiveProposalResponse:
            raise RuntimeError("detached macro failure")

        runtime._macro_task = asyncio.create_task(fail())
        await asyncio.sleep(0)
        assert runtime._macro_task.done()

        await runtime.close()

    asyncio.run(exercise())

    assert client.closed is True
