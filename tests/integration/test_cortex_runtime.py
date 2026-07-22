from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from rtscortex.config import (
    AgentSettings,
    CortexHIMAEnsembleMemberSettings,
    CortexMacroSettings,
    CortexPlaybookSettings,
    CortexSettings,
    CortexTacticalSettings,
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
from rtscortex.cortex import HIMAEnsemblePolicyClient, SituationAssessment, TacticalIntent
from rtscortex.evaluation import compute_cortex_observability
from rtscortex.memory import EventStore
from rtscortex.playbook import CortexPlaybookReviewer, PlaybookStore
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
from rtscortex.policy.models import (
    PolicyActionAssessment,
    PolicyActionClassification,
)
from rtscortex.providers import FakeProvider
from rtscortex.runtime import CortexRuntimeEngine


class _FakeMacroClient:
    def __init__(
        self,
        output: str | list[str] = "Actions: ['Pylon']",
        *,
        model_id: str = "SNUMPR/Protoss-a",
    ) -> None:
        self.outputs = [output] if isinstance(output, str) else output
        if not self.outputs:
            raise ValueError("fake macro client requires at least one output")
        self.contexts: list[HIMAInputContext] = []
        self.closed = False
        self.model_id = model_id

    async def health(self) -> HIMALiveHealth:
        return HIMALiveHealth(
            model_id=self.model_id,
            model_revision=HIMA_PINNED_REVISIONS[self.model_id],
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


class _TimeoutOnceMacroClient(_FakeMacroClient):
    async def propose(
        self,
        context: HIMAInputContext,
        *,
        request_id: str | None = None,
    ) -> HIMALiveProposalResponse:
        if not self.contexts:
            self.contexts.append(context)
            await asyncio.sleep(0)
            raise TimeoutError("macro request timed out once")
        return await super().propose(context, request_id=request_id)


class _RecoveringMacroSidecar:
    def __init__(self, client: _FakeMacroClient) -> None:
        self.client = client
        self.restart_count = 0
        self.closed = False

    async def start(self) -> HIMALiveHealth:
        return await self.client.health()

    async def restart(self) -> HIMALiveHealth:
        self.restart_count += 1
        return await self.client.health()

    async def close(self) -> None:
        self.closed = True


class _EmptyShadowTacticalProvider:
    provider_id = "empty-shadow-tactical"
    provider_version = "1.0"

    def evaluate(
        self,
        observation: ObservationEnvelope,
        situation: SituationAssessment,
    ) -> list[TacticalIntent]:
        del observation, situation
        return []


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


def test_cortex_runtime_records_three_specialist_race_brain_cycle(tmp_path: Path) -> None:
    base = _config(tmp_path, macro=False)
    members = [
        CortexHIMAEnsembleMemberSettings.model_validate(
            {"candidate": f"protoss-{cluster}", "model_path": f"/tmp/{cluster}"}
        )
        for cluster in ("a", "b", "c")
    ]
    config = base.model_copy(
        update={
            "cortex": base.cortex.model_copy(
                update={
                    "macro": CortexMacroSettings(
                        kind="hima_ensemble",
                        ensemble_members=members,
                        allow_unlicensed_weights=True,
                    )
                }
            )
        }
    )
    client = HIMAEnsemblePolicyClient(
        {
            "a": _FakeMacroClient(
                "Actions: ['Pylon']",
                model_id="SNUMPR/Protoss-a",
            ),
            "b": _FakeMacroClient(
                "Actions: ['Gateway']",
                model_id="SNUMPR/Protoss-b",
            ),
            "c": _FakeMacroClient(
                "Actions: ['Stargate']",
                model_id="SNUMPR/Protoss-c",
            ),
        },
        race="protoss",
    )
    store = _store(tmp_path)
    runtime = CortexRuntimeEngine(
        config=config,
        store=store,
        provider=FakeProvider(),
        macro_client=client,
    )

    async def exercise() -> None:
        await runtime.start()
        await runtime.tick(_macro_observation(step_id=0, game_loop=0))
        assert runtime._macro_task is not None
        await runtime._macro_task
        batch = await runtime.tick(_macro_observation(step_id=1, game_loop=1))
        assert [command.name for command in batch.commands] == ["Build_Pylon_Screen"]

    asyncio.run(exercise())
    coordinated = store.events_of_type("cortex-run", "episode-1", "race_brain_coordinated")
    assert len(coordinated) == 1
    assert coordinated[0].payload["selected_member_id"] == "hima-protoss-a"
    assert len(coordinated[0].payload["members"]) == 3
    asyncio.run(runtime.close())


def test_cortex_runtime_dispatches_proactive_tactical_focus_fire(tmp_path: Path) -> None:
    runtime = CortexRuntimeEngine(
        config=_config(tmp_path, macro=False),
        store=_store(tmp_path),
        provider=FakeProvider(),
    )
    observation = ObservationEnvelope(
        run_id="cortex-run",
        episode_id="episode-1",
        step_id=0,
        game_loop=0,
        state=SC2State(
            economy=EconomyState(army_supply=4, supply_used=16, supply_cap=23),
            own_units=[UnitState(unit_id="0x10", unit_type="Adept", alliance="self")],
            visible_enemies=[UnitState(unit_id="0x20", unit_type="Zergling", alliance="enemy")],
        ),
        available_actions=[
            AvailableAction(
                name="Attack_Unit",
                argument_names=["tag"],
                argument_types=[ActionArgumentType.TAG],
                actor_scopes=["CombatGroup/Adept-1"],
                argument_candidates=[["0x20"]],
            )
        ],
    )

    batch = asyncio.run(runtime.tick(observation))

    assert len(batch.commands) == 1
    assert batch.commands[0].name == "Attack_Unit"
    assert batch.commands[0].arguments == ["0x20"]
    assert batch.commands[0].source is ActionSource.PLANNER
    lineage = runtime._command_lineages[batch.commands[0].command_id]
    assert lineage.source_role.value == "tactical"
    assert lineage.responsibility == "focus_fire"
    assert lineage.strategic_intent_id is not None
    assert lineage.arbiter_mode == "shadow"
    assert lineage.intent_decision == "selected"
    asyncio.run(runtime.close())


def test_tactical_shadow_records_without_changing_active_action(tmp_path: Path) -> None:
    config = _config(tmp_path, macro=False)
    config = config.model_copy(
        update={
            "cortex": config.cortex.model_copy(
                update={"tactical": CortexTacticalSettings(kind="model_shadow")}
            )
        }
    )
    store = _store(tmp_path)
    runtime = CortexRuntimeEngine(
        config=config,
        store=store,
        provider=FakeProvider(),
        shadow_tactical_provider=_EmptyShadowTacticalProvider(),
    )
    observation = ObservationEnvelope(
        run_id="cortex-run",
        episode_id="episode-1",
        step_id=0,
        game_loop=0,
        state=SC2State(
            economy=EconomyState(army_supply=4, supply_used=16, supply_cap=23),
            own_units=[UnitState(unit_id="0x10", unit_type="Adept", alliance="self")],
            visible_enemies=[UnitState(unit_id="0x20", unit_type="Zergling", alliance="enemy")],
        ),
        available_actions=[
            AvailableAction(
                name="Attack_Unit",
                argument_names=["tag"],
                argument_types=[ActionArgumentType.TAG],
                actor_scopes=["CombatGroup/Adept-1"],
                argument_candidates=[["0x20"]],
            )
        ],
    )

    batch = asyncio.run(runtime.tick(observation))

    assert [command.name for command in batch.commands] == ["Attack_Unit"]
    shadows = store.events_of_type("cortex-run", "episode-1", "tactical_policy_shadow")
    assert len(shadows) == 1
    assert shadows[0].payload["shadow_intents"] == []
    asyncio.run(runtime.close())


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


def test_slow_hima_plan_ttl_starts_at_acceptance_game_loop(tmp_path: Path) -> None:
    client = _BlockingFirstMacroClient()
    store = _store(tmp_path)
    runtime = CortexRuntimeEngine(
        config=_config(tmp_path),
        store=store,
        provider=FakeProvider(),
        macro_client=client,
    )

    async def exercise() -> None:
        await runtime.start()
        first = await runtime.tick(_macro_observation(step_id=0, game_loop=0))
        assert first.planner_pending is True
        assert runtime._macro_task is not None

        client.release_first.set()
        await runtime._macro_task
        accepted = await runtime.tick(_macro_observation(step_id=1, game_loop=500))

        assert [command.name for command in accepted.commands] == ["Build_Pylon_Screen"]
        assert runtime._macro_plan is not None
        assert runtime._macro_plan.created_game_loop == 500
        assert runtime._macro_plan.expires_game_loop == 948
        event = store.events_of_type("cortex-run", "episode-1", "macro_plan_accepted")[-1]
        assert event.payload["proposal_source_game_loop"] == 0
        assert event.payload["accepted_game_loop"] == 500
        assert event.payload["acceptance_delay_game_loops"] == 500
        await runtime.close()

    asyncio.run(exercise())


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
            "environment": config.environment.model_copy(update={"pause_until_first_plan": True})
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


def test_missing_prerequisite_plan_is_retained_without_hot_loop(
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
        deferred = await runtime.tick(blocked.model_copy(update={"step_id": 1, "game_loop": 1}))
        assert deferred.commands == []
        assert deferred.planner_pending is False
        assert runtime._macro_plan is not None
        assert runtime._macro_plan.steps[0].status.value == "deferred"
        assert runtime._macro_plan.steps[0].reason == "missing_prerequisite_pylon"
        assert runtime._macro_plan_frozen is False
        assert runtime._urgent_replan_requested is False
        await runtime.tick(blocked.model_copy(update={"step_id": 2, "game_loop": 2}))
        assert len(client.contexts) == 1
        assert not store.events_of_type("cortex-run", "episode-1", "macro_plan_rejected")
        accepted = store.events_of_type("cortex-run", "episode-1", "macro_plan_accepted")
        assert accepted[-1].payload["runtime_frontier"] == "Build_Gateway_Screen"
        deferred_events = store.events_of_type(
            "cortex-run", "episode-1", "macro_frontier_deferred"
        )
        assert len(deferred_events) == 1
        assert deferred_events[0].payload["reason"] == "missing_prerequisite_pylon"
        assert not store.events_of_type("cortex-run", "episode-1", "specialist_failed")
        await runtime.close()

    asyncio.run(exercise())


def test_missing_technology_prerequisite_is_closed_by_technology_agent(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    runtime = CortexRuntimeEngine(
        config=_config(tmp_path),
        store=store,
        provider=FakeProvider(),
        macro_client=_FakeMacroClient("Actions: ['Stargate']"),
    )
    initial = ObservationEnvelope(
        run_id="cortex-run",
        episode_id="episode-1",
        step_id=0,
        game_loop=0,
        state=SC2State(
            economy=EconomyState(
                minerals=200,
                vespene=200,
                supply_used=12,
                supply_cap=23,
                workers=12,
            ),
            own_structures=[
                UnitState(unit_id="0x1", unit_type="Pylon", alliance="self"),
                UnitState(
                    unit_id="0x2",
                    unit_type="Gateway",
                    alliance="self",
                    status="constructing",
                ),
            ],
        ),
        available_actions=[],
    )
    gateway_complete = initial.model_copy(
        update={
            "step_id": 2,
            "game_loop": 2,
            "state": initial.state.model_copy(
                update={
                    "own_structures": [
                        UnitState(unit_id="0x1", unit_type="Pylon", alliance="self"),
                        UnitState(unit_id="0x2", unit_type="Gateway", alliance="self"),
                    ]
                }
            ),
            "available_actions": [
                AvailableAction(
                    name="Build_CyberneticsCore_Screen",
                    argument_names=["screen"],
                    argument_types=[ActionArgumentType.POSITION],
                    actor_scopes=["Builder/Probe-1"],
                    argument_candidates=[[[70, 90]]],
                )
            ],
        }
    )

    async def exercise() -> None:
        await runtime.start()
        await runtime.tick(initial)
        for _ in range(5):
            await asyncio.sleep(0)
        waiting = await runtime.tick(initial.model_copy(update={"step_id": 1, "game_loop": 1}))
        assert waiting.commands == []
        assert runtime._macro_plan is not None
        assert runtime._macro_plan_frozen is False
        assert not store.events_of_type("cortex-run", "episode-1", "macro_plan_rejected")

        batch = await runtime.tick(gateway_complete)

        assert [command.name for command in batch.commands] == ["Build_CyberneticsCore_Screen"]
        assert runtime._macro_plan.steps[0].semantic_action == "BUILD STARGATE"
        assert runtime._macro_plan.steps[0].status.value == "deferred"
        preemptions = store.events_of_type("cortex-run", "episode-1", "macro_frontier_preempted")
        assert preemptions[-1].payload["reason"] == "prerequisite_closure"
        role_events = store.events_of_type("cortex-run", "episode-1", "role_intent_emitted")
        assert role_events[-1].payload["intent"]["role"] == "technology"
        lineage = store.events_of_type("cortex-run", "episode-1", "command_lineage")
        assert lineage[-1].payload["macro_step_ordinal"] is None
        await runtime.close()

    asyncio.run(exercise())


def test_resource_deferred_frontier_satisfies_required_startup_barrier(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    config = config.model_copy(
        update={
            "environment": config.environment.model_copy(update={"pause_until_first_plan": True})
        }
    )
    observation = _macro_observation(step_id=0, game_loop=0)
    observation = observation.model_copy(
        update={
            "state": observation.state.model_copy(
                update={"economy": observation.state.economy.model_copy(update={"minerals": 0})}
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


def test_supply_emergency_pylon_preempts_blocked_technology_frontier(
    tmp_path: Path,
) -> None:
    client = _FakeMacroClient("Actions: ['Stargate', 'Pylon']")
    runtime = CortexRuntimeEngine(
        config=_config(tmp_path),
        store=_store(tmp_path),
        provider=FakeProvider(),
        macro_client=client,
    )
    observation = _macro_observation(step_id=0, game_loop=0).model_copy(
        update={
            "state": _macro_observation(step_id=0, game_loop=0).state.model_copy(
                update={
                    "economy": EconomyState(
                        minerals=250,
                        vespene=0,
                        supply_used=14,
                        supply_cap=15,
                        workers=14,
                    )
                }
            )
        }
    )

    async def exercise() -> None:
        await runtime.start()
        await runtime.tick(observation)
        for _ in range(5):
            await asyncio.sleep(0)
        batch = await runtime.tick(observation.model_copy(update={"step_id": 1, "game_loop": 1}))

        assert [command.name for command in batch.commands] == ["Build_Pylon_Screen"]
        assert runtime._macro_plan is not None
        assert runtime._macro_plan_frozen is False
        preemptions = runtime.store.events_of_type(
            "cortex-run", "episode-1", "macro_frontier_preempted"
        )
        assert preemptions[-1].payload["reason"] == "supply_emergency"
        assert preemptions[-1].payload["blocked_action"] == "BUILD STARGATE"
        await runtime.close()

    asyncio.run(exercise())


def test_macro_skips_redundant_pylon_when_supply_headroom_is_already_large(
    tmp_path: Path,
) -> None:
    runtime = CortexRuntimeEngine(
        config=_config(tmp_path),
        store=_store(tmp_path),
        provider=FakeProvider(),
        macro_client=_FakeMacroClient("Actions: ['Pylon']"),
    )
    observation = _macro_observation(step_id=0, game_loop=0).model_copy(
        update={
            "state": _macro_observation(step_id=0, game_loop=0).state.model_copy(
                update={
                    "economy": EconomyState(
                        minerals=500,
                        supply_used=12,
                        supply_cap=31,
                        workers=12,
                    )
                }
            )
        }
    )

    async def exercise() -> None:
        await runtime.start()
        await runtime.tick(observation)
        for _ in range(5):
            await asyncio.sleep(0)
        batch = await runtime.tick(observation.model_copy(update={"step_id": 1, "game_loop": 1}))

        assert batch.commands == []
        assert runtime._macro_plan is not None
        assert runtime._macro_plan.steps[0].status.value == "obsolete"
        events = runtime.store.events_of_type("cortex-run", "episode-1", "macro_step_deduplicated")
        assert events[-1].payload["free_supply"] == 19
        await runtime.close()

    asyncio.run(exercise())


def test_macro_defers_duplicate_supply_provider_while_one_is_constructing(
    tmp_path: Path,
) -> None:
    runtime = CortexRuntimeEngine(
        config=_config(tmp_path),
        store=_store(tmp_path),
        provider=FakeProvider(),
        macro_client=_FakeMacroClient("Actions: ['Pylon']"),
    )
    base = _macro_observation(step_id=0, game_loop=0)
    observation = base.model_copy(
        update={
            "state": base.state.model_copy(
                update={
                    "economy": EconomyState(
                        minerals=500,
                        supply_used=15,
                        supply_cap=15,
                        workers=12,
                    ),
                    "own_structures": [
                        UnitState(
                            unit_id="0x1",
                            unit_type="Pylon",
                            alliance="self",
                            status="constructing",
                        )
                    ],
                }
            )
        }
    )

    async def exercise() -> None:
        await runtime.start()
        await runtime.tick(observation)
        for _ in range(5):
            await asyncio.sleep(0)
        batch = await runtime.tick(
            observation.model_copy(update={"step_id": 1, "game_loop": 1})
        )

        assert batch.commands == []
        assert runtime._macro_plan is not None
        assert runtime._macro_plan.steps[0].status.value == "deferred"
        events = runtime.store.events_of_type(
            "cortex-run", "episode-1", "macro_structure_deferred"
        )
        assert len(events) == 1
        assert events[0].payload["reason"] == "same_structure_in_progress"
        assert events[0].payload["target_structure"] == "Pylon"
        await runtime.close()

    asyncio.run(exercise())


def test_macro_defers_duplicate_tech_structure_while_one_is_constructing(
    tmp_path: Path,
) -> None:
    runtime = CortexRuntimeEngine(
        config=_config(tmp_path),
        store=_store(tmp_path),
        provider=FakeProvider(),
        macro_client=_FakeMacroClient("Actions: ['Gateway']"),
    )
    observation = ObservationEnvelope(
        run_id="cortex-run",
        episode_id="episode-1",
        step_id=0,
        game_loop=0,
        state=SC2State(
            economy=EconomyState(
                minerals=500,
                supply_used=12,
                supply_cap=23,
                workers=12,
            ),
            own_structures=[
                UnitState(unit_id="0x1", unit_type="Pylon", alliance="self"),
                UnitState(
                    unit_id="0x2",
                    unit_type="Gateway",
                    alliance="self",
                    status="constructing",
                ),
            ],
        ),
        available_actions=[
            AvailableAction(
                name="Build_Gateway_Screen",
                argument_names=["screen"],
                argument_types=[ActionArgumentType.POSITION],
                actor_scopes=["Builder/Probe-1"],
                argument_candidates=[[[70, 90]]],
            )
        ],
    )

    async def exercise() -> None:
        await runtime.start()
        await runtime.tick(observation)
        for _ in range(5):
            await asyncio.sleep(0)
        batch = await runtime.tick(
            observation.model_copy(update={"step_id": 1, "game_loop": 1})
        )

        assert batch.commands == []
        events = runtime.store.events_of_type(
            "cortex-run", "episode-1", "macro_structure_deferred"
        )
        assert len(events) == 1
        assert events[0].payload["target_structure"] == "Gateway"
        await runtime.close()

    asyncio.run(exercise())


def test_redundant_pylon_skip_advances_to_next_legal_step_in_same_tick(
    tmp_path: Path,
) -> None:
    runtime = CortexRuntimeEngine(
        config=_config(tmp_path),
        store=_store(tmp_path),
        provider=FakeProvider(),
        macro_client=_FakeMacroClient("Actions: ['Pylon', 'Gateway']"),
    )
    observation = ObservationEnvelope(
        run_id="cortex-run",
        episode_id="episode-1",
        step_id=0,
        game_loop=0,
        state=SC2State(
            economy=EconomyState(
                minerals=500,
                supply_used=12,
                supply_cap=31,
                workers=12,
            ),
            own_structures=[UnitState(unit_id="0x1", unit_type="Pylon", alliance="self")],
        ),
        available_actions=[
            AvailableAction(
                name="Build_Pylon_Screen",
                argument_names=["screen"],
                argument_types=[ActionArgumentType.POSITION],
                actor_scopes=["Builder/Probe-1"],
                argument_candidates=[[[65, 90]]],
            ),
            AvailableAction(
                name="Build_Gateway_Screen",
                argument_names=["screen"],
                argument_types=[ActionArgumentType.POSITION],
                actor_scopes=["Builder/Probe-1"],
                argument_candidates=[[[70, 90]]],
            ),
        ],
    )

    async def exercise() -> None:
        await runtime.start()
        await runtime.tick(observation)
        for _ in range(5):
            await asyncio.sleep(0)
        batch = await runtime.tick(observation.model_copy(update={"step_id": 1, "game_loop": 1}))

        assert [command.name for command in batch.commands] == ["Build_Gateway_Screen"]
        deduplicated = runtime.store.events_of_type(
            "cortex-run", "episode-1", "macro_step_deduplicated"
        )
        assert deduplicated[-1].payload["semantic_action"] == "BUILD PYLON"
        await runtime.close()

    asyncio.run(exercise())


def test_gas_blocked_stargate_uses_legal_zealot_fallback(tmp_path: Path) -> None:
    client = _FakeMacroClient("Actions: ['Stargate', 'Zealot', 'Nexus']")
    runtime = CortexRuntimeEngine(
        config=_config(tmp_path),
        store=_store(tmp_path),
        provider=FakeProvider(),
        macro_client=client,
    )
    observation = ObservationEnvelope(
        run_id="cortex-run",
        episode_id="episode-1",
        step_id=0,
        game_loop=0,
        state=SC2State(
            economy=EconomyState(
                minerals=500,
                vespene=0,
                supply_used=20,
                supply_cap=31,
                workers=18,
            ),
            own_structures=[
                UnitState(unit_id="0x1", unit_type="Gateway", alliance="self"),
                UnitState(unit_id="0x2", unit_type="CyberneticsCore", alliance="self"),
            ],
        ),
        available_actions=[
            AvailableAction(
                name="Train_Zealot",
                actor_scopes=["Developer/Empty"],
                argument_candidates=None,
            ),
            AvailableAction(
                name="Build_Nexus_Near",
                argument_names=["tag"],
                argument_types=[ActionArgumentType.TAG],
                actor_scopes=["Builder/Probe-1"],
                argument_candidates=[["0x99"]],
            ),
        ],
    )

    async def exercise() -> None:
        await runtime.start()
        await runtime.tick(observation)
        for _ in range(5):
            await asyncio.sleep(0)
        batch = await runtime.tick(observation.model_copy(update={"step_id": 1, "game_loop": 1}))

        assert [command.name for command in batch.commands] == ["Train_Zealot"]
        preemptions = runtime.store.events_of_type(
            "cortex-run", "episode-1", "macro_frontier_preempted"
        )
        assert preemptions[-1].payload["reason"] == "resource_fallback"
        assert preemptions[-1].payload["blocked_reason"] == "insufficient_vespene"
        await runtime.close()

    asyncio.run(exercise())


def test_terran_gas_blocked_addon_builds_first_refinery(tmp_path: Path) -> None:
    base = _config(tmp_path)
    config = base.model_copy(
        update={"environment": base.environment.model_copy(update={"agent_race": "terran"})}
    )
    runtime = CortexRuntimeEngine(
        config=config,
        store=_store(tmp_path),
        provider=FakeProvider(),
        macro_client=_FakeMacroClient(
            "Actions: ['BarracksReactor']",
            model_id="SNUMPR/Terran-a",
        ),
    )
    observation = ObservationEnvelope(
        run_id="cortex-run",
        episode_id="episode-1",
        step_id=0,
        game_loop=0,
        state=SC2State(
            economy=EconomyState(
                minerals=200,
                vespene=0,
                supply_used=15,
                supply_cap=23,
                workers=12,
            ),
            own_structures=[
                UnitState(unit_id="0x1", unit_type="CommandCenter", alliance="self"),
                UnitState(unit_id="0x2", unit_type="Barracks", alliance="self"),
            ],
        ),
        available_actions=[
            AvailableAction(
                name="Build_Refinery_Near",
                argument_names=["tag"],
                argument_types=[ActionArgumentType.TAG],
                actor_scopes=["Builder/SCV-1"],
                argument_candidates=[["0x99"]],
            )
        ],
    )
    proposal = HIMAProposalParser(race="terran").parse("Actions: ['BarracksReactor']")
    blocked = PolicyActionAssessment(
        ordinal=0,
        source_action="BUILD BARRACKSREACTOR",
        runtime_action="Build_BarracksReactor",
        classification=PolicyActionClassification.MAPPED_DEFERRED,
        reason_code="insufficient_vespene",
        is_runtime_frontier=True,
    )

    fallback = runtime._fallback_frontier(proposal, observation, blocked)

    assert fallback is not None
    assert fallback.source_action == "BUILD REFINERY"
    assert fallback.runtime_action == "Build_Refinery_Near"
    assert runtime._fallback_reason(blocked, fallback, observation) == "prerequisite_closure"


@pytest.mark.parametrize(
    ("supply_used", "expected_action"),
    [(28, "Build_Pylon_Screen"), (20, "Build_Nexus_Near")],
)
def test_gas_blocked_stargate_uses_supply_or_expansion_fallback(
    tmp_path: Path,
    supply_used: int,
    expected_action: str,
) -> None:
    client = _FakeMacroClient("Actions: ['Stargate', 'Pylon', 'Nexus']")
    runtime = CortexRuntimeEngine(
        config=_config(tmp_path),
        store=_store(tmp_path),
        provider=FakeProvider(),
        macro_client=client,
    )
    observation = ObservationEnvelope(
        run_id="cortex-run",
        episode_id="episode-1",
        step_id=0,
        game_loop=0,
        state=SC2State(
            economy=EconomyState(
                minerals=500,
                vespene=0,
                supply_used=supply_used,
                supply_cap=31,
                workers=18,
            ),
            own_structures=[UnitState(unit_id="0x2", unit_type="CyberneticsCore", alliance="self")],
        ),
        available_actions=[
            AvailableAction(
                name="Build_Pylon_Screen",
                argument_names=["screen"],
                argument_types=[ActionArgumentType.POSITION],
                actor_scopes=["Builder/Probe-1"],
                argument_candidates=[[[65, 90]]],
            ),
            AvailableAction(
                name="Build_Nexus_Near",
                argument_names=["tag"],
                argument_types=[ActionArgumentType.TAG],
                actor_scopes=["Builder/Probe-1"],
                argument_candidates=[["0x99"]],
            ),
        ],
    )

    async def exercise() -> None:
        await runtime.start()
        await runtime.tick(observation)
        for _ in range(5):
            await asyncio.sleep(0)
        batch = await runtime.tick(observation.model_copy(update={"step_id": 1, "game_loop": 1}))
        assert [command.name for command in batch.commands] == [expected_action]
        await runtime.close()

    asyncio.run(exercise())


def test_saturated_main_base_gas_frontier_expands_instead_of_stalling(
    tmp_path: Path,
) -> None:
    runtime = CortexRuntimeEngine(
        config=_config(tmp_path),
        store=_store(tmp_path),
        provider=FakeProvider(),
        macro_client=_FakeMacroClient(
            "Actions: ['Assimilator', 'Assimilator', 'Assimilator', 'Nexus', 'Zealot']"
        ),
    )
    observation = ObservationEnvelope(
        run_id="cortex-run",
        episode_id="episode-1",
        step_id=0,
        game_loop=0,
        state=SC2State(
            economy=EconomyState(
                minerals=900,
                vespene=300,
                supply_used=20,
                supply_cap=47,
                workers=20,
            ),
            own_structures=[
                UnitState(unit_id="0x1", unit_type="Nexus", alliance="self"),
                UnitState(unit_id="0x2", unit_type="Assimilator", alliance="self"),
                UnitState(unit_id="0x3", unit_type="Assimilator", alliance="self"),
                UnitState(unit_id="0x4", unit_type="Gateway", alliance="self"),
            ],
        ),
        available_actions=[
            AvailableAction(
                name="Build_Nexus_Near",
                argument_names=["tag"],
                argument_types=[ActionArgumentType.TAG],
                actor_scopes=["Builder/Probe-1"],
                argument_candidates=[["0x99"]],
            ),
            AvailableAction(
                name="Train_Zealot",
                actor_scopes=["Developer/Empty"],
                argument_candidates=None,
            ),
        ],
    )

    proposal = HIMAProposalParser().parse(
        "Actions: ['Assimilator', 'Assimilator', 'Assimilator', 'Nexus', 'Zealot']"
    )
    blocked = PolicyActionAssessment(
        ordinal=2,
        source_action="BUILD ASSIMILATOR",
        runtime_action="Build_Assimilator_Near",
        classification=PolicyActionClassification.MAPPED_DEFERRED,
        reason_code="action_unavailable_now",
        is_runtime_frontier=True,
    )

    fallback = runtime._fallback_frontier(proposal, observation, blocked)

    assert fallback is not None
    assert fallback.source_action == "BUILD NEXUS"
    assert fallback.runtime_action == "Build_Nexus_Near"


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
        assert [context.observation.episode_id for context in client.contexts] == ["episode-1"]
        for _ in range(5):
            await asyncio.sleep(0)
        await runtime.tick(second.model_copy(update={"step_id": 1, "game_loop": 1}))
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
        dispatched = await runtime.tick(_macro_observation(step_id=1, game_loop=1))
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


def test_completed_episode_emits_strategic_consequence_and_review_summary(
    tmp_path: Path,
) -> None:
    base = _config(tmp_path, macro=False)
    config = base.model_copy(
        update={
            "cortex": base.cortex.model_copy(
                update={
                    "playbook": CortexPlaybookSettings(
                        enabled=True,
                        database_path=tmp_path / "playbook.sqlite3",
                    )
                }
            )
        }
    )
    playbook = PlaybookStore(tmp_path / "playbook.sqlite3")
    runtime = CortexRuntimeEngine(
        config=config,
        store=_store(tmp_path),
        provider=FakeProvider(),
        macro_client=None,
        playbook_store=playbook,
        playbook_reviewer=CortexPlaybookReviewer(playbook),
    )
    for step_id, game_loop in ((10, 1_000), (20, 1_224)):
        runtime.store.append_event(
            run_id="cortex-run",
            episode_id="episode-1",
            step_id=step_id,
            event_type="situation_assessed",
            payload={
                "game_loop": game_loop,
                "phase": "combat",
                "threat_level": "high",
                "economy_status": "stable",
                "army_readiness": "ready",
                "own_force": {"estimated_resource_value": 800, "total_units": 8},
                "visible_enemy_force": {
                    "estimated_resource_value": 700,
                    "total_units": 7,
                },
                "bases": {"own_base_count": 2, "own_production_capacity": 4},
                "scouting": {"enemy_visible": True},
            },
        )

    runtime.end_episode(
        EpisodeResult(
            run_id="cortex-run",
            episode_id="episode-1",
            scenario="Simple64",
            seed=0,
            outcome=EpisodeOutcome.DEFEAT,
            steps=20,
        )
    )

    consequences = runtime.store.events_of_type(
        "cortex-run",
        "episode-1",
        "strategic_consequence_attributed",
    )
    reviews = runtime.store.events_of_type(
        "cortex-run",
        "episode-1",
        "postgame_review_completed",
    )
    assert len(consequences) == 1
    assert consequences[0].payload["consequence_type"] == "threat_unanswered"
    assert reviews[0].payload["strategic_consequence_count"] == 1
    assert reviews[0].payload["strategic_consequence_counts"] == {
        "threat_unanswered": 1
    }
    asyncio.run(runtime.close())


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
        timeout_batch = await runtime.tick(_macro_observation(step_id=1, game_loop=1))
        assert timeout_batch.idle_reason is not None
        assert timeout_batch.idle_reason.value == "planner_timeout"
        assert runtime._macro_requests_suspended is True

        with pytest.raises(RuntimeError, match="required HIMA macro specialist is suspended"):
            await runtime.tick(second_episode)

        await runtime.close()

    asyncio.run(exercise())


def test_timed_out_macro_specialist_restarts_and_resumes_requests(tmp_path: Path) -> None:
    client = _TimeoutOnceMacroClient()
    sidecar = _RecoveringMacroSidecar(client)
    runtime = CortexRuntimeEngine(
        config=_config(tmp_path),
        store=_store(tmp_path),
        provider=FakeProvider(),
        macro_client=client,
        macro_sidecar=sidecar,
    )

    async def exercise() -> None:
        await runtime.start()
        await runtime.tick(_macro_observation(step_id=0, game_loop=0))
        for _ in range(5):
            await asyncio.sleep(0)
        await runtime.tick(_macro_observation(step_id=1, game_loop=1))
        for _ in range(10):
            await asyncio.sleep(0)

        assert sidecar.restart_count == 1
        assert runtime._macro_requests_suspended is False

        retry = await runtime.tick(_macro_observation(step_id=2, game_loop=2))
        assert retry.planner_pending is True
        for _ in range(5):
            await asyncio.sleep(0)
        resumed = await runtime.tick(_macro_observation(step_id=3, game_loop=3))
        assert [command.name for command in resumed.commands] == ["Build_Pylon_Screen"]
        recovered = runtime.store.events_of_type("cortex-run", "episode-1", "specialist_recovered")
        assert len(recovered) == 1
        assert recovered[0].payload["restart_attempt"] == 1
        await runtime.close()

    asyncio.run(exercise())
    assert sidecar.closed is True


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
            own_units=[UnitState(unit_id="0x10", unit_type="Adept", alliance="self")],
            visible_enemies=[UnitState(unit_id="0x20", unit_type="Zergling", alliance="enemy")],
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
    candidate_sets = recovered.events_of_type("reflex-run", "episode-1", "candidate_set_built")
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
    updates = recovered.events_of_type("cortex-run", "episode-1", "macro_step_updated")
    executions = recovered.events_of_type("cortex-run", "episode-1", "execution")
    assert len(updates) == 1
    assert len(executions) == 1
    recovered.close()


def test_completed_refresh_is_accepted_while_old_plan_command_is_inflight(
    tmp_path: Path,
) -> None:
    client = _FakeMacroClient(["Actions: ['Pylon', 'Pylon']", "Actions: ['Gateway']"])
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

        old_plan_batch = await runtime.tick(_macro_observation(step_id=2, game_loop=112))
        assert len(old_plan_batch.commands) == 1
        assert runtime._macro_task is not None
        await runtime._macro_task
        await runtime.tick(_macro_observation(step_id=3, game_loop=113))
        assert runtime._macro_plan is not None
        assert runtime._macro_plan.plan_id != first_plan_id
        replacement_plan_id = runtime._macro_plan.plan_id
        assert runtime._macro_inflight_command_id == old_plan_batch.commands[0].command_id
        assert len(store.events_of_type("cortex-run", "episode-1", "macro_plan_accepted")) == 2

        runtime.record_execution(successful_report(old_plan_batch))
        current = _macro_observation(step_id=4, game_loop=114, pylon=True)
        current.available_actions.append(
            AvailableAction(
                name="Build_Gateway_Screen",
                argument_names=["screen"],
                argument_types=[ActionArgumentType.POSITION],
                actor_scopes=["Builder/Probe-1"],
                argument_candidates=[[[70, 90]]],
            )
        )
        await runtime.tick(current)
        assert runtime._macro_plan is not None
        assert runtime._macro_plan.plan_id == replacement_plan_id
        assert len(store.events_of_type("cortex-run", "episode-1", "macro_plan_accepted")) == 2
        revalidated = store.events_of_type("cortex-run", "episode-1", "macro_proposal_revalidated")
        assert revalidated == []
        await runtime.close()

    asyncio.run(exercise())


def test_fixed_macro_start_cadence_is_not_blocked_by_inflight_effect(tmp_path: Path) -> None:
    runtime = CortexRuntimeEngine(
        config=_config(tmp_path),
        store=_store(tmp_path),
        provider=FakeProvider(),
        macro_client=_FakeMacroClient(),
    )
    runtime._last_planner_started_game_loop = 0
    runtime._macro_inflight_command_id = "old-plan-command"

    assert runtime._should_start_macro(_macro_observation(step_id=2, game_loop=112)) is True


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
        first_batch = await first_runtime.tick(_macro_observation(step_id=1, game_loop=1))
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
        batch = await recovered_runtime.tick(_macro_observation(step_id=4, game_loop=114))
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
