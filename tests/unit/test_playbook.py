from __future__ import annotations

from pathlib import Path

from rtscortex.contracts import (
    ActionSource,
    EpisodeOutcome,
    EpisodeResult,
    ExecutionReport,
    ExecutionStage,
    ExecutionStatus,
)
from rtscortex.cortex import GamePhase
from rtscortex.memory import EventStore
from rtscortex.playbook import (
    CortexPlaybookReviewer,
    DecisionQuality,
    LessonStatus,
    PlaybookContext,
    PlaybookQuery,
    PlaybookStore,
)


def _episode_events(root: Path, run_id: str) -> tuple[EventStore, EpisodeResult]:
    store = EventStore(root / f"{run_id}.sqlite3", root / f"{run_id}.jsonl")
    store.append_event(
        run_id=run_id,
        episode_id="episode",
        step_id=0,
        event_type="situation_assessed",
        payload={"phase": "technology"},
    )
    store.append_event(
        run_id=run_id,
        episode_id="episode",
        step_id=1,
        event_type="command_lineage",
        payload={
            "command_id": f"{run_id}:command",
            "macro_plan_id": f"{run_id}:plan",
            "semantic_action": "BUILD STARGATE",
            "lineage": {"source_role": "macro"},
        },
    )
    store.append_event(
        run_id=run_id,
        episode_id="episode",
        step_id=2,
        event_type="execution",
        payload=ExecutionReport(
            run_id=run_id,
            episode_id="episode",
            step_id=2,
            command_id=f"{run_id}:command",
            success=True,
            action_name="Build_Stargate_Screen",
            actor="Builder/Probe-1",
            source=ActionSource.PLANNER,
            requested_arguments=[[65, 90]],
            status=ExecutionStatus.SUCCEEDED,
            execution_stage=ExecutionStage.EFFECT_VERIFICATION,
        ),
    )
    return store, EpisodeResult(
        run_id=run_id,
        episode_id="episode",
        scenario="Simple64",
        seed=0,
        outcome=EpisodeOutcome.VICTORY,
        steps=2,
    )


def test_playbook_promotes_only_repeated_outcome_backed_experience(tmp_path: Path) -> None:
    playbook = PlaybookStore(tmp_path / "playbook.sqlite3")
    reviewer = CortexPlaybookReviewer(playbook, promotion_support=2)

    first_store, first_result = _episode_events(tmp_path, "run-1")
    first_cases, first_lessons = reviewer.review_episode(
        first_store.events_after("run-1", 0, 100, episode_id="episode"),
        first_result,
        agent_race="protoss",
        opponent_race="zerg",
    )
    second_store, second_result = _episode_events(tmp_path, "run-2")
    _, second_lessons = reviewer.review_episode(
        second_store.events_after("run-2", 0, 100, episode_id="episode"),
        second_result,
        agent_race="protoss",
        opponent_race="zerg",
    )

    assert first_cases[0].quality is DecisionQuality.ADVANTAGE_GAINED
    assert first_lessons[0].status is LessonStatus.CANDIDATE
    assert second_lessons[0].status is LessonStatus.PROMOTED
    selection = playbook.retrieve(
        PlaybookQuery(
            context=PlaybookContext(
                agent_race="protoss",
                opponent_race="zerg",
                phase=GamePhase.TECHNOLOGY,
                map_name="Simple64",
            )
        )
    )
    assert selection.lesson_ids == (second_lessons[0].lesson_id,)

    first_store.close()
    second_store.close()
    playbook.close()


def test_playbook_does_not_promote_bridge_failure_as_strategy(tmp_path: Path) -> None:
    playbook = PlaybookStore(tmp_path / "playbook.sqlite3")
    reviewer = CortexPlaybookReviewer(playbook, promotion_support=1)
    store, result = _episode_events(tmp_path, "run")
    # Replace the successful journal with a separate failed command so the reviewer sees
    # both an outcome-backed success and a Bridge diagnostic case.
    store.append_event(
        run_id="run",
        episode_id="episode",
        step_id=3,
        event_type="command_lineage",
        payload={
            "command_id": "failed-command",
            "semantic_action": "BUILD NEXUS",
            "lineage": {"source_role": "macro"},
        },
    )
    store.append_event(
        run_id="run",
        episode_id="episode",
        step_id=4,
        event_type="execution",
        payload=ExecutionReport(
            run_id="run",
            episode_id="episode",
            step_id=4,
            command_id="failed-command",
            success=False,
            action_name="Build_Nexus_Near",
            actor="Builder/Probe-1",
            source=ActionSource.PLANNER,
            status=ExecutionStatus.FAILED,
            execution_stage=ExecutionStage.TRANSLATION,
            failure_code="invalid_expansion_anchor",
        ),
    )

    cases, lessons = reviewer.review_episode(
        store.events_after("run", 0, 100, episode_id="episode"),
        result,
        agent_race="protoss",
        opponent_race="zerg",
    )

    failed = next(case for case in cases if case.command_id == "failed-command")
    assert failed.quality is DecisionQuality.EXECUTION_ERROR
    assert all(lesson.recommended_action != "BUILD NEXUS" for lesson in lessons)

    store.close()
    playbook.close()


def test_playbook_records_rejected_macro_proposal_as_cortex_error(tmp_path: Path) -> None:
    playbook = PlaybookStore(tmp_path / "playbook.sqlite3")
    reviewer = CortexPlaybookReviewer(playbook, promotion_support=1)
    store, result = _episode_events(tmp_path, "run")
    store.append_event(
        run_id="run",
        episode_id="episode",
        step_id=0,
        event_type="macro_plan_rejected",
        payload={
            "classification": "unsupported_by_runtime",
            "reason": "no_runtime_frontier",
            "proposal": {"steps": [{"canonical_action": "RESEARCH PSIONIC STORM"}]},
        },
    )

    cases, _ = reviewer.review_episode(
        store.events_after("run", 0, 100, episode_id="episode"),
        result,
        agent_race="protoss",
        opponent_race="zerg",
    )

    rejected = next(case for case in cases if case.command_id.startswith("proposal:"))
    assert rejected.quality is DecisionQuality.STRATEGIC_ERROR
    assert rejected.failure_owner.value == "cortex"
    assert rejected.semantic_action == "RESEARCH PSIONIC STORM"

    store.close()
    playbook.close()
