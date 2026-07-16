from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from rtscortex.cli.app import app
from rtscortex.contracts import (
    ActionSource,
    ExecutionReport,
    ExecutionStage,
    ExecutionStatus,
)
from rtscortex.cortex import (
    CandidateCompiler,
    CommandLineage,
    DeterministicCandidateExecutor,
    ExecutorCorpusError,
    ExecutorSplit,
    MacroIntent,
    benchmark_executor_corpus,
    build_executor_corpus,
    executor_episode_split,
    load_executor_corpus,
    verify_executor_corpus,
)
from rtscortex.memory import EventStore, read_event_log
from tests.helpers import make_observation


def test_executor_corpus_builds_minimized_episode_isolated_samples(tmp_path: Path) -> None:
    journal = _write_cortex_journal(tmp_path)

    result = build_executor_corpus([journal.parent], tmp_path / "corpus")
    verification = verify_executor_corpus(result.manifest_path, verify_sources=True)
    samples = load_executor_corpus(result.manifest_path)

    assert verification.valid, verification.errors
    assert len(samples) == 2
    assert result.manifest.conservation.selection_events == 3
    assert result.manifest.conservation.included_samples == 2
    assert result.manifest.conservation.excluded_selections == 1
    assert result.manifest.conservation.selected_labels == 1
    assert result.manifest.conservation.abstained_labels == 1
    assert result.manifest.conservation.terminal_outcomes_linked == 1
    assert result.manifest.exclusion_reasons == {"duplicate_selection_id": 1}
    assert not Path(result.manifest.sources[0].journal_path).is_absolute()
    assert result.manifest.duplicates.duplicate_selection_events == 1
    assert result.manifest.distributions.selected_action == {"Attack_Unit": 1}
    assert result.manifest.distributions.selection_status == {
        "abstained": 1,
        "selected": 1,
    }
    assert all(
        sample.split
        is executor_episode_split(
            sample.run_id,
            sample.episode_id,
            seed=result.manifest.split_seed,
        )
        for sample in samples
    )
    assert len(
        {
            sample.split
            for sample in samples
            if sample.episode_id == "episode-selected"
        }
    ) == 1

    selected = next(sample for sample in samples if sample.label.selected_candidate_id)
    assert selected.command_id is not None
    assert selected.command_id.startswith("command:")
    assert selected.has_macro_plan
    assert selected.intent_action_names == ["Attack_Unit"]
    assert selected.terminal_outcome is not None
    assert selected.terminal_outcome.status is ExecutionStatus.SUCCEEDED
    assert selected.observation_features.visible_enemy_counts == {"Zergling": 1}

    encoded = b"".join(path.read_bytes() for path in result.split_paths.values())
    assert b"A compact test observation" not in encoded
    assert b"image_uri" not in encoded
    assert b"candidate_arguments" not in encoded
    assert b'"arguments"' not in encoded
    assert b"0x1" not in encoded
    assert {
        "runtime_observation_fingerprint",
        "runtime_candidate_id",
        "runtime_selection_id",
        "runtime_intent_id",
        "runtime_command_id",
        "runtime_macro_plan_id",
        "runtime_situation_assessment_id",
        "raw_journal_sha256",
    } <= set(result.manifest.redacted_fields)

    persisted = encoded + result.manifest_path.read_bytes()
    for raw_id in _raw_runtime_identifiers(journal):
        assert raw_id.encode() not in persisted
    raw_journal_sha256 = hashlib.sha256(journal.read_bytes()).hexdigest()
    assert raw_journal_sha256.encode() not in persisted


def test_executor_corpus_is_byte_deterministic_and_detects_tampering(
    tmp_path: Path,
) -> None:
    journal = _write_cortex_journal(tmp_path)
    first = build_executor_corpus([journal], tmp_path / "first")
    second = build_executor_corpus([journal], tmp_path / "second")

    for split in ExecutorSplit:
        assert first.split_paths[split].read_bytes() == second.split_paths[split].read_bytes()
    assert first.manifest.corpus_fingerprint == second.manifest.corpus_fingerprint

    populated = next(path for path in first.split_paths.values() if path.stat().st_size)
    payload = json.loads(populated.read_text(encoding="utf-8").splitlines()[0])
    payload["raw_rgb"] = "forbidden"
    populated.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    verification = verify_executor_corpus(first.manifest_path)

    assert not verification.valid
    assert any("SHA256" in error for error in verification.errors)
    assert any("cannot be decoded" in error for error in verification.errors)


def test_corpus_local_ids_depend_only_on_the_safe_projection(tmp_path: Path) -> None:
    first_journal = _write_cortex_journal(tmp_path / "raw-a", enemy_tag="0x1")
    second_journal = _write_cortex_journal(
        tmp_path / "raw-b",
        enemy_tag="0xdeadbeef",
    )

    first = build_executor_corpus([first_journal], tmp_path / "corpus-a")
    second = build_executor_corpus([second_journal], tmp_path / "corpus-b")
    first_samples = {sample.episode_id: sample for sample in first.samples}
    second_samples = {sample.episode_id: sample for sample in second.samples}

    assert first.manifest.sources[0].source_fingerprint == (
        second.manifest.sources[0].source_fingerprint
    )
    for episode_id, first_sample in first_samples.items():
        second_sample = second_samples[episode_id]
        assert first_sample.sample_id == second_sample.sample_id
        assert first_sample.observation_fingerprint == (
            second_sample.observation_fingerprint
        )
        assert first_sample.intent_id == second_sample.intent_id
        assert [candidate.candidate_id for candidate in first_sample.candidates] == [
            candidate.candidate_id for candidate in second_sample.candidates
        ]
        assert first_sample.label.selection_id == second_sample.label.selection_id
        assert first_sample.label.selected_candidate_id == (
            second_sample.label.selected_candidate_id
        )
        assert first_sample.command_id == second_sample.command_id

    assert _raw_runtime_identifiers(first_journal) != _raw_runtime_identifiers(
        second_journal
    )


def test_source_verification_checks_safe_payload_changes(tmp_path: Path) -> None:
    journal = _write_cortex_journal(tmp_path)
    result = build_executor_corpus([journal], tmp_path / "corpus")
    records = [json.loads(line) for line in journal.read_text().splitlines()]
    observation = next(
        record for record in records if record["event_type"] == "observation"
    )
    observation["payload"]["state"]["economy"]["minerals"] += 1
    journal.write_text(
        "".join(
            json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )

    verification = verify_executor_corpus(result.manifest_path, verify_sources=True)

    assert not verification.valid
    assert any("safe structure changed" in error for error in verification.errors)


def test_executor_corpus_rejects_nested_pipeline_identity_mismatch(
    tmp_path: Path,
) -> None:
    journal = _write_cortex_journal(tmp_path)
    records = [json.loads(line) for line in journal.read_text().splitlines()]
    intent = next(
        record
        for record in records
        if record["event_type"] == "intent_emitted"
        and record["episode_id"] == "episode-selected"
    )
    intent["payload"]["intent"]["intent_id"] = "intent-tampered"
    _write_jsonl_records(journal, records)

    result = build_executor_corpus([journal], tmp_path / "corpus")

    assert result.manifest.exclusion_reasons["pipeline_identity_mismatch"] == 1
    assert all(sample.episode_id != "episode-selected" for sample in result.samples)


def test_executor_corpus_rejects_observation_wrapper_identity_mismatch(
    tmp_path: Path,
) -> None:
    journal = _write_cortex_journal(tmp_path)
    records = [json.loads(line) for line in journal.read_text().splitlines()]
    observation = next(
        record
        for record in records
        if record["event_type"] == "observation"
        and record["episode_id"] == "episode-selected"
    )
    observation["payload"]["run_id"] = "stale-run-id"
    _write_jsonl_records(journal, records)

    result = build_executor_corpus([journal], tmp_path / "corpus")

    assert result.manifest.exclusion_reasons["pipeline_identity_mismatch"] == 1
    assert all(sample.episode_id != "episode-selected" for sample in result.samples)


def test_executor_corpus_rejects_execution_action_identity_mismatch(
    tmp_path: Path,
) -> None:
    journal = _write_cortex_journal(tmp_path)
    records = [json.loads(line) for line in journal.read_text().splitlines()]
    execution = next(
        record
        for record in records
        if record["event_type"] == "execution"
        and record["episode_id"] == "episode-selected"
    )
    execution["payload"]["action_name"] = "Train_Oracle"
    _write_jsonl_records(journal, records)

    result = build_executor_corpus([journal], tmp_path / "corpus")

    assert result.manifest.exclusion_reasons["execution_identity_mismatch"] == 1
    assert all(sample.episode_id != "episode-selected" for sample in result.samples)


def test_executor_corpus_rejects_an_empty_export(tmp_path: Path) -> None:
    run_dir = tmp_path / "empty-run"
    store = EventStore(run_dir / "events.sqlite3", run_dir / "events.jsonl")
    try:
        store.append_event(
            run_id="run-empty",
            episode_id="episode-empty",
            step_id=0,
            event_type="episode_started",
            payload={"seed": 0},
        )
    finally:
        store.close()

    with pytest.raises(ExecutorCorpusError, match="no valid executor samples"):
        build_executor_corpus([run_dir], tmp_path / "empty-corpus")


def test_executor_corpus_rejects_overlapping_source_namespaces(tmp_path: Path) -> None:
    journal = _write_cortex_journal(tmp_path)

    with pytest.raises(ExecutorCorpusError, match="duplicate observation identity"):
        build_executor_corpus([journal, journal], tmp_path / "overlap-corpus")


def test_duplicate_detection_uses_semantic_features_across_episode_ids(
    tmp_path: Path,
) -> None:
    episode_by_split: dict[ExecutorSplit, str] = {}
    for ordinal in range(1_000):
        episode_id = f"episode-{ordinal}"
        split = executor_episode_split("run-cortex", episode_id)
        episode_by_split.setdefault(split, episode_id)
        if len(episode_by_split) >= 2:
            break
    first_episode, second_episode = tuple(episode_by_split.values())[:2]
    assert executor_episode_split("run-cortex", first_episode) is not (
        executor_episode_split("run-cortex", second_episode)
    )
    run_dir = tmp_path / "semantic-duplicates"
    store = EventStore(run_dir / "events.sqlite3", run_dir / "events.jsonl")
    try:
        _write_selection(
            store,
            episode_id=first_episode,
            step_id=1,
            include_enemy=True,
            action_name="Attack_Unit",
            command_id=None,
        )
        _write_selection(
            store,
            episode_id=second_episode,
            step_id=1,
            include_enemy=True,
            action_name="Attack_Unit",
            command_id=None,
        )
    finally:
        store.close()

    result = build_executor_corpus([run_dir], tmp_path / "semantic-corpus")
    verification = verify_executor_corpus(result.manifest_path)

    assert verification.valid, verification.errors
    assert result.manifest.duplicates.repeated_observation_fingerprints == 0
    assert result.manifest.duplicates.repeated_semantic_feature_fingerprints == 0
    assert result.manifest.exclusion_reasons == {
        "cross_split_semantic_duplicate": 1,
    }
    assert len(result.samples) == 1
    assert len({sample.split for sample in result.samples}) == 1


def test_saved_candidate_benchmark_reports_agreement_without_live_latency_claim(
    tmp_path: Path,
) -> None:
    journal = _write_cortex_journal(tmp_path)
    result = build_executor_corpus([journal], tmp_path / "corpus")

    benchmark = benchmark_executor_corpus(
        result.manifest_path,
        repetitions=3,
        split="all",
    )

    assert benchmark.benchmark_kind == "saved_candidate_ranking"
    assert benchmark.reconstructs_full_observation is False
    assert benchmark.sample_count == 2
    assert benchmark.measured_calls == 6
    assert benchmark.split == "all"
    assert benchmark.corpus_fingerprint == result.manifest.corpus_fingerprint
    assert benchmark.agreement_count == 1
    assert benchmark.agreement_rate == 1.0
    assert benchmark.overall_agreement_count == 2
    assert benchmark.overall_agreement_rate == 1.0
    assert benchmark.abstain_match_count == 1
    assert benchmark.prediction_distribution == {"abstain": 1, "selected": 1}
    assert benchmark.latency_us_p99 >= 0.0


def test_saved_policy_preserves_live_compile_order_for_tied_candidate_ranks(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "tie-run"
    store = EventStore(run_dir / "events.sqlite3", run_dir / "events.jsonl")
    try:
        _write_selection(
            store,
            episode_id="episode-tie",
            step_id=1,
            include_enemy=True,
            action_name="Attack_Unit",
            command_id="command-tie",
            duplicate_attack_actions=True,
        )
    finally:
        store.close()
    result = build_executor_corpus([run_dir], tmp_path / "tie-corpus")
    samples = load_executor_corpus(result.manifest_path)

    assert len(samples) == 1
    assert [
        candidate.features.compile_ordinal for candidate in samples[0].candidates
    ] == [0, 1]
    assert samples[0].label.selected_candidate_id == samples[0].candidates[0].candidate_id

    benchmark = benchmark_executor_corpus(
        result.manifest_path,
        repetitions=1,
        split="all",
    )

    assert benchmark.agreement_count == 1
    assert benchmark.agreement_rate == 1.0


def test_executor_corpus_cli_build_verify_and_benchmark(tmp_path: Path) -> None:
    journal = _write_cortex_journal(tmp_path)
    output = tmp_path / "cli-corpus"
    runner = CliRunner()

    built = runner.invoke(
        app,
        [
            "executor-corpus",
            "build",
            str(journal.parent),
            "--output-dir",
            str(output),
        ],
    )
    verified = runner.invoke(
        app,
        ["executor-corpus", "verify", str(output / "manifest.json")],
    )
    benchmarked = runner.invoke(
        app,
        [
            "executor-benchmark",
            str(output / "manifest.json"),
            "--repetitions",
            "2",
            "--split",
            "all",
        ],
    )

    assert built.exit_code == 0, built.output
    assert f"Manifest: {output / 'manifest.json'}" in built.output
    assert verified.exit_code == 0, verified.output
    assert json.loads(verified.output)["sample_count"] == 2
    assert benchmarked.exit_code == 0, benchmarked.output
    assert json.loads(benchmarked.output)["measured_calls"] == 4


def _write_cortex_journal(tmp_path: Path, *, enemy_tag: str = "0x1") -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    store = EventStore(run_dir / "events.sqlite3", run_dir / "events.jsonl")
    try:
        selection_payload = _write_selection(
            store,
            episode_id="episode-selected",
            step_id=1,
            include_enemy=True,
            action_name="Attack_Unit",
            command_id="command-selected",
            enemy_tag=enemy_tag,
        )
        store.append_event(
            run_id="run-cortex",
            episode_id="episode-selected",
            step_id=1,
            event_type="executor_selection",
            payload=selection_payload,
        )
        _write_selection(
            store,
            episode_id="episode-abstained",
            step_id=2,
            include_enemy=False,
            action_name="Train_Oracle",
            command_id=None,
            enemy_tag=enemy_tag,
        )
    finally:
        store.close()
    return run_dir / "events.jsonl"


def _write_jsonl_records(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def _write_selection(
    store: EventStore,
    *,
    episode_id: str,
    step_id: int,
    include_enemy: bool,
    action_name: str,
    command_id: str | None,
    enemy_tag: str = "0x1",
    duplicate_attack_actions: bool = False,
) -> dict[str, object]:
    observation = make_observation(
        run_id="run-cortex",
        episode_id=episode_id,
        step_id=step_id,
        game_loop=step_id * 64,
        include_enemy=include_enemy,
    )
    if include_enemy and enemy_tag != "0x1":
        enemy = observation.state.visible_enemies[0].model_copy(
            update={"unit_id": enemy_tag}
        )
        state = observation.state.model_copy(update={"visible_enemies": [enemy]})
        attack = observation.available_actions[0].model_copy(
            update={"argument_candidates": [[enemy_tag]]}
        )
        observation = observation.model_copy(
            update={
                "state": state,
                "available_actions": [attack, *observation.available_actions[1:]],
            }
        )
    if duplicate_attack_actions:
        first_enemy = observation.state.visible_enemies[0]
        second_enemy = first_enemy.model_copy(update={"unit_id": "0x2"})
        first_attack = observation.available_actions[0].model_copy(
            update={"argument_candidates": [[first_enemy.unit_id]]}
        )
        second_attack = first_attack.model_copy(
            update={"argument_candidates": [[second_enemy.unit_id]]}
        )
        observation = observation.model_copy(
            update={
                "state": observation.state.model_copy(
                    update={"visible_enemies": [first_enemy, second_enemy]}
                ),
                "available_actions": [
                    first_attack,
                    second_attack,
                    *observation.available_actions[1:],
                ],
            }
        )
    intent = MacroIntent(
        intent_id=f"intent-{episode_id}",
        run_id=observation.run_id,
        episode_id=observation.episode_id,
        step_id=observation.step_id,
        created_game_loop=observation.game_loop,
        objective="Advance the current macro objective",
        action_names=[action_name],
        actor_scopes=["army"] if action_name == "Attack_Unit" else ["Developer/Empty"],
        ttl_game_loops=112,
        source_id="hima-protoss-a",
        source_version="95348eea",
        macro_plan_id=f"plan-{episode_id.removeprefix('episode-')}",
    )
    context = CandidateCompiler().compile(observation, intent)
    selection = DeterministicCandidateExecutor().select(context).model_copy(
        update={"latency_ms": 0.25}
    )
    store.append_event(
        run_id=observation.run_id,
        episode_id=observation.episode_id,
        step_id=observation.step_id,
        event_type="observation",
        payload=observation,
    )
    store.append_event(
        run_id=observation.run_id,
        episode_id=observation.episode_id,
        step_id=observation.step_id,
        event_type="intent_emitted",
        payload={
            "intent_id": intent.intent_id,
            "role": "macro",
            "intent": intent.model_dump(mode="json"),
        },
    )
    store.append_event(
        run_id=observation.run_id,
        episode_id=observation.episode_id,
        step_id=observation.step_id,
        event_type="candidate_set_built",
        payload={
            "intent_id": intent.intent_id,
            "role": "macro",
            "candidate_count": len(context.candidates),
            "candidates": [candidate.model_dump(mode="json") for candidate in context.candidates],
        },
    )
    selection_payload: dict[str, object] = {
        **selection.model_dump(mode="json"),
        "selected_candidate_id": selection.candidate_id,
        "role": "macro",
        "fallback": False,
    }
    store.append_event(
        run_id=observation.run_id,
        episode_id=observation.episode_id,
        step_id=observation.step_id,
        event_type="executor_selection",
        payload=selection_payload,
    )
    if command_id is None:
        return selection_payload
    assert selection.candidate_id is not None
    command = CandidateCompiler().materialize(context, selection, command_id=command_id)
    lineage = CommandLineage(
        command_id=command.command_id,
        intent_id=intent.intent_id,
        candidate_id=selection.candidate_id,
        selection_id=selection.selection_id,
        source_role=intent.source_role,
        source_id=intent.source_id,
        source_version=intent.source_version,
        executor_id=selection.executor_id,
        executor_version=selection.executor_version,
        macro_plan_id=intent.macro_plan_id,
        selected_game_loop=observation.game_loop,
    )
    store.append_event(
        run_id=observation.run_id,
        episode_id=observation.episode_id,
        step_id=observation.step_id,
        event_type="command_lineage",
        payload={"lineage": lineage.model_dump(mode="json")},
    )
    for status in ("dispatched", "succeeded"):
        store.append_event(
            run_id=observation.run_id,
            episode_id=observation.episode_id,
            step_id=observation.step_id,
            event_type="command_lifecycle",
            payload={
                "command": command.model_dump(mode="json"),
                "status": status,
                "reason": None,
                "game_loop": observation.game_loop,
            },
        )
    store.append_event(
        run_id=observation.run_id,
        episode_id=observation.episode_id,
        step_id=observation.step_id,
        event_type="execution",
        payload=ExecutionReport(
            run_id=observation.run_id,
            episode_id=observation.episode_id,
            step_id=observation.step_id,
            command_id=command.command_id,
            success=True,
            action_name=command.name,
            actor=command.actor,
            source=ActionSource.PLANNER,
            requested_arguments=command.arguments,
            resolved_arguments=command.arguments,
            status=ExecutionStatus.SUCCEEDED,
            execution_stage=ExecutionStage.PYSC2_ACCEPTANCE,
        ),
    )
    return selection_payload


_RAW_RUNTIME_ID_FIELDS = frozenset(
    {
        "candidate_id",
        "selection_id",
        "intent_id",
        "command_id",
        "observation_fingerprint",
        "macro_plan_id",
        "situation_assessment_id",
    }
)


def _raw_runtime_identifiers(journal: Path) -> set[str]:
    identifiers: set[str] = set()

    def collect(value: object) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                if key in _RAW_RUNTIME_ID_FIELDS and isinstance(nested, str):
                    identifiers.add(nested)
                collect(nested)
        elif isinstance(value, list):
            for nested in value:
                collect(nested)

    for event in read_event_log(journal):
        collect(event.payload)
    return identifiers
