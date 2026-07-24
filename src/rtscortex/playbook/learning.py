"""Read completed runs into a separate cross-run CortexPlaybook learning store."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from rtscortex.contracts import EpisodeResult
from rtscortex.memory import StoredEvent, read_event_log
from rtscortex.playbook.promotion import PlaybookPromotionSweep, PromotionSweepResult
from rtscortex.playbook.reviewer import CortexPlaybookReviewer
from rtscortex.playbook.store import PlaybookStore


@dataclass(frozen=True, slots=True)
class LearnedEpisode:
    run_id: str
    episode_id: str
    seed: int
    case_count: int
    lesson_count: int


@dataclass(frozen=True, slots=True)
class PlaybookLearningResult:
    learned_episodes: tuple[LearnedEpisode, ...]
    promotion_sweep: PromotionSweepResult


class PlaybookRunLearner:
    """Review immutable run journals without mutating their frozen Playbook copies."""

    def __init__(
        self,
        store: PlaybookStore,
        *,
        promotion_support: int = 2,
    ) -> None:
        self.store = store
        self.reviewer = CortexPlaybookReviewer(
            store,
            promotion_support=promotion_support,
        )

    def learn(
        self,
        run_directories: tuple[Path, ...],
        *,
        agent_race: str,
        opponent_race: str,
    ) -> PlaybookLearningResult:
        if not run_directories:
            raise ValueError("at least one run directory is required")
        learned: list[LearnedEpisode] = []
        source_directories: dict[str, Path] = {}
        for raw_run_directory in run_directories:
            run_directory = raw_run_directory.expanduser()
            events = _read_completed_run(
                run_directory,
                agent_race=agent_race,
                opponent_race=opponent_race,
            )
            results = _episode_results(events)
            for result in results:
                episode_events = tuple(
                    event
                    for event in events
                    if event.run_id == result.run_id
                    and event.episode_id == result.episode_id
                )
                cases, lessons = self.reviewer.review_episode(
                    episode_events,
                    result,
                    agent_race=agent_race,
                    opponent_race=opponent_race,
                )
                source_directories[result.run_id] = run_directory
                learned.append(
                    LearnedEpisode(
                        run_id=result.run_id,
                        episode_id=result.episode_id,
                        seed=result.seed,
                        case_count=len(cases),
                        lesson_count=len(lessons),
                    )
                )
        sweep = PlaybookPromotionSweep(
            self.store,
            run_directories=source_directories,
        ).run()
        return PlaybookLearningResult(
            learned_episodes=tuple(learned),
            promotion_sweep=sweep,
        )


def _read_completed_run(
    run_directory: Path,
    *,
    agent_race: str,
    opponent_race: str,
) -> tuple[StoredEvent, ...]:
    if not run_directory.is_dir():
        raise ValueError(f"run directory does not exist: {run_directory}")
    journal_path = run_directory / "events.jsonl"
    if not journal_path.is_file():
        raise ValueError(f"run journal does not exist: {journal_path}")
    _verify_races(
        run_directory / "config.yaml",
        agent_race=agent_race,
        opponent_race=opponent_race,
    )
    events = tuple(read_event_log(journal_path))
    if not events:
        raise ValueError(f"run journal is empty: {journal_path}")
    if not any(event.event_type == "episode_result" for event in events):
        raise ValueError(f"run has no completed episode result: {run_directory}")
    return events


def _verify_races(
    config_path: Path,
    *,
    agent_race: str,
    opponent_race: str,
) -> None:
    if not config_path.is_file():
        raise ValueError(f"run config does not exist: {config_path}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    environment = payload.get("environment") if isinstance(payload, dict) else None
    if not isinstance(environment, dict):
        raise ValueError(f"run config has no environment mapping: {config_path}")
    actual = (environment.get("agent_race"), environment.get("opponent_race"))
    expected = (agent_race, opponent_race)
    if actual != expected:
        raise ValueError(
            "run race mismatch: "
            f"expected {agent_race} vs {opponent_race}, got {actual[0]} vs {actual[1]}"
        )


def _episode_results(events: tuple[StoredEvent, ...]) -> tuple[EpisodeResult, ...]:
    results: dict[tuple[str, str], EpisodeResult] = {}
    for event in events:
        if event.event_type != "episode_result":
            continue
        result = EpisodeResult.model_validate(event.payload)
        results[(result.run_id, result.episode_id)] = result
    return tuple(results.values())
