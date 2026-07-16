"""Strongly typed YAML configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, TypeAlias

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

RaceName: TypeAlias = Literal["random", "protoss", "terran", "zerg"]
HIMACandidate: TypeAlias = Literal[
    "protoss-a",
    "protoss-b",
    "protoss-c",
    "terran-a",
    "terran-b",
    "terran-c",
    "zerg-a",
    "zerg-b",
    "zerg-c",
]
BotDifficulty: TypeAlias = Literal[
    "very_easy",
    "easy",
    "medium",
    "medium_hard",
    "hard",
    "harder",
    "very_hard",
    "cheat_vision",
    "cheat_money",
    "cheat_insane",
]
BotBuild: TypeAlias = Literal["random", "rush", "timing", "power", "macro", "air"]


class SettingsModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RunSettings(SettingsModel):
    output_root: Path = Path("~/scratch/outputs/RTSCortex")
    runtime_root: Path = Path("~/fastscratch/rtscortex_runtime")
    seed: int = 0


class EnvironmentSettings(SettingsModel):
    adapter: Literal["mock", "llm_pysc2"] = "mock"
    scenario: str = "pvz_task1_level1"
    max_steps: int = Field(default=6, ge=1)
    agent_race: RaceName = "protoss"
    opponent_race: RaceName = "random"
    opponent_difficulty: BotDifficulty = "very_hard"
    opponent_build: BotBuild = "random"
    step_mul: int = Field(default=1, ge=1)
    game_steps_per_episode: int | None = Field(default=None, ge=1)
    simulation_speed_multiplier: float | None = Field(default=None, gt=0.0, le=1.0)
    pause_until_first_plan: bool = False
    sc2_path: Path | None = None
    worker_python: Path = Path("~/fastscratch/envs/rtscortex-llm-pysc2/bin/python")
    pending_plan_step_delay_seconds: float = Field(default=0.0, ge=0.0)
    action_effect_timeout_game_loops: int = Field(default=112, ge=1)
    server_ready_timeout_seconds: float = Field(default=15.0, gt=0.0)
    shutdown_timeout_seconds: float = Field(default=10.0, gt=0.0)


class RuntimeSettings(SettingsModel):
    deterministic: bool = True
    planning_interval_game_loops: int = Field(default=16, ge=1)
    planner_command_ttl_game_loops: int = Field(default=16, ge=1)
    planner_timeout_seconds: float = Field(default=15.0, gt=0.0)
    max_actions: int = Field(default=5, ge=1)

    @model_validator(mode="before")
    @classmethod
    def default_command_ttl_to_planning_interval(cls, value: Any) -> Any:
        if not isinstance(value, dict) or "planner_command_ttl_game_loops" in value:
            return value
        normalized = dict(value)
        normalized["planner_command_ttl_game_loops"] = normalized.get(
            "planning_interval_game_loops", 16
        )
        return normalized


AgentVariant: TypeAlias = Literal[
    "noop",
    "reflex_only",
    "planner_only",
    "planner_reflection_memory_reflex",
    "cortex",
]


class AgentSettings(SettingsModel):
    variant: AgentVariant = "planner_reflection_memory_reflex"


class CortexSituationSettings(SettingsModel):
    kind: Literal["deterministic"] = "deterministic"


class CortexHIMAEnsembleMemberSettings(SettingsModel):
    candidate: HIMACandidate
    model_path: Path
    device: str = "cuda:0"


class CortexMacroSettings(SettingsModel):
    kind: Literal["disabled", "hima", "hima_ensemble"] = "disabled"
    candidate: Literal["protoss-a", "protoss-b", "protoss-c"] = "protoss-a"
    python_executable: Path = Path("~/fastscratch/envs/rtscortex-hima/bin/python")
    model_path: Path | None = None
    device: str = "cuda:0"
    allow_unlicensed_weights: bool = False
    required: bool = True
    interval_game_loops: int = Field(default=112, ge=1)
    plan_ttl_game_loops: int = Field(default=448, ge=1)
    timeout_seconds: float = Field(default=12.0, gt=0.0)
    max_new_tokens: int = Field(default=512, ge=1)
    restart_limit: int = Field(default=1, ge=0)
    ensemble_members: list[CortexHIMAEnsembleMemberSettings] = Field(default_factory=list)
    coordinator: Literal["deterministic_v1"] = "deterministic_v1"

    @model_validator(mode="after")
    def require_hima_model_path(self) -> CortexMacroSettings:
        if self.kind == "hima" and self.model_path is None:
            raise ValueError("cortex HIMA macro policy requires model_path")
        if self.kind == "hima_ensemble":
            candidates = [member.candidate for member in self.ensemble_members]
            if len(candidates) != 3 or len(set(candidates)) != 3:
                raise ValueError("HIMA ensemble requires exactly three distinct members")
            races = {candidate.rsplit("-", 1)[0] for candidate in candidates}
            clusters = {candidate.rsplit("-", 1)[1] for candidate in candidates}
            if len(races) != 1 or clusters != {"a", "b", "c"}:
                raise ValueError(
                    "HIMA ensemble members must be the a/b/c checkpoints for one race"
                )
        elif self.ensemble_members:
            raise ValueError("ensemble_members are only valid for kind=hima_ensemble")
        return self


class CortexTacticalSettings(SettingsModel):
    kind: Literal["deterministic_reflex"] = "deterministic_reflex"


class CortexExecutorSettings(SettingsModel):
    kind: Literal["deterministic"] = "deterministic"
    timeout_ms: float = Field(default=10.0, gt=0.0)
    fallback: Literal["deterministic"] = "deterministic"
    supply_emergency_free_supply: int = Field(default=2, ge=0)
    resource_fallback_pylon_free_supply: int = Field(default=4, ge=0)


class CortexExplanationSettings(SettingsModel):
    enabled: bool = False


class CortexPlaybookSettings(SettingsModel):
    enabled: bool = False
    database_path: Path = Path("~/scratch/outputs/RTSCortex/cortex-playbook.sqlite3")
    top_k: int = Field(default=6, ge=1, le=20)
    min_confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    promotion_support: int = Field(default=2, ge=1)
    include_candidates: bool = False


class CortexSettings(SettingsModel):
    situation: CortexSituationSettings = Field(default_factory=CortexSituationSettings)
    macro: CortexMacroSettings = Field(default_factory=CortexMacroSettings)
    tactical: CortexTacticalSettings = Field(default_factory=CortexTacticalSettings)
    executor: CortexExecutorSettings = Field(default_factory=CortexExecutorSettings)
    explanation: CortexExplanationSettings = Field(default_factory=CortexExplanationSettings)
    playbook: CortexPlaybookSettings = Field(default_factory=CortexPlaybookSettings)


class ReflexSettings(SettingsModel):
    enabled: bool = True
    low_health_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    target_latency_ms: float = Field(default=50.0, gt=0.0)


class MemorySettings(SettingsModel):
    short_term_window: int = Field(default=20, ge=1)


class ContextSettings(SettingsModel):
    max_prompt_chars: int = Field(default=9_000, ge=2_000)
    max_recent_events: int = Field(default=8, ge=1)
    max_lessons: int = Field(default=6, ge=1)
    max_episode_summaries: int = Field(default=1, ge=0)


class ProviderSettings(SettingsModel):
    kind: Literal["fake", "openai_compatible"] = "fake"
    model: str = "fake-planner-v1"
    base_url: str = "http://127.0.0.1:8000/v1"
    api_key_env: str = "RTSCORTEX_LLM_API_KEY"
    timeout_seconds: float = Field(default=30.0, gt=0.0)
    max_tokens: int | None = Field(default=None, ge=1)
    enable_thinking: bool | None = None
    prompt_cost_per_million_tokens: float = Field(default=0.0, ge=0.0)
    completion_cost_per_million_tokens: float = Field(default=0.0, ge=0.0)


class EvaluationSettings(SettingsModel):
    seeds: list[int] = Field(default_factory=lambda: [0, 1, 2], min_length=1)


class ConsoleSettings(SettingsModel):
    enabled: bool = False
    port: int = Field(default=8765, ge=1, le=65_535)
    frame_fps: float = Field(default=2.0, gt=0.0, le=30.0)
    rgb_screen_size: int = Field(default=256, ge=64, le=2_048)
    rgb_minimap_size: int = Field(default=128, ge=32, le=1_024)
    jpeg_quality: int = Field(default=75, ge=1, le=95)
    stale_after_seconds: float = Field(default=2.0, gt=0.0)
    frontend_event_limit: int = Field(default=5_000, ge=100, le=100_000)


class ExperimentConfig(SettingsModel):
    run: RunSettings = Field(default_factory=RunSettings)
    environment: EnvironmentSettings = Field(default_factory=EnvironmentSettings)
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    cortex: CortexSettings = Field(default_factory=CortexSettings)
    reflex: ReflexSettings = Field(default_factory=ReflexSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    context: ContextSettings = Field(default_factory=ContextSettings)
    provider: ProviderSettings = Field(default_factory=ProviderSettings)
    evaluation: EvaluationSettings = Field(default_factory=EvaluationSettings)
    console: ConsoleSettings = Field(default_factory=ConsoleSettings)

    @model_validator(mode="after")
    def validate_race_brain_matches_agent(self) -> ExperimentConfig:
        if self.cortex.macro.kind != "hima_ensemble":
            return self
        race = self.cortex.macro.ensemble_members[0].candidate.rsplit("-", 1)[0]
        if self.environment.agent_race != race:
            raise ValueError(
                "HIMA ensemble race must match environment.agent_race; "
                f"received {race} brain for {self.environment.agent_race} agent"
            )
        return self

    def expanded(self) -> ExperimentConfig:
        data = self.model_dump()
        data["run"]["output_root"] = self.run.output_root.expanduser()
        data["run"]["runtime_root"] = self.run.runtime_root.expanduser()
        if self.environment.sc2_path is not None:
            data["environment"]["sc2_path"] = self.environment.sc2_path.expanduser()
        data["environment"]["worker_python"] = self.environment.worker_python.expanduser()
        data["cortex"]["macro"]["python_executable"] = (
            self.cortex.macro.python_executable.expanduser()
        )
        if self.cortex.macro.model_path is not None:
            data["cortex"]["macro"]["model_path"] = self.cortex.macro.model_path.expanduser()
        for member in data["cortex"]["macro"]["ensemble_members"]:
            member["model_path"] = Path(member["model_path"]).expanduser()
        data["cortex"]["playbook"]["database_path"] = (
            self.cortex.playbook.database_path.expanduser()
        )
        return ExperimentConfig.model_validate(data)


def load_config(path: Path) -> ExperimentConfig:
    with path.open(encoding="utf-8") as stream:
        payload = yaml.safe_load(stream) or {}
    return ExperimentConfig.model_validate(payload).expanded()
