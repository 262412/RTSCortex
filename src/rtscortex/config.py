"""Strongly typed YAML configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, TypeAlias

import yaml
from pydantic import BaseModel, ConfigDict, Field


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
    sc2_path: Path | None = None
    worker_python: Path = Path("~/fastscratch/envs/rtscortex-llm-pysc2/bin/python")
    pending_plan_step_delay_seconds: float = Field(default=0.0, ge=0.0)
    server_ready_timeout_seconds: float = Field(default=15.0, gt=0.0)
    shutdown_timeout_seconds: float = Field(default=10.0, gt=0.0)


class RuntimeSettings(SettingsModel):
    deterministic: bool = True
    planning_interval_game_loops: int = Field(default=16, ge=1)
    planner_timeout_seconds: float = Field(default=15.0, gt=0.0)
    max_actions: int = Field(default=5, ge=1)


AgentVariant: TypeAlias = Literal[
    "noop",
    "reflex_only",
    "planner_only",
    "planner_reflection_memory_reflex",
]


class AgentSettings(SettingsModel):
    variant: AgentVariant = "planner_reflection_memory_reflex"


class ReflexSettings(SettingsModel):
    enabled: bool = True
    low_health_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    target_latency_ms: float = Field(default=50.0, gt=0.0)


class MemorySettings(SettingsModel):
    short_term_window: int = Field(default=20, ge=1)


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


class ExperimentConfig(SettingsModel):
    run: RunSettings = Field(default_factory=RunSettings)
    environment: EnvironmentSettings = Field(default_factory=EnvironmentSettings)
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    reflex: ReflexSettings = Field(default_factory=ReflexSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    provider: ProviderSettings = Field(default_factory=ProviderSettings)
    evaluation: EvaluationSettings = Field(default_factory=EvaluationSettings)

    def expanded(self) -> ExperimentConfig:
        data = self.model_dump()
        data["run"]["output_root"] = self.run.output_root.expanduser()
        data["run"]["runtime_root"] = self.run.runtime_root.expanduser()
        if self.environment.sc2_path is not None:
            data["environment"]["sc2_path"] = self.environment.sc2_path.expanduser()
        data["environment"]["worker_python"] = self.environment.worker_python.expanduser()
        return ExperimentConfig.model_validate(data)


def load_config(path: Path) -> ExperimentConfig:
    with path.open(encoding="utf-8") as stream:
        payload = yaml.safe_load(stream) or {}
    return ExperimentConfig.model_validate(payload).expanded()
