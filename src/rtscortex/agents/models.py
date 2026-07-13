"""Structured outputs produced by deliberative modules."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AgentOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ReflectionOutput(AgentOutput):
    summary: str = Field(max_length=500)
    lessons: list[str] = Field(default_factory=list, max_length=3)


class ActionProposal(AgentOutput):
    actor: str
    name: str
    arguments: list[Any] = Field(default_factory=list)
    priority: int = Field(default=50, ge=0, le=89)
    ttl_game_loops: int = Field(default=32, ge=1)


class PlanningOutput(AgentOutput):
    strategic_goal: str = Field(max_length=200)
    steps: list[str] = Field(default_factory=list, max_length=3)
    proposed_actions: list[ActionProposal] = Field(default_factory=list, max_length=3)
