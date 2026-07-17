"""Typed goal definitions and deterministic progress reports."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from rtscortex.contracts.models import ContractModel


class GoalRequirementKind(StrEnum):
    """State facts that can be verified without an LLM."""

    STRUCTURE = "structure"
    UNIT = "unit"
    UPGRADE = "upgrade"


class GoalProgressStatus(StrEnum):
    """High-level state of one goal at an observation tick."""

    ACTIONABLE = "actionable"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    ACHIEVED = "achieved"


class GoalBlockerKind(StrEnum):
    """Stable blocker taxonomy for prompts, events, and the Live Console."""

    GOAL_DEPENDENCY = "goal_dependency"
    MISSING_PREREQUISITE = "missing_prerequisite"
    PREREQUISITE_IN_PROGRESS = "prerequisite_in_progress"
    EFFECT_IN_PROGRESS = "effect_in_progress"
    INSUFFICIENT_MINERALS = "insufficient_minerals"
    INSUFFICIENT_VESPENE = "insufficient_vespene"
    INSUFFICIENT_SUPPLY = "insufficient_supply"
    ACTION_UNAVAILABLE = "action_unavailable"
    NO_PROGRESS_ACTION = "no_progress_action"


class GoalRequirement(ContractModel):
    """One explicit, measurable fact required by a strategic goal."""

    requirement_id: str = Field(min_length=1, max_length=120)
    kind: GoalRequirementKind
    target: str = Field(min_length=1, max_length=120)
    count: int = Field(default=1, ge=1)
    action_name: str | None = Field(default=None, min_length=1, max_length=120)
    depends_on: list[str] = Field(default_factory=list)
    description: str = Field(default="", max_length=240)

    @model_validator(mode="after")
    def validate_dependencies(self) -> GoalRequirement:
        if self.requirement_id in self.depends_on:
            raise ValueError("a goal requirement cannot depend on itself")
        if len(self.depends_on) != len(set(self.depends_on)):
            raise ValueError("goal requirement dependencies must be unique")
        return self


class GoalSpec(ContractModel):
    """A strategic goal expressed as deterministic state requirements."""

    goal_id: str = Field(default="active_goal", min_length=1, max_length=120)
    strategic_goal: str = Field(min_length=1, max_length=240)
    requirements: list[GoalRequirement] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_requirement_graph(self) -> GoalSpec:
        ids = [requirement.requirement_id for requirement in self.requirements]
        if len(ids) != len(set(ids)):
            raise ValueError("goal requirement IDs must be unique")
        known_ids = set(ids)
        for requirement in self.requirements:
            unknown = set(requirement.depends_on) - known_ids
            if unknown:
                rendered = ", ".join(sorted(unknown))
                raise ValueError(f"unknown goal requirement dependencies: {rendered}")

        dependencies = {
            requirement.requirement_id: requirement.depends_on for requirement in self.requirements
        }
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(requirement_id: str) -> None:
            if requirement_id in visiting:
                raise ValueError("goal requirement dependencies must be acyclic")
            if requirement_id in visited:
                return
            visiting.add(requirement_id)
            for dependency_id in dependencies[requirement_id]:
                visit(dependency_id)
            visiting.remove(requirement_id)
            visited.add(requirement_id)

        for requirement_id in ids:
            visit(requirement_id)
        return self


class GoalProgressItem(ContractModel):
    """Measured progress for one requirement at the current observation."""

    requirement_id: str
    kind: GoalRequirementKind
    target: str
    required_count: int = Field(ge=1)
    current_count: int = Field(ge=0)
    in_progress_count: int = Field(default=0, ge=0)
    description: str = ""


class GoalProgressBlocker(ContractModel):
    """A reason why one missing requirement cannot advance immediately."""

    requirement_id: str
    kind: GoalBlockerKind
    detail: str
    action_name: str | None = None


class GoalProgressReport(ContractModel):
    """Objective goal state shared by Reflection, Planning, and the Console."""

    run_id: str
    episode_id: str
    step_id: int = Field(ge=0)
    game_loop: int = Field(ge=0)
    goal_id: str
    strategic_goal: str
    status: GoalProgressStatus
    achieved: list[GoalProgressItem] = Field(default_factory=list)
    missing: list[GoalProgressItem] = Field(default_factory=list)
    blockers: list[GoalProgressBlocker] = Field(default_factory=list)
    advancing_actions: list[str] = Field(default_factory=list)
    unique_next_action: str | None = None
    defensive_hold_required: bool = False

    @model_validator(mode="after")
    def validate_action_summary(self) -> GoalProgressReport:
        if len(self.advancing_actions) != len(set(self.advancing_actions)):
            raise ValueError("advancing actions must be unique")
        expected = self.advancing_actions[0] if len(self.advancing_actions) == 1 else None
        if self.unique_next_action != expected:
            raise ValueError(
                "unique_next_action must be set exactly when one advancing action exists"
            )
        if self.status == GoalProgressStatus.ACHIEVED and self.missing:
            raise ValueError("an achieved goal cannot have missing requirements")
        if self.status != GoalProgressStatus.ACHIEVED and not self.missing:
            raise ValueError("a non-achieved goal must have missing requirements")
        return self
