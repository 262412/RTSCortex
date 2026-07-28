"""Stable semantic operation identities shared across Cortex decision ticks."""

from __future__ import annotations

import hashlib
import json

from pydantic import Field

from rtscortex.contracts.models import ContractModel


def _identity(prefix: str, payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return f"{prefix}:{hashlib.sha256(encoded).hexdigest()}"


class OperationKey(ContractModel):
    """Identity for one semantic operation that may span multiple command attempts."""

    run_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    role: str = Field(min_length=1)
    action_family: str = Field(min_length=1)
    semantic_actor: str = Field(min_length=1)
    target_key: str = Field(min_length=1)
    epoch: int = Field(default=0, ge=0)

    @property
    def operation_id(self) -> str:
        return _identity("operation", self.model_dump(mode="json"))


class AttemptKey(ContractModel):
    """One concrete dispatch attempt belonging to a semantic operation."""

    operation_id: str = Field(pattern=r"^operation:[0-9a-f]{64}$")
    command_id: str = Field(min_length=1)
    attempt_ordinal: int = Field(default=0, ge=0)

    @property
    def attempt_id(self) -> str:
        return _identity("attempt", self.model_dump(mode="json"))


class PlacementReservationKey(ContractModel):
    """Concrete placement lease used by validation, dispatch and verification."""

    operation_id: str | None = Field(default=None, pattern=r"^operation:[0-9a-f]{64}$")
    command_id: str = Field(min_length=1)
    action_name: str = Field(min_length=1)
    builder_tag: int = Field(gt=0)
    ability_name: str = Field(min_length=1)
    world_target: tuple[float, float]
    placement_revision: str = Field(min_length=1)

    @property
    def reservation_id(self) -> str:
        return _identity("placement", self.model_dump(mode="json"))


class RetreatCommitmentKey(ContractModel):
    operation_id: str = Field(pattern=r"^operation:[0-9a-f]{64}$")
    actor_tags: tuple[int, ...]
    threat_signature: str = Field(min_length=1)
    destination: tuple[float, float] | str

    @property
    def commitment_id(self) -> str:
        return _identity("retreat", self.model_dump(mode="json"))


class EngagementKey(ContractModel):
    operation_id: str = Field(pattern=r"^operation:[0-9a-f]{64}$")
    actor_tags: tuple[int, ...]
    ability_name: str = Field(min_length=1)
    target_tag: int = Field(gt=0)

    @property
    def engagement_id(self) -> str:
        return _identity("engagement", self.model_dump(mode="json"))


class ExpansionGoalKey(ContractModel):
    run_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    race: str = Field(min_length=1)
    desired_base_count: int = Field(ge=2)
    strategic_revision: int = Field(default=0, ge=0)

    @property
    def goal_id(self) -> str:
        return _identity("expansion", self.model_dump(mode="json"))


class ExpansionGoalState(ContractModel):
    goal_id: str = Field(pattern=r"^expansion:[0-9a-f]{64}$")
    baseline_base_count: int = Field(ge=0)
    desired_base_count: int = Field(ge=2)
    observed_base_count: int = Field(ge=0)
    strategic_revision: int = Field(ge=0)
    current_candidate_epoch: int = Field(default=0, ge=0)
    exhausted_candidate_epochs: tuple[int, ...] = ()
    retry_budget: int = Field(default=8, ge=0)
    cooldown_until_game_loop: int = Field(default=0, ge=0)
    active_reservation_id: str | None = None
    terminal_state: str | None = None


__all__ = [
    "AttemptKey",
    "EngagementKey",
    "ExpansionGoalKey",
    "ExpansionGoalState",
    "OperationKey",
    "PlacementReservationKey",
    "RetreatCommitmentKey",
]
