"""Stable semantic operation identities shared across Cortex decision ticks."""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import Field, model_validator

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


class AuthoritativeBuildCircuitState(ContractModel):
    """Durable semantic boundary opened by authoritative Build rejections."""

    operation_id: str = Field(pattern=r"^operation:[0-9a-f]{64}$")
    action_name: str | None = Field(default=None, pattern=r"^Build_.+")
    streak: int = Field(ge=1)
    threshold: Literal[3] = 3
    circuit_open: bool
    failure_count: int = Field(ge=1)
    last_failure_code: str | None = Field(default=None, min_length=1)
    opened_command_id: str | None = Field(default=None, min_length=1)
    opened_attempt_ordinal: int | None = Field(default=None, ge=0)
    opened_game_loop: int = Field(ge=0)
    builder_tag: int | None = Field(default=None, gt=0)
    ability_id: int | None = Field(default=None, ge=0)
    world_target: tuple[float, float] | None = None
    material_legality_identity: str | None = Field(
        default=None,
        pattern=r"^build-legality:[0-9a-f]{64}$",
    )
    blocked_semantic_material_identity: str | None = Field(
        default=None,
        pattern=r"^semantic-build-material:[0-9a-f]{64}$",
    )
    material_evidence_valid: bool
    invalid_evidence_reasons: tuple[str, ...] = ()
    operation_epoch_changed: bool = False

    @model_validator(mode="after")
    def validate_material_evidence(self) -> AuthoritativeBuildCircuitState:
        if self.material_evidence_valid:
            if (
                self.builder_tag is None
                or self.ability_id is None
                or self.ability_id <= 0
                or self.world_target is None
                or self.action_name is None
                or self.last_failure_code is None
                or self.opened_command_id is None
                or self.material_legality_identity is None
                or self.blocked_semantic_material_identity is None
            ):
                raise ValueError("valid circuit material evidence requires complete typed identity")
            if self.invalid_evidence_reasons:
                raise ValueError("valid circuit material evidence cannot include invalid reasons")
        elif not self.invalid_evidence_reasons:
            raise ValueError("invalid circuit material evidence requires typed reasons")
        return self


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
    phase: str = Field(default="active", pattern=r"^(active|waiting_for_candidates)$")
    terminal_state: str | None = None


__all__ = [
    "AuthoritativeBuildCircuitState",
    "AttemptKey",
    "EngagementKey",
    "ExpansionGoalKey",
    "ExpansionGoalState",
    "OperationKey",
    "PlacementReservationKey",
    "RetreatCommitmentKey",
]
