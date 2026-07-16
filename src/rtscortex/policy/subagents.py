"""Shadow-only policy subagents and the built-in comparison catalog."""

from __future__ import annotations

import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Protocol

from rtscortex.agents.context import model_observation
from rtscortex.agents.models import (
    ActionProposal,
    planning_output_model,
    project_planning_observation,
)
from rtscortex.contracts.interfaces import LLMProvider
from rtscortex.policy.models import (
    MacroPolicyProposal,
    PolicyAvailability,
    PolicyAvailabilityStatus,
    PolicyObservationFixture,
    PolicyProposal,
    PolicyProviderKind,
    PolicySubagentSpec,
)


class PolicySubagent(Protocol):
    """Produce advisory proposals; this interface has no dispatch capability."""

    spec: PolicySubagentSpec

    async def propose(
        self,
        fixture: PolicyObservationFixture,
    ) -> PolicyProposal | MacroPolicyProposal: ...


@dataclass(frozen=True)
class PolicySubagentRegistration:
    """Bind a catalog entry to its optional local implementation."""

    spec: PolicySubagentSpec
    availability: PolicyAvailability
    subagent: PolicySubagent | None = None

    def __post_init__(self) -> None:
        if self.availability.status is PolicyAvailabilityStatus.AVAILABLE:
            if self.subagent is None:
                raise ValueError("available policies require a subagent implementation")
            if self.subagent.spec != self.spec:
                raise ValueError("registered subagent spec does not match the catalog spec")


QWEN3_8B_SPEC = PolicySubagentSpec(
    subagent_id="qwen3-8b-current",
    display_name="Current Qwen3-8B planner",
    provider_kind=PolicyProviderKind.OPENAI_COMPATIBLE,
    model_id="Qwen/Qwen3-8B",
    role="general planner baseline",
    race="any",
    action_interface="RTSCortex AvailableAction structured output",
    requires_external_weights=False,
    license_id="Apache-2.0",
)

HIMA_PROTOSS_SPECS = tuple(
    PolicySubagentSpec(
        subagent_id=f"hima-protoss-{specialist}",
        display_name=f"HIMA Protoss-{specialist}",
        provider_kind=PolicyProviderKind.HUGGING_FACE_TRANSFORMERS,
        model_id=f"SNUMPR/Protoss-{specialist}",
        role=f"upstream Protoss specialist cluster {specialist.upper()}",
        race="Protoss",
        action_interface="HIMA GitHub JSON state to ordered Protoss macro actions",
        requires_external_weights=True,
        license_id=None,
    )
    for specialist in ("a", "b", "c")
)

_HIMA_RACES: tuple[Literal["Protoss", "Terran", "Zerg"], ...] = (
    "Protoss",
    "Terran",
    "Zerg",
)
HIMA_RACE_SPECS = MappingProxyType({
    race.lower(): tuple(
        PolicySubagentSpec(
            subagent_id=f"hima-{race.lower()}-{specialist}",
            display_name=f"HIMA {race}-{specialist}",
            provider_kind=PolicyProviderKind.HUGGING_FACE_TRANSFORMERS,
            model_id=f"SNUMPR/{race}-{specialist}",
            role=f"upstream {race} specialist cluster {specialist.upper()}",
            race=race,
            action_interface=f"HIMA GitHub JSON state to ordered {race} macro actions",
            requires_external_weights=True,
            license_id=None,
        )
        for specialist in ("a", "b", "c")
    )
    for race in _HIMA_RACES
})
HIMA_ALL_SPECS = tuple(
    spec for race in ("protoss", "terran", "zerg") for spec in HIMA_RACE_SPECS[race]
)

HIERNET_SC2_SPEC = PolicySubagentSpec(
    subagent_id="hiernet-sc2-protoss",
    display_name="HierNet-SC2 Protoss",
    provider_kind=PolicyProviderKind.TENSORFLOW_CHECKPOINT,
    model_id="liuruoze/HierNet-SC2",
    role="hierarchical macro policy",
    race="Protoss",
    action_interface="HierNet action index adapter required",
    requires_external_weights=True,
    license_id="Apache-2.0",
)


class LLMPlanningPolicySubagent:
    """Adapt an existing structured-output LLM to the shadow proposal interface."""

    def __init__(
        self,
        provider: LLMProvider,
        *,
        spec: PolicySubagentSpec = QWEN3_8B_SPEC,
    ) -> None:
        self.provider = provider
        self.spec = spec

    async def propose(self, fixture: PolicyObservationFixture) -> PolicyProposal:
        observation = project_planning_observation(fixture.observation)
        compact_observation, _ = model_observation(observation)
        # Exact action candidates already carry every dispatchable screen/minimap value.
        compact_observation.pop("spatial_context", None)
        response_type = planning_output_model(observation)
        output = await self.provider.generate(
            response_type,
            system_prompt=(
                "You are a shadow-only StarCraft II policy advisor. Propose only actions "
                "allowed by the supplied structured schema. Your output is advisory and "
                "will never be dispatched directly. Treat deterministic goal_progress as "
                "ground truth. When the goal can advance and no defensive hold is required, "
                "prefer a goal-advancing action and do not propose Stop or Hold_Position."
            ),
            user_prompt=json.dumps(
                {
                    "observation": compact_observation,
                    "goal_spec": (
                        fixture.goal_spec.model_dump(mode="json")
                        if fixture.goal_spec is not None
                        else None
                    ),
                    "goal_progress": (
                        fixture.goal_progress.model_dump(mode="json")
                        if fixture.goal_progress is not None
                        else None
                    ),
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        )
        return PolicyProposal(
            strategic_goal=output.strategic_goal,
            steps=list(output.steps),
            proposed_actions=[
                ActionProposal.model_validate(action.model_dump())
                for action in output.proposed_actions
            ],
        )


def built_in_policy_specs() -> tuple[PolicySubagentSpec, ...]:
    """Return the stable candidate order used by comparison reports."""

    return (QWEN3_8B_SPEC, *HIMA_PROTOSS_SPECS, HIERNET_SC2_SPEC)


def default_shadow_registrations(
    *,
    current_qwen: PolicySubagent | None = None,
) -> tuple[PolicySubagentRegistration, ...]:
    """Build a no-download catalog with explicit availability for every candidate."""

    if current_qwen is None:
        qwen_registration = PolicySubagentRegistration(
            spec=QWEN3_8B_SPEC,
            availability=PolicyAvailability(
                status=PolicyAvailabilityStatus.SKIPPED,
                reason="current OpenAI-compatible Qwen endpoint was not configured",
            ),
        )
    else:
        qwen_registration = PolicySubagentRegistration(
            spec=QWEN3_8B_SPEC,
            availability=PolicyAvailability(status=PolicyAvailabilityStatus.AVAILABLE),
            subagent=current_qwen,
        )

    hima = tuple(
        PolicySubagentRegistration(
            spec=spec,
            availability=PolicyAvailability(
                status=PolicyAvailabilityStatus.UNAVAILABLE,
                reason="local weights are not configured; no download attempted",
            ),
        )
        for spec in HIMA_PROTOSS_SPECS
    )
    hiernet = PolicySubagentRegistration(
        spec=HIERNET_SC2_SPEC,
        availability=PolicyAvailability(
            status=PolicyAvailabilityStatus.UNAVAILABLE,
            reason="adapter_not_implemented; no download attempted",
        ),
    )
    return (qwen_registration, *hima, hiernet)
