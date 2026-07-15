"""One-candidate worker used by the Policy Comparison v0.2 orchestrator."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from rtscortex.policy.comparison import HIMAWorkerRequest, validate_hima_model_path
from rtscortex.policy.corpus import load_policy_corpus
from rtscortex.policy.hima.observation import HIMAObservationAdapter
from rtscortex.policy.hima.parser import HIMAProposalParser
from rtscortex.policy.hima.subagent import HIMAPolicySubagent, TransformersHIMAGenerator
from rtscortex.policy.models import PolicyAvailability, PolicyAvailabilityStatus
from rtscortex.policy.shadow import PolicyShadowRunner
from rtscortex.policy.subagents import HIMA_PROTOSS_SPECS, PolicySubagentRegistration


def run_worker(request_path: Path, response_path: Path) -> None:
    """Load one local HIMA checkpoint, evaluate all fixtures, then persist a response."""

    request = HIMAWorkerRequest.model_validate_json(request_path.read_text(encoding="utf-8"))
    validate_hima_model_path(request.model_id, request.model_path)
    spec = next(
        (candidate for candidate in HIMA_PROTOSS_SPECS if candidate.model_id == request.model_id),
        None,
    )
    if spec is None:
        raise ValueError(f"no HIMA policy spec exists for {request.model_id}")
    fixtures = load_policy_corpus(request.manifest_path)
    generator = TransformersHIMAGenerator(
        request.model_path,
        model_id=request.model_id,
        allow_unlicensed_weights=request.allow_unlicensed_weights,
        device=request.device,
    )
    subagent = HIMAPolicySubagent(
        spec,
        generator,
        HIMAObservationAdapter(),
        HIMAProposalParser(),
    )
    comparison = asyncio.run(
        PolicyShadowRunner().compare(
            fixtures,
            [
                PolicySubagentRegistration(
                    spec=spec,
                    availability=PolicyAvailability(
                        status=PolicyAvailabilityStatus.AVAILABLE
                    ),
                    subagent=subagent,
                )
            ],
        )
    )
    response_path.parent.mkdir(parents=True, exist_ok=True)
    response_path.write_text(comparison.model_dump_json(indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    arguments = parser.parse_args()
    run_worker(arguments.request, arguments.response)


if __name__ == "__main__":
    main()
