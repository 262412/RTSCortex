from __future__ import annotations

import rtscortex.policy as policy
import rtscortex.policy.hima as hima


def test_policy_v02_public_api_is_exported_from_stable_packages() -> None:
    policy_names = {
        "MacroPolicyProposal",
        "PolicyActionAssessment",
        "PolicyActionClassification",
        "PolicyActionClassificationCounts",
        "PolicyFixtureStratum",
        "PolicyGenerationMetadata",
        "PolicyComparisonConfig",
        "build_policy_corpus_from_file",
        "load_policy_corpus",
        "verify_policy_corpus",
        "run_policy_comparison",
        "write_policy_comparison_reports",
        "HIMAObservationAdapter",
        "HIMAInputContext",
        "HIMALivePolicyClient",
        "HIMALiveProposalRequest",
        "HIMALiveProposalResponse",
        "HIMAProposalParser",
        "HIMAMacroActionMapper",
        "HIMAPolicySubagent",
        "TransformersHIMAGenerator",
    }
    hima_names = {
        "HIMAObservationAdapter",
        "HIMAInputContext",
        "HIMALivePolicyClient",
        "HIMALiveProposalRequest",
        "HIMALiveProposalResponse",
        "HIMAProposalParser",
        "HIMAMacroActionMapper",
        "HIMAPolicySubagent",
        "TransformersHIMAGenerator",
        "HIMA_PINNED_REVISIONS",
    }

    assert policy_names <= set(policy.__all__)
    assert all(hasattr(policy, name) for name in policy_names)
    assert hima_names <= set(hima.__all__)
    assert all(hasattr(hima, name) for name in hima_names)
