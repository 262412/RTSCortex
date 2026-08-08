"""Build fail-closed evidence for promoting one soft Playbook rule to hard."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from rtscortex.memory import StoredEvent
from rtscortex.playbook.models import (
    PlaybookRule,
    PlaybookRuleCategory,
    PlaybookRuleEffect,
    PlaybookRuleStatus,
    PlaybookRuleStrength,
)
from rtscortex.playbook.readiness import (
    PlaybookHardQualificationManifest,
    PlaybookHardQualificationRunEvidence,
)
from rtscortex.playbook.semantics import evaluation_kind


@dataclass(frozen=True, slots=True)
class HardQualificationEvaluationMetrics:
    """Terminal counterfactual observations for one exact probe rule."""

    shadow_would_block_application_count: int
    resolved_counterfactual_count: int
    unresolved_counterfactual_count: int
    execution_false_block_count: int
    invalid_counterfactual_evidence_count: int

    @property
    def execution_false_block_rate(self) -> float:
        if self.resolved_counterfactual_count == 0:
            return 0.0
        return self.execution_false_block_count / self.resolved_counterfactual_count


def analyze_hard_qualification_evaluations(
    events: Iterable[StoredEvent],
    *,
    rule_id: str,
) -> HardQualificationEvaluationMetrics:
    """Reduce journal events to the latest terminal state of each probe evaluation."""

    applications: set[str] = set()
    evaluations: dict[str, tuple[int, dict[str, object]]] = {}
    for event in events:
        payload = event.payload
        if payload.get("rule_id") != rule_id:
            continue
        if event.event_type == "playbook_rule_applied":
            application_id = payload.get("application_id")
            if (
                isinstance(application_id, str)
                and payload.get("reason") == "shadow_would_block"
                and payload.get("blocked") is False
            ):
                applications.add(application_id)
        elif event.event_type == "playbook_rule_evaluated":
            evaluation_id = payload.get("evaluation_id")
            if not isinstance(evaluation_id, str):
                continue
            previous = evaluations.get(evaluation_id)
            if previous is None or event.event_id > previous[0]:
                evaluations[evaluation_id] = (event.event_id, dict(payload))

    resolved = 0
    unresolved = 0
    false_blocks = 0
    invalid = 0
    evaluated_application_ids: set[str] = set()
    for _, evaluation in evaluations.values():
        application_id = evaluation.get("application_id")
        if isinstance(application_id, str):
            evaluated_application_ids.add(application_id)
        if not _is_complete_hard_probe_evaluation(evaluation):
            invalid += 1
            continue
        if evaluation.get("counterfactual_observable") is not True:
            unresolved += 1
            continue
        false_block = evaluation.get("execution_false_block", evaluation.get("false_block"))
        if not isinstance(false_block, bool):
            unresolved += 1
            continue
        resolved += 1
        false_blocks += false_block

    unresolved += len(applications - evaluated_application_ids)
    return HardQualificationEvaluationMetrics(
        shadow_would_block_application_count=len(applications),
        resolved_counterfactual_count=resolved,
        unresolved_counterfactual_count=unresolved,
        execution_false_block_count=false_blocks,
        invalid_counterfactual_evidence_count=invalid,
    )


def build_hard_qualification_manifest(
    *,
    parent: PlaybookRule,
    baseline_sha256: str,
    probe_baseline_sha256: str,
    git_sha: str,
    sc2_patch: str,
    runs: Sequence[PlaybookHardQualificationRunEvidence],
) -> PlaybookHardQualificationManifest:
    """Create a manifest only when every source seed has real bounded evidence."""

    reasons = _qualification_rejection_reasons(
        parent=parent,
        probe_baseline_sha256=probe_baseline_sha256,
        git_sha=git_sha,
        sc2_patch=sc2_patch,
        runs=runs,
    )
    if reasons:
        raise ValueError("hard qualification rejected: " + ", ".join(reasons))

    ordered_runs = tuple(sorted(runs, key=lambda item: (item.seed_id, item.run_id)))
    resolved = sum(item.resolved_counterfactual_count for item in ordered_runs)
    unresolved = sum(item.unresolved_counterfactual_count for item in ordered_runs)
    false_blocks = sum(item.execution_false_block_count for item in ordered_runs)
    overflow = sum(item.analysis_evidence_overflow_count for item in ordered_runs)
    return PlaybookHardQualificationManifest(
        schema_version="1.1",
        artifact_kind="playbook-hard-qualification",
        parent_rule_id=parent.rule_id,
        parent_canonical_key=parent.canonical_key,
        baseline_sha256=baseline_sha256,
        probe_baseline_sha256=probe_baseline_sha256,
        git_sha=git_sha,
        sc2_patch=sc2_patch,
        qualification_seed_ids=tuple(
            sorted(set(parent.source_seeds) - set(parent.censored_source_seeds))
        ),
        source_run_ids=tuple(
            sorted(set(parent.source_run_ids) - set(parent.censored_source_run_ids))
        ),
        qualification_runs=ordered_runs,
        engineering_accepted=True,
        counterfactual_evidence_accepted=True,
        analysis_evidence_overflow_count=overflow,
        counterfactual_resolved_count=resolved,
        counterfactual_unresolved_count=unresolved,
        counterfactual_false_block_count=false_blocks,
        counterfactual_false_block_rate=false_blocks / resolved,
        shadow_state_count=parent.shadow_state_count,
        execution_false_block_count=parent.false_block_count,
        execution_false_block_rate=parent.false_block_rate,
    )


def _qualification_rejection_reasons(
    *,
    parent: PlaybookRule,
    probe_baseline_sha256: str,
    git_sha: str,
    sc2_patch: str,
    runs: Sequence[PlaybookHardQualificationRunEvidence],
) -> tuple[str, ...]:
    reasons: list[str] = []
    qualification_seeds = set(parent.source_seeds) - set(parent.censored_source_seeds)
    qualification_runs = set(parent.source_run_ids) - set(parent.censored_source_run_ids)
    observed_seeds = {item.seed_id for item in runs}
    observed_run_ids = {item.run_id for item in runs}
    if parent.status is not PlaybookRuleStatus.ACTIVE:
        reasons.append("parent_not_active")
    if parent.strength is not PlaybookRuleStrength.SOFT:
        reasons.append("parent_not_soft")
    if parent.effect is not PlaybookRuleEffect.AVOID:
        reasons.append("parent_effect_not_avoid")
    if evaluation_kind(parent.category).value != "execution_guard":
        reasons.append("strategic_parent_requires_paired_ab")
    if parent.category not in {
        PlaybookRuleCategory.ENGINE_INVARIANT,
        PlaybookRuleCategory.EXECUTION_GUARD,
        PlaybookRuleCategory.TACTICAL_RESPONSE,
    }:
        reasons.append("parent_category_not_execution")
    if parent.code_revision != git_sha:
        reasons.append("git_sha")
    if parent.sc2_patch != sc2_patch:
        reasons.append("sc2_patch")
    if len(qualification_seeds) < 3 or observed_seeds != qualification_seeds:
        reasons.append("qualification_seed_coverage")
    if len(qualification_runs) < 3:
        reasons.append("source_run_coverage")
    if len(observed_run_ids) != len(runs):
        reasons.append("qualification_run_identity")
    if observed_run_ids & qualification_runs:
        reasons.append("qualification_run_source_overlap")
    fingerprints = {item.source_attestation_fingerprint for item in runs}
    if len(fingerprints) != 1:
        reasons.append("source_attestation")
    if any(item.git_sha != git_sha for item in runs):
        reasons.append("git_sha")
    if any(item.sc2_patch != sc2_patch for item in runs):
        reasons.append("sc2_patch")
    if any(
        item.playbook_before_sha256 != probe_baseline_sha256
        or item.playbook_after_sha256 != probe_baseline_sha256
        for item in runs
    ):
        reasons.append("probe_baseline_identity")
    if any(not item.natural_terminal for item in runs):
        reasons.append("natural_terminal")
    if any(not item.engineering_accepted for item in runs):
        reasons.append("engineering_accepted")
    if any(item.analysis_evidence_overflow_count for item in runs):
        reasons.append("analysis_evidence_overflow")
    if any(item.invalid_counterfactual_evidence_count for item in runs):
        reasons.append("invalid_counterfactual_evidence")
    if any(item.shadow_would_block_application_count == 0 for item in runs):
        reasons.append("shadow_would_block_coverage")
    if any(item.resolved_counterfactual_count == 0 for item in runs):
        reasons.append("resolved_counterfactual_coverage")
    unresolved = sum(item.unresolved_counterfactual_count for item in runs)
    if unresolved:
        reasons.append("unresolved_counterfactuals")
    resolved = sum(item.resolved_counterfactual_count for item in runs)
    false_blocks = sum(item.execution_false_block_count for item in runs)
    if resolved == 0 or false_blocks / resolved > 0.01:
        reasons.append("execution_false_block_rate")
    return tuple(dict.fromkeys(reasons))


def _is_complete_hard_probe_evaluation(evaluation: dict[str, object]) -> bool:
    return (
        evaluation.get("strength_at_evaluation") == "hard"
        and evaluation.get("status_at_evaluation") == "active"
        and evaluation.get("shadow_decision") == "would_block"
        and evaluation.get("rule_kind") == "execution_guard"
        and isinstance(evaluation.get("application_id"), str)
        and isinstance(evaluation.get("counterfactual_key"), str)
        and isinstance(evaluation.get("counterfactual_signature"), str)
        and isinstance(evaluation.get("behavior_before_hash"), str)
        and isinstance(evaluation.get("decision_epoch"), int)
        and isinstance(evaluation.get("counterfactual_observable"), bool)
    )
