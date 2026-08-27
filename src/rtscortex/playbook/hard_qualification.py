"""Build fail-closed evidence for promoting one soft Playbook rule to hard."""

from __future__ import annotations

import string
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from rtscortex.memory import StoredEvent
from rtscortex.playbook.models import (
    PlaybookRule,
    PlaybookRuleApplication,
    PlaybookRuleCategory,
    PlaybookRuleEffect,
    PlaybookRuleEvaluation,
    PlaybookRuleStatus,
    PlaybookRuleStrength,
)
from rtscortex.playbook.readiness import (
    PlaybookHardQualificationManifest,
    PlaybookHardQualificationRunEvidence,
    _probe_fingerprints,
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
    structurally_unobservable_counterfactual_count: int = 0
    invalid_application_count: int = 0
    application_without_evaluation_count: int = 0
    evaluation_without_application_count: int = 0

    @property
    def not_selected_counterfactual_count(self) -> int:
        """Compatibility alias for the structural ``not_selected`` bucket."""

        return self.structurally_unobservable_counterfactual_count

    @property
    def counterfactual_not_selected_count(self) -> int:
        return self.structurally_unobservable_counterfactual_count

    @property
    def counterfactual_structurally_unobservable_count(self) -> int:
        return self.structurally_unobservable_counterfactual_count

    @property
    def classified_application_count(self) -> int:
        return (
            self.resolved_counterfactual_count
            + self.unresolved_counterfactual_count
            + self.structurally_unobservable_counterfactual_count
            + self.invalid_application_count
        )

    @property
    def application_conservation_count(self) -> int:
        return self.classified_application_count

    @property
    def application_conservation_valid(self) -> bool:
        """Whether every probe application belongs to one classification bucket."""

        return self.application_conservation_count == self.shadow_would_block_application_count

    @property
    def execution_false_block_rate(self) -> float:
        if self.resolved_counterfactual_count == 0:
            return 0.0
        return self.execution_false_block_count / self.resolved_counterfactual_count


def analyze_hard_qualification_evaluations(
    events: Iterable[StoredEvent],
    *,
    rule_id: str,
    expected_rule_fingerprint: str | None = None,
    expected_predicate_fingerprint: str | None = None,
) -> HardQualificationEvaluationMetrics:
    """Reduce journal events to the latest terminal state of each probe evaluation."""

    application_events: list[dict[str, object]] = []
    evaluations: dict[str, tuple[int, dict[str, object]]] = {}
    malformed_evaluation_count = 0
    for event in events:
        payload = event.payload
        if payload.get("rule_id") != rule_id:
            continue
        if event.event_type == "playbook_rule_applied":
            application_events.append(dict(payload))
        elif event.event_type == "playbook_rule_evaluated":
            evaluation_id = payload.get("evaluation_id")
            if not isinstance(evaluation_id, str):
                malformed_evaluation_count += 1
                continue
            previous = evaluations.get(evaluation_id)
            if previous is None or event.event_id > previous[0]:
                evaluations[evaluation_id] = (event.event_id, dict(payload))

    resolved = 0
    unresolved = 0
    false_blocks = 0
    structurally_unobservable = 0
    application_without_evaluation = 0
    evaluation_without_application = malformed_evaluation_count
    application_id_counts = Counter(
        application_id
        for application in application_events
        if isinstance((application_id := application.get("application_id")), str)
    )
    known_application_ids = set(application_id_counts)
    valid_applications: dict[str, dict[str, object]] = {}
    invalid_application_count = 0
    for application in application_events:
        application_id = application.get("application_id")
        if (
            not isinstance(application_id, str)
            or application_id_counts[application_id] != 1
            or not _is_complete_hard_probe_application(
                application,
                expected_rule_fingerprint=expected_rule_fingerprint,
                expected_predicate_fingerprint=expected_predicate_fingerprint,
            )
        ):
            invalid_application_count += 1
            continue
        valid_applications[application_id] = application

    evaluations_by_application: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for _, evaluation in evaluations.values():
        application_id = evaluation.get("application_id")
        if not isinstance(application_id, str) or application_id not in known_application_ids:
            evaluation_without_application += 1
            continue
        evaluations_by_application[application_id].append(evaluation)

    for application_id, application in valid_applications.items():
        matching_evaluations = evaluations_by_application[application_id]
        if not matching_evaluations:
            application_without_evaluation += 1
            invalid_application_count += 1
            continue
        if len(matching_evaluations) != 1:
            invalid_application_count += 1
            evaluation_without_application += len(matching_evaluations) - 1
            continue
        evaluation = matching_evaluations[0]
        if not _is_complete_hard_probe_evaluation(
            evaluation,
            expected_rule_fingerprint=expected_rule_fingerprint,
            expected_predicate_fingerprint=expected_predicate_fingerprint,
        ) or _application_evaluation_schema_mismatch(application, evaluation):
            invalid_application_count += 1
            continue
        if evaluation.get("actual_outcome") == "not_selected":
            if (
                evaluation.get("counterfactual_observable") is not False
                or evaluation.get("execution_false_block", evaluation.get("false_block"))
                is not None
            ):
                invalid_application_count += 1
                continue
            structurally_unobservable += 1
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

    invalid = invalid_application_count + evaluation_without_application
    return HardQualificationEvaluationMetrics(
        shadow_would_block_application_count=len(application_events),
        resolved_counterfactual_count=resolved,
        unresolved_counterfactual_count=unresolved,
        execution_false_block_count=false_blocks,
        invalid_counterfactual_evidence_count=invalid,
        structurally_unobservable_counterfactual_count=structurally_unobservable,
        invalid_application_count=invalid_application_count,
        application_without_evaluation_count=application_without_evaluation,
        evaluation_without_application_count=evaluation_without_application,
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
    structurally_unobservable = sum(
        item.structurally_unobservable_counterfactual_count for item in ordered_runs
    )
    invalid_applications = sum(item.invalid_application_count for item in ordered_runs)
    application_without_evaluation = sum(
        item.application_without_evaluation_count for item in ordered_runs
    )
    evaluation_without_application = sum(
        item.evaluation_without_application_count for item in ordered_runs
    )
    fingerprints = {(item.rule_fingerprint, item.predicate_fingerprint) for item in ordered_runs}
    if len(fingerprints) != 1:
        raise ValueError("hard qualification rejected: fingerprint_identity")
    rule_fingerprint, predicate_fingerprint = next(iter(fingerprints))
    return PlaybookHardQualificationManifest(
        schema_version="1.2",
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
        counterfactual_structurally_unobservable_count=structurally_unobservable,
        counterfactual_invalid_application_count=invalid_applications,
        application_without_evaluation_count=application_without_evaluation,
        evaluation_without_application_count=evaluation_without_application,
        rule_fingerprint=rule_fingerprint,
        predicate_fingerprint=predicate_fingerprint,
        shadow_state_count=parent.shadow_state_count,
        execution_false_block_count=parent.false_block_count,
        execution_false_block_rate=parent.false_block_rate,
        typed_retry_opportunity_count_by_seed={
            str(item.seed_id): item.typed_retry_opportunity_count
            for item in ordered_runs
            if item.typed_retry_opportunity_count is not None
        },
        typed_retry_application_count_by_seed={
            str(item.seed_id): item.typed_retry_application_count
            for item in ordered_runs
            if item.typed_retry_application_count is not None
        },
        typed_retry_coverage_unavailable_by_seed={
            str(item.seed_id): tuple(item.typed_retry_coverage_reasons or ())
            for item in ordered_runs
            if item.typed_retry_coverage_unavailable or item.typed_retry_coverage_reasons
        },
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
    if parent.category is PlaybookRuleCategory.EXECUTION_GUARD and parent.retry_guard is None:
        reasons.append("parent_missing_typed_retry_binding")
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
    if any(
        not item.application_conservation_valid
        or item.invalid_application_count
        or item.application_without_evaluation_count
        or item.evaluation_without_application_count
        for item in runs
    ):
        reasons.append("application_conservation")
    if len({(item.rule_fingerprint, item.predicate_fingerprint) for item in runs}) != 1:
        reasons.append("fingerprint_identity")
    expected_fingerprints = _probe_fingerprints(parent)
    if any(
        item.rule_fingerprint != expected_fingerprints[0]
        or item.predicate_fingerprint != expected_fingerprints[1]
        for item in runs
    ):
        reasons.append("fingerprint_mismatch")
    if any(item.shadow_would_block_application_count == 0 for item in runs):
        reasons.append("shadow_would_block_coverage")
    typed_retry_required = (
        parent.retry_guard is not None
        and evaluation_kind(parent.category).value == "execution_guard"
    )
    typed_retry_present = typed_retry_required or any(
        item.typed_retry_opportunity_count is not None
        or item.typed_retry_application_count is not None
        or item.typed_retry_coverage_unavailable is not None
        or item.typed_retry_coverage_reasons is not None
        for item in runs
    )
    if typed_retry_present:
        if any(
            item.typed_retry_coverage_unavailable is not False
            or item.typed_retry_opportunity_count is None
            or item.typed_retry_application_count is None
            or item.typed_retry_coverage_reasons is None
            or (typed_retry_required and item.config_sha256 is None)
            for item in runs
        ):
            reasons.append("typed_retry_coverage_unavailable")
        if any(
            item.typed_retry_opportunity_count is not None
            and item.typed_retry_opportunity_count <= 0
            for item in runs
        ):
            reasons.append("typed_retry_opportunity_coverage")
        if any(
            item.typed_retry_application_count is not None
            and item.typed_retry_application_count <= 0
            for item in runs
        ):
            reasons.append("typed_retry_application_coverage")
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


def _is_complete_hard_probe_application(
    application: dict[str, object],
    *,
    expected_rule_fingerprint: str | None,
    expected_predicate_fingerprint: str | None,
) -> bool:
    try:
        observed = PlaybookRuleApplication.model_validate(application)
    except ValueError:
        return False
    if (
        observed.reason != "shadow_would_block"
        or observed.blocked is not False
        or observed.rule_kind is None
        or observed.rule_kind.value != "execution_guard"
        or not _is_fingerprint(observed.rule_fingerprint)
        or not _is_fingerprint(observed.predicate_fingerprint)
    ):
        return False
    if (
        expected_rule_fingerprint is not None
        and observed.rule_fingerprint != expected_rule_fingerprint
    ):
        return False
    if (
        expected_predicate_fingerprint is not None
        and observed.predicate_fingerprint != expected_predicate_fingerprint
    ):
        return False
    return True


def _is_complete_hard_probe_evaluation(
    evaluation: dict[str, object],
    *,
    expected_rule_fingerprint: str | None,
    expected_predicate_fingerprint: str | None,
) -> bool:
    try:
        observed = PlaybookRuleEvaluation.model_validate(evaluation)
    except ValueError:
        return False
    if (
        observed.strength_at_evaluation is not PlaybookRuleStrength.HARD
        or observed.status_at_evaluation is not PlaybookRuleStatus.ACTIVE
        or observed.shadow_decision != "would_block"
        or observed.rule_kind.value != "execution_guard"
        or observed.counterfactual_signature is None
        or observed.behavior_before_hash is None
        or observed.decision_epoch is None
        or not _is_fingerprint(observed.rule_fingerprint)
        or not _is_fingerprint(observed.predicate_fingerprint)
    ):
        return False
    if (
        expected_rule_fingerprint is not None
        and observed.rule_fingerprint != expected_rule_fingerprint
    ):
        return False
    if (
        expected_predicate_fingerprint is not None
        and observed.predicate_fingerprint != expected_predicate_fingerprint
    ):
        return False
    return True


def _application_evaluation_schema_mismatch(
    application: dict[str, object],
    evaluation: dict[str, object],
) -> bool:
    """Ensure the evaluation is evidence for the exact emitted application."""

    for field in (
        "application_id",
        "rule_id",
        "run_id",
        "episode_id",
        "target_kind",
        "target_id",
        "rule_kind",
        "rule_fingerprint",
        "predicate_fingerprint",
    ):
        if application.get(field) != evaluation.get(field):
            return True
    return False


def _is_fingerprint(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in string.hexdigits for character in value)
        and value == value.lower()
    )
