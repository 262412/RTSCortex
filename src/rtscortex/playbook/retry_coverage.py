"""Replay exact typed retry opportunities from immutable run journals.

The live candidate guard owns the retry predicate.  This module only rebuilds
the runtime inputs (terminal feedback and candidate lineage) from journal
events and then calls :func:`typed_retry_binding_matches`; it does not provide
an approximate historical matcher.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from rtscortex.memory import StoredEvent
from rtscortex.playbook.guards import (
    RecentTerminalFeedback,
    candidate_signature,
    typed_candidate_predicate_matches,
    typed_retry_binding_matches,
)
from rtscortex.playbook.models import (
    PlaybookRetryGuardBinding,
    PlaybookRule,
    playbook_predicate_fingerprint,
    playbook_rule_fingerprint,
)


@dataclass(frozen=True, slots=True)
class TypedRetryOpportunityBinding:
    """Exact runtime identity and freshness proof for one typed opportunity."""

    opportunity_id: str
    failure_code: str
    candidate_signature: str
    operation_id: str
    next_attempt_ordinal: int
    candidate_game_loop: int
    terminal_game_loop: int
    freshness_age_game_loops: int
    max_age_game_loops: int
    rule_fingerprint: str
    predicate_fingerprint: str

    def as_dict(self) -> dict[str, object]:
        return {
            "opportunity_id": self.opportunity_id,
            "failure_code": self.failure_code,
            "candidate_signature": self.candidate_signature,
            "operation_id": self.operation_id,
            "next_attempt_ordinal": self.next_attempt_ordinal,
            "candidate_game_loop": self.candidate_game_loop,
            "terminal_game_loop": self.terminal_game_loop,
            "freshness_age_game_loops": self.freshness_age_game_loops,
            "max_age_game_loops": self.max_age_game_loops,
            "rule_fingerprint": self.rule_fingerprint,
            "predicate_fingerprint": self.predicate_fingerprint,
        }


@dataclass(frozen=True, slots=True)
class TypedRetryCoverage:
    """Typed retry coverage reconstructed for one source seed and rule."""

    typed_retry_opportunity_count: int = 0
    typed_retry_application_count: int = 0
    unavailable_reasons: tuple[str, ...] = ()
    opportunity_ids: tuple[str, ...] = ()
    application_ids: tuple[str, ...] = ()
    opportunity_bindings: tuple[TypedRetryOpportunityBinding, ...] = ()

    @property
    def typed_retry_coverage_unavailable(self) -> bool:
        return bool(self.unavailable_reasons)

    @property
    def available(self) -> bool:
        return not self.typed_retry_coverage_unavailable

    def as_dict(self) -> dict[str, object]:
        return {
            "typed_retry_opportunity_count": self.typed_retry_opportunity_count,
            "typed_retry_application_count": self.typed_retry_application_count,
            "typed_retry_coverage_unavailable": self.typed_retry_coverage_unavailable,
            "unavailable_reasons": list(self.unavailable_reasons),
            "opportunity_ids": list(self.opportunity_ids),
            "application_ids": list(self.application_ids),
            "opportunity_bindings": [binding.as_dict() for binding in self.opportunity_bindings],
        }


@dataclass(frozen=True, slots=True)
class TypedRetryCoverageBySeed:
    """Coverage and fail-closed reasons for all source seeds of one rule."""

    by_seed: dict[int, TypedRetryCoverage] = field(default_factory=dict)

    @property
    def typed_retry_opportunity_count_by_seed(self) -> dict[int, int]:
        return {
            seed: coverage.typed_retry_opportunity_count
            for seed, coverage in sorted(self.by_seed.items())
        }

    @property
    def typed_retry_application_count_by_seed(self) -> dict[int, int]:
        return {
            seed: coverage.typed_retry_application_count
            for seed, coverage in sorted(self.by_seed.items())
        }

    @property
    def unavailable_reasons_by_seed(self) -> dict[int, tuple[str, ...]]:
        return {
            seed: coverage.unavailable_reasons
            for seed, coverage in sorted(self.by_seed.items())
            if coverage.unavailable_reasons
        }

    @property
    def typed_retry_coverage_unavailable(self) -> bool:
        return any(coverage.typed_retry_coverage_unavailable for coverage in self.by_seed.values())

    @property
    def has_real_opportunity_every_seed(self) -> bool:
        return bool(self.by_seed) and all(
            coverage.available and coverage.typed_retry_opportunity_count > 0
            for coverage in self.by_seed.values()
        )

    @property
    def has_real_application_every_seed(self) -> bool:
        return bool(self.by_seed) and all(
            coverage.available and coverage.typed_retry_application_count > 0
            for coverage in self.by_seed.values()
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "typed_retry_opportunity_count_by_seed": {
                str(seed): count
                for seed, count in self.typed_retry_opportunity_count_by_seed.items()
            },
            "typed_retry_application_count_by_seed": {
                str(seed): count
                for seed, count in self.typed_retry_application_count_by_seed.items()
            },
            "typed_retry_coverage_unavailable": self.typed_retry_coverage_unavailable,
            "unavailable_reasons_by_seed": {
                str(seed): list(reasons)
                for seed, reasons in self.unavailable_reasons_by_seed.items()
            },
            "by_seed": {
                str(seed): coverage.as_dict() for seed, coverage in sorted(self.by_seed.items())
            },
        }


@dataclass(frozen=True, slots=True)
class _CandidateLineage:
    candidate_id: str
    operation_id: str
    attempt_ordinal: int
    command_id: str | None
    situation_assessment_id: str | None
    game_loop: int
    event_id: int


@dataclass(frozen=True, slots=True)
class _TerminalFailure:
    feedback: RecentTerminalFeedback
    command_id: str | None
    event_id: int


def analyze_typed_retry_coverage(
    events: Iterable[StoredEvent],
    *,
    rule: PlaybookRule,
    expected_rule_fingerprint: str | None = None,
    expected_predicate_fingerprint: str | None = None,
    context_values: Mapping[str, object] | None = None,
) -> TypedRetryCoverage:
    """Reconstruct runtime-typed retry opportunities and applications.

    A candidate only counts when its terminal failure, exact candidate
    signature, semantic operation lineage, next attempt ordinal, freshness
    window, and predicate fingerprint all survive journal reconstruction.
    Missing or mismatched source fields are reported as unavailable instead of
    being replaced by ``shadow_state_count``.
    """

    event_list = tuple(events)
    context_values = {} if context_values is None else dict(context_values)
    binding = rule.retry_guard
    if not isinstance(binding, PlaybookRetryGuardBinding):
        return TypedRetryCoverage(unavailable_reasons=("missing_typed_retry_binding",))

    expected_rule_fingerprint = expected_rule_fingerprint or playbook_rule_fingerprint(rule)
    expected_predicate_fingerprint = (
        expected_predicate_fingerprint or playbook_predicate_fingerprint(rule)
    )

    lineages_by_candidate: dict[str, list[_CandidateLineage]] = defaultdict(list)
    lineages_by_command: dict[str, list[_CandidateLineage]] = defaultdict(list)
    unavailable: set[str] = set()
    for event in event_list:
        if event.event_type != "command_lineage":
            continue
        lineage = event.payload.get("lineage")
        payload = lineage if isinstance(lineage, dict) else event.payload
        candidate_id = _nonempty_string(payload.get("candidate_id"))
        operation_id = _nonempty_string(payload.get("operation_id"))
        attempt_ordinal = _nonnegative_int(payload.get("attempt_ordinal"))
        command_id = _optional_string(payload.get("command_id"))
        situation_assessment_id = _optional_string(payload.get("situation_assessment_id"))
        game_loop = _nonnegative_int(
            payload.get("selected_game_loop", payload.get("game_loop", event.step_id))
        )
        if candidate_id is None or operation_id is None or attempt_ordinal is None:
            unavailable.add("missing_operation_lineage")
            continue
        if game_loop is None:
            unavailable.add("missing_operation_lineage")
            continue
        item = _CandidateLineage(
            candidate_id=candidate_id,
            operation_id=operation_id,
            attempt_ordinal=attempt_ordinal,
            command_id=command_id,
            situation_assessment_id=situation_assessment_id,
            game_loop=game_loop,
            event_id=event.event_id,
        )
        lineages_by_candidate[candidate_id].append(item)
        if command_id is not None:
            lineages_by_command[command_id].append(item)

    failures: list[_TerminalFailure] = []
    for event in event_list:
        if event.event_type == "playbook_case_recorded":
            raw_feedback = event.payload.get("retry_feedback")
            if raw_feedback is None:
                continue
            feedback = _feedback_from_payload(raw_feedback, binding=binding)
            if feedback is None:
                action_name = (
                    raw_feedback.get("action_name") if isinstance(raw_feedback, dict) else None
                )
                if action_name == _bound_action(rule):
                    unavailable.add("incomplete_terminal_feedback")
                continue
            failures.append(
                _TerminalFailure(
                    feedback=feedback,
                    command_id=_optional_string(event.payload.get("command_id")),
                    event_id=event.event_id,
                )
            )
        elif event.event_type == "execution":
            # Older journals may not have playbook_case_recorded yet.  Only
            # synthesize feedback when all signature inputs are present.
            if event.payload.get("action_name") != _bound_action(rule):
                continue
            if event.payload.get("failure_code") != binding.failure_code:
                continue
            if event.payload.get("status") not in (None, "failed"):
                continue
            feedback = _feedback_from_execution(event.payload, binding=binding)
            if feedback is None:
                unavailable.add("incomplete_terminal_feedback")
                continue
            failures.append(
                _TerminalFailure(
                    feedback=feedback,
                    command_id=_optional_string(event.payload.get("command_id")),
                    event_id=event.event_id,
                )
            )

    # A source may contain both an execution report and its derived case.  The
    # case carries the same typed feedback; keep the higher-level case only.
    failures = _deduplicate_failures(failures)

    candidate_applications = _candidate_applications(
        event_list,
        rule_id=rule.rule_id,
        expected_rule_fingerprint=expected_rule_fingerprint,
        expected_predicate_fingerprint=expected_predicate_fingerprint,
        unavailable=unavailable,
    )
    assessments = {
        assessment_id: _assessment_values(event.payload)
        for event in event_list
        if event.event_type == "situation_assessed"
        for assessment_id in [_optional_string(event.payload.get("assessment_id"))]
        if assessment_id is not None
    }

    opportunity_ids: list[str] = []
    application_ids: list[str] = []
    opportunity_bindings: list[TypedRetryOpportunityBinding] = []

    # The source run generally predates the parent rule, so it has no
    # ``playbook_rule_applied`` event.  Rebuild hypothetical applications from
    # the immutable candidate set and the selected command lineage instead.
    # This still uses the same condition/action/role and typed retry matcher as
    # the live candidate guard; no broad shadow counter is consulted.
    candidate_replays = _candidate_replays(
        event_list,
        lineages_by_candidate=lineages_by_candidate,
        rule=rule,
        known_candidate_ids={
            candidate_id
            for _, _, application in candidate_applications
            if application.get("rule_id") == rule.rule_id
            for candidate_id in [_nonempty_string(application.get("target_id"))]
            if candidate_id is not None
        },
        unavailable=unavailable,
    )
    candidate_applications = [*candidate_applications, *candidate_replays]
    for application_id, app_event_id, application in candidate_applications:
        candidate_id = _nonempty_string(application.get("target_id"))
        signature = _fingerprint(application.get("counterfactual_signature"))
        game_loop = _nonnegative_int(application.get("game_loop"))
        action_name = _nonempty_string(application.get("action_name"))
        if candidate_id is None or signature is None or game_loop is None or action_name is None:
            # This is only a hard error for the parent's own application.  A
            # terminal-feedback application has no candidate identity and is
            # intentionally ignored by _candidate_applications.
            if application.get("rule_id") == rule.rule_id:
                unavailable.add("incomplete_candidate_application")
            continue
        lineage = _lineage_for_application(
            lineages_by_candidate.get(candidate_id, ()),
            application_event_id=app_event_id,
            game_loop=game_loop,
        )
        if lineage is None:
            unavailable.add("missing_operation_lineage")
            continue
        predicate_match = _runtime_predicate_matches(
            rule,
            application,
            lineage=lineage,
            assessments=assessments,
            context_values=context_values,
        )
        if predicate_match is None:
            unavailable.add("missing_retry_predicate_inputs")
            continue
        if not predicate_match:
            continue
        matching_failure = next(
            (
                failure
                for failure in failures
                if typed_retry_binding_matches(
                    binding,
                    target_kind="candidate",
                    counterfactual_signature=signature,
                    recent_feedback=(failure.feedback,),
                    operation_id=lineage.operation_id,
                    attempt_ordinal=lineage.attempt_ordinal,
                    game_loop=game_loop,
                )
                and failure.feedback.action_name == action_name
            ),
            None,
        )
        if matching_failure is None:
            continue
        opportunity_id = (
            f"{matching_failure.feedback.operation_id}|{signature}|{lineage.attempt_ordinal}|"
            f"{game_loop}"
        )
        if opportunity_id not in opportunity_ids:
            opportunity_ids.append(opportunity_id)
            opportunity_bindings.append(
                TypedRetryOpportunityBinding(
                    opportunity_id=opportunity_id,
                    failure_code=matching_failure.feedback.failure_code,
                    candidate_signature=signature,
                    operation_id=lineage.operation_id,
                    next_attempt_ordinal=lineage.attempt_ordinal,
                    candidate_game_loop=game_loop,
                    terminal_game_loop=matching_failure.feedback.terminal_game_loop,
                    freshness_age_game_loops=(
                        game_loop - matching_failure.feedback.terminal_game_loop
                    ),
                    max_age_game_loops=binding.max_age_game_loops,
                    rule_fingerprint=expected_rule_fingerprint,
                    predicate_fingerprint=expected_predicate_fingerprint,
                )
            )
        if application.get("rule_id") == rule.rule_id and application_id is not None:
            if application_id not in application_ids:
                application_ids.append(application_id)

    if candidate_applications and not lineages_by_candidate:
        unavailable.add("missing_operation_lineage")
    return TypedRetryCoverage(
        typed_retry_opportunity_count=len(opportunity_ids),
        typed_retry_application_count=len(application_ids),
        unavailable_reasons=tuple(sorted(unavailable)),
        opportunity_ids=tuple(opportunity_ids),
        application_ids=tuple(application_ids),
        opportunity_bindings=tuple(opportunity_bindings),
    )


def analyze_typed_retry_coverage_by_seed(
    events_by_seed: Mapping[int, Iterable[StoredEvent]],
    *,
    rule: PlaybookRule,
    expected_rule_fingerprint: str | None = None,
    expected_predicate_fingerprint: str | None = None,
    context_values_by_seed: Mapping[int, Mapping[str, object]] | None = None,
) -> TypedRetryCoverageBySeed:
    """Replay one exact typed matcher per source seed."""

    return TypedRetryCoverageBySeed(
        by_seed={
            int(seed): analyze_typed_retry_coverage(
                events,
                rule=rule,
                expected_rule_fingerprint=expected_rule_fingerprint,
                expected_predicate_fingerprint=expected_predicate_fingerprint,
                context_values=(
                    None
                    if context_values_by_seed is None
                    else context_values_by_seed.get(int(seed))
                ),
            )
            for seed, events in sorted(events_by_seed.items())
        }
    )


def _candidate_applications(
    events: Sequence[StoredEvent],
    *,
    rule_id: str,
    expected_rule_fingerprint: str,
    expected_predicate_fingerprint: str,
    unavailable: set[str],
) -> list[tuple[str | None, int, dict[str, Any]]]:
    applications: list[tuple[str | None, int, dict[str, Any]]] = []
    for event in events:
        if event.event_type != "playbook_rule_applied":
            continue
        payload = dict(event.payload)
        if payload.get("target_kind") != "candidate":
            continue
        # Applications emitted for another rule are unrelated to this
        # parent's typed retry opportunity.  Ignore them before validating
        # fingerprints so malformed evidence from a sibling rule cannot make
        # this parent's coverage unavailable.
        if payload.get("rule_id") != rule_id:
            continue
        parent_fingerprint_valid = True
        if payload.get("rule_fingerprint") != expected_rule_fingerprint:
            unavailable.add("rule_fingerprint_mismatch")
            parent_fingerprint_valid = False
        if payload.get("predicate_fingerprint") != expected_predicate_fingerprint:
            unavailable.add("predicate_fingerprint_mismatch")
            parent_fingerprint_valid = False
        # The terminal-feedback cooldown application is deliberately excluded;
        # it has no candidate signature and is not a Playbook rule decision.
        if payload.get("reason") in {
            "recent_terminal_failure_block",
            "recent_terminal_failure_cooldown",
        }:
            continue
        if payload.get("rule_id") == rule_id and not parent_fingerprint_valid:
            continue
        applications.append(
            (
                _optional_string(payload.get("application_id")),
                event.event_id,
                payload,
            )
        )
    return applications


def _candidate_replays(
    events: Sequence[StoredEvent],
    *,
    lineages_by_candidate: Mapping[str, Sequence[_CandidateLineage]],
    rule: PlaybookRule,
    known_candidate_ids: set[str],
    unavailable: set[str],
) -> list[tuple[str | None, int, dict[str, Any]]]:
    """Materialize hypothetical candidate applications from source journals.

    Natural-terminal source runs usually were recorded before the parent rule
    existed, and therefore contain no ``playbook_rule_applied`` rows.  The
    candidate-set and command-lineage records are nevertheless immutable
    runtime inputs, so pair them to recover the exact candidate identity and
    role that a live candidate guard would have evaluated.
    """

    candidate_sets: dict[str, list[tuple[int, str | None, dict[str, Any]]]] = defaultdict(list)
    for event in events:
        if event.event_type != "candidate_set_built":
            continue
        role = _optional_string(event.payload.get("role"))
        raw_candidates = event.payload.get("candidates")
        if not isinstance(raw_candidates, (list, tuple)):
            unavailable.add("missing_candidate_set")
            continue
        for raw_candidate in raw_candidates:
            if not isinstance(raw_candidate, dict):
                unavailable.add("missing_candidate_set")
                continue
            candidate_id = _nonempty_string(raw_candidate.get("candidate_id"))
            if candidate_id is None:
                unavailable.add("missing_candidate_set")
                continue
            candidate_sets[candidate_id].append((event.event_id, role, dict(raw_candidate)))

    replays: list[tuple[str | None, int, dict[str, Any]]] = []
    for candidate_id, lineages in lineages_by_candidate.items():
        observed_sets = candidate_sets.get(candidate_id, ())
        if not observed_sets:
            if candidate_id not in known_candidate_ids:
                unavailable.add("missing_candidate_set")
            continue
        for lineage in lineages:
            prior_sets = [item for item in observed_sets if item[0] <= lineage.event_id]
            if not prior_sets:
                unavailable.add("missing_candidate_set")
                continue
            _, role, candidate = max(prior_sets, key=lambda item: item[0])
            action_name = _nonempty_string(candidate.get("action_name"))
            actor = _nonempty_string(candidate.get("actor"))
            arguments = candidate.get("arguments")
            if action_name is None or actor is None or not isinstance(arguments, (list, tuple)):
                unavailable.add("incomplete_candidate_set")
                continue
            replays.append(
                (
                    None,
                    lineage.event_id,
                    {
                        "rule_id": rule.rule_id,
                        "target_kind": "candidate",
                        "target_id": candidate_id,
                        "action_name": action_name,
                        "role": role,
                        "counterfactual_signature": candidate_signature(
                            action_name,
                            actor,
                            arguments,
                        ),
                        "game_loop": lineage.game_loop,
                    },
                )
            )
    return replays


def _lineage_for_application(
    lineages: Sequence[_CandidateLineage],
    *,
    application_event_id: int,
    game_loop: int,
) -> _CandidateLineage | None:
    """Bind a guard application to selection lineage from the same decision loop.

    Candidate guard events are persisted before the selected command lineage,
    so event ordering alone cannot identify the attempt. The runtime's exact
    selected game loop is authoritative; event IDs only disambiguate records
    from that loop.
    """

    matching = [lineage for lineage in lineages if lineage.game_loop == game_loop]
    if not matching:
        return None
    after_application = [
        lineage for lineage in matching if lineage.event_id >= application_event_id
    ]
    if after_application:
        return min(after_application, key=lambda lineage: lineage.event_id)
    return max(matching, key=lambda lineage: lineage.event_id)


def _assessment_values(payload: Mapping[str, object]) -> dict[str, object]:
    raw_assessment = payload.get("assessment")
    source = raw_assessment if isinstance(raw_assessment, Mapping) else payload
    values: dict[str, object] = {}
    for condition_field in (
        "phase",
        "threat_level",
        "economy_status",
        "army_readiness",
    ):
        value = source.get(condition_field)
        if isinstance(value, str) and value:
            values[condition_field] = value
    tags = source.get("tags")
    if isinstance(tags, (list, tuple)):
        values["alert"] = tuple(item for item in tags if isinstance(item, str))
    return values


def _runtime_predicate_matches(
    rule: PlaybookRule,
    application: Mapping[str, object],
    *,
    lineage: _CandidateLineage,
    assessments: Mapping[str, Mapping[str, object]],
    context_values: Mapping[str, object],
) -> bool | None:
    """Rebuild the same values consumed by ``guards._evaluate``.

    ``None`` means a required retry-time value was not observable; callers must
    fail closed rather than treating it as a non-match or using broad counts.
    """

    values = dict(context_values)
    assessment_id = lineage.situation_assessment_id
    if assessment_id is not None:
        assessment = assessments.get(assessment_id)
        if assessment is None:
            return None
        values.update(assessment)
    required_fields = {condition.field for condition in rule.conditions}
    if any(field not in values for field in required_fields):
        return None
    action_name = _nonempty_string(application.get("action_name"))
    if action_name is None:
        return None
    values["action_name"] = action_name
    role = _optional_string(application.get("role"))
    if rule.role_ids or "role" in required_fields:
        if role is None:
            return None
        values["role"] = role
    elif role is not None:
        values["role"] = role
    return typed_candidate_predicate_matches(
        rule,
        action_name=action_name,
        role=role,
        values=values,
    )


def _feedback_from_payload(
    raw: object,
    *,
    binding: PlaybookRetryGuardBinding,
) -> RecentTerminalFeedback | None:
    if not isinstance(raw, dict):
        return None
    action_name = _nonempty_string(raw.get("action_name"))
    failure_code = _nonempty_string(raw.get("failure_code"))
    signature = _fingerprint(raw.get("candidate_signature"))
    operation_id = _nonempty_string(raw.get("operation_id"))
    attempt_ordinal = _nonnegative_int(raw.get("attempt_ordinal"))
    terminal_game_loop = _nonnegative_int(raw.get("terminal_game_loop"))
    max_age = _positive_int(raw.get("max_age_game_loops"))
    if (
        action_name is None
        or failure_code is None
        or signature is None
        or operation_id is None
        or attempt_ordinal is None
        or terminal_game_loop is None
        or max_age is None
    ):
        return None
    return RecentTerminalFeedback(
        signature=signature,
        action_name=action_name,
        actor="historical",
        failure_code=failure_code,
        command_id="historical",
        operation_id=operation_id,
        attempt_ordinal=attempt_ordinal,
        terminal_game_loop=terminal_game_loop,
        expires_game_loop=terminal_game_loop + min(max_age, binding.max_age_game_loops),
        hard_suppression=False,
    )


def _feedback_from_execution(
    payload: Mapping[str, object],
    *,
    binding: PlaybookRetryGuardBinding,
) -> RecentTerminalFeedback | None:
    action_name = _nonempty_string(payload.get("action_name"))
    actor = _nonempty_string(payload.get("actor"))
    failure_code = _nonempty_string(payload.get("failure_code"))
    operation_id = _nonempty_string(payload.get("operation_id"))
    attempt_ordinal = _nonnegative_int(payload.get("attempt_ordinal"))
    if (
        action_name is None
        or actor is None
        or failure_code is None
        or operation_id is None
        or attempt_ordinal is None
    ):
        return None
    arguments = payload.get("requested_arguments") or payload.get("resolved_arguments")
    if not isinstance(arguments, (list, tuple)):
        return None
    signature = candidate_signature(action_name, actor, arguments)
    effect_evidence = payload.get("effect_evidence")
    terminal_game_loop = (
        _nonnegative_int(effect_evidence.get("accepted_game_loop"))
        if isinstance(effect_evidence, dict)
        else None
    )
    if terminal_game_loop is None:
        terminal_game_loop = _nonnegative_int(payload.get("step_id"))
    if terminal_game_loop is None:
        return None
    return RecentTerminalFeedback(
        signature=signature,
        action_name=action_name,
        actor=actor,
        failure_code=failure_code,
        command_id=_optional_string(payload.get("command_id")) or "historical",
        operation_id=operation_id,
        attempt_ordinal=attempt_ordinal,
        terminal_game_loop=terminal_game_loop,
        expires_game_loop=terminal_game_loop + binding.max_age_game_loops,
        hard_suppression=False,
    )


def _deduplicate_failures(failures: Sequence[_TerminalFailure]) -> list[_TerminalFailure]:
    observed: dict[tuple[str, str, str, int, int], _TerminalFailure] = {}
    for failure in failures:
        feedback = failure.feedback
        key = (
            feedback.failure_code,
            feedback.signature,
            feedback.operation_id,
            feedback.attempt_ordinal,
            feedback.terminal_game_loop,
        )
        observed[key] = max(observed.get(key, failure), failure, key=lambda item: item.event_id)
    return list(observed.values())


def _bound_action(rule: PlaybookRule) -> str | None:
    return rule.action_names[0] if len(rule.action_names) == 1 else None


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _nonempty_string(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _nonnegative_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _positive_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _fingerprint(value: object) -> str | None:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        return None
    return value if all(character in "0123456789abcdef" for character in value) else None
