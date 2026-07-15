"""Deterministic machine and Markdown reports for policy comparisons."""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from rtscortex.policy.hima.models import (
    HIMA_ADAPTER_VERSION,
    HIMA_PARSER_VERSION,
    HIMA_VOCABULARY_VERSION,
)
from rtscortex.policy.hima.subagent import HIMA_PINNED_REVISIONS
from rtscortex.policy.models import (
    MacroPolicyProposal,
    PolicyActionAssessment,
    PolicyActionClassification,
    PolicyFixtureStratum,
    PolicyShadowComparison,
    PolicyShadowRecord,
    PolicyShadowStatus,
)

COMPARISON_FILENAME = "comparison.json"
REPORT_FILENAME = "report.md"
_STRATA = tuple(item.value for item in PolicyFixtureStratum)
_UNLABELED_STRATUM = "unlabeled"
_CLASSIFICATIONS = tuple(item.value for item in PolicyActionClassification)
_MAPPED_CLASSIFICATIONS = (
    PolicyActionClassification.MAPPED_FUTURE.value,
    PolicyActionClassification.MAPPED_LEGAL_NOW.value,
    PolicyActionClassification.MAPPED_DEFERRED.value,
    PolicyActionClassification.ILLEGAL_ACTION.value,
    PolicyActionClassification.OBSOLETE.value,
)
_RUNTIME_OUTCOMES = (
    PolicyActionClassification.MAPPED_LEGAL_NOW.value,
    PolicyActionClassification.MAPPED_DEFERRED.value,
    PolicyActionClassification.ILLEGAL_ACTION.value,
    PolicyActionClassification.OBSOLETE.value,
)


@dataclass(frozen=True)
class PolicyComparisonReportArtifacts:
    """Files and deterministic aggregate data produced for one comparison."""

    comparison_path: Path
    report_path: Path
    summary: dict[str, object]


class PolicyComparisonReportError(ValueError):
    """Raised when policy comparison artifacts cannot be written."""


def build_policy_comparison_summary(
    comparison: PolicyShadowComparison,
) -> dict[str, object]:
    """Aggregate raw records using global counts rather than record-level rates."""

    fixture_strata = {
        fixture.fixture_id: (
            fixture.primary_stratum.value
            if fixture.primary_stratum is not None
            else _UNLABELED_STRATUM
        )
        for fixture in comparison.fixtures
    }
    corpus_strata = Counter(
        fixture_strata.get(fixture_id, _UNLABELED_STRATUM)
        for fixture_id in comparison.fixture_ids
    )
    for stratum in (*_STRATA, _UNLABELED_STRATUM):
        corpus_strata.setdefault(stratum, 0)

    candidate_ids = _ordered_unique(
        [
            *comparison.candidate_ids,
            *(record.spec.subagent_id for record in comparison.records),
        ]
    )
    candidates: dict[str, object] = {}
    for candidate_id in candidate_ids:
        records = [
            record
            for record in comparison.records
            if record.spec.subagent_id == candidate_id
        ]
        candidate = _aggregate_records(records)
        candidate["by_stratum"] = {
            stratum: _aggregate_records(
                [
                    record
                    for record in records
                    if fixture_strata.get(record.fixture_id, _UNLABELED_STRATUM)
                    == stratum
                ]
            )
            for stratum in (*_STRATA, _UNLABELED_STRATUM)
        }
        candidates[candidate_id] = candidate

    fixture_runs = {fixture.observation.run_id for fixture in comparison.fixtures}
    fixture_episodes = {
        (fixture.observation.run_id, fixture.observation.episode_id)
        for fixture in comparison.fixtures
    }
    return {
        "format_version": "0.2",
        "comparison_version": comparison.comparison_version,
        "candidate_order": candidate_ids,
        "corpus": {
            "fixtures": len(comparison.fixture_ids),
            "runs": len(fixture_runs),
            "episodes": len(fixture_episodes),
            "labeled_fixtures": sum(corpus_strata[stratum] for stratum in _STRATA),
            "unlabeled_fixtures": corpus_strata[_UNLABELED_STRATUM],
            "strata": {
                stratum: corpus_strata[stratum]
                for stratum in (*_STRATA, _UNLABELED_STRATUM)
            },
        },
        "candidates": candidates,
    }


def render_policy_comparison_report(comparison: PolicyShadowComparison) -> str:
    """Render a concise report from the same global aggregates as the JSON API."""

    summary = build_policy_comparison_summary(comparison)
    corpus = _as_dict(summary["corpus"])
    strata = _as_dict(corpus["strata"])
    candidates = _as_dict(summary["candidates"])
    candidate_order = _as_string_list(summary["candidate_order"])

    lines = [
        "# Policy Comparison v0.2",
        "",
        "This is an offline, shadow-only comparison. No proposal in this report was "
        "dispatched to the RTSCortex Runtime, Bridge, or StarCraft II.",
        "",
        "## Corpus coverage",
        "",
        f"Fixtures: `{corpus['fixtures']}` across `{corpus['runs']}` runs and "
        f"`{corpus['episodes']}` episodes.",
        "",
        "| Primary stratum | Fixtures |",
        "|---|---:|",
    ]
    lines.extend(
        f"| `{stratum}` | {strata[stratum]} |"
        for stratum in (*_STRATA, _UNLABELED_STRATUM)
    )

    lines.extend(
        [
            "",
            "## Candidate availability and completion",
            "",
            "| Candidate | Model | Availability | Completed | Unavailable | "
            "Skipped | Failed | Completion | Latency p50 | Latency p95 |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for candidate_id in candidate_order:
        candidate = _as_dict(candidates[candidate_id])
        identity = _as_dict(candidate["identity"])
        availability = _as_dict(candidate["availability"])
        outcomes = _as_dict(candidate["outcomes"])
        latency = _as_dict(candidate["latency_ms"])
        lines.append(
            f"| `{candidate_id}` | `{identity['model_id']}` | "
            f"`{availability['status']}` | {outcomes['completed']} | "
            f"{outcomes['unavailable']} | {outcomes['skipped']} | "
            f"{outcomes['failed']} | {_format_ratio(candidate['completion_rate'])} | "
            f"{_format_latency(latency['p50'])} | "
            f"{_format_latency(latency['p95'])} |"
        )

    reasons = [
        (candidate_id, reason)
        for candidate_id in candidate_order
        for reason in _as_string_list(
            _as_dict(_as_dict(candidates[candidate_id])["availability"])["reasons"]
        )
    ]
    if reasons:
        lines.extend(["", "Availability notes:", ""])
        lines.extend(f"- `{candidate_id}`: {reason}" for candidate_id, reason in reasons)

    lines.extend(
        [
            "",
            "## Model and adapter provenance",
            "",
            "| Candidate | Model | Pinned revision | Observed revision | "
            "Adapter | Parser | Vocabulary |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for candidate_id in candidate_order:
        identity = _as_dict(_as_dict(candidates[candidate_id])["identity"])
        lines.append(
            f"| `{candidate_id}` | `{identity['model_id']}` | "
            f"{_format_versions(identity['pinned_revision'])} | "
            f"{_format_versions(identity['observed_model_revisions'])} | "
            f"{_format_versions(identity['adapter_versions'])} | "
            f"{_format_versions(identity['parser_versions'])} | "
            f"{_format_versions(identity['vocabulary_versions'])} |"
        )

    lines.extend(
        [
            "",
            "## Parser validity and Runtime mapping",
            "",
            "| Candidate | Count unit | Discovered | Parsed known | Parse errors | "
            "Parse validity | Runtime mapped | Unsupported | Mapping coverage |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for candidate_id in candidate_order:
        candidate = _as_dict(candidates[candidate_id])
        parser = _as_dict(candidate["parser"])
        mapping = _as_dict(candidate["mapping"])
        for unit in ("logical", "effective"):
            parser_unit = _as_dict(parser[unit])
            mapping_unit = _as_dict(mapping[unit])
            applicable = bool(parser["applicable"])
            lines.append(
                f"| `{candidate_id}` | `{unit}` | "
                f"{_format_when(applicable, parser_unit['discovered_actions'])} | "
                f"{_format_when(applicable, parser_unit['parsed_known_actions'])} | "
                f"{_format_when(applicable, parser_unit['parse_errors'])} | "
                f"{_format_when_ratio(applicable, parser_unit['parse_validity'])} | "
                f"{_format_when(applicable, mapping_unit['runtime_mapped_actions'])} | "
                f"{_format_when(applicable, mapping_unit['unsupported_by_runtime'])} | "
                f"{_format_when_ratio(applicable, mapping_unit['coverage'])} |"
            )

    lines.extend(
        [
            "",
            "## Logical and repeat-weighted classifications",
            "",
            "| Candidate | Count unit | Parse error | Unsupported | Future | Legal now | "
            "Deferred | Illegal | Obsolete | Total | Conserved |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for candidate_id in candidate_order:
        classification = _as_dict(_as_dict(candidates[candidate_id])["classification"])
        applicable = bool(classification["applicable"])
        for unit in ("logical", "effective"):
            metrics = _as_dict(classification[unit])
            counts = _as_dict(metrics["counts"])
            lines.append(
                f"| `{candidate_id}` | `{unit}` | "
                f"{_format_when(applicable, counts['parse_error'])} | "
                f"{_format_when(applicable, counts['unsupported_by_runtime'])} | "
                f"{_format_when(applicable, counts['mapped_future'])} | "
                f"{_format_when(applicable, counts['mapped_legal_now'])} | "
                f"{_format_when(applicable, counts['mapped_deferred'])} | "
                f"{_format_when(applicable, counts['illegal_action'])} | "
                f"{_format_when(applicable, counts['obsolete'])} | "
                f"{_format_when(applicable, metrics['total'])} | "
                f"{_format_when(applicable, metrics['conserved'])} |"
            )

    lines.extend(
        [
            "",
            "## Sequence and Runtime frontiers",
            "",
            "The sequence frontier is the first logical model action. The Runtime frontier "
            "skips actions that RTSCortex does not own, such as automatically managed Probe "
            "production. Soft availability gaps may be skipped to the earliest action that "
            "validates now; resource, supply, and prerequisite blockers preserve sequence "
            "order.",
            "",
            "| Candidate | Frontier | Count unit | Parse error | Unsupported | Future | "
            "Legal now | Deferred | Illegal | Obsolete | Total | Runtime conserved | "
            "Illegal rate |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|",
        ]
    )
    for candidate_id in candidate_order:
        candidate = _as_dict(candidates[candidate_id])
        frontier = _as_dict(candidate["frontier"])
        applicable = bool(frontier["applicable"])
        for frontier_name in ("sequence", "runtime"):
            frontier_group = _as_dict(frontier[frontier_name])
            for unit in ("logical", "effective"):
                metrics = _as_dict(frontier_group[unit])
                counts = _as_dict(metrics["counts"])
                lines.append(
                    f"| `{candidate_id}` | `{frontier_name}` | `{unit}` | "
                    f"{_format_when(applicable, counts['parse_error'])} | "
                    f"{_format_when(applicable, counts['unsupported_by_runtime'])} | "
                    f"{_format_when(applicable, counts['mapped_future'])} | "
                    f"{_format_when(applicable, counts['mapped_legal_now'])} | "
                    f"{_format_when(applicable, counts['mapped_deferred'])} | "
                    f"{_format_when(applicable, counts['illegal_action'])} | "
                    f"{_format_when(applicable, counts['obsolete'])} | "
                    f"{_format_when(applicable, metrics['total'])} | "
                    f"{_format_when(applicable, metrics['runtime_outcome_conserved'])} | "
                    f"{_format_when_ratio(applicable, metrics['illegal_action_rate'])} |"
                )

    lines.extend(
        [
            "",
            "## Goal progress and control safety",
            "",
            "| Candidate | Action proposals | Legal | Legal rate | Goal opportunities | "
            "Goal advancing | Advancement rate | Control violations |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for candidate_id in candidate_order:
        goal = _as_dict(_as_dict(candidates[candidate_id])["goal_and_control"])
        lines.append(
            f"| `{candidate_id}` | {goal['proposed_actions']} | "
            f"{goal['legal_actions']} | {_format_ratio(goal['legal_action_rate'])} | "
            f"{goal['goal_opportunity_proposals']} | "
            f"{goal['goal_advancing_actions']} | "
            f"{_format_ratio(goal['goal_advancing_action_rate'])} | "
            f"{goal['control_action_violations']} |"
        )

    lines.extend(
        [
            "",
            "## Availability and completion by corpus stratum",
            "",
            "| Stratum | Candidate | Fixtures | Completed | Unavailable | Skipped | "
            "Failed | Completion | Latency p50 | Latency p95 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for stratum in _STRATA:
        for candidate_id in candidate_order:
            by_stratum = _as_dict(_as_dict(candidates[candidate_id])["by_stratum"])
            metrics = _as_dict(by_stratum[stratum])
            outcomes = _as_dict(metrics["outcomes"])
            latency = _as_dict(metrics["latency_ms"])
            lines.append(
                f"| `{stratum}` | `{candidate_id}` | {metrics['fixtures']} | "
                f"{outcomes['completed']} | {outcomes['unavailable']} | "
                f"{outcomes['skipped']} | {outcomes['failed']} | "
                f"{_format_ratio(metrics['completion_rate'])} | "
                f"{_format_latency(latency['p50'])} | "
                f"{_format_latency(latency['p95'])} |"
            )

    lines.extend(
        [
            "",
            "## Logical Runtime outcomes by corpus stratum",
            "",
            "| Stratum | Candidate | Unsupported | Runtime legal | Deferred | Illegal | "
            "Obsolete |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for stratum in _STRATA:
        for candidate_id in candidate_order:
            by_stratum = _as_dict(_as_dict(candidates[candidate_id])["by_stratum"])
            metrics = _as_dict(by_stratum[stratum])
            parser = _as_dict(metrics["parser"])
            frontier = _as_dict(metrics["frontier"])
            lines.append(
                f"| `{stratum}` | `{candidate_id}` | "
                f"{_format_applicable_value(parser, 'unsupported_by_runtime')} | "
                f"{_format_applicable_value(frontier, 'mapped_legal_now')} | "
                f"{_format_applicable_value(frontier, 'mapped_deferred')} | "
                f"{_format_applicable_value(frontier, 'illegal_actions')} | "
                f"{_format_applicable_value(frontier, 'obsolete')} |"
            )

    lines.extend(
        [
            "",
            "## Quality by corpus stratum",
            "",
            "| Stratum | Candidate | Count unit | Parse validity | Mapping coverage | "
            "Classification conserved | Mapping denominator conserved | "
            "Goal advancement | Control violations |",
            "|---|---|---|---:|---:|---|---|---:|---:|",
        ]
    )
    for stratum in _STRATA:
        for candidate_id in candidate_order:
            by_stratum = _as_dict(_as_dict(candidates[candidate_id])["by_stratum"])
            metrics = _as_dict(by_stratum[stratum])
            parser = _as_dict(metrics["parser"])
            mapping = _as_dict(metrics["mapping"])
            classification = _as_dict(metrics["classification"])
            goal = _as_dict(metrics["goal_and_control"])
            applicable = bool(parser["applicable"])
            for unit in ("logical", "effective"):
                parser_unit = _as_dict(parser[unit])
                mapping_unit = _as_dict(mapping[unit])
                classification_unit = _as_dict(classification[unit])
                lines.append(
                    f"| `{stratum}` | `{candidate_id}` | `{unit}` | "
                    f"{_format_when_ratio(applicable, parser_unit['parse_validity'])} | "
                    f"{_format_when_ratio(applicable, mapping_unit['coverage'])} | "
                    f"{_format_when(applicable, classification_unit['conserved'])} | "
                    f"{_format_when(applicable, mapping_unit['denominator_conserved'])} | "
                    f"{_format_ratio(goal['goal_advancing_action_rate'])} | "
                    f"{goal['control_action_violations']} |"
                )

    lines.extend(
        [
            "",
            "## Metric definitions",
            "",
            "- All rates are computed from aggregate counts across records; "
            "per-record rates are not averaged.",
            "- Logical counts assign one unit to each action assessment; effective counts "
            "weight the same classification by its `repeat` value.",
            "- Logical and effective conservation are checked independently.",
            "- `parse_validity = parsed_known / (parsed_known + parse_error)` using the "
            "same logical or effective count unit.",
            "- `mapping_coverage = runtime_mapped / (runtime_mapped + unsupported)` using "
            "the same count unit; parse errors are excluded.",
            "- Runtime-frontier `illegal_action_rate = illegal / "
            "(legal_now + deferred + illegal + obsolete)`.",
            "- `unsupported_by_runtime`, `parse_error`, and `illegal_action` are separate "
            "terminal classifications and are never merged.",
            "- Native-only v0.1 records show `N/A` for HIMA parser, mapping, and frontier metrics.",
            "",
        ]
    )
    return "\n".join(lines)


def write_policy_comparison_reports(
    comparison: PolicyShadowComparison,
    output_dir: Path,
) -> PolicyComparisonReportArtifacts:
    """Write the lossless comparison JSON and a deterministic Markdown report."""

    resolved_output_dir = output_dir.expanduser().resolve()
    comparison_path = resolved_output_dir / COMPARISON_FILENAME
    report_path = resolved_output_dir / REPORT_FILENAME
    summary = build_policy_comparison_summary(comparison)
    report = render_policy_comparison_report(comparison)
    try:
        resolved_output_dir.mkdir(parents=True, exist_ok=True)
        comparison_path.write_text(
            json.dumps(
                comparison.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        report_path.write_text(report, encoding="utf-8")
    except OSError as error:
        raise PolicyComparisonReportError(
            f"Could not write policy comparison reports below {resolved_output_dir}: {error}"
        ) from error
    return PolicyComparisonReportArtifacts(
        comparison_path=comparison_path,
        report_path=report_path,
        summary=summary,
    )


def _aggregate_records(records: Sequence[PolicyShadowRecord]) -> dict[str, object]:
    status_counts = Counter(record.status.value for record in records)
    for status in PolicyShadowStatus:
        status_counts.setdefault(status.value, 0)
    availability_counts = Counter(record.availability.status.value for record in records)
    availability_statuses = sorted(availability_counts)
    if not availability_statuses:
        availability_status = "unknown"
    elif len(availability_statuses) == 1:
        availability_status = availability_statuses[0]
    else:
        availability_status = "mixed"
    availability_reasons = sorted(
        {
            record.availability.reason
            for record in records
            if record.availability.reason is not None
        }
    )

    identity = _candidate_identity(records)
    macro_records = [
        record
        for record in records
        if isinstance(record.proposal, MacroPolicyProposal)
    ]
    macro_applicable = bool(macro_records)
    assessments = [
        assessment
        for record in macro_records
        for assessment in record.action_assessments
    ]
    logical_counts = _classification_counts(assessments, effective=False)
    effective_counts = _classification_counts(assessments, effective=True)
    logical_classification = _classification_payload(
        logical_counts,
        expected_total=len(assessments),
        applicable=macro_applicable,
    )
    effective_classification = _classification_payload(
        effective_counts,
        expected_total=sum(assessment.repeat for assessment in assessments),
        applicable=macro_applicable,
    )
    logical_mapping = _mapping_payload(logical_counts, applicable=macro_applicable)
    effective_mapping = _mapping_payload(effective_counts, applicable=macro_applicable)
    sequence_frontier_assessments = [
        assessment for assessment in assessments if assessment.is_logical_frontier
    ]
    runtime_frontier_assessments = [
        assessment for assessment in assessments if assessment.is_runtime_frontier
    ]
    sequence_frontier = _frontier_payload(
        sequence_frontier_assessments,
        runtime=False,
        applicable=macro_applicable,
    )
    runtime_frontier = _frontier_payload(
        runtime_frontier_assessments,
        runtime=True,
        applicable=macro_applicable,
    )

    logical_total = _as_int(logical_classification["total"])
    logical_parse_errors = logical_counts[PolicyActionClassification.PARSE_ERROR.value]
    logical_unsupported = logical_counts[
        PolicyActionClassification.UNSUPPORTED_BY_RUNTIME.value
    ]
    logical_mapped_future = logical_counts[
        PolicyActionClassification.MAPPED_FUTURE.value
    ]
    logical_mapped_legal = logical_counts[
        PolicyActionClassification.MAPPED_LEGAL_NOW.value
    ]
    logical_mapped_deferred = logical_counts[
        PolicyActionClassification.MAPPED_DEFERRED.value
    ]
    logical_illegal = logical_counts[PolicyActionClassification.ILLEGAL_ACTION.value]
    logical_obsolete = logical_counts[PolicyActionClassification.OBSOLETE.value]
    logical_runtime_mapped = sum(logical_counts[name] for name in _MAPPED_CLASSIFICATIONS)
    logical_parsed = logical_total - logical_parse_errors

    proposed = sum(record.proposed_action_count for record in records)
    legal = sum(record.legal_action_count for record in records)
    goal_advancing = sum(record.goal_advancing_action_count for record in records)
    goal_opportunity_records = [
        record for record in records if record.goal_advancing_action_rate is not None
    ]
    goal_opportunity_proposals = sum(
        record.proposed_action_count for record in goal_opportunity_records
    )
    attempted_latencies = [
        record.latency_ms
        for record in records
        if record.status in {PolicyShadowStatus.COMPLETED, PolicyShadowStatus.FAILED}
    ]

    return {
        "fixtures": len(records),
        "identity": identity,
        "availability": {
            "status": availability_status,
            "counts": dict(sorted(availability_counts.items())),
            "reasons": availability_reasons,
        },
        "outcomes": {
            status.value: status_counts[status.value] for status in PolicyShadowStatus
        },
        "completion_rate": _ratio(
            status_counts[PolicyShadowStatus.COMPLETED.value],
            len(records),
        ),
        "latency_ms": {
            "sample_count": len(attempted_latencies),
            "p50": _percentile(attempted_latencies, 0.50),
            "p95": _percentile(attempted_latencies, 0.95),
        },
        "classification": {
            "applicable": macro_applicable,
            "logical": logical_classification,
            "effective": effective_classification,
        },
        "parser": {
            "applicable": macro_applicable,
            "logical": _parser_payload(logical_counts, applicable=macro_applicable),
            "effective": _parser_payload(effective_counts, applicable=macro_applicable),
            # Stable logical-count aliases retained for existing report consumers.
            "discovered_macro_steps": logical_total,
            "parsed_known_actions": logical_parsed,
            "effective_actions": _as_int(effective_classification["total"]),
            "parse_errors": logical_parse_errors,
            "parse_validity": (
                _ratio(logical_parsed, logical_total) if macro_applicable else None
            ),
            "unsupported_by_runtime": logical_unsupported,
            "runtime_mapped_actions": logical_runtime_mapped,
            "mapped_future": logical_mapped_future,
            "mapped_legal_now": logical_mapped_legal,
            "mapped_deferred": logical_mapped_deferred,
            "illegal_actions": logical_illegal,
            "obsolete": logical_obsolete,
            "classified_steps": _as_int(logical_classification["classified_total"]),
            "classification_conserved": logical_classification["conserved"],
        },
        "mapping": {
            "applicable": macro_applicable,
            "logical": logical_mapping,
            "effective": effective_mapping,
            "runtime_mapped_actions": logical_runtime_mapped,
            "unsupported_by_runtime": logical_unsupported,
            "mapping_coverage": logical_mapping["coverage"],
            "classification_conserved": logical_classification["conserved"],
            "parsed_mapping_conserved": logical_mapping["denominator_conserved"],
        },
        "frontier": {
            "applicable": macro_applicable,
            "sequence": sequence_frontier,
            "runtime": runtime_frontier,
            # Stable logical runtime-frontier aliases retained for report consumers.
            "mapped_legal_now": _frontier_count(
                runtime_frontier,
                "logical",
                PolicyActionClassification.MAPPED_LEGAL_NOW.value,
            ),
            "mapped_deferred": _frontier_count(
                runtime_frontier,
                "logical",
                PolicyActionClassification.MAPPED_DEFERRED.value,
            ),
            "illegal_actions": _frontier_count(
                runtime_frontier,
                "logical",
                PolicyActionClassification.ILLEGAL_ACTION.value,
            ),
            "obsolete": _frontier_count(
                runtime_frontier,
                "logical",
                PolicyActionClassification.OBSOLETE.value,
            ),
            "evaluated_actions": _nested_int(
                runtime_frontier,
                "logical",
                "evaluated_actions",
            ),
            "illegal_rate_denominator": _nested_int(
                runtime_frontier,
                "logical",
                "evaluated_actions",
            ),
            "illegal_action_rate": _nested_value(
                runtime_frontier,
                "logical",
                "illegal_action_rate",
            ),
            "excluded_unsupported_actions": logical_unsupported,
            "excluded_future_actions": logical_mapped_future,
            "excluded_parse_errors": logical_parse_errors,
        },
        "goal_and_control": {
            "proposed_actions": proposed,
            "legal_actions": legal,
            "legal_action_rate": _ratio(legal, proposed),
            "goal_opportunity_fixtures": len(goal_opportunity_records),
            "goal_opportunity_proposals": goal_opportunity_proposals,
            "goal_advancing_actions": goal_advancing,
            "goal_advancing_action_rate": _ratio(
                goal_advancing,
                goal_opportunity_proposals,
            ),
            "control_action_violations": sum(
                record.control_action_violation_count for record in records
            ),
        },
    }


def _candidate_identity(records: Sequence[PolicyShadowRecord]) -> dict[str, object]:
    first = records[0] if records else None
    model_id = first.spec.model_id if first is not None else ""
    macro_proposals = [
        record.proposal
        for record in records
        if isinstance(record.proposal, MacroPolicyProposal)
    ]
    generation_metadata = [
        proposal.generation_metadata
        for proposal in macro_proposals
        if proposal.generation_metadata is not None
    ]
    pinned_revision = HIMA_PINNED_REVISIONS.get(model_id)
    is_hima = pinned_revision is not None
    adapter_versions = {proposal.adapter_version for proposal in macro_proposals}
    parser_versions = {proposal.parser_version for proposal in macro_proposals}
    vocabulary_versions = {proposal.vocabulary_version for proposal in macro_proposals}
    if is_hima:
        adapter_versions.add(HIMA_ADAPTER_VERSION)
        parser_versions.add(HIMA_PARSER_VERSION)
        vocabulary_versions.add(HIMA_VOCABULARY_VERSION)
    return {
        "display_name": first.spec.display_name if first is not None else "",
        "model_id": model_id,
        "provider_kind": (
            first.spec.provider_kind.value if first is not None else "unknown"
        ),
        "pinned_revision": pinned_revision,
        "observed_model_revisions": sorted(
            {metadata.model_revision for metadata in generation_metadata}
        ),
        "adapter_versions": sorted(adapter_versions),
        "parser_versions": sorted(parser_versions),
        "vocabulary_versions": sorted(vocabulary_versions),
        "checkpoint_verified": (
            all(metadata.checkpoint_verified for metadata in generation_metadata)
            if generation_metadata
            else None
        ),
        "license_acknowledged": (
            all(metadata.license_acknowledged for metadata in generation_metadata)
            if generation_metadata
            else None
        ),
    }


def _classification_counts(
    assessments: Sequence[PolicyActionAssessment],
    *,
    effective: bool,
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for assessment in assessments:
        counts[assessment.classification.value] += assessment.repeat if effective else 1
    for classification in _CLASSIFICATIONS:
        counts.setdefault(classification, 0)
    return counts


def _classification_payload(
    counts: Counter[str],
    *,
    expected_total: int,
    applicable: bool,
) -> dict[str, object]:
    classified_total = sum(counts[name] for name in _CLASSIFICATIONS)
    return {
        "applicable": applicable,
        "counts": {name: counts[name] for name in _CLASSIFICATIONS},
        "total": expected_total,
        "classified_total": classified_total,
        "conserved": classified_total == expected_total,
    }


def _parser_payload(
    counts: Counter[str],
    *,
    applicable: bool,
) -> dict[str, object]:
    discovered = sum(counts[name] for name in _CLASSIFICATIONS)
    parse_errors = counts[PolicyActionClassification.PARSE_ERROR.value]
    parsed = discovered - parse_errors
    return {
        "applicable": applicable,
        "discovered_actions": discovered,
        "parsed_known_actions": parsed,
        "parse_errors": parse_errors,
        "parse_validity": _ratio(parsed, discovered) if applicable else None,
    }


def _mapping_payload(
    counts: Counter[str],
    *,
    applicable: bool,
) -> dict[str, object]:
    mapped = sum(counts[name] for name in _MAPPED_CLASSIFICATIONS)
    unsupported = counts[PolicyActionClassification.UNSUPPORTED_BY_RUNTIME.value]
    denominator = mapped + unsupported
    parse_errors = counts[PolicyActionClassification.PARSE_ERROR.value]
    classified = sum(counts[name] for name in _CLASSIFICATIONS)
    return {
        "applicable": applicable,
        "runtime_mapped_actions": mapped,
        "unsupported_by_runtime": unsupported,
        "matching_denominator": denominator,
        "coverage": _ratio(mapped, denominator) if applicable else None,
        "parse_errors_excluded": parse_errors,
        "denominator_conserved": denominator + parse_errors == classified,
    }


def _frontier_payload(
    assessments: Sequence[PolicyActionAssessment],
    *,
    runtime: bool,
    applicable: bool,
) -> dict[str, object]:
    logical_counts = _classification_counts(assessments, effective=False)
    effective_counts = _classification_counts(assessments, effective=True)
    return {
        "applicable": applicable,
        "logical": _frontier_unit_payload(
            logical_counts,
            expected_total=len(assessments),
            runtime=runtime,
        ),
        "effective": _frontier_unit_payload(
            effective_counts,
            expected_total=sum(assessment.repeat for assessment in assessments),
            runtime=runtime,
        ),
    }


def _frontier_unit_payload(
    counts: Counter[str],
    *,
    expected_total: int,
    runtime: bool,
) -> dict[str, object]:
    classified_total = sum(counts[name] for name in _CLASSIFICATIONS)
    evaluated = sum(counts[name] for name in _RUNTIME_OUTCOMES)
    illegal = counts[PolicyActionClassification.ILLEGAL_ACTION.value]
    unexpected = classified_total - evaluated if runtime else 0
    return {
        "counts": {name: counts[name] for name in _CLASSIFICATIONS},
        "total": expected_total,
        "classified_total": classified_total,
        "conserved": classified_total == expected_total,
        "evaluated_actions": evaluated,
        "unexpected_runtime_classifications": unexpected,
        "runtime_outcome_conserved": (evaluated == expected_total if runtime else None),
        "illegal_action_rate": _ratio(illegal, evaluated),
    }


def _frontier_count(frontier: dict[str, object], unit: str, classification: str) -> int:
    payload = _as_dict(frontier[unit])
    counts = _as_dict(payload["counts"])
    return _as_int(counts[classification])


def _nested_int(payload: dict[str, object], group: str, key: str) -> int:
    return _as_int(_as_dict(payload[group])[key])


def _nested_value(payload: dict[str, object], group: str, key: str) -> object:
    return _as_dict(payload[group])[key]


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * quantile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _ordered_unique(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _as_dict(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("policy report aggregate is not a dictionary")
    return value


def _as_string_list(value: object) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise TypeError("policy report aggregate is not a list of strings")
    return value


def _as_int(value: object) -> int:
    if not isinstance(value, int):
        raise TypeError("policy report aggregate is not an integer")
    return value


def _format_ratio(value: object) -> str:
    if value is None:
        return "N/A"
    if not isinstance(value, int | float):
        raise TypeError("policy report ratio is not numeric")
    return f"{value:.1%}"


def _format_latency(value: object) -> str:
    if value is None:
        return "N/A"
    if not isinstance(value, int | float):
        raise TypeError("policy report latency is not numeric")
    return f"{value:.1f} ms"


def _format_versions(value: object) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, str):
        return f"`{value}`"
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return ", ".join(f"`{item}`" for item in value) if value else "N/A"
    raise TypeError("policy report version provenance has an invalid type")


def _format_when(applicable: bool, value: object) -> str:
    return str(value) if applicable and value is not None else "N/A"


def _format_when_ratio(applicable: bool, value: object) -> str:
    return _format_ratio(value) if applicable else "N/A"


def _format_applicable_value(metrics: dict[str, object], key: str) -> str:
    if not metrics.get("applicable", True):
        return "N/A"
    return str(metrics[key])
