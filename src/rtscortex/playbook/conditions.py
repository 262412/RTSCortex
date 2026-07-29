"""Shared Playbook condition matching semantics."""

from __future__ import annotations

from collections.abc import Mapping

from rtscortex.playbook.models import PlaybookCondition, PlaybookConditionOperator


def condition_matches(
    condition: PlaybookCondition,
    values: Mapping[str, object],
) -> bool:
    """Evaluate one condition with the same semantics in every Playbook path."""

    if condition.field not in values:
        return False
    actual = values[condition.field]
    expected = condition.value
    if condition.operator is PlaybookConditionOperator.EQ:
        return _equal(actual, expected)
    if condition.operator is PlaybookConditionOperator.IN:
        return isinstance(expected, tuple) and any(_equal(actual, option) for option in expected)
    if condition.operator is PlaybookConditionOperator.CONTAINS:
        if isinstance(actual, str) and isinstance(expected, str):
            return expected.casefold() in actual.casefold()
        if isinstance(actual, (tuple, list, set)):
            return any(_equal(item, expected) for item in actual)
        return False
    if condition.operator is PlaybookConditionOperator.GTE:
        return (
            isinstance(actual, (int, float))
            and isinstance(expected, (int, float))
            and actual >= expected
        )
    if condition.operator is PlaybookConditionOperator.LTE:
        return (
            isinstance(actual, (int, float))
            and isinstance(expected, (int, float))
            and actual <= expected
        )
    return False


def _equal(actual: object, expected: object) -> bool:
    if isinstance(actual, str) and isinstance(expected, str):
        return actual.casefold() == expected.casefold()
    return actual == expected
