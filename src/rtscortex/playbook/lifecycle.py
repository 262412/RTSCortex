"""Conservative promotion and invalidation rules for executable Playbook v2 rules."""

from __future__ import annotations

from dataclasses import dataclass

from rtscortex.playbook.models import (
    PlaybookRule,
    PlaybookRuleCategory,
    PlaybookRuleEffect,
    PlaybookRuleStatus,
    PlaybookRuleStrength,
)


@dataclass(frozen=True, slots=True)
class StrategicABEvidence:
    paired_seed_count: int = 0
    repeat_error_reduction: float = 0.0
    task_score_improvement: float = 0.0
    win_rate_delta: float = 0.0


class PlaybookRuleLifecycle:
    """Apply explicit evidence gates; no model output can bypass these checks."""

    def revalidate_legacy(self, rule: PlaybookRule) -> PlaybookRule:
        if rule.status is not PlaybookRuleStatus.LEGACY:
            raise ValueError("only legacy rules require schema revalidation")
        return rule.model_copy(
            update={
                "status": PlaybookRuleStatus.CANDIDATE,
                "strength": PlaybookRuleStrength.ADVISORY,
            }
        )

    def promote_to_soft(self, rule: PlaybookRule) -> PlaybookRule:
        if rule.status is not PlaybookRuleStatus.CANDIDATE:
            raise ValueError("only candidate rules can be promoted to soft")
        if len(set(rule.source_run_ids)) < 2:
            raise ValueError("soft promotion requires evidence from two runs")
        if rule.confidence < 0.75 or rule.contradiction_count:
            raise ValueError("soft promotion confidence or contradiction gate failed")
        return rule.model_copy(
            update={
                "status": PlaybookRuleStatus.ACTIVE,
                "strength": PlaybookRuleStrength.SOFT,
            }
        )

    def promote_to_hard(
        self,
        rule: PlaybookRule,
        *,
        current_code_revision: str,
        current_sc2_patch: str,
        strategic_ab: StrategicABEvidence | None = None,
    ) -> PlaybookRule:
        if rule.status is not PlaybookRuleStatus.ACTIVE:
            raise ValueError("only active soft rules can be promoted to hard")
        if rule.strength is not PlaybookRuleStrength.SOFT:
            raise ValueError("hard promotion requires a soft rule")
        if len(set(rule.source_seeds)) < 3:
            raise ValueError("hard promotion requires evidence from three seeds")
        if rule.confidence < 0.9 or rule.contradiction_count:
            raise ValueError("hard promotion confidence or contradiction gate failed")
        if rule.code_revision != current_code_revision or rule.sc2_patch != current_sc2_patch:
            raise ValueError("hard promotion revision gate failed")
        if rule.shadow_state_count < 48 or rule.false_block_rate > 0.01:
            raise ValueError("hard promotion shadow coverage gate failed")
        if _is_strategic_blocking_rule(rule):
            if strategic_ab is None or strategic_ab.paired_seed_count < 3:
                raise ValueError("strategic hard promotion requires three paired seeds")
            improved = (
                strategic_ab.repeat_error_reduction >= 0.5
                or strategic_ab.task_score_improvement >= 0.1
            )
            if not improved or strategic_ab.win_rate_delta < 0.0:
                raise ValueError("strategic hard promotion quality gate failed")
        return rule.model_copy(update={"strength": PlaybookRuleStrength.HARD})

    def record_contradiction(self, rule: PlaybookRule, *, seed: int) -> PlaybookRule:
        if seed in rule.contradiction_seeds:
            return rule
        contradiction_seeds = (*rule.contradiction_seeds, seed)
        contradictions = len(contradiction_seeds)
        status = rule.status
        if contradictions >= 3:
            status = PlaybookRuleStatus.RETIRED
        elif contradictions >= 2:
            status = PlaybookRuleStatus.SUSPENDED
        return rule.model_copy(
            update={
                "confidence": max(0.0, rule.confidence - 0.15),
                "contradiction_count": contradictions,
                "contradiction_seeds": contradiction_seeds,
                "status": status,
            }
        )

    def invalidate_revision(
        self,
        rule: PlaybookRule,
        *,
        current_code_revision: str,
    ) -> PlaybookRule:
        execution_rule = rule.category in {
            PlaybookRuleCategory.ENGINE_INVARIANT,
            PlaybookRuleCategory.EXECUTION_GUARD,
            PlaybookRuleCategory.TACTICAL_RESPONSE,
        }
        if execution_rule and rule.code_revision != current_code_revision:
            return rule.model_copy(update={"status": PlaybookRuleStatus.SUSPENDED})
        return rule


def _is_strategic_blocking_rule(rule: PlaybookRule) -> bool:
    return rule.category in {
        PlaybookRuleCategory.RACE_MACRO,
        PlaybookRuleCategory.MATCHUP_STRATEGY,
        PlaybookRuleCategory.MAP_SPECIFIC,
    } and rule.effect in {PlaybookRuleEffect.REQUIRE, PlaybookRuleEffect.FORBID}
