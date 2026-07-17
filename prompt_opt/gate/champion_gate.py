"""
Champion/Challenger gate.

Evaluates the optimizer's best candidate on the frozen test split and
decides whether to promote it to champion status.

Pre-registered promotion criteria (must be decided BEFORE looking at test results):
  - Challenger must win >= min_win_rate of paired comparisons
  - The 95% CI on win_rate must exclude 0.50 (null hypothesis)
  - No guardrail metric may regress by more than max_guardrail_regression_pct
  - Regression suite pass rate must be 100%
  - (Optional) Human spot-check approval

Gate result is immutable once computed and is attached to the prompt registry entry.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from prompt_opt.config import settings
from prompt_opt.eval.harness import EvalHarness, PromptCandidate
from prompt_opt.eval.guardrails import GuardrailChecker, RegressionCase
from prompt_opt.eval.metrics import EvalSummary
from prompt_opt.gate.statistical import StatTestResult, run_statistical_tests
from prompt_opt.ingestion.schema import ConversationRecord

logger = logging.getLogger(__name__)


# ─── Gate criteria ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GateCriteria:
    """
    Pre-registered promotion thresholds.

    IMPORTANT: set these before running the gate, not after observing results.
    """

    min_win_rate: float = field(default_factory=lambda: settings.gate_min_win_rate)
    ci_level: float = field(default_factory=lambda: settings.gate_ci_level)
    ci_must_exclude_null: bool = True   # CI must strictly exclude 0.50
    max_guardrail_regression_pct: float = field(
        default_factory=lambda: settings.gate_max_guardrail_regression_pct
    )
    regression_suite_required_pass_rate: float = 1.0  # 100% required
    require_human_spot_check: bool = True
    spot_check_sample_size: int = 5


# ─── Gate result ─────────────────────────────────────────────────────────────


@dataclass
class GateResult:
    decision: Literal["PASS", "FAIL", "PENDING_HUMAN"]
    challenger_hash: str
    champion_hash: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # Statistical test results
    stat_test: StatTestResult | None = None

    # Eval summaries
    challenger_eval: EvalSummary | None = None
    champion_eval: EvalSummary | None = None

    # Criteria check details
    criteria_checks: list[dict] = field(default_factory=list)
    failure_reasons: list[str] = field(default_factory=list)

    # Human spot-check
    spot_check_items: list[dict] = field(default_factory=list)
    human_approved: bool | None = None  # None = pending
    human_reviewer: str | None = None
    human_notes: str | None = None

    def to_dict(self) -> dict:
        return {
            "decision": self.decision,
            "challenger_hash": self.challenger_hash,
            "champion_hash": self.champion_hash,
            "timestamp": self.timestamp,
            "failure_reasons": self.failure_reasons,
            "criteria_checks": self.criteria_checks,
            "stat_test": self.stat_test.summary() if self.stat_test else None,
            "challenger_eval": self.challenger_eval.to_dict() if self.challenger_eval else None,
            "champion_eval": self.champion_eval.to_dict() if self.champion_eval else None,
            "human_approved": self.human_approved,
        }


# ─── ChampionGate ─────────────────────────────────────────────────────────────


class ChampionGate:
    """
    Independent, low-frequency gate that decides real-world prompt promotion.

    The optimizer never directly sees test split scores — only dev split scores.
    This gate is the only entity that touches the frozen test split.

    Args:
        harness:               Configured EvalHarness.
        criteria:              Pre-registered promotion thresholds.
        guardrail_checker:     GuardrailChecker for regression suite.
        regression_cases:      Loaded RegressionCase objects.
    """

    def __init__(
        self,
        harness: EvalHarness,
        criteria: GateCriteria | None = None,
        guardrail_checker: GuardrailChecker | None = None,
        regression_cases: list[RegressionCase] | None = None,
    ):
        self.harness = harness
        self.criteria = criteria or GateCriteria()
        self.guardrail_checker = guardrail_checker or GuardrailChecker()
        self.regression_cases = regression_cases or []

    def _generate_spot_check(
        self,
        challenger_eval: EvalSummary,
        champion_eval: EvalSummary,
        n: int,
    ) -> list[dict]:
        """Pick n head-to-heads from the test set for human review."""
        import random

        examples = challenger_eval.per_example[:n]
        spot_items = []
        for ex in examples:
            spot_items.append(
                {
                    "example_id": ex.example_id,
                    "category": ex.category,
                    "challenger_score": ex.challenger_score,
                    "champion_score": ex.champion_score,
                    "outcome": ex.outcome,
                    "judge_rationale": ex.judge_rationale[:200],
                }
            )
        return spot_items

    def evaluate(
        self,
        challenger: PromptCandidate,
        champion: PromptCandidate,
        test_conversations: list[ConversationRecord],
        champion_responses: dict[str, str] | None = None,
        regression_responses: list[tuple[RegressionCase, str]] | None = None,
    ) -> GateResult:
        """
        Run the full gate evaluation on the frozen test split.

        Args:
            challenger:           Candidate prompt from optimizer.
            champion:             Current champion prompt.
            test_conversations:   Frozen test split conversations.
            champion_responses:   Pre-computed champion responses (cache them!).
            regression_responses: (RegressionCase, response_text) pairs.

        Returns:
            GateResult with PASS / FAIL / PENDING_HUMAN decision.
        """
        logger.info(
            "Running champion gate: challenger=%s vs champion=%s on %d test examples",
            challenger.content_hash,
            champion.content_hash,
            len(test_conversations),
        )

        failure_reasons: list[str] = []
        criteria_checks: list[dict] = []

        # ── Evaluate challenger on test split ────────────────────────────────
        challenger_eval = self.harness.evaluate(
            challenger,
            test_conversations,
            split="test",
            champion_responses=champion_responses,
            regression_data=regression_responses,
        )

        # ── Evaluate champion baseline (if no pre-computed responses) ────────
        champion_eval = None
        if not champion_responses:
            champion_eval = self.harness.evaluate(
                champion, test_conversations, split="test"
            )

        # ── Statistical test ─────────────────────────────────────────────────
        outcomes = [ex.outcome or "tie" for ex in challenger_eval.per_example]
        challenger_scores = [ex.challenger_score for ex in challenger_eval.per_example]
        champion_scores = [ex.champion_score or 0.0 for ex in challenger_eval.per_example]

        stat_result = run_statistical_tests(
            outcomes,
            challenger_scores,
            champion_scores,
            ci_level=self.criteria.ci_level,
        )

        # ── Check criteria ────────────────────────────────────────────────────

        # 1. Win rate
        win_rate_pass = stat_result.win_rate >= self.criteria.min_win_rate
        criteria_checks.append({
            "criterion": "win_rate",
            "value": stat_result.win_rate,
            "threshold": self.criteria.min_win_rate,
            "passed": win_rate_pass,
        })
        if not win_rate_pass:
            failure_reasons.append(
                f"Win rate {stat_result.win_rate:.3f} < min {self.criteria.min_win_rate}"
            )

        # 2. CI excludes null (0.50)
        ci_pass = not self.criteria.ci_must_exclude_null or stat_result.ci_excludes_null
        criteria_checks.append({
            "criterion": "ci_excludes_null",
            "value": f"[{stat_result.ci_lower:.3f}, {stat_result.ci_upper:.3f}]",
            "threshold": "CI must exclude 0.50",
            "passed": ci_pass,
        })
        if not ci_pass:
            failure_reasons.append(
                f"CI [{stat_result.ci_lower:.3f}, {stat_result.ci_upper:.3f}] does not exclude 0.50"
            )

        # 3. Guardrail regression
        guard_pass = challenger_eval.guardrail_passed
        reg_regression_pct = 0.0
        if not guard_pass:
            reg_regression_pct = 1.0  # treat full failure as 100% regression
        criteria_checks.append({
            "criterion": "guardrail_regression",
            "value": reg_regression_pct,
            "threshold": self.criteria.max_guardrail_regression_pct,
            "passed": reg_regression_pct <= self.criteria.max_guardrail_regression_pct,
        })
        if reg_regression_pct > self.criteria.max_guardrail_regression_pct:
            failure_reasons.append(
                f"Guardrail failed: {challenger_eval.guardrail_details}"
            )

        # 4. Regression suite pass rate
        reg_suite_rate = challenger_eval.guardrail_details.get("regression_pass_rate", 1.0)
        reg_suite_pass = reg_suite_rate >= self.criteria.regression_suite_required_pass_rate
        criteria_checks.append({
            "criterion": "regression_suite_pass_rate",
            "value": reg_suite_rate,
            "threshold": self.criteria.regression_suite_required_pass_rate,
            "passed": reg_suite_pass,
        })
        if not reg_suite_pass:
            failures = challenger_eval.guardrail_details.get("regression_failures", [])
            failure_reasons.append(
                f"Regression suite failed ({reg_suite_rate:.0%}): {failures}"
            )

        # ── Determine decision ───────────────────────────────────────────────
        if failure_reasons:
            decision: Literal["PASS", "FAIL", "PENDING_HUMAN"] = "FAIL"
        elif self.criteria.require_human_spot_check:
            decision = "PENDING_HUMAN"
        else:
            decision = "PASS"

        # Generate spot-check items
        spot_check = self._generate_spot_check(
            challenger_eval,
            champion_eval or challenger_eval,
            self.criteria.spot_check_sample_size,
        )

        result = GateResult(
            decision=decision,
            challenger_hash=challenger.content_hash,
            champion_hash=champion.content_hash,
            stat_test=stat_result,
            challenger_eval=challenger_eval,
            champion_eval=champion_eval,
            criteria_checks=criteria_checks,
            failure_reasons=failure_reasons,
            spot_check_items=spot_check,
        )

        logger.info(
            "Gate decision: %s (win_rate=%.3f, CI=[%.3f, %.3f], reasons=%s)",
            decision,
            stat_result.win_rate,
            stat_result.ci_lower,
            stat_result.ci_upper,
            failure_reasons or "none",
        )
        return result

    def approve_spot_check(
        self,
        gate_result: GateResult,
        reviewer: str,
        notes: str = "",
        approved: bool = True,
    ) -> GateResult:
        """Record human spot-check decision. Call this after human review."""
        if gate_result.decision != "PENDING_HUMAN":
            raise ValueError(f"Gate is not pending human review (status={gate_result.decision})")

        gate_result.human_approved = approved
        gate_result.human_reviewer = reviewer
        gate_result.human_notes = notes

        if approved:
            gate_result.decision = "PASS"
            logger.info("Gate APPROVED by %s: %s", reviewer, notes)
        else:
            gate_result.decision = "FAIL"
            gate_result.failure_reasons.append(f"Human reviewer {reviewer!r} rejected: {notes}")
            logger.info("Gate REJECTED by %s: %s", reviewer, notes)

        return gate_result
