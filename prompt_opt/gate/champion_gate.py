"""
Champion/Challenger gate.

Evaluates the optimizer's best candidate on the frozen test split and
decides whether to promote it to champion status.

Pre-registered promotion criteria (must be decided BEFORE looking at test results;
the exact thresholds used are stored in every result):
  - At least `min_test_examples` paired examples (a verdict on a handful of examples
    means nothing)
  - Challenger's win rate (ties count half) must be >= min_win_rate
  - The bootstrap CI on the win rate must exclude 0.50 (null: no difference)
  - A one-sided Wilcoxon test on the per-example score differences must be significant
  - Guardrails, each against its own threshold:
      refusal-rate shift, output-length distribution shift, token growth,
      format compliance (when measured)
  - The regression suite must have run, and its pass rate must be 100%
  - (Optional) Human spot-check approval

A criterion that could not be evaluated because its input is missing (e.g. no
regression suite) FAILS the gate rather than passing vacuously.

Gate results are written once and never modified; a human review is recorded next to
the result, not in it.
"""

from __future__ import annotations

import logging
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Literal

from prompt_opt.config import settings
from prompt_opt.eval.guardrails import RegressionCase, check_regression_case
from prompt_opt.eval.harness import EvalHarness, PromptCandidate
from prompt_opt.eval.metrics import EvalSummary, PerExampleResult
from prompt_opt.gate.statistical import StatTestResult, run_statistical_tests
from prompt_opt.ingestion.schema import ConversationRecord

logger = logging.getLogger(__name__)

# Caps on the text stored per spot-check item (the reviewer needs the substance, not a novel)
_QUERY_CAP = 1500
_RESPONSE_CAP = 2000
_RATIONALE_CAP = 1000


# ─── Gate criteria ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GateCriteria:
    """
    Pre-registered promotion thresholds.

    IMPORTANT: set these before running the gate, not after observing results.
    """

    min_test_examples: int = field(default_factory=lambda: settings.gate_min_test_examples)
    min_win_rate: float = field(default_factory=lambda: settings.gate_min_win_rate)
    ci_level: float = field(default_factory=lambda: settings.gate_ci_level)
    ci_must_exclude_null: bool = True   # CI must strictly exclude 0.50
    require_significant_score_shift: bool = True  # one-sided Wilcoxon p < 1 - ci_level

    # Guardrails
    max_refusal_rate_shift: float = field(
        default_factory=lambda: settings.gate_max_guardrail_regression_pct
    )  # absolute, either direction
    max_token_growth_pct: float = 0.30       # fractional increase in mean output tokens
    length_shift_alpha: float | None = 0.01  # KS test on output lengths; None disables

    regression_suite_required_pass_rate: float = 1.0  # 100% required
    require_regression_suite: bool = True   # the suite must have actually run
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
    n_paired: int = 0

    # Eval summaries
    challenger_eval: EvalSummary | None = None
    champion_eval: EvalSummary | None = None

    # Criteria check details
    criteria: dict = field(default_factory=dict)   # the thresholds this verdict was made under
    criteria_checks: list[dict] = field(default_factory=list)
    failure_reasons: list[str] = field(default_factory=list)

    # Human spot-check
    spot_check_items: list[dict] = field(default_factory=list)
    human_approved: bool | None = None  # None = pending
    human_reviewer: str | None = None
    human_notes: str | None = None

    def record_human_review(
        self,
        reviewer: str,
        notes: str = "",
        approved: bool = True,
    ) -> GateResult:
        """
        Record a human spot-check decision on this result.

        Mutates decision PENDING_HUMAN → PASS (approved) or FAIL (rejected).
        """
        if self.decision != "PENDING_HUMAN":
            raise ValueError(f"Gate is not pending human review (status={self.decision})")

        self.human_approved = approved
        self.human_reviewer = reviewer
        self.human_notes = notes

        if approved:
            self.decision = "PASS"
            logger.info("Gate APPROVED by %s: %s", reviewer, notes)
        else:
            self.decision = "FAIL"
            self.failure_reasons.append(f"Human reviewer {reviewer!r} rejected: {notes}")
            logger.info("Gate REJECTED by %s: %s", reviewer, notes)

        return self

    def to_dict(self) -> dict:
        return {
            "decision": self.decision,
            "challenger_hash": self.challenger_hash,
            "champion_hash": self.champion_hash,
            "timestamp": self.timestamp,
            "n_paired": self.n_paired,
            "criteria": self.criteria,
            "failure_reasons": self.failure_reasons,
            "criteria_checks": self.criteria_checks,
            "stat_test": self.stat_test.summary() if self.stat_test else None,
            "challenger_eval": self.challenger_eval.to_dict() if self.challenger_eval else None,
            "champion_eval": self.champion_eval.to_dict() if self.champion_eval else None,
            "spot_check_items": self.spot_check_items,
            "human_approved": self.human_approved,
            "human_reviewer": self.human_reviewer,
            "human_notes": self.human_notes,
        }


def _cap(text: str | None, limit: int) -> str:
    text = text or ""
    return text if len(text) <= limit else text[: limit - 1] + "…"


# ─── ChampionGate ─────────────────────────────────────────────────────────────


class ChampionGate:
    """
    Independent, low-frequency gate that decides real-world prompt promotion.

    The optimizer never directly sees test split scores — only dev split scores.
    This gate is the only entity that touches the frozen test split.

    Args:
        harness:               Configured EvalHarness.
        criteria:              Pre-registered promotion thresholds.
        regression_cases:      Loaded RegressionCase objects (used to tell "no suite exists"
                               from "the suite exists but was not run" in failure messages).
    """

    def __init__(
        self,
        harness: EvalHarness,
        criteria: GateCriteria | None = None,
        regression_cases: list[RegressionCase] | None = None,
    ):
        self.harness = harness
        self.criteria = criteria or GateCriteria()
        self.regression_cases = regression_cases or []

    def _generate_spot_check(
        self,
        paired: list[PerExampleResult],
        conversations: dict[str, ConversationRecord],
        n: int,
        seed: int,
    ) -> list[dict]:
        """
        Pick n head-to-heads for human review, with the content a reviewer needs.

        The sample cycles through losses, wins and ties (a challenger that passed the
        statistics still deserves a look at the cases where the judge said it lost)
        and is reproducible: it is seeded by the challenger's hash.
        """
        groups: dict[str, list[PerExampleResult]] = {"loss": [], "win": [], "tie": []}
        for ex in paired:
            groups[ex.outcome or "tie"].append(ex)
        rng = random.Random(seed)
        for group in groups.values():
            rng.shuffle(group)

        picked: list[PerExampleResult] = []
        while len(picked) < n and any(groups.values()):
            for outcome in ("loss", "win", "tie"):
                if groups[outcome] and len(picked) < n:
                    picked.append(groups[outcome].pop())

        items = []
        for ex in picked:
            conv = conversations.get(ex.example_id)
            user_turns = [m.content for m in conv.user_turns if m.content] if conv else []
            items.append(
                {
                    "example_id": ex.example_id,
                    "category": ex.category,
                    "user_query": _cap(user_turns[-1] if user_turns else "", _QUERY_CAP),
                    "challenger_response": _cap(ex.response_text, _RESPONSE_CAP),
                    "champion_response": _cap(ex.champion_response_text, _RESPONSE_CAP),
                    "challenger_score": ex.challenger_score,
                    "champion_score": ex.champion_score,
                    "outcome": ex.outcome,
                    "judge_rationale": _cap(ex.judge_rationale, _RATIONALE_CAP),
                }
            )
        return items

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
            regression_responses: (RegressionCase, response_text) pairs, replayed through
                                  the challenger.

        Returns:
            GateResult with PASS / FAIL / PENDING_HUMAN decision.
        """
        crit = self.criteria
        logger.info(
            "Running champion gate: challenger=%s vs champion=%s on %d test examples",
            challenger.content_hash,
            champion.content_hash,
            len(test_conversations),
        )

        failure_reasons: list[str] = []
        criteria_checks: list[dict] = []

        def check(name: str, value, threshold, passed: bool, reason: str = "") -> None:
            criteria_checks.append(
                {"criterion": name, "value": value, "threshold": threshold, "passed": passed}
            )
            if not passed:
                failure_reasons.append(reason or f"{name}: {value} (threshold {threshold})")

        # ── Establish champion baseline responses ────────────────────────────
        # Paired comparison (same inputs, both prompts) is required for the
        # statistical tests. If pre-computed champion responses were not
        # supplied, evaluate the champion first and reuse its response texts.
        champion_eval = None
        if not champion_responses:
            champion_eval = self.harness.evaluate(
                champion, test_conversations, split="test"
            )
            champion_responses = {
                ex.example_id: ex.response_text
                for ex in champion_eval.per_example
                if ex.response_text
            }

        # ── Evaluate challenger on test split (paired vs champion) ──────────
        challenger_eval = self.harness.evaluate(
            challenger,
            test_conversations,
            split="test",
            champion_responses=champion_responses,
            regression_data=regression_responses,
        )

        # Only genuinely paired examples enter the statistics
        paired = [
            ex for ex in challenger_eval.per_example
            if ex.outcome is not None and ex.champion_score is not None
        ]
        n_paired = len(paired)
        stat_result = run_statistical_tests(
            [ex.outcome for ex in paired],  # type: ignore[misc]
            [ex.challenger_score for ex in paired],
            [ex.champion_score for ex in paired],  # type: ignore[misc]
            ci_level=crit.ci_level,
        )

        # ── Check criteria ────────────────────────────────────────────────────

        # 1. Enough evidence to decide anything
        check(
            "min_test_examples",
            n_paired,
            crit.min_test_examples,
            n_paired >= crit.min_test_examples,
            f"Only {n_paired} paired test examples (minimum {crit.min_test_examples}); "
            f"the verdict would not be statistically meaningful",
        )

        if n_paired:
            # 2. Win rate (ties count half)
            check(
                "win_rate",
                round(stat_result.win_rate, 4),
                crit.min_win_rate,
                stat_result.win_rate >= crit.min_win_rate,
                f"Win rate {stat_result.win_rate:.3f} < min {crit.min_win_rate} "
                f"({stat_result.n_wins}W / {stat_result.n_losses}L / {stat_result.n_ties}T, "
                f"ties count half)",
            )

            # 3. CI excludes null (0.50)
            ci_pass = not crit.ci_must_exclude_null or stat_result.ci_excludes_null
            ci_text = f"[{stat_result.ci_lower:.3f}, {stat_result.ci_upper:.3f}]"
            check(
                "ci_excludes_null",
                ci_text,
                "CI must exclude 0.50",
                ci_pass,
                f"CI {ci_text} does not exclude 0.50",
            )

            # 4. Per-example score shift is significant (Wilcoxon, one-sided)
            if crit.require_significant_score_shift:
                check(
                    "score_shift_significance",
                    f"p={stat_result.wilcoxon_pvalue:.4f}",
                    f"p < {stat_result.alpha:.3f}",
                    stat_result.statistically_significant,
                    f"Score improvement not significant (Wilcoxon p="
                    f"{stat_result.wilcoxon_pvalue:.4f}, {stat_result.n_nonzero_diffs} "
                    f"non-zero differences)",
                )

        # 5. Guardrails: measured values (from the paired evaluation) against the
        #    gate's own thresholds. A guardrail that could not be measured is reported
        #    as such and does not fail on its own.
        gd = challenger_eval.guardrail_details or {}

        refusal_delta = gd.get("refusal_rate_delta")
        check(
            "refusal_rate_shift",
            "n/a" if refusal_delta is None else round(refusal_delta, 4),
            f"|shift| <= {crit.max_refusal_rate_shift}",
            refusal_delta is None or abs(refusal_delta) <= crit.max_refusal_rate_shift,
            f"Refusal rate shifted by {refusal_delta:+.3f} vs champion "
            f"(max {crit.max_refusal_rate_shift})" if refusal_delta is not None else "",
        )

        if crit.length_shift_alpha is not None:
            ks_p = gd.get("length_ks_pvalue")
            check(
                "output_length_shift",
                "n/a" if ks_p is None else f"KS p={ks_p:.4f}",
                f"p > {crit.length_shift_alpha}",
                ks_p is None or ks_p > crit.length_shift_alpha,
                f"Output length distribution shifted significantly vs champion "
                f"(KS p={ks_p})" if ks_p is not None else "",
            )

        token_delta = gd.get("token_count_delta_pct")
        check(
            "token_growth",
            "n/a" if token_delta is None else round(token_delta, 4),
            f"<= {crit.max_token_growth_pct}",
            token_delta is None or token_delta <= crit.max_token_growth_pct,
            f"Mean output tokens grew {token_delta:+.1%} vs champion "
            f"(max {crit.max_token_growth_pct:.0%})" if token_delta is not None else "",
        )

        fmt_rate = gd.get("format_compliance_rate")
        if fmt_rate is not None:
            check(
                "format_compliance",
                round(fmt_rate, 4),
                "guardrail threshold",
                bool(gd.get("format_compliance_pass", True)),
                f"Format compliance {fmt_rate:.3f} below threshold",
            )

        # 6. Regression suite: must have run, and pass at the required rate
        n_reg = len(regression_responses or [])
        if n_reg == 0:
            if crit.require_regression_suite:
                why = (
                    "the suite exists but was not run"
                    if self.regression_cases
                    else "no regression suite was provided"
                )
                check(
                    "regression_suite_pass_rate",
                    "not run",
                    crit.regression_suite_required_pass_rate,
                    False,
                    f"Regression suite did not run ({why}); a gate without it cannot verify "
                    f"known-bad behaviour",
                )
            else:
                check(
                    "regression_suite_pass_rate",
                    "not run (allowed)",
                    crit.regression_suite_required_pass_rate,
                    True,
                )
        else:
            failures = []
            for case, response in regression_responses:  # type: ignore[union-attr]
                passed, reason = check_regression_case(case, response)
                if not passed:
                    failures.append(f"{case.conversation_id}: {reason}")
            reg_rate = (n_reg - len(failures)) / n_reg
            check(
                "regression_suite_pass_rate",
                round(reg_rate, 4),
                crit.regression_suite_required_pass_rate,
                reg_rate >= crit.regression_suite_required_pass_rate,
                f"Regression suite failed ({reg_rate:.0%}): {failures}",
            )

        # ── Determine decision ───────────────────────────────────────────────
        if failure_reasons:
            decision: Literal["PASS", "FAIL", "PENDING_HUMAN"] = "FAIL"
        elif crit.require_human_spot_check:
            decision = "PENDING_HUMAN"
        else:
            decision = "PASS"

        # Spot-check items (kept even for FAIL: useful to understand why)
        spot_check = self._generate_spot_check(
            paired,
            {c.conversation_id: c for c in test_conversations},
            crit.spot_check_sample_size,
            seed=int(challenger.content_hash, 16),
        )

        result = GateResult(
            decision=decision,
            challenger_hash=challenger.content_hash,
            champion_hash=champion.content_hash,
            stat_test=stat_result,
            n_paired=n_paired,
            challenger_eval=challenger_eval,
            champion_eval=champion_eval,
            criteria=asdict(crit),
            criteria_checks=criteria_checks,
            failure_reasons=failure_reasons,
            spot_check_items=spot_check,
        )

        logger.info(
            "Gate decision: %s (win_rate=%.3f, CI=[%.3f, %.3f], n=%d, reasons=%s)",
            decision,
            stat_result.win_rate,
            stat_result.ci_lower,
            stat_result.ci_upper,
            n_paired,
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
        return gate_result.record_human_review(
            reviewer=reviewer, notes=notes, approved=approved
        )
