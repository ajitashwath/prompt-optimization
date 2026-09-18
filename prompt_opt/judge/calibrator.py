"""
Judge calibration: measures agreement between LLM-judge and human gold labels.

Metrics computed:
  - Cohen's kappa (pairwise preference: A/B/tie)
  - Spearman correlation (absolute Likert scores)
  - Pearson correlation
  - Confusion matrix
  - Bias diagnostics:
      * Length bias: does judge prefer longer responses?
      * Position bias: does it prefer whichever response appears first?
      * Style bias: markdown vs prose correlation

Output: CalibrationReport (serialisable dict/JSON).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from scipy import stats

from prompt_opt.config import settings
from prompt_opt.ingestion.schema import ConversationRecord
from prompt_opt.judge.judge import LLMJudge, PairwiseJudgeResult
from prompt_opt.labeling.gold_store import GoldStore

logger = logging.getLogger(__name__)


# ─── Agreement metrics helpers ───────────────────────────────────────────────

_PREF_MAP = {"A": 0, "B": 1, "tie": 2}


def _cohen_kappa(y1: list[int], y2: list[int]) -> float:
    """Compute Cohen's kappa for two raters."""
    from sklearn.metrics import cohen_kappa_score  # type: ignore[import]

    return float(cohen_kappa_score(y1, y2))


def _spearman(x: list[float], y: list[float]) -> tuple[float, float]:
    """Return (rho, p_value). Returns (0.0, 1.0) for constant inputs (undefined)."""
    if len(set(x)) < 2 or len(set(y)) < 2:
        return 0.0, 1.0
    result = stats.spearmanr(x, y)
    rho = float(result.statistic)
    if np.isnan(rho):
        return 0.0, 1.0
    return rho, float(result.pvalue)


def _pearson(x: list[float], y: list[float]) -> tuple[float, float]:
    """Return (r, p_value). Returns (0.0, 1.0) for constant inputs (undefined)."""
    if len(set(x)) < 2 or len(set(y)) < 2:
        return 0.0, 1.0
    r, p = stats.pearsonr(x, y)
    if np.isnan(r):
        return 0.0, 1.0
    return float(r), float(p)


# ─── Report data class ───────────────────────────────────────────────────────


@dataclass
class BiasReport:
    length_bias_rho: float          # Spearman rho: judge score difference (A-B) ~ length difference
    length_bias_pvalue: float
    position_bias_rate: float       # share of decisive per-ordering verdicts that picked the
                                    # FIRST-listed response (0.5 = no position bias)
    style_bias_rho: float | None = None   # correlation with markdown presence (optional)
    human_length_rho: float = 0.0   # same correlation for the human labels (baseline)
    order_inconsistency_rate: float | None = None  # items whose two orderings disagree
    n_position_verdicts: int = 0    # decisive (non-tie) verdicts behind position_bias_rate

    @property
    def has_significant_length_bias(self) -> bool:
        """The judge tracks length significantly AND clearly more than the humans do."""
        excess = self.length_bias_rho - self.human_length_rho
        return (
            abs(self.length_bias_rho) > 0.2
            and self.length_bias_pvalue < 0.05
            and abs(excess) > 0.15
        )

    @property
    def has_significant_position_bias(self) -> bool:
        """Needs >= 10 decisive verdicts; a fair judge picks first/second ~50/50."""
        return self.n_position_verdicts >= 10 and abs(self.position_bias_rate - 0.5) > 0.1


@dataclass
class CalibrationReport:
    judge_version: str
    rubric_version: str
    n_items: int
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # Agreement
    kappa: float = 0.0
    kappa_interpretation: str = ""
    spearman_rho: float = 0.0
    spearman_pvalue: float = 1.0
    pearson_r: float = 0.0
    pearson_pvalue: float = 1.0

    # Human → Judge agreement rate (exact match on A/B/tie)
    agreement_rate: float = 0.0

    # Bias
    bias: BiasReport | None = None

    # Confusion matrix (rows=human, cols=judge) for A/B/tie
    confusion_matrix: list[list[int]] = field(default_factory=list)

    # Per-item results
    item_results: list[dict] = field(default_factory=list)

    # Pass/fail
    passed: bool = False
    failure_reasons: list[str] = field(default_factory=list)

    # Gold items that could not be used
    n_judge_failures: int = 0
    n_skipped_no_context: int = 0

    def to_dict(self) -> dict:
        d = {
            "judge_version": self.judge_version,
            "rubric_version": self.rubric_version,
            "n_items": self.n_items,
            "timestamp": self.timestamp,
            "agreement_rate": self.agreement_rate,
            "kappa": self.kappa,
            "kappa_interpretation": self.kappa_interpretation,
            "spearman_rho": self.spearman_rho,
            "pearson_r": self.pearson_r,
            "passed": self.passed,
            "failure_reasons": self.failure_reasons,
            "confusion_matrix": self.confusion_matrix,
            "n_judge_failures": self.n_judge_failures,
            "n_skipped_no_context": self.n_skipped_no_context,
        }
        if self.bias:
            d["bias"] = {
                "length_bias_rho": self.bias.length_bias_rho,
                "length_bias_pvalue": self.bias.length_bias_pvalue,
                "position_bias_rate": self.bias.position_bias_rate,
                "n_position_verdicts": self.bias.n_position_verdicts,
                "order_inconsistency_rate": self.bias.order_inconsistency_rate,
                "human_length_rho": self.bias.human_length_rho,
                "has_significant_length_bias": self.bias.has_significant_length_bias,
                "has_significant_position_bias": self.bias.has_significant_position_bias,
            }
        return d


def _interpret_kappa(k: float) -> str:
    if k < 0:
        return "poor (worse than chance)"
    if k < 0.2:
        return "slight"
    if k < 0.4:
        return "fair"
    if k < 0.6:
        return "moderate"
    if k < 0.8:
        return "substantial"
    return "almost perfect"


# ─── Calibrator ──────────────────────────────────────────────────────────────


class JudgeCalibrator:
    """
    Calibrates an LLMJudge against a human-labeled gold set.

    Usage::

        calibrator = JudgeCalibrator(judge, gold_store)
        report = calibrator.calibrate(rubric_version="v1.0")
        if report.passed:
            judge_registry.freeze(judge, report)
    """

    def __init__(
        self,
        judge: LLMJudge,
        gold_store: GoldStore,
        *,
        min_kappa: float | None = None,
        min_spearman: float | None = None,
        max_failure_rate: float = 0.2,
    ):
        self.judge = judge
        self.gold_store = gold_store
        self.min_kappa = settings.calibration_min_kappa if min_kappa is None else min_kappa
        self.min_spearman = (
            settings.calibration_min_correlation if min_spearman is None else min_spearman
        )
        self.max_failure_rate = max_failure_rate

    def calibrate(
        self,
        rubric_version: str,
        conversations: dict[str, ConversationRecord] | None = None,
        judge_version: str = "auto",
    ) -> CalibrationReport:
        """
        Run the judge over all labeled gold items for `rubric_version`
        and compute agreement / bias metrics.

        Args:
            rubric_version:  Which rubric version to evaluate against.
            conversations:   Map from conversation_id -> ConversationRecord. When given,
                             the judge sees each item's real conversation (as it does at
                             eval time) and items whose conversation is missing are
                             skipped and counted. If None, the judge is run against a
                             "[context unavailable]" stub - only meaningful for tests.
            judge_version:   Identifier for this judge configuration.
        """
        labeled = self.gold_store.get_labeled(rubric_version=rubric_version)
        if not labeled:
            logger.warning("No labeled items found for rubric_version=%s", rubric_version)
            return CalibrationReport(
                judge_version=judge_version,
                rubric_version=rubric_version,
                n_items=0,
                passed=False,
                failure_reasons=["No labeled gold items found"],
            )

        logger.info("Calibrating judge over %d gold items ...", len(labeled))
        if conversations is None:
            logger.warning(
                "Calibrating WITHOUT conversation context: the judge sees a stub instead "
                "of the real conversation, unlike at eval time."
            )

        human_prefs: list[int] = []
        judge_prefs: list[int] = []
        judge_scores_a: list[float] = []
        judge_scores_b: list[float] = []
        len_a_list: list[int] = []
        len_b_list: list[int] = []
        verdict_pairs: list[tuple[str | None, str | None]] = []
        item_results: list[dict] = []
        n_judge_failures = 0
        n_no_context = 0

        for item in labeled:
            if item.preferred is None:
                continue

            if conversations is not None:
                conv = conversations.get(item.conversation_id)
                if conv is None:
                    n_no_context += 1
                    logger.warning(
                        "Skipping gold item %s: conversation %r not found",
                        item.id,
                        item.conversation_id,
                    )
                    continue
            else:
                from prompt_opt.ingestion.schema import Message, MessageRole

                conv = ConversationRecord(
                    conversation_id=item.conversation_id,
                    messages=[Message(role=MessageRole.USER, content="[context unavailable]")],
                )

            try:
                judge_result: PairwiseJudgeResult = self.judge.score_pairwise(
                    conv, item.response_a, item.response_b
                )
            except Exception as exc:  # noqa: BLE001
                n_judge_failures += 1
                logger.warning("Judge call failed for item %s: %s", item.id, exc)
                continue

            human_prefs.append(_PREF_MAP.get(item.preferred or "tie", 2))
            judge_prefs.append(_PREF_MAP.get(judge_result.preferred, 2))
            judge_scores_a.append(judge_result.aggregate_score_a)
            judge_scores_b.append(judge_result.aggregate_score_b)
            len_a_list.append(len(item.response_a))
            len_b_list.append(len(item.response_b))
            verdict_pairs.append((judge_result.verdict_ab, judge_result.verdict_ba))

            item_results.append(
                {
                    "item_id": item.id,
                    "human_preferred": item.preferred,
                    "judge_preferred": judge_result.preferred,
                    "agree": item.preferred == judge_result.preferred,
                    "judge_score_a": judge_result.aggregate_score_a,
                    "judge_score_b": judge_result.aggregate_score_b,
                    "verdict_ab": judge_result.verdict_ab,
                    "verdict_ba": judge_result.verdict_ba,
                }
            )

        n = len(human_prefs)
        attempted = n + n_judge_failures
        if n < 5:
            detail = ""
            if n_judge_failures or n_no_context:
                detail = (
                    f" ({n_judge_failures} judge call failures, "
                    f"{n_no_context} items skipped for missing conversation context)"
                )
            return CalibrationReport(
                judge_version=judge_version,
                rubric_version=rubric_version,
                n_items=n,
                passed=False,
                failure_reasons=[
                    f"Too few valid items for calibration (got {n}, need >=5){detail}"
                ],
                n_judge_failures=n_judge_failures,
                n_skipped_no_context=n_no_context,
            )

        # Agreement
        kappa = _cohen_kappa(human_prefs, judge_prefs)
        kappa_undefined = math.isnan(kappa)
        if kappa_undefined:
            kappa = 0.0
        agreement_rate = sum(h == j for h, j in zip(human_prefs, judge_prefs, strict=True)) / n

        # Continuous correlation (use diff of scores as proxy for preference strength)
        score_diffs = [a - b for a, b in zip(judge_scores_a, judge_scores_b, strict=True)]
        human_binary = [1 if h == 0 else (-1 if h == 1 else 0) for h in human_prefs]
        spearman_rho, spearman_p = _spearman(score_diffs, human_binary)
        pearson_r, pearson_p = _pearson(score_diffs, human_binary)

        # Confusion matrix (rows=human, cols=judge) for A/B/tie
        cm = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        for h, j in zip(human_prefs, judge_prefs, strict=True):
            cm[h][j] += 1

        # Length bias: does the judge's preference follow the length difference (A - B),
        # and more strongly than the humans' own preference does?
        length_diffs = [float(a - b) for a, b in zip(len_a_list, len_b_list, strict=True)]
        len_rho, len_p = _spearman(score_diffs, length_diffs)
        human_len_rho, _ = _spearman([float(v) for v in human_binary], length_diffs)

        # Position bias: across BOTH orderings of every pair, how often did the judge pick
        # the first-listed response? A position-neutral judge picks it ~50% of the time
        # whatever the gold set's A/B label balance. verdict_ab / verdict_ba are in
        # original labels: A listed first in pass 1, B listed first in pass 2.
        first_picks = second_picks = considered = inconsistent = 0
        for v_ab, v_ba in verdict_pairs:
            if v_ab is None or v_ba is None:
                continue
            considered += 1
            first_picks += (v_ab == "A") + (v_ba == "B")
            second_picks += (v_ab == "B") + (v_ba == "A")
            inconsistent += v_ab != v_ba
        n_decisive = first_picks + second_picks
        pos_bias_rate = first_picks / n_decisive if n_decisive else 0.5
        order_inconsistency = inconsistent / considered if considered else None

        bias = BiasReport(
            length_bias_rho=len_rho,
            length_bias_pvalue=len_p,
            position_bias_rate=pos_bias_rate,
            human_length_rho=human_len_rho,
            order_inconsistency_rate=order_inconsistency,
            n_position_verdicts=n_decisive,
        )

        # Pass/fail
        failure_reasons: list[str] = []
        if kappa_undefined:
            failure_reasons.append(
                "Cohen's kappa is undefined (human or judge verdicts have no variation); "
                "the gold set needs a mix of A / B / tie outcomes"
            )
        elif kappa < self.min_kappa:
            failure_reasons.append(f"kappa={kappa:.3f} < threshold={self.min_kappa}")
        if abs(spearman_rho) < self.min_spearman:
            failure_reasons.append(
                f"spearman_rho={spearman_rho:.3f} < threshold={self.min_spearman}"
            )
        if bias.has_significant_length_bias:
            failure_reasons.append(
                f"Significant length bias detected (judge rho={len_rho:.3f}, p={len_p:.4f}, "
                f"human rho={human_len_rho:.3f})"
            )
        if bias.has_significant_position_bias:
            failure_reasons.append(
                f"Significant position bias detected (first-listed picked "
                f"{pos_bias_rate:.0%} of {n_decisive} decisive verdicts)"
            )
        if n_judge_failures / attempted > self.max_failure_rate:
            failure_reasons.append(
                f"{n_judge_failures}/{attempted} judge calls failed "
                f"(limit {self.max_failure_rate:.0%}); results may not be representative"
            )

        report = CalibrationReport(
            judge_version=judge_version,
            rubric_version=rubric_version,
            n_items=n,
            kappa=kappa,
            kappa_interpretation=_interpret_kappa(kappa),
            spearman_rho=spearman_rho,
            spearman_pvalue=spearman_p,
            pearson_r=pearson_r,
            pearson_pvalue=pearson_p,
            agreement_rate=agreement_rate,
            bias=bias,
            confusion_matrix=cm,
            item_results=item_results,
            passed=len(failure_reasons) == 0,
            failure_reasons=failure_reasons,
            n_judge_failures=n_judge_failures,
            n_skipped_no_context=n_no_context,
        )

        if report.passed:
            logger.info("Judge calibration PASSED (kappa=%.3f, spearman=%.3f)", kappa, spearman_rho)
        else:
            logger.warning("Judge calibration FAILED: %s", failure_reasons)

        return report
