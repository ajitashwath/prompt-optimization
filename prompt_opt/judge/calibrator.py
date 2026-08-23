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
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from scipy import stats

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
    r, p = stats.pearsonr(x, y)
    return float(r), float(p)


# ─── Report data class ───────────────────────────────────────────────────────


@dataclass
class BiasReport:
    length_bias_rho: float          # Spearman rho: judge_preference_score ~ len(winner)
    length_bias_pvalue: float
    position_bias_rate: float       # fraction of items where judge preferred A first
    style_bias_rho: float | None = None   # correlation with markdown presence (optional)

    @property
    def has_significant_length_bias(self) -> bool:
        return abs(self.length_bias_rho) > 0.2 and self.length_bias_pvalue < 0.05

    @property
    def has_significant_position_bias(self) -> bool:
        return self.position_bias_rate > 0.6  # >60% preference for first item → suspect


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
        }
        if self.bias:
            d["bias"] = {
                "length_bias_rho": self.bias.length_bias_rho,
                "length_bias_pvalue": self.bias.length_bias_pvalue,
                "position_bias_rate": self.bias.position_bias_rate,
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
        min_kappa: float = 0.6,
        min_spearman: float = 0.7,
    ):
        self.judge = judge
        self.gold_store = gold_store
        self.min_kappa = min_kappa
        self.min_spearman = min_spearman

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
            conversations:   Map from conversation_id → ConversationRecord.
                             If None, the judge is run without conversation context
                             (only response text is used).
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

        logger.info("Calibrating judge over %d gold items …", len(labeled))

        human_prefs: list[int] = []
        judge_prefs: list[int] = []
        judge_scores_a: list[float] = []
        judge_scores_b: list[float] = []
        len_a_list: list[int] = []
        len_b_list: list[int] = []
        item_results: list[dict] = []

        for item in labeled:
            if item.preferred is None:
                continue

            conv = conversations.get(item.conversation_id) if conversations else None

            try:
                if conv is not None:
                    judge_result: PairwiseJudgeResult = self.judge.score_pairwise(
                        conv, item.response_a, item.response_b
                    )
                else:
                    # Minimal ConversationRecord with just the text
                    from prompt_opt.ingestion.schema import Message, MessageRole

                    stub = ConversationRecord(
                        conversation_id=item.conversation_id,
                        messages=[
                            Message(role=MessageRole.USER, content="[context unavailable]")
                        ],
                    )
                    judge_result = self.judge.score_pairwise(
                        stub, item.response_a, item.response_b
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Judge call failed for item %s: %s", item.id, exc)
                continue

            human_prefs.append(_PREF_MAP.get(item.preferred or "tie", 2))
            judge_prefs.append(_PREF_MAP.get(judge_result.preferred, 2))
            judge_scores_a.append(judge_result.aggregate_score_a)
            judge_scores_b.append(judge_result.aggregate_score_b)
            len_a_list.append(len(item.response_a))
            len_b_list.append(len(item.response_b))

            item_results.append(
                {
                    "item_id": item.id,
                    "human_preferred": item.preferred,
                    "judge_preferred": judge_result.preferred,
                    "agree": item.preferred == judge_result.preferred,
                    "judge_score_a": judge_result.aggregate_score_a,
                    "judge_score_b": judge_result.aggregate_score_b,
                }
            )

        n = len(human_prefs)
        if n < 5:
            return CalibrationReport(
                judge_version=judge_version,
                rubric_version=rubric_version,
                n_items=n,
                passed=False,
                failure_reasons=[f"Too few valid items for calibration (got {n}, need ≥5)"],
            )

        # Agreement
        kappa = _cohen_kappa(human_prefs, judge_prefs)
        agreement_rate = sum(h == j for h, j in zip(human_prefs, judge_prefs)) / n

        # Continuous correlation (use diff of scores as proxy for preference strength)
        score_diffs = [a - b for a, b in zip(judge_scores_a, judge_scores_b)]
        human_binary = [1 if h == 0 else (-1 if h == 1 else 0) for h in human_prefs]
        spearman_rho, spearman_p = _spearman(score_diffs, human_binary)
        pearson_r, pearson_p = _pearson(score_diffs, human_binary)

        # Confusion matrix (rows=human, cols=judge) for A/B/tie
        cm = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        for h, j in zip(human_prefs, judge_prefs):
            cm[h][j] += 1

        # Bias: length bias
        len_winner = [
            len_a_list[i] if judge_prefs[i] == 0 else len_b_list[i]
            for i in range(n)
        ]
        len_rho, len_p = _spearman(
            [float(lw) for lw in len_winner],
            [float(s) for s in score_diffs],
        )

        # Position bias: in pairwise, judge always "prefers A" when uncertain
        pos_bias_rate = sum(1 for j in judge_prefs if j == 0) / n  # fraction choosing A

        bias = BiasReport(
            length_bias_rho=len_rho,
            length_bias_pvalue=len_p,
            position_bias_rate=pos_bias_rate,
        )

        # Pass/fail
        failure_reasons: list[str] = []
        if kappa < self.min_kappa:
            failure_reasons.append(
                f"kappa={kappa:.3f} < threshold={self.min_kappa}"
            )
        if abs(spearman_rho) < self.min_spearman:
            failure_reasons.append(
                f"spearman_rho={spearman_rho:.3f} < threshold={self.min_spearman}"
            )
        if bias.has_significant_length_bias:
            failure_reasons.append(
                f"Significant length bias detected (rho={len_rho:.3f}, p={len_p:.4f})"
            )
        if bias.has_significant_position_bias:
            failure_reasons.append(
                f"Significant position bias detected (rate={pos_bias_rate:.2f})"
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
        )

        if report.passed:
            logger.info("Judge calibration PASSED (kappa=%.3f, spearman=%.3f)", kappa, spearman_rho)
        else:
            logger.warning("Judge calibration FAILED: %s", failure_reasons)

        return report
