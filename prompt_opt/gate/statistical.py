"""
Statistical significance testing for the champion/challenger gate.

Two tests are run (belt + suspenders):
  1. Paired bootstrap (10 000 resamples) — non-parametric, robust to non-normality.
  2. Wilcoxon signed-rank test — parametric-ish, handles ties correctly.

Both are computed on per-example win/loss/tie vectors from the eval harness.

Returns:
  - win_rate: fraction of examples where challenger wins
  - confidence_interval: (lower, upper) at requested CI level
  - p_value: from Wilcoxon test
  - ci_excludes_null: True if CI excludes 0.50 (null hypothesis win_rate)

Null hypothesis: win_rate == 0.50 (coin flip — no difference).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class StatTestResult:
    """Result of a paired statistical comparison."""

    n_examples: int
    n_wins: int
    n_losses: int
    n_ties: int

    win_rate: float      # n_wins / (n_wins + n_losses + n_ties)
    loss_rate: float
    tie_rate: float

    # Paired bootstrap CI on win_rate
    ci_lower: float
    ci_upper: float
    ci_level: float      # e.g. 0.95

    # Wilcoxon signed-rank
    wilcoxon_statistic: float
    wilcoxon_pvalue: float

    # Summary flags
    ci_excludes_null: bool   # CI strictly excludes 0.50
    statistically_significant: bool   # pvalue < alpha

    @property
    def alpha(self) -> float:
        return 1 - self.ci_level

    def summary(self) -> dict:
        return {
            "n_examples": self.n_examples,
            "win_rate": round(self.win_rate, 4),
            "loss_rate": round(self.loss_rate, 4),
            "tie_rate": round(self.tie_rate, 4),
            "ci_lower": round(self.ci_lower, 4),
            "ci_upper": round(self.ci_upper, 4),
            "ci_level": self.ci_level,
            "ci_excludes_null": self.ci_excludes_null,
            "wilcoxon_pvalue": round(self.wilcoxon_pvalue, 6),
            "statistically_significant": self.statistically_significant,
        }


def paired_bootstrap_ci(
    outcomes: list[str],  # "win" | "loss" | "tie" per example
    *,
    n_resamples: int = 10_000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Compute a bootstrap confidence interval on win_rate.

    Args:
        outcomes:    Per-example outcomes ("win", "loss", "tie").
        n_resamples: Number of bootstrap resamples.
        ci_level:    Confidence level (e.g. 0.95 for 95% CI).
        seed:        Random seed for reproducibility.

    Returns:
        (win_rate, ci_lower, ci_upper)
    """
    rng = np.random.default_rng(seed)
    n = len(outcomes)

    # Encode: win=1, tie=0, loss=-1
    encoded = np.array([1 if o == "win" else (-1 if o == "loss" else 0) for o in outcomes])

    # Observed win rate
    win_rate = float(np.sum(encoded == 1)) / n

    # Bootstrap distribution of win rates
    boot_win_rates = np.empty(n_resamples)
    for i in range(n_resamples):
        sample = rng.choice(encoded, size=n, replace=True)
        boot_win_rates[i] = float(np.sum(sample == 1)) / n

    alpha = 1 - ci_level
    ci_lower = float(np.percentile(boot_win_rates, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_win_rates, 100 * (1 - alpha / 2)))

    return win_rate, ci_lower, ci_upper


def wilcoxon_test(
    challenger_scores: list[float],
    champion_scores: list[float],
    *,
    alternative: str = "greater",
) -> tuple[float, float]:
    """
    Wilcoxon signed-rank test on per-example score differences.

    Args:
        challenger_scores: Challenger scores per example.
        champion_scores:   Champion scores on the same examples.
        alternative:       "greater" (challenger > champion), "two-sided", "less".

    Returns:
        (statistic, pvalue)
    """
    if len(challenger_scores) != len(champion_scores):
        raise ValueError("challenger and champion score lists must be same length")
    if len(challenger_scores) < 5:
        logger.warning("Too few samples for Wilcoxon test (n=%d < 5)", len(challenger_scores))
        return 0.0, 1.0

    diffs = [c - ch for c, ch in zip(challenger_scores, champion_scores)]
    # scipy's wilcoxon requires non-zero differences (ties are dropped)
    non_zero_diffs = [d for d in diffs if abs(d) > 1e-9]
    if len(non_zero_diffs) < 5:
        logger.warning("Too many ties for Wilcoxon test (%d non-zero diffs)", len(non_zero_diffs))
        return 0.0, 1.0

    result = stats.wilcoxon(non_zero_diffs, alternative=alternative)
    return float(result.statistic), float(result.pvalue)


def run_statistical_tests(
    outcomes: list[str],
    challenger_scores: list[float],
    champion_scores: list[float],
    *,
    ci_level: float = 0.95,
    n_resamples: int = 10_000,
) -> StatTestResult:
    """
    Run the full statistical test suite.

    Args:
        outcomes:           Per-example outcome strings ("win"/"loss"/"tie").
        challenger_scores:  Challenger aggregate scores per example.
        champion_scores:    Champion aggregate scores per example.
        ci_level:           Confidence level for bootstrap CI.
        n_resamples:        Bootstrap resamples.

    Returns:
        StatTestResult with all metrics.
    """
    n = len(outcomes)
    n_wins = sum(1 for o in outcomes if o == "win")
    n_losses = sum(1 for o in outcomes if o == "loss")
    n_ties = sum(1 for o in outcomes if o == "tie")

    win_rate, ci_lower, ci_upper = paired_bootstrap_ci(
        outcomes, ci_level=ci_level, n_resamples=n_resamples
    )

    wil_stat, wil_p = wilcoxon_test(challenger_scores, champion_scores)

    alpha = 1 - ci_level
    ci_excludes_null = ci_lower > 0.50
    statistically_significant = wil_p < alpha

    return StatTestResult(
        n_examples=n,
        n_wins=n_wins,
        n_losses=n_losses,
        n_ties=n_ties,
        win_rate=win_rate,
        loss_rate=n_losses / n if n else 0.0,
        tie_rate=n_ties / n if n else 0.0,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_level=ci_level,
        wilcoxon_statistic=wil_stat,
        wilcoxon_pvalue=wil_p,
        ci_excludes_null=ci_excludes_null,
        statistically_significant=statistically_significant,
    )
