"""
Statistical significance testing for the champion/challenger gate.

Two tests are run (belt + suspenders):
  1. Paired bootstrap (10 000 resamples) — non-parametric CI on the win rate.
  2. Wilcoxon signed-rank test — one-sided test on the per-example score differences.

Both are computed on paired per-example results from the eval harness.

Win rate: each example is a win (1), tie (0.5) or loss (0) for the challenger, so

    win_rate = (wins + 0.5 * ties) / n.

Counting ties as half keeps the null hypothesis honest ("all ties" is exactly 0.5,
no difference) and stops a mostly-tied but clearly favourable comparison
(30 wins / 10 losses / 60 ties) from looking like a 30% win rate. `decisive_win_rate`
(wins / (wins + losses)) is reported for information.

Returns:
  - win_rate: tie-half win rate
  - confidence_interval: (lower, upper) at requested CI level
  - wilcoxon_pvalue: one-sided (challenger > champion)
  - ci_excludes_null: True if the CI lower bound is above 0.50 (null: no difference)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

_MIN_WILCOXON_DIFFS = 5


@dataclass
class StatTestResult:
    """Result of a paired statistical comparison."""

    n_examples: int
    n_wins: int
    n_losses: int
    n_ties: int

    win_rate: float      # (n_wins + 0.5 * n_ties) / n_examples
    loss_rate: float     # n_losses / n_examples
    tie_rate: float      # n_ties / n_examples

    # Paired bootstrap CI on win_rate
    ci_lower: float
    ci_upper: float
    ci_level: float      # e.g. 0.95

    # Wilcoxon signed-rank (one-sided: challenger > champion)
    wilcoxon_statistic: float
    wilcoxon_pvalue: float

    # Summary flags
    ci_excludes_null: bool   # CI lower bound > 0.50
    statistically_significant: bool   # pvalue < alpha

    # wins / (wins + losses); None if there was no decisive comparison
    decisive_win_rate: float | None = None
    # Examples with a score difference large enough to enter the Wilcoxon test
    n_nonzero_diffs: int = 0

    @property
    def alpha(self) -> float:
        return 1 - self.ci_level

    def summary(self) -> dict:
        return {
            "n_examples": self.n_examples,
            "n_wins": self.n_wins,
            "n_losses": self.n_losses,
            "n_ties": self.n_ties,
            "win_rate": round(self.win_rate, 4),
            "decisive_win_rate": (
                round(self.decisive_win_rate, 4) if self.decisive_win_rate is not None else None
            ),
            "loss_rate": round(self.loss_rate, 4),
            "tie_rate": round(self.tie_rate, 4),
            "ci_lower": round(self.ci_lower, 4),
            "ci_upper": round(self.ci_upper, 4),
            "ci_level": self.ci_level,
            "ci_excludes_null": self.ci_excludes_null,
            "wilcoxon_pvalue": round(self.wilcoxon_pvalue, 6),
            "n_nonzero_diffs": self.n_nonzero_diffs,
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
    Compute a bootstrap confidence interval on the (tie-half) win rate.

    Resampling examples with replacement gives category counts that follow a
    multinomial distribution over (win, tie, loss), so the bootstrap distribution
    is drawn directly from it — exact, and vectorised.

    Args:
        outcomes:    Per-example outcomes ("win", "loss", "tie").
        n_resamples: Number of bootstrap resamples.
        ci_level:    Confidence level (e.g. 0.95 for 95% CI).
        seed:        Random seed for reproducibility.

    Returns:
        (win_rate, ci_lower, ci_upper). With no outcomes there is no information:
        (0.0, 0.0, 1.0).
    """
    n = len(outcomes)
    if n == 0:
        return 0.0, 0.0, 1.0

    n_wins = sum(1 for o in outcomes if o == "win")
    n_ties = sum(1 for o in outcomes if o == "tie")
    n_losses = n - n_wins - n_ties
    win_rate = (n_wins + 0.5 * n_ties) / n

    rng = np.random.default_rng(seed)
    probs = np.array([n_wins, n_ties, n_losses], dtype=float) / n
    counts = rng.multinomial(n, probs / probs.sum(), size=n_resamples)
    boot_win_rates = (counts[:, 0] + 0.5 * counts[:, 1]) / n

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
        (statistic, pvalue). (0.0, 1.0) — "no evidence" — if fewer than 5 examples
        have a non-zero difference.
    """
    if len(challenger_scores) != len(champion_scores):
        raise ValueError("challenger and champion score lists must be same length")
    if len(challenger_scores) < _MIN_WILCOXON_DIFFS:
        logger.warning("Too few samples for Wilcoxon test (n=%d < 5)", len(challenger_scores))
        return 0.0, 1.0

    diffs = [c - ch for c, ch in zip(challenger_scores, champion_scores, strict=True)]
    # scipy's wilcoxon requires non-zero differences (ties are dropped)
    non_zero_diffs = [d for d in diffs if abs(d) > 1e-9]
    if len(non_zero_diffs) < _MIN_WILCOXON_DIFFS:
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
    n_nonzero = sum(
        1 for c, ch in zip(challenger_scores, champion_scores, strict=True) if abs(c - ch) > 1e-9
    )

    alpha = 1 - ci_level
    ci_excludes_null = n > 0 and ci_lower > 0.50
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
        decisive_win_rate=n_wins / (n_wins + n_losses) if (n_wins + n_losses) else None,
        n_nonzero_diffs=n_nonzero,
    )
