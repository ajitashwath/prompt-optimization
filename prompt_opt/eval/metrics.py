"""
Eval metrics: aggregation, win-rate, category breakdown.

All computations are over per-example JudgeResult objects.
Prefers paired comparisons (challenger vs champion on same input) over
independent absolute scores because paired tests have much lower variance.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PerExampleResult:
    example_id: str
    category: str
    challenger_score: float
    champion_score: float | None
    outcome: Literal["win", "loss", "tie"] | None  # relative to challenger
    judge_rationale: str = ""
    response_text: str = ""            # challenger's generated response
    champion_response_text: str | None = None  # champion's response (paired mode)


@dataclass
class CategoryBreakdown:
    category: str
    n: int
    mean_score: float
    win_rate: float | None       # None if no champion baseline
    loss_rate: float | None


@dataclass
class EvalSummary:
    """Complete evaluation result for a prompt candidate on a split."""

    prompt_hash: str
    split: str
    judge_version: str
    n_examples: int

    # Aggregate
    mean_score: float
    median_score: float
    std_score: float

    # Paired comparison (if champion provided)
    win_rate: float | None = None
    loss_rate: float | None = None
    tie_rate: float | None = None

    # Per-category breakdown
    category_breakdown: list[CategoryBreakdown] = field(default_factory=list)

    # Per-example (for diagnostics / GEPA's reflective mutation)
    per_example: list[PerExampleResult] = field(default_factory=list)

    # Guardrail summary (populated by harness)
    guardrail_passed: bool = True
    guardrail_details: dict = field(default_factory=dict)

    def worst_categories(self, n: int = 3) -> list[CategoryBreakdown]:
        """Return the n categories with lowest win-rate or mean score."""
        return sorted(
            self.category_breakdown,
            key=lambda c: c.win_rate if c.win_rate is not None else c.mean_score,
        )[:n]

    def failing_examples(self, threshold: float = 0.4) -> list[PerExampleResult]:
        """Return examples where challenger scored below threshold."""
        return [e for e in self.per_example if e.challenger_score < threshold]

    def to_dict(self) -> dict:
        return {
            "prompt_hash": self.prompt_hash,
            "split": self.split,
            "judge_version": self.judge_version,
            "n_examples": self.n_examples,
            "mean_score": self.mean_score,
            "median_score": self.median_score,
            "std_score": self.std_score,
            "win_rate": self.win_rate,
            "loss_rate": self.loss_rate,
            "tie_rate": self.tie_rate,
            "guardrail_passed": self.guardrail_passed,
            "category_breakdown": [
                {
                    "category": c.category,
                    "n": c.n,
                    "mean_score": c.mean_score,
                    "win_rate": c.win_rate,
                }
                for c in self.category_breakdown
            ],
        }


def _outcome(
    challenger: float, champion: float, margin: float = 0.02
) -> Literal["win", "loss", "tie"]:
    if challenger > champion + margin:
        return "win"
    if champion > challenger + margin:
        return "loss"
    return "tie"


def compute_eval_summary(
    prompt_hash: str,
    split: str,
    judge_version: str,
    per_example: list[PerExampleResult],
    guardrail_result=None,
) -> EvalSummary:
    """
    Aggregate per-example results into an EvalSummary.

    Args:
        prompt_hash:    Hash of the evaluated prompt.
        split:          "train" | "dev" | "test"
        judge_version:  Version of judge used.
        per_example:    List of PerExampleResult.
        guardrail_result: GuardrailResult (optional).
    """
    if not per_example:
        return EvalSummary(
            prompt_hash=prompt_hash,
            split=split,
            judge_version=judge_version,
            n_examples=0,
            mean_score=0.0,
            median_score=0.0,
            std_score=0.0,
        )

    scores = [e.challenger_score for e in per_example]
    mean_score = float(np.mean(scores))
    median_score = float(np.median(scores))
    std_score = float(np.std(scores))

    # Paired win/loss/tie
    outcomes = [e.outcome for e in per_example if e.outcome is not None]
    n_outcomes = len(outcomes)
    win_rate = sum(1 for o in outcomes if o == "win") / n_outcomes if n_outcomes else None
    loss_rate = sum(1 for o in outcomes if o == "loss") / n_outcomes if n_outcomes else None
    tie_rate = sum(1 for o in outcomes if o == "tie") / n_outcomes if n_outcomes else None

    # Category breakdown
    by_cat: dict[str, list[PerExampleResult]] = defaultdict(list)
    for e in per_example:
        by_cat[e.category].append(e)

    breakdowns: list[CategoryBreakdown] = []
    for cat, items in sorted(by_cat.items()):
        cat_scores = [i.challenger_score for i in items]
        cat_outcomes = [i.outcome for i in items if i.outcome is not None]
        n_cat_out = len(cat_outcomes)
        breakdowns.append(
            CategoryBreakdown(
                category=cat,
                n=len(items),
                mean_score=float(np.mean(cat_scores)),
                win_rate=(
                    sum(1 for o in cat_outcomes if o == "win") / n_cat_out
                    if n_cat_out
                    else None
                ),
                loss_rate=(
                    sum(1 for o in cat_outcomes if o == "loss") / n_cat_out
                    if n_cat_out
                    else None
                ),
            )
        )

    guardrail_passed = True
    guardrail_details: dict = {}
    if guardrail_result is not None:
        guardrail_passed = guardrail_result.overall_pass
        guardrail_details = guardrail_result.summary()

    return EvalSummary(
        prompt_hash=prompt_hash,
        split=split,
        judge_version=judge_version,
        n_examples=len(per_example),
        mean_score=mean_score,
        median_score=median_score,
        std_score=std_score,
        win_rate=win_rate,
        loss_rate=loss_rate,
        tie_rate=tie_rate,
        category_breakdown=breakdowns,
        per_example=per_example,
        guardrail_passed=guardrail_passed,
        guardrail_details=guardrail_details,
    )
