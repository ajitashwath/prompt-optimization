"""
Budget controller: hard caps on LLM calls, wall-clock time, and USD spend.

Shared across optimizer strategies. Raises BudgetExceededError when any limit
is hit so the optimizer can gracefully terminate and return its best candidate.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from prompt_opt.config import settings

logger = logging.getLogger(__name__)


class BudgetExceededError(Exception):
    """Raised when any budget cap is hit during optimization."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"Budget exceeded: {reason}")


# Approximate cost per 1K tokens for common models (input, output)
_MODEL_COSTS: dict[str, tuple[float, float]] = {
    "openai/gpt-4o": (0.005, 0.015),
    "openai/gpt-4o-mini": (0.00015, 0.0006),
    "anthropic/claude-3-5-sonnet-20241022": (0.003, 0.015),
    "anthropic/claude-3-haiku-20240307": (0.00025, 0.00125),
    "google/gemini-1.5-flash": (0.00035, 0.00105),
    "google/gemini-1.5-pro": (0.0035, 0.0105),
}

_DEFAULT_COST_PER_1K = (0.001, 0.003)  # conservative default


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    input_rate, output_rate = _MODEL_COSTS.get(model, _DEFAULT_COST_PER_1K)
    return (input_tokens / 1000) * input_rate + (output_tokens / 1000) * output_rate


@dataclass
class BudgetController:
    """
    Tracks and enforces budget limits across an optimization run.

    Usage::

        budget = BudgetController()
        budget.record_call(model="openai/gpt-4o-mini", input_tokens=200, output_tokens=100)
        budget.check()  # raises BudgetExceededError if any limit hit
    """

    max_llm_calls: int = field(default_factory=lambda: settings.max_llm_calls_per_run)
    max_wall_seconds: int = field(default_factory=lambda: settings.max_wall_seconds_per_run)
    max_usd_spend: float = field(default_factory=lambda: settings.max_usd_spend_per_run)

    # Tracked usage (mutable)
    _llm_calls: int = field(default=0, init=False)
    _usd_spend: float = field(default=0.0, init=False)
    _start_time: float = field(default_factory=time.time, init=False)

    def record_call(
        self,
        model: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
        n_calls: int = 1,
    ) -> None:
        """Record one or more LLM API calls."""
        self._llm_calls += n_calls
        cost = estimate_cost_usd(model, input_tokens, output_tokens) * n_calls
        self._usd_spend += cost
        logger.debug(
            "Budget: calls=%d/%d, spend=$%.4f/$%.2f, elapsed=%.0fs/%ds",
            self._llm_calls,
            self.max_llm_calls,
            self._usd_spend,
            self.max_usd_spend,
            self.elapsed_seconds,
            self.max_wall_seconds,
        )

    def check(self) -> None:
        """Raise BudgetExceededError if any cap is hit."""
        if self._llm_calls >= self.max_llm_calls:
            raise BudgetExceededError(
                f"LLM call cap reached ({self._llm_calls}/{self.max_llm_calls})"
            )
        if self._usd_spend >= self.max_usd_spend:
            raise BudgetExceededError(
                f"USD spend cap reached (${self._usd_spend:.4f}/${self.max_usd_spend:.2f})"
            )
        elapsed = self.elapsed_seconds
        if elapsed >= self.max_wall_seconds:
            raise BudgetExceededError(
                f"Wall-clock cap reached ({elapsed:.0f}s/{self.max_wall_seconds}s)"
            )

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self._start_time

    @property
    def remaining_calls(self) -> int:
        return max(0, self.max_llm_calls - self._llm_calls)

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.max_usd_spend - self._usd_spend)

    def summary(self) -> dict:
        return {
            "llm_calls": self._llm_calls,
            "max_llm_calls": self.max_llm_calls,
            "usd_spend": round(self._usd_spend, 4),
            "max_usd_spend": self.max_usd_spend,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "max_wall_seconds": self.max_wall_seconds,
        }
