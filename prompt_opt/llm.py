"""
Central litellm access point.

All LLM calls in the codebase go through this module so that global client
behaviour (e.g. suppressing litellm's debug/feedback banners on handled
exceptions) and budget metering are configured in exactly one place.

`get_litellm()` returns the configured module; `completion()` is the metered
entry point used by the eval harness, the judge and the optimizers' proposers.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

logger = logging.getLogger(__name__)

# The BudgetController of the optimizer run currently executing in this
# thread/task, if any (see `track_budget`).
_active_budget: ContextVar[Any] = ContextVar("prompt_opt_active_budget", default=None)


def get_litellm():
    """Return the litellm module with project-wide settings applied."""
    import litellm  # type: ignore[import]

    # litellm prints "Give Feedback / Get Help ..." banners to stderr on every
    # handled exception; we handle failures ourselves, so silence the noise.
    litellm.suppress_debug_info = True
    return litellm


@contextmanager
def track_budget(budget: Any) -> Iterator[Any]:
    """
    Meter every `completion()` made inside the block against *budget*.

    Before each call the budget's caps are checked (raising BudgetExceededError
    mid-evaluation instead of overshooting), and after each call the response's
    real usage/cost is recorded.
    """
    token = _active_budget.set(budget)
    try:
        yield budget
    finally:
        _active_budget.reset(token)


def completion(**kwargs: Any) -> Any:
    """`litellm.completion`, metered against the active budget (if any)."""
    budget = _active_budget.get()
    if budget is not None:
        budget.check()
    response = get_litellm().completion(**kwargs)
    if budget is not None:
        budget.record_response(str(kwargs.get("model", "")), response)
    return response
