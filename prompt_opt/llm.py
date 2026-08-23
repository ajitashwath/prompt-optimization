"""
Central litellm access point.

All LLM calls in the codebase go through `get_litellm()` so that global
client behaviour (e.g. suppressing litellm's debug/feedback banners on
handled exceptions) is configured in exactly one place.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def get_litellm():
    """Return the litellm module with project-wide settings applied."""
    import litellm  # type: ignore[import]

    # litellm prints "Give Feedback / Get Help ..." banners to stderr on every
    # handled exception; we handle failures ourselves, so silence the noise.
    litellm.suppress_debug_info = True
    return litellm
