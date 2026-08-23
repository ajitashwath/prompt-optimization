"""
PII scrubbing pass.

Two-stage pipeline:
  1. Regex-based: deterministic patterns (emails, phones, SSNs, card numbers, IPs)
  2. NER-based: spaCy en_core_web_sm detects PERSON / ORG / GPE named entities

Produces a scrubbed string plus a redaction log for audit purposes.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import spacy

logger = logging.getLogger(__name__)


# ─── Regex Patterns ──────────────────────────────────────────────────────────

# Ordered from most-specific to least-specific to avoid partial matches.
_REGEX_PATTERNS: list[tuple[str, str, str]] = [
    # label, replacement, pattern
    ("EMAIL", "[EMAIL]", r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"),
    ("SSN", "[SSN]", r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"),
    ("CARD", "[CARD]", r"\b(?:\d[ -]?){13,16}\b"),
    ("PHONE_US", "[PHONE]", r"\b(?:\+?1[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)?\d{3}[\s.-]?\d{4}\b"),
    ("IP", "[IP]", r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    ("URL", "[URL]", r"https?://[^\s<>\"']+"),
    ("ZIP", "[ZIP]", r"\b\d{5}(?:-\d{4})?\b"),
]

_COMPILED: list[tuple[str, str, re.Pattern]] = [
    (label, replacement, re.compile(pattern, re.IGNORECASE))
    for label, replacement, pattern in _REGEX_PATTERNS
]

# NER entity types to redact
_NER_TYPES: frozenset[str] = frozenset({"PERSON", "ORG", "GPE", "LOC", "FAC"})


# ─── Data classes ────────────────────────────────────────────────────────────


@dataclass
class RedactionEntry:
    label: str
    original: str
    start: int
    end: int


@dataclass
class ScrubResult:
    scrubbed_text: str
    redactions: list[RedactionEntry] = field(default_factory=list)
    regex_count: int = 0
    ner_count: int = 0

    @property
    def total_redactions(self) -> int:
        return self.regex_count + self.ner_count


# ─── spaCy loader ────────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def _load_nlp() -> spacy.Language | None:
    """Lazy-load spaCy model; returns None if unavailable (graceful degradation)."""
    try:
        import spacy  # type: ignore[import]

        return spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "spaCy model unavailable (%s). NER-based PII scrubbing disabled. "
            "Run: python -m spacy download en_core_web_sm",
            exc,
        )
        return None


# ─── Core scrub function ─────────────────────────────────────────────────────


def scrub(text: str, *, use_ner: bool = True) -> ScrubResult:
    """
    Scrub PII from *text* and return the redacted version plus an audit log.

    Args:
        text:    Raw input string.
        use_ner: Whether to run the NER pass (disable for speed if spaCy unavailable).

    Returns:
        ScrubResult with scrubbed_text and a list of RedactionEntry records.
    """
    redactions: list[RedactionEntry] = []
    working = text

    # ── Stage 1: Regex ───────────────────────────────────────────────────────
    regex_count = 0
    for label, replacement, pattern in _COMPILED:
        for match in list(pattern.finditer(working)):
            redactions.append(
                RedactionEntry(
                    label=label,
                    original=match.group(),
                    start=match.start(),
                    end=match.end(),
                )
            )
        new_text = pattern.sub(replacement, working)
        if new_text != working:
            regex_count += len(pattern.findall(working))
            working = new_text

    # ── Stage 2: NER ────────────────────────────────────────────────────────
    ner_count = 0
    if use_ner:
        nlp = _load_nlp()
        if nlp is not None:
            doc = nlp(working)
            # Process in reverse order to preserve char offsets
            ents = [e for e in doc.ents if e.label_ in _NER_TYPES]
            for ent in reversed(ents):
                tag = f"[{ent.label_}]"
                redactions.append(
                    RedactionEntry(
                        label=ent.label_,
                        original=ent.text,
                        start=ent.start_char,
                        end=ent.end_char,
                    )
                )
                working = working[: ent.start_char] + tag + working[ent.end_char :]
                ner_count += 1

    return ScrubResult(
        scrubbed_text=working,
        redactions=redactions,
        regex_count=regex_count,
        ner_count=ner_count,
    )


def scrub_conversation_text(text: str) -> str:
    """Convenience wrapper that just returns the scrubbed string."""
    return scrub(text).scrubbed_text
