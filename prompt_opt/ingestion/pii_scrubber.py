"""
PII scrubbing pass.

Two-stage pipeline, resolved together on the ORIGINAL text:
  1. Pattern-based: secrets, emails, URLs, payment cards (Luhn-checked), SSNs, IPs,
     phone numbers, street addresses, ZIP codes and self-introduced names.
  2. NER-based (optional): spaCy en_core_web_sm detects PERSON / ORG / GPE / ... entities.

Every candidate span is found in the original text, overlaps are resolved by priority
(a secret beats an email beats a URL beats a card ...), and the text is rewritten in one
pass. Redaction offsets therefore refer to the original text.

Precision matters as much as recall: this text feeds evaluation, so redacting every
7-digit or 5-digit number (`1234567`, `10000`) would corrupt math and coding conversations.
Numbers are redacted only when they validate (card: Luhn; SSN: real ranges; IP: octets <= 255)
or have a supporting context word / format (phone: separators or "phone", "call", ...; ZIP: a
"zip code" label or a state code before it).

Patterns are best-effort. Names and addresses in free text are only partly covered without NER;
enable it (`--use-ner`, `PII_USE_NER=true`) for more.

Produces a scrubbed string plus a redaction log (label and offsets only, never the PII itself).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import spacy

logger = logging.getLogger(__name__)

# spaCy's default max_length is 1M characters; stay far below it.
_NER_CHUNK_CHARS = 50_000


# ─── Validators ──────────────────────────────────────────────────────────────


def _digits(value: str) -> str:
    return re.sub(r"\D", "", value)


def _luhn_ok(value: str) -> bool:
    digits = _digits(value)
    if not 13 <= len(digits) <= 19 or len(set(digits)) == 1:
        return False
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _ip_ok(value: str) -> bool:
    return all(0 <= int(part) <= 255 for part in value.split("."))


def _intl_phone_ok(value: str) -> bool:
    return 9 <= len(_digits(value)) <= 15


def _context_phone_ok(value: str) -> bool:
    return 7 <= len(_digits(value)) <= 15


# ─── Patterns ────────────────────────────────────────────────────────────────

_US_STATES = (
    "AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV "
    "NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY"
).split()


@dataclass(frozen=True)
class _Spec:
    label: str
    replacement: str
    pattern: re.Pattern
    priority: int                       # lower wins when spans overlap
    group: int = 0                      # which group is the sensitive part
    validator: Callable[[str], bool] | None = None


def _spec(label, priority, pattern, *, group=0, validator=None, flags=0, replacement=None):
    return _Spec(
        label=label,
        replacement=replacement or f"[{label}]",
        pattern=re.compile(pattern, flags),
        priority=priority,
        group=group,
        validator=validator,
    )


_SPECS: list[_Spec] = [
    # Credentials and keys
    _spec("SECRET", 0, r"\bsk-(?:ant-|proj-)?[A-Za-z0-9_\-]{20,}"),
    _spec("SECRET", 0, r"\bAKIA[0-9A-Z]{16}\b"),
    _spec("SECRET", 0, r"\bgh[pousr]_[A-Za-z0-9]{30,}"),
    _spec("SECRET", 0, r"\bxox[abprs]-[A-Za-z0-9\-]{10,}"),
    _spec("SECRET", 0, r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}"),
    _spec(
        "SECRET", 0,
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----",
    ),
    _spec(
        "SECRET", 0,
        r"\b(?:password|passwd|secret|api[_-]?key|access[_-]?token|auth[_-]?token)\b"
        r"\s*[:=]\s*['\"]?([^\s'\"]{8,})",
        group=1, flags=re.IGNORECASE,
    ),
    # Contact details
    _spec("EMAIL", 1, r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"),
    _spec("URL", 2, r"https?://[^\s<>\"']+", flags=re.IGNORECASE),
    # Payment card: 13-19 digits, optionally grouped, and it must pass the Luhn checksum
    _spec("CARD", 3, r"(?<![\d-])(?:\d[ -]?){12,18}\d(?![\d-])", validator=_luhn_ok),
    # SSN: structured form with valid ranges, or 9 digits right after an SSN label
    _spec("SSN", 4, r"(?<![\d-])(?!000|666|9\d\d)\d{3}[- ](?!00)\d{2}[- ](?!0000)\d{4}(?![\d-])"),
    _spec(
        "SSN", 4,
        r"\b(?:ssn|social security(?: number)?)\b\D{0,20}?(\d{9})(?!\d)",
        group=1, flags=re.IGNORECASE,
    ),
    _spec("IP", 5, r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])", validator=_ip_ok),
    _spec("IP", 5, r"\b(?:[0-9A-Fa-f]{1,4}:){7}[0-9A-Fa-f]{1,4}\b"),
    # Phone: formatted (separators between the groups), international, or labelled
    _spec("PHONE", 6, r"(?<![\d-])(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}(?![\d-])"),
    _spec(
        "PHONE", 6, r"(?<![\w])\+\d{1,3}[\s.-]?\(?\d{1,4}\)?(?:[\s.-]?\d{2,4}){2,4}(?!\d)",
        validator=_intl_phone_ok,
    ),
    _spec(
        "PHONE", 6,
        r"\b(?:phone|tel|telephone|mobile|cell|whatsapp|fax|call|text)\b[^\d\n]{0,15}"
        r"(\+?\d[\d\s().-]{5,}\d)",
        group=1, validator=_context_phone_ok, flags=re.IGNORECASE,
    ),
    # Street address ("221 Baker Street") and self-introduced names ("my name is Jane Doe")
    _spec(
        "ADDRESS", 7,
        r"\b\d{1,6}\s+(?:[A-Z][A-Za-z]+\s+){1,3}"
        r"(?:Street|Avenue|Road|Boulevard|Lane|St\.?|Ave\.?|Rd\.?|Blvd\.?|Ln\.?)(?![A-Za-z])",
    ),
    _spec(
        "PERSON", 9,
        r"\b(?:[Mm]y name is|[Ii] am called|[Cc]all me)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b",
        group=1,
    ),
    # ZIP: only with a label, or after a US state code ("Springfield, IL 62704")
    _spec(
        "ZIP", 8,
        r"\b(?:zip(?:\s*code)?|postal\s*code|postcode)\b\D{0,5}(\d{5}(?:-\d{4})?)\b",
        group=1, flags=re.IGNORECASE,
    ),
    _spec("ZIP", 8, rf"\b(?:{'|'.join(_US_STATES)})\s+(\d{{5}}(?:-\d{{4}})?)\b", group=1),
]

# NER entity types to redact
_NER_TYPES: frozenset[str] = frozenset({"PERSON", "ORG", "GPE", "LOC", "FAC"})
_NER_PRIORITY = 10


# ─── Data classes ────────────────────────────────────────────────────────────


@dataclass
class RedactionEntry:
    """One redaction. Offsets refer to the ORIGINAL text; the PII itself is not stored."""

    label: str
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


@dataclass
class _Span:
    start: int
    end: int
    label: str
    replacement: str
    priority: int
    from_ner: bool = False


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


# ─── Span finding ────────────────────────────────────────────────────────────


def _regex_spans(text: str) -> list[_Span]:
    spans: list[_Span] = []
    for spec in _SPECS:
        for match in spec.pattern.finditer(text):
            start, end = match.span(spec.group)
            if start < 0:
                continue
            if spec.label == "URL":  # trailing sentence punctuation is not part of the URL
                while end > start and text[end - 1] in ".,;:!?)]}\"'":
                    end -= 1
            value = text[start:end]
            if spec.validator is not None and not spec.validator(value):
                continue
            spans.append(_Span(start, end, spec.label, spec.replacement, spec.priority))
    return spans


def _ner_spans(text: str, nlp: Any, types: frozenset[str]) -> list[_Span]:
    """Named-entity spans, computed in bounded chunks (spaCy refuses very long inputs)."""
    spans: list[_Span] = []
    for offset in range(0, len(text), _NER_CHUNK_CHARS):
        chunk = text[offset : offset + _NER_CHUNK_CHARS]
        for ent in nlp(chunk).ents:
            if ent.label_ in types:
                spans.append(
                    _Span(
                        offset + ent.start_char,
                        offset + ent.end_char,
                        ent.label_,
                        f"[{ent.label_}]",
                        _NER_PRIORITY,
                        from_ner=True,
                    )
                )
    return spans


def _resolve(spans: list[_Span]) -> list[_Span]:
    """Keep non-overlapping spans, preferring higher priority, then longer, then earlier."""
    accepted: list[_Span] = []
    for span in sorted(spans, key=lambda s: (s.priority, -(s.end - s.start), s.start)):
        if span.end <= span.start:
            continue
        if all(span.end <= a.start or span.start >= a.end for a in accepted):
            accepted.append(span)
    return sorted(accepted, key=lambda s: s.start)


# ─── Core scrub function ─────────────────────────────────────────────────────


def scrub(
    text: str,
    *,
    use_ner: bool = True,
    ner_types: frozenset[str] | set[str] | None = None,
) -> ScrubResult:
    """
    Scrub PII from *text* and return the redacted version plus an audit log.

    Args:
        text:      Raw input string.
        use_ner:   Whether to add the NER pass (needs spaCy + en_core_web_sm; skipped with a
                   warning if unavailable).
        ner_types: spaCy entity labels to redact (default: PERSON, ORG, GPE, LOC, FAC).

    Returns:
        ScrubResult with scrubbed_text and a list of RedactionEntry records.
    """
    spans = _regex_spans(text)
    if use_ner:
        nlp = _load_nlp()
        if nlp is not None:
            spans.extend(_ner_spans(text, nlp, frozenset(ner_types or _NER_TYPES)))

    accepted = _resolve(spans)
    if not accepted:
        return ScrubResult(scrubbed_text=text)

    pieces: list[str] = []
    cursor = 0
    for span in accepted:
        pieces.append(text[cursor : span.start])
        pieces.append(span.replacement)
        cursor = span.end
    pieces.append(text[cursor:])

    return ScrubResult(
        scrubbed_text="".join(pieces),
        redactions=[RedactionEntry(s.label, s.start, s.end) for s in accepted],
        regex_count=sum(1 for s in accepted if not s.from_ner),
        ner_count=sum(1 for s in accepted if s.from_ner),
    )


def scrub_structure(obj: Any, *, use_ner: bool = True) -> tuple[Any, list[RedactionEntry]]:
    """
    Scrub every string inside a nested dict / list / str (e.g. tool-call arguments).

    Non-string leaves are returned unchanged. Dict keys are left alone. The returned
    redactions are pooled (their offsets are per-string and only their labels are meaningful).
    """
    redactions: list[RedactionEntry] = []

    def walk(value: Any) -> Any:
        if isinstance(value, str):
            result = scrub(value, use_ner=use_ner)
            redactions.extend(result.redactions)
            return result.scrubbed_text
        if isinstance(value, dict):
            return {k: walk(v) for k, v in value.items()}
        if isinstance(value, list):
            return [walk(v) for v in value]
        return value

    return walk(obj), redactions


def scrub_conversation_text(text: str) -> str:
    """Convenience wrapper that just returns the scrubbed string."""
    return scrub(text).scrubbed_text
