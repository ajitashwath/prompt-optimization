"""
Conversation ingestion orchestrator.

Reads raw JSON-Lines conversation logs, applies PII scrubbing, deduplication,
category classification (heuristic), and stratified splitting.

Robustness rules:
  * One bad record never aborts an ingest: it is skipped, counted and reported
    (`Ingestor.last_report`), with the reason.
  * A conversation is never silently truncated: an unusable message (unknown role, ...) makes
    the whole conversation a reported skip rather than dropping a turn from the middle.
  * Tool calls are preserved (native format and OpenAI-style `tool_calls` / `function_call`).

Entry-point for the `prompt-opt ingest` CLI command.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from prompt_opt.ingestion.deduplicator import DedupResult, deduplicate
from prompt_opt.ingestion.pii_scrubber import scrub, scrub_structure
from prompt_opt.ingestion.schema import (
    ConversationRecord,
    DifficultyBand,
    Message,
    MessageRole,
    RawConversation,
    ToolCall,
    WeakSignal,
)
from prompt_opt.ingestion.stratifier import StratifyResult, stratify

logger = logging.getLogger(__name__)


# ─── Heuristic category classifier ──────────────────────────────────────────


def _normalise_apostrophes(text: str) -> str:
    return text.replace("’", "'").replace("‘", "'")


def _keyword_pattern(words: list[str]) -> re.Pattern:
    """Whole-word match with common inflections (`calculate` -> `calculated`, `calculating`)."""
    alternatives = "|".join(re.escape(w) for w in words)
    return re.compile(rf"\b(?:{alternatives})(?:s|es|ed|ing|d)?\b", re.IGNORECASE)


# Listed in tie-break order. Whole words only: "whatever" is not "hate", "pharmacy" is not "harm".
_CATEGORY_PATTERNS: dict[str, re.Pattern] = {
    category: _keyword_pattern(words)
    for category, words in {
        "safety": ["harm", "illegal", "violence", "weapon", "drug", "self-harm", "hate"],
        "coding": ["code", "python", "javascript", "function", "bug", "error", "exception"],
        "math": ["calculate", "equation", "solve", "integral", "derivative", "probability"],
        "creative": ["write a story", "poem", "creative", "fiction", "imagine"],
        "factual_qa": ["what is", "who is", "when did", "where is", "why does"],
        "tool_use": ["search", "browse", "fetch", "execute", "api"],
        "multi_turn_correction": ["that's wrong", "incorrect", "you made a mistake", "try again"],
    }.items()
}

_CORRECTION_PATTERN = _keyword_pattern(["that's wrong", "incorrect", "you made a mistake"])


def _user_text(conv: ConversationRecord) -> str:
    return _normalise_apostrophes(
        " ".join(m.content or "" for m in conv.messages if m.role == MessageRole.USER)
    )


def _classify_category(conv: ConversationRecord) -> str:
    """
    Keyword-based category classifier: the category with the most distinct keyword hits in
    the user turns wins (ties go to the earlier category). Replace with a model in production.
    """
    text = _user_text(conv)
    best, best_score = "general", 0
    for category, pattern in _CATEGORY_PATTERNS.items():
        score = len({match.lower() for match in pattern.findall(text)})
        if score > best_score:
            best, best_score = category, score
    return best


def _classify_difficulty(conv: ConversationRecord) -> DifficultyBand:
    """Estimate difficulty from structural signals."""
    n_turns = len(conv.user_turns)
    has_correction = bool(_CORRECTION_PATTERN.search(_user_text(conv)))
    if has_correction or n_turns > 5:
        return DifficultyBand.HARD
    if n_turns > 2:
        return DifficultyBand.MEDIUM
    return DifficultyBand.EASY


# ─── Raw message parsing ────────────────────────────────────────────────────


class IngestSkipError(Exception):
    """A record that cannot be turned into a usable conversation (reported, not fatal)."""


_ROLE_ALIASES = {
    "human": "user",
    "ai": "assistant",
    "bot": "assistant",
    "model": "assistant",
    "function": "tool",
    "developer": "system",
}


def _normalise_content(value: Any) -> str | None:
    """Text content from a string, None, or an OpenAI-style list of content parts."""
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for part in value:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "\n".join(parts) or None
    return json.dumps(value, ensure_ascii=False, default=str)


def _parse_tool_calls(raw_calls: Any) -> list[ToolCall]:
    """Tool calls in the native format or the OpenAI one; malformed entries are skipped."""
    if isinstance(raw_calls, dict):
        raw_calls = [raw_calls]
    calls: list[ToolCall] = []
    for i, item in enumerate(raw_calls or []):
        if not isinstance(item, dict):
            continue
        function = item.get("function") if isinstance(item.get("function"), dict) else {}
        name = item.get("name") or function.get("name")
        if not name:
            logger.debug("Skipping tool call without a name: %s", item)
            continue
        arguments = item.get("arguments", function.get("arguments", {}))
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError:
                arguments = {"_raw": arguments}
        if not isinstance(arguments, dict):
            arguments = {"_value": arguments}
        result = item.get("result")
        if result is not None and not isinstance(result, str):
            result = json.dumps(result, ensure_ascii=False, default=str)
        calls.append(
            ToolCall(
                call_id=str(item.get("call_id") or item.get("id") or f"call_{i}"),
                name=str(name),
                arguments=arguments,
                result=result,
            )
        )
    return calls


def _parse_message(msg: Any) -> tuple[Message, str | None]:
    """(Message, tool_call_id of a tool-result message). Raises IngestSkipError if unusable."""
    if not isinstance(msg, dict):
        raise IngestSkipError(f"message is not an object ({type(msg).__name__})")
    role_raw = str(msg.get("role", "user")).strip().lower()
    role_raw = _ROLE_ALIASES.get(role_raw, role_raw)
    try:
        role = MessageRole(role_raw)
    except ValueError:
        raise IngestSkipError(f"unknown message role {role_raw!r}") from None

    raw_calls = msg.get("tool_calls")
    if not raw_calls and msg.get("function_call"):
        raw_calls = [msg["function_call"]]
    name = msg.get("name")
    return (
        Message(
            role=role,
            content=_normalise_content(msg.get("content")),
            tool_calls=_parse_tool_calls(raw_calls),
            name=str(name) if name is not None else None,
        ),
        str(msg["tool_call_id"]) if msg.get("tool_call_id") else None,
    )


def _parse_timestamp(value: str | None) -> datetime:
    """Naive UTC datetime from an ISO-8601 string (any offset); now() if absent/invalid."""
    if value:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            logger.debug("Unparseable created_at %r; using the ingestion time", value)
        else:
            if parsed.tzinfo is not None:
                parsed = parsed.astimezone(UTC).replace(tzinfo=None)
            return parsed
    return datetime.utcnow()


# ─── Report ─────────────────────────────────────────────────────────────────


@dataclass
class IngestReport:
    """What happened to the input: nothing is dropped without being counted here."""

    n_lines: int = 0                 # non-empty input lines (file input)
    n_unreadable_lines: int = 0      # not valid JSON / not a RawConversation
    n_parsed: int = 0                # RawConversation objects received
    skipped: Counter = field(default_factory=Counter)   # reason -> count
    n_duplicates_exact: int = 0
    n_duplicates_near: int = 0
    n_duplicates_of_reference: int = 0    # of the duplicates: copies of frozen test items
    dedup_method: str = ""
    redactions: Counter = field(default_factory=Counter)   # PII label -> count
    n_kept: int = 0

    @property
    def n_skipped(self) -> int:
        return sum(self.skipped.values())

    def summary(self) -> dict:
        return {
            "lines": self.n_lines,
            "unreadable_lines": self.n_unreadable_lines,
            "conversations": self.n_parsed,
            "skipped": dict(self.skipped),
            "duplicates_exact": self.n_duplicates_exact,
            "duplicates_near": self.n_duplicates_near,
            "duplicates_of_frozen_test": self.n_duplicates_of_reference,
            "dedup_method": self.dedup_method,
            "redactions": dict(self.redactions),
            "kept": self.n_kept,
        }


# ─── Main ingestor ────────────────────────────────────────────────────────────


class Ingestor:
    """
    Full ingestion pipeline: parse → scrub → classify → dedup → split.

    Usage::

        ingestor = Ingestor()
        result = ingestor.ingest_file(Path("logs.jsonl"))
        # result.train / result.dev / result.test are ready for downstream use
        # ingestor.last_report says what was skipped, deduplicated and redacted
    """

    def __init__(self, *, use_ner: bool = True):
        self.use_ner = use_ner
        self.last_report: IngestReport | None = None

    def _parse_raw(self, raw: RawConversation) -> ConversationRecord:
        """RawConversation -> ConversationRecord (pre-scrub). Raises IngestSkipError if unusable."""
        if not raw.messages:
            raise IngestSkipError("no messages")

        messages: list[Message] = []
        tool_result_ids: list[tuple[int, str]] = []
        for msg in raw.messages:
            parsed, tool_call_id = _parse_message(msg)
            if tool_call_id:
                tool_result_ids.append((len(messages), tool_call_id))
            messages.append(parsed)

        if not any(m.role == MessageRole.USER and m.content for m in messages):
            raise IngestSkipError("no user turn with content")

        # Attach tool results to the calls they answer (OpenAI `tool_call_id` linkage)
        for idx, call_id in tool_result_ids:
            for earlier in messages[:idx]:
                for call in earlier.tool_calls:
                    if call.call_id == call_id and call.result is None:
                        call.result = messages[idx].content

        return ConversationRecord(
            conversation_id=raw.conversation_id,
            messages=messages,
            category=raw.category or "uncategorized",
            weak_signal=WeakSignal(raw.weak_signal or "none"),
            source_system=raw.source_system,
            created_at=_parse_timestamp(raw.created_at),
        )

    def _scrub_record(self, conv: ConversationRecord, report: IngestReport) -> ConversationRecord:
        """PII-scrub message content, message names and tool-call arguments/results."""
        scrubbed_messages = []
        for msg in conv.messages:
            updates: dict[str, Any] = {}
            if msg.content:
                result = scrub(msg.content, use_ner=self.use_ner)
                updates["content"] = result.scrubbed_text
                report.redactions.update(r.label for r in result.redactions)
            if msg.name:
                result = scrub(msg.name, use_ner=False)
                updates["name"] = result.scrubbed_text
                report.redactions.update(r.label for r in result.redactions)
            if msg.tool_calls:
                calls = []
                for call in msg.tool_calls:
                    arguments, found = scrub_structure(call.arguments, use_ner=self.use_ner)
                    report.redactions.update(r.label for r in found)
                    result_text = call.result
                    if result_text:
                        scrubbed = scrub(result_text, use_ner=self.use_ner)
                        report.redactions.update(r.label for r in scrubbed.redactions)
                        result_text = scrubbed.scrubbed_text
                    calls.append(call.model_copy(update={"arguments": arguments,
                                                         "result": result_text}))
                updates["tool_calls"] = calls
            scrubbed_messages.append(msg.model_copy(update=updates) if updates else msg)
        return conv.model_copy(
            update={
                "messages": scrubbed_messages,
                "pii_scrubbed": True,
                "ingested_at": datetime.utcnow(),
            }
        )

    def _enrich_record(self, conv: ConversationRecord) -> ConversationRecord:
        """Add category, difficulty, and structural flags."""
        updates: dict = {}
        if conv.category in ("uncategorized",):
            updates["category"] = _classify_category(conv)
        updates["difficulty"] = _classify_difficulty(conv)
        updates["has_tool_use"] = any(
            m.tool_calls or m.role == MessageRole.TOOL for m in conv.messages
        )
        updates["is_multi_turn"] = len(conv.user_turns) > 1
        return conv.model_copy(update=updates)

    def ingest_records(
        self,
        raw_convs: list[RawConversation],
        *,
        train_ratio: float = 0.60,
        dev_ratio: float = 0.20,
        test_ratio: float = 0.20,
        reference: list[ConversationRecord] | None = None,
        _report: IngestReport | None = None,
    ) -> StratifyResult:
        """
        Full pipeline over a list of RawConversation objects.

        Steps: parse → scrub → enrich → dedup → stratify. Unusable records are skipped and
        counted in `last_report`, never fatal. `reference` (e.g. the frozen test split) makes
        dedup drop conversations that duplicate those items.
        """
        report = _report or IngestReport()
        self.last_report = report
        report.n_parsed = len(raw_convs)
        logger.info("Ingesting %d raw conversations …", len(raw_convs))

        # 1-3. Parse, scrub, enrich — one record at a time so one bad record costs one record
        records: list[ConversationRecord] = []
        seen_ids: set[str] = set()
        for raw in raw_convs:
            try:
                if raw.conversation_id in seen_ids:
                    raise IngestSkipError("duplicate conversation_id")
                record = self._parse_raw(raw)
                record = self._scrub_record(record, report)
                record = self._enrich_record(record)
            except IngestSkipError as exc:
                report.skipped[str(exc)] += 1
                logger.warning("Skipping conversation %r: %s", raw.conversation_id, exc)
                continue
            except Exception as exc:  # noqa: BLE001 - never let one record abort the ingest
                reason = f"invalid record ({type(exc).__name__})"
                report.skipped[reason] += 1
                logger.warning("Skipping conversation %r: %s: %s", raw.conversation_id,
                               type(exc).__name__, exc)
                continue
            seen_ids.add(raw.conversation_id)
            records.append(record)

        # 4. Dedup
        dedup_result: DedupResult = deduplicate(records, reference=reference)
        kept = dedup_result.kept
        report.n_duplicates_exact = dedup_result.n_exact
        report.n_duplicates_near = dedup_result.n_near
        report.n_duplicates_of_reference = dedup_result.n_vs_reference
        report.dedup_method = dedup_result.method
        report.n_kept = len(kept)
        logger.info("After dedup: %d kept, %d removed", dedup_result.n_kept, dedup_result.n_removed)

        # 5. Stratify
        return stratify(
            kept,
            train_ratio=train_ratio,
            dev_ratio=dev_ratio,
            test_ratio=test_ratio,
        )

    def ingest_file(
        self,
        path: Path,
        *,
        train_ratio: float = 0.60,
        dev_ratio: float = 0.20,
        test_ratio: float = 0.20,
        reference: list[ConversationRecord] | None = None,
    ) -> StratifyResult:
        """
        Load a JSON-Lines file of raw conversations and run the full pipeline.

        Lines that are not valid JSON / not a valid RawConversation are skipped and counted.
        """
        report = IngestReport()
        raw_convs: list[RawConversation] = []
        with open(path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                report.n_lines += 1
                try:
                    data = json.loads(line)
                    raw_convs.append(RawConversation(**data))
                except Exception as exc:  # noqa: BLE001
                    report.n_unreadable_lines += 1
                    logger.warning("Line %d parse error: %s — skipping", lineno, exc)

        if not raw_convs:
            logger.warning("No valid conversations found in %s", path)
            self.last_report = report
            return StratifyResult(train=[], dev=[], test=[])

        return self.ingest_records(
            raw_convs,
            train_ratio=train_ratio,
            dev_ratio=dev_ratio,
            test_ratio=test_ratio,
            reference=reference,
            _report=report,
        )

    def ingest_jsonl_string(
        self, content: str, *, reference: list[ConversationRecord] | None = None
    ) -> StratifyResult:
        """Convenience: ingest from a JSONL string (useful for tests)."""
        report = IngestReport()
        raw_convs = []
        for line in (line.strip() for line in content.splitlines()):
            if not line:
                continue
            report.n_lines += 1
            try:
                raw_convs.append(RawConversation(**json.loads(line)))
            except Exception as exc:  # noqa: BLE001
                report.n_unreadable_lines += 1
                logger.warning("Parse error: %s", exc)
        return self.ingest_records(raw_convs, reference=reference, _report=report)
