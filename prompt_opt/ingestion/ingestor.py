"""
Conversation ingestion orchestrator.

Reads raw JSON-Lines conversation logs, applies PII scrubbing, deduplication,
category classification (heuristic), and stratified splitting.

Entry-point for the `prompt-opt ingest` CLI command.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from prompt_opt.ingestion.deduplicator import DedupResult, deduplicate
from prompt_opt.ingestion.pii_scrubber import scrub
from prompt_opt.ingestion.schema import (
    ConversationRecord,
    DifficultyBand,
    Message,
    MessageRole,
    RawConversation,
    WeakSignal,
)
from prompt_opt.ingestion.stratifier import StratifyResult, stratify

logger = logging.getLogger(__name__)


# ─── Heuristic category classifier ──────────────────────────────────────────


_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "safety": ["harm", "illegal", "violence", "weapon", "drug", "self-harm", "hate"],
    "coding": ["code", "python", "javascript", "function", "bug", "error", "exception"],
    "math": ["calculate", "equation", "solve", "integral", "derivative", "probability"],
    "creative": ["write a story", "poem", "creative", "fiction", "imagine"],
    "factual_qa": ["what is", "who is", "when did", "where is", "why does"],
    "tool_use": ["search", "browse", "fetch", "call", "execute", "run"],
    "multi_turn_correction": ["that's wrong", "incorrect", "you made a mistake", "try again"],
}


def _classify_category(conv: ConversationRecord) -> str:
    """Simple keyword-based category classifier. Replace with a proper model in production."""
    text = " ".join(
        m.content or "" for m in conv.messages if m.role == MessageRole.USER
    ).lower()

    for category, keywords in _CATEGORY_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return category
    return "general"


def _classify_difficulty(conv: ConversationRecord) -> DifficultyBand:
    """Estimate difficulty from structural signals."""
    n_turns = len(conv.user_turns)
    has_correction = any(
        m.content and ("that's wrong" in m.content.lower() or "incorrect" in m.content.lower())
        for m in conv.messages
        if m.role == MessageRole.USER
    )
    if has_correction or n_turns > 5:
        return DifficultyBand.HARD
    if n_turns > 2:
        return DifficultyBand.MEDIUM
    return DifficultyBand.EASY


# ─── Main ingestor ────────────────────────────────────────────────────────────


class Ingestor:
    """
    Full ingestion pipeline: parse → scrub → classify → dedup → split.

    Usage::

        ingestor = Ingestor()
        result = ingestor.ingest_file(Path("logs.jsonl"))
        # result.train / result.dev / result.test are ready for downstream use
    """

    def __init__(self, *, use_ner: bool = True):
        self.use_ner = use_ner

    def _parse_raw(self, raw: RawConversation) -> ConversationRecord:
        """Convert a RawConversation into a ConversationRecord (pre-scrub)."""
        messages = []
        for msg in raw.messages:
            try:
                messages.append(
                    Message(
                        role=MessageRole(msg.get("role", "user")),
                        content=msg.get("content"),
                        name=msg.get("name"),
                    )
                )
            except Exception:  # noqa: BLE001
                logger.debug("Skipping malformed message: %s", msg)

        created_at = datetime.utcnow()
        if raw.created_at:
            try:
                created_at = datetime.fromisoformat(raw.created_at.rstrip("Z"))
            except ValueError:
                pass

        return ConversationRecord(
            conversation_id=raw.conversation_id,
            messages=messages,
            category=raw.category or "uncategorized",
            weak_signal=WeakSignal(raw.weak_signal or "none"),
            source_system=raw.source_system,
            created_at=created_at,
        )

    def _scrub_record(self, conv: ConversationRecord) -> ConversationRecord:
        """Apply PII scrubbing to all message content in place."""
        scrubbed_messages = []
        for msg in conv.messages:
            if msg.content:
                result = scrub(msg.content, use_ner=self.use_ner)
                scrubbed_messages.append(msg.model_copy(update={"content": result.scrubbed_text}))
            else:
                scrubbed_messages.append(msg)
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
        updates["has_tool_use"] = any(m.tool_calls for m in conv.messages)
        updates["is_multi_turn"] = len(conv.user_turns) > 1
        return conv.model_copy(update=updates)

    def ingest_records(
        self,
        raw_convs: list[RawConversation],
        *,
        train_ratio: float = 0.60,
        dev_ratio: float = 0.20,
        test_ratio: float = 0.20,
    ) -> StratifyResult:
        """
        Full pipeline over a list of RawConversation objects.

        Steps: parse → scrub → enrich → dedup → stratify.
        """
        logger.info("Ingesting %d raw conversations …", len(raw_convs))

        # 1. Parse
        records = [self._parse_raw(r) for r in raw_convs]

        # 2. PII scrub
        records = [self._scrub_record(r) for r in records]

        # 3. Enrich (category + difficulty)
        records = [self._enrich_record(r) for r in records]

        # 4. Dedup
        dedup_result: DedupResult = deduplicate(records)
        kept = dedup_result.kept
        logger.info("After dedup: %d kept, %d removed", dedup_result.n_kept, dedup_result.n_removed)

        # 5. Stratify
        split_result: StratifyResult = stratify(
            kept,
            train_ratio=train_ratio,
            dev_ratio=dev_ratio,
            test_ratio=test_ratio,
        )

        return split_result

    def ingest_file(
        self,
        path: Path,
        *,
        train_ratio: float = 0.60,
        dev_ratio: float = 0.20,
        test_ratio: float = 0.20,
    ) -> StratifyResult:
        """
        Load a JSON-Lines file of raw conversations and run the full pipeline.

        Each line must be a valid RawConversation JSON object.
        """
        raw_convs: list[RawConversation] = []
        with open(path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    raw_convs.append(RawConversation(**data))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Line %d parse error: %s — skipping", lineno, exc)

        if not raw_convs:
            logger.warning("No valid conversations found in %s", path)
            return StratifyResult(train=[], dev=[], test=[])

        return self.ingest_records(
            raw_convs,
            train_ratio=train_ratio,
            dev_ratio=dev_ratio,
            test_ratio=test_ratio,
        )

    def ingest_jsonl_string(self, content: str) -> StratifyResult:
        """Convenience: ingest from a JSONL string (useful for tests)."""
        tmp_lines = [line.strip() for line in content.splitlines() if line.strip()]
        raw_convs = []
        for line in tmp_lines:
            try:
                raw_convs.append(RawConversation(**json.loads(line)))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Parse error: %s", exc)
        return self.ingest_records(raw_convs)
