"""
Core data schemas for conversation records flowing through the pipeline.

All models use Pydantic v2 for validation and serialization.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


# ─── Enumerations ─────────────────────────────────────────────────────────────


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class WeakSignal(str, Enum):
    POSITIVE = "positive"   # thumbs-up, task completion, etc.
    NEGATIVE = "negative"   # thumbs-down, retry, escalation, etc.
    NONE = "none"           # no implicit signal


class DataSplit(str, Enum):
    TRAIN = "train"
    DEV = "dev"
    TEST = "test"
    UNLABELED = "unlabeled"  # not yet curated


class DifficultyBand(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    UNKNOWN = "unknown"


# ─── Message ─────────────────────────────────────────────────────────────────


class ToolCall(BaseModel):
    """Represents a single tool call made by the assistant."""
    call_id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: str | None = None


class Message(BaseModel):
    """A single turn in a conversation."""
    role: MessageRole
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    name: str | None = None  # for tool messages

    @field_validator("content", mode="before")
    @classmethod
    def _coerce_none(cls, v: Any) -> str | None:
        if v == "":
            return None
        return v


# ─── Conversation Record ──────────────────────────────────────────────────────


class ConversationRecord(BaseModel):
    """
    The canonical unit of data flowing through the pipeline.

    Produced by the ingestor after PII scrubbing, dedup, and stratification.
    """

    # Identity
    conversation_id: str = Field(description="Unique identifier for this conversation")
    raw_transcript_ref: str | None = Field(
        None, description="Pointer to blob store for the original (unredacted) transcript"
    )

    # Content
    messages: list[Message] = Field(min_length=1)

    # Stratification / labeling
    category: str = Field(default="uncategorized")
    weak_signal: WeakSignal = Field(default=WeakSignal.NONE)
    difficulty: DifficultyBand = Field(default=DifficultyBand.UNKNOWN)
    has_tool_use: bool = False
    is_multi_turn: bool = False

    # Split assignment
    split: DataSplit = Field(default=DataSplit.UNLABELED)

    # Metadata
    pii_scrubbed: bool = False
    is_duplicate: bool = False
    curation_version: str = "0.1.0"
    source_system: str = "unknown"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    ingested_at: datetime | None = None

    @property
    def user_turns(self) -> list[Message]:
        return [m for m in self.messages if m.role == MessageRole.USER]

    @property
    def assistant_turns(self) -> list[Message]:
        return [m for m in self.messages if m.role == MessageRole.ASSISTANT]

    @property
    def system_prompt(self) -> str | None:
        for m in self.messages:
            if m.role == MessageRole.SYSTEM:
                return m.content
        return None

    def without_system_prompt(self) -> list[Message]:
        """Return messages with system message stripped (for prompt-injection evaluation)."""
        return [m for m in self.messages if m.role != MessageRole.SYSTEM]


# ─── Raw Ingestion Format ─────────────────────────────────────────────────────


class RawConversation(BaseModel):
    """
    Expected input format for raw production logs.

    Adapters should transform from any source format into this model,
    which the ingestor then converts into ConversationRecord.
    """

    conversation_id: str
    messages: list[dict[str, Any]]  # raw dicts before strict validation
    category: str | None = None
    weak_signal: Literal["positive", "negative", "none"] | None = None
    source_system: str = "unknown"
    created_at: str | None = None  # ISO-8601 string; parsed later
    metadata: dict[str, Any] = Field(default_factory=dict)
