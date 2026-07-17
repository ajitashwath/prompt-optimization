"""
Gold-labeled dataset store.

Manages the human-labeled seed set used for judge calibration.
Uses SQLite via SQLAlchemy (sync) for simplicity; can migrate to Postgres later.

Schema:
  gold_labels(id, conversation_id, response_a, response_b, preferred,
              rationale, labeler_id, rubric_version, dimension_scores_json,
              created_at, adjudicated)
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    String,
    Text,
    create_engine,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

logger = logging.getLogger(__name__)


# ─── ORM Model ───────────────────────────────────────────────────────────────


class Base(DeclarativeBase):
    pass


class GoldLabelRow(Base):
    __tablename__ = "gold_labels"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id = Column(String, index=True, nullable=False)
    response_a = Column(Text, nullable=False)
    response_b = Column(Text, nullable=False)
    preferred = Column(String, nullable=True)  # "A" | "B" | "tie" | None (not yet labeled)
    rationale = Column(Text, nullable=True)
    labeler_id = Column(String, nullable=True)
    rubric_version = Column(String, nullable=False)
    dimension_scores_json = Column(Text, nullable=True)  # JSON dict
    created_at = Column(DateTime, default=datetime.utcnow)
    adjudicated = Column(Boolean, default=False)


# ─── Data class returned to callers ─────────────────────────────────────────


@dataclass
class GoldLabel:
    id: str
    conversation_id: str
    response_a: str
    response_b: str
    preferred: Literal["A", "B", "tie"] | None  # None = unlabeled
    rationale: str | None
    labeler_id: str | None
    rubric_version: str
    dimension_scores: dict[str, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    adjudicated: bool = False


def _row_to_label(row: GoldLabelRow) -> GoldLabel:
    return GoldLabel(
        id=row.id,
        conversation_id=row.conversation_id,
        response_a=row.response_a,
        response_b=row.response_b,
        preferred=row.preferred,
        rationale=row.rationale,
        labeler_id=row.labeler_id,
        rubric_version=row.rubric_version,
        dimension_scores=json.loads(row.dimension_scores_json or "{}"),
        created_at=row.created_at or datetime.utcnow(),
        adjudicated=row.adjudicated or False,
    )


# ─── Store ───────────────────────────────────────────────────────────────────


class GoldStore:
    """
    CRUD interface for the human-labeled gold set.

    Thread-safe via SQLAlchemy session-per-call pattern.
    """

    def __init__(self, db_url: str = "sqlite:///./data/prompt_opt.db"):
        self._engine = create_engine(db_url, echo=False)
        Base.metadata.create_all(self._engine)
        self._Session = sessionmaker(self._engine)

    def add_item(
        self,
        conversation_id: str,
        response_a: str,
        response_b: str,
        rubric_version: str,
    ) -> str:
        """Add an unlabeled item to the annotation queue. Returns item id."""
        with self._Session() as session:
            row = GoldLabelRow(
                id=str(uuid.uuid4()),
                conversation_id=conversation_id,
                response_a=response_a,
                response_b=response_b,
                rubric_version=rubric_version,
            )
            session.add(row)
            session.commit()
            logger.debug("Added gold item %s for conversation %s", row.id, conversation_id)
            return row.id

    def submit_label(
        self,
        item_id: str,
        preferred: Literal["A", "B", "tie"],
        rationale: str,
        labeler_id: str,
        dimension_scores: dict[str, float] | None = None,
    ) -> None:
        """Record a human's annotation for a queue item."""
        with self._Session() as session:
            row = session.get(GoldLabelRow, item_id)
            if row is None:
                raise KeyError(f"Gold item {item_id!r} not found")
            row.preferred = preferred
            row.rationale = rationale
            row.labeler_id = labeler_id
            row.dimension_scores_json = json.dumps(dimension_scores or {})
            session.commit()

    def get_unlabeled(self, limit: int = 10) -> list[GoldLabel]:
        """Return up to `limit` items that haven't been labeled yet."""
        with self._Session() as session:
            rows = session.scalars(
                select(GoldLabelRow)
                .where(GoldLabelRow.preferred.is_(None))
                .limit(limit)
            ).all()
            return [_row_to_label(r) for r in rows]

    def get_labeled(self, *, rubric_version: str | None = None) -> list[GoldLabel]:
        """Return all labeled items, optionally filtered by rubric version."""
        with self._Session() as session:
            q = select(GoldLabelRow).where(GoldLabelRow.preferred.is_not(None))
            if rubric_version:
                q = q.where(GoldLabelRow.rubric_version == rubric_version)
            rows = session.scalars(q).all()
            return [_row_to_label(r) for r in rows]

    def count_labeled(self) -> int:
        with self._Session() as session:
            return session.query(GoldLabelRow).filter(GoldLabelRow.preferred.is_not(None)).count()

    def count_total(self) -> int:
        with self._Session() as session:
            return session.query(GoldLabelRow).count()
