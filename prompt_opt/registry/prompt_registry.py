"""
Prompt Registry — content-addressed, versioned storage for prompt candidates.

Every prompt version is immutable once written.
Rollback = pointing champion pointer back to a previous version_id.

Storage: SQLite (versions table) + JSON files in blob_store_path for full prompt text.

Schema:
  prompt_versions(version_id, content_hash, semantic_version, system_prompt_text,
                  few_shot_demos_json, optimizer_run_id, eval_scores_json,
                  gate_result_json, approval_record_json, is_champion, deployed_at,
                  created_at)
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from sqlalchemy import Boolean, Column, DateTime, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, sessionmaker

logger = logging.getLogger(__name__)


class _Base(DeclarativeBase):
    pass


class PromptVersionRow(_Base):
    __tablename__ = "prompt_versions"

    version_id = Column(String, primary_key=True)
    content_hash = Column(String, unique=True, index=True, nullable=False)
    semantic_version = Column(String, nullable=False)
    system_prompt_text = Column(Text, nullable=False)
    few_shot_demos_json = Column(Text, nullable=True)
    optimizer_run_id = Column(String, nullable=True)
    eval_scores_json = Column(Text, nullable=True)
    gate_result_json = Column(Text, nullable=True)
    approval_record_json = Column(Text, nullable=True)
    is_champion = Column(Boolean, default=False, nullable=False)
    deployed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


@dataclass
class PromptVersion:
    version_id: str
    content_hash: str
    semantic_version: str
    system_prompt: str
    few_shot_demos: list[dict] | None = None
    optimizer_run_id: str | None = None
    eval_scores: dict = field(default_factory=dict)
    gate_result: dict = field(default_factory=dict)
    approval_record: dict = field(default_factory=dict)
    is_champion: bool = False
    deployed_at: datetime | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "version_id": self.version_id,
            "content_hash": self.content_hash,
            "semantic_version": self.semantic_version,
            "is_champion": self.is_champion,
            "eval_scores": self.eval_scores,
            "created_at": self.created_at.isoformat(),
            "deployed_at": self.deployed_at.isoformat() if self.deployed_at else None,
        }


def _row_to_version(row: PromptVersionRow) -> PromptVersion:
    return PromptVersion(
        version_id=row.version_id,
        content_hash=row.content_hash,
        semantic_version=row.semantic_version,
        system_prompt=row.system_prompt_text,
        few_shot_demos=json.loads(row.few_shot_demos_json) if row.few_shot_demos_json else None,
        optimizer_run_id=row.optimizer_run_id,
        eval_scores=json.loads(row.eval_scores_json or "{}"),
        gate_result=json.loads(row.gate_result_json or "{}"),
        approval_record=json.loads(row.approval_record_json or "{}"),
        is_champion=bool(row.is_champion),
        deployed_at=row.deployed_at,
        created_at=row.created_at or datetime.utcnow(),
    )


class PromptRegistry:
    """
    Content-addressed, append-only prompt version registry.

    Usage::

        registry = PromptRegistry()
        version_id = registry.register(candidate, eval_summary, gate_result)
        registry.promote(version_id)   # sets as new champion
        registry.rollback()            # reverts to previous champion
    """

    def __init__(self, db_url: str = "sqlite:///./data/prompt_opt.db"):
        self._engine = create_engine(db_url, echo=False)
        _Base.metadata.create_all(self._engine)
        self._Session = sessionmaker(self._engine)

    @staticmethod
    def _compute_hash(system_prompt: str, few_shot_demos: list | None) -> str:
        payload = json.dumps(
            {"system": system_prompt, "demos": few_shot_demos or []}, sort_keys=True
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    @staticmethod
    def _next_semantic_version(n: int) -> str:
        major = n // 100
        minor = (n % 100) // 10
        patch = n % 10
        return f"{major}.{minor}.{patch}"

    def register(
        self,
        system_prompt: str,
        few_shot_demos: list[dict] | None = None,
        optimizer_run_id: str | None = None,
        eval_scores: dict | None = None,
        gate_result_dict: dict | None = None,
    ) -> str:
        """
        Register a new prompt version. Returns version_id.

        If a version with the same content_hash already exists, returns
        the existing version_id (idempotent).
        """
        content_hash = self._compute_hash(system_prompt, few_shot_demos)

        with self._Session() as session:
            existing = session.scalars(
                select(PromptVersionRow).where(PromptVersionRow.content_hash == content_hash)
            ).first()
            if existing:
                logger.debug("Prompt already registered: %s", existing.version_id)
                return existing.version_id

            n = session.query(PromptVersionRow).count() + 1
            version_id = f"prompt_v{n}"

            row = PromptVersionRow(
                version_id=version_id,
                content_hash=content_hash,
                semantic_version=self._next_semantic_version(n),
                system_prompt_text=system_prompt,
                few_shot_demos_json=json.dumps(few_shot_demos) if few_shot_demos else None,
                optimizer_run_id=optimizer_run_id,
                eval_scores_json=json.dumps(eval_scores or {}),
                gate_result_json=json.dumps(gate_result_dict or {}),
                is_champion=False,
            )
            session.add(row)
            session.commit()
            logger.info("Registered prompt version %s (hash=%s)", version_id, content_hash)
            return version_id

    def promote(self, version_id: str, approver: str = "system") -> None:
        """
        Promote a registered version to champion.

        Demotes the current champion first.
        """
        with self._Session() as session:
            # Demote current champion
            current_champs = session.scalars(
                select(PromptVersionRow).where(PromptVersionRow.is_champion == True)  # noqa: E712
            ).all()
            for champ in current_champs:
                champ.is_champion = False

            # Promote new champion
            row = session.get(PromptVersionRow, version_id)
            if row is None:
                raise KeyError(f"Version {version_id!r} not found in registry")
            row.is_champion = True
            row.deployed_at = datetime.utcnow()
            row.approval_record_json = json.dumps(
                {"approver": approver, "approved_at": datetime.utcnow().isoformat()}
            )
            session.commit()
            logger.info("Promoted %s to champion (approved by %s)", version_id, approver)

    def rollback(self) -> str | None:
        """
        Rollback to the previous champion.

        Returns the version_id of the new (reverted) champion, or None if no history.
        """
        with self._Session() as session:
            # Get all deployed versions in order
            rows = session.scalars(
                select(PromptVersionRow)
                .where(PromptVersionRow.deployed_at.is_not(None))
                .order_by(PromptVersionRow.deployed_at.desc())
            ).all()

            if len(rows) < 2:
                logger.warning("No previous version to roll back to")
                return None

            # rows[0] is current champion; rows[1] is previous
            current = rows[0]
            previous = rows[1]

            current.is_champion = False
            previous.is_champion = True
            previous.deployed_at = datetime.utcnow()
            session.commit()
            logger.info(
                "Rolled back: %s → %s (reverted champion)",
                current.version_id,
                previous.version_id,
            )
            return previous.version_id

    def get_champion(self) -> PromptVersion | None:
        with self._Session() as session:
            row = session.scalars(
                select(PromptVersionRow).where(PromptVersionRow.is_champion == True)  # noqa: E712
            ).first()
            return _row_to_version(row) if row else None

    def get(self, version_id: str) -> PromptVersion | None:
        with self._Session() as session:
            row = session.get(PromptVersionRow, version_id)
            return _row_to_version(row) if row else None

    def list_versions(self, limit: int = 20) -> list[PromptVersion]:
        with self._Session() as session:
            rows = session.scalars(
                select(PromptVersionRow)
                .order_by(PromptVersionRow.created_at.desc())
                .limit(limit)
            ).all()
            return [_row_to_version(r) for r in rows]

    def history(self) -> list[dict]:
        """Return a human-readable version history."""
        versions = self.list_versions(limit=50)
        return [v.to_dict() for v in reversed(versions)]
