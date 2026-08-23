"""
SQLite-backed cache for (prompt_hash, example_id, judge_version, model_version) → score.

Prevents re-paying for unchanged evaluations across optimizer iterations.
Cache keys are deterministic so the same evaluation is never duplicated.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, sessionmaker

logger = logging.getLogger(__name__)


class _Base(DeclarativeBase):
    pass


class CacheRow(_Base):
    __tablename__ = "eval_cache"

    cache_key = Column(String, primary_key=True)
    prompt_hash = Column(String, index=True)
    example_id = Column(String, index=True)
    judge_version = Column(String)
    model_version = Column(String)
    score = Column(Float, nullable=False)
    score_breakdown_json = Column(Text, nullable=True)
    rationale = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


@dataclass
class CachedScore:
    score: float
    score_breakdown: dict
    rationale: str


class EvalCache:
    """
    Persistent cache for eval scores.

    Cache key = SHA256(prompt_hash + "|" + example_id + "|" + judge_version + "|" + model_version)
    """

    def __init__(self, db_url: str = "sqlite:///./data/prompt_opt.db"):
        self._engine = create_engine(db_url, echo=False)
        _Base.metadata.create_all(self._engine)
        self._Session = sessionmaker(self._engine)
        self._hits = 0
        self._misses = 0

    @staticmethod
    def make_key(prompt_hash: str, example_id: str, judge_version: str, model_version: str) -> str:
        raw = f"{prompt_hash}|{example_id}|{judge_version}|{model_version}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(
        self,
        prompt_hash: str,
        example_id: str,
        judge_version: str,
        model_version: str,
    ) -> CachedScore | None:
        key = self.make_key(prompt_hash, example_id, judge_version, model_version)
        with self._Session() as session:
            row = session.get(CacheRow, key)
            if row is None:
                self._misses += 1
                return None
            self._hits += 1
            return CachedScore(
                score=row.score,
                score_breakdown=json.loads(row.score_breakdown_json or "{}"),
                rationale=row.rationale or "",
            )

    def put(
        self,
        prompt_hash: str,
        example_id: str,
        judge_version: str,
        model_version: str,
        score: float,
        score_breakdown: dict | None = None,
        rationale: str = "",
    ) -> None:
        key = self.make_key(prompt_hash, example_id, judge_version, model_version)
        with self._Session() as session:
            row = CacheRow(
                cache_key=key,
                prompt_hash=prompt_hash,
                example_id=example_id,
                judge_version=judge_version,
                model_version=model_version,
                score=score,
                score_breakdown_json=json.dumps(score_breakdown or {}),
                rationale=rationale,
            )
            session.merge(row)
            session.commit()

    @property
    def stats(self) -> dict:
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / max(1, self._hits + self._misses),
        }

    def invalidate_prompt(self, prompt_hash: str) -> int:
        """Remove all cached scores for a specific prompt (e.g. after prompt fix)."""
        with self._Session() as session:
            rows = session.scalars(
                select(CacheRow).where(CacheRow.prompt_hash == prompt_hash)
            ).all()
            n = len(rows)
            for row in rows:
                session.delete(row)
            session.commit()
        logger.info("Invalidated %d cache entries for prompt_hash=%s", n, prompt_hash)
        return n
