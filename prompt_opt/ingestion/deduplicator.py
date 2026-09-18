"""
Near-duplicate detection for conversation records.

Strategy (conversations are processed in order; the first occurrence is kept):
  1. Exact duplicates: identical user text after case/whitespace normalisation. Cheap,
     deterministic, and independent of any embedding service.
  2. Near duplicates: cosine similarity of the user-turn text against every conversation
     kept so far, using one nearest-neighbour search per conversation over the kept
     set only (FAISS if installed, else numpy).
       * Embeddings (litellm, batched) when available.
       * If embeddings are unavailable, a deterministic lexical fallback: hashed character
         n-gram vectors (texts of >= 40 characters only). It catches near-identical text
         (spacing, punctuation, small edits), not paraphrases. A failed embedding service
         never silently disables deduplication.
  3. Optional `reference` records (e.g. the frozen test split) are seeded into the index as
     "already accepted": a new conversation that is a near-duplicate of one of them is dropped,
     so paraphrases of test items cannot leak into train/dev. Reference records are never
     returned in `kept`.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import numpy as np

from prompt_opt.config import settings
from prompt_opt.ingestion.schema import ConversationRecord

logger = logging.getLogger(__name__)

_EMBED_BATCH = 64
_EMBED_MAX_CHARS = 8000        # keep each text within typical embedding-model input limits
_LEXICAL_FEATURES = 4096
# Character n-gram similarity is unreliable on short strings ("question number 1" vs "... 2"
# score 0.97 yet are different prompts), so the offline fallback only compares texts at least
# this long; shorter ones are deduplicated only when identical.
_LEXICAL_MIN_CHARS = 40


@dataclass
class DedupResult:
    kept: list[ConversationRecord]
    duplicates: list[tuple[ConversationRecord, str]]  # (dup, representative_id)
    method: str = "embedding"    # "embedding" | "lexical" (near-duplicate stage)
    n_exact: int = 0             # duplicates found by exact normalised-text match
    n_near: int = 0              # duplicates found by similarity
    n_vs_reference: int = 0      # of the above, duplicates of a reference (e.g. frozen test) item

    @property
    def n_kept(self) -> int:
        return len(self.kept)

    @property
    def n_removed(self) -> int:
        return len(self.duplicates)


# ─── Text / vectors ──────────────────────────────────────────────────────────


def _dedup_text(conv: ConversationRecord) -> str:
    """The text compared for duplication: the concatenated user turns."""
    text = " ".join(m.content or "" for m in conv.user_turns).strip()
    return text or f"[empty:{conv.conversation_id}]"


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


def _embed_batch(batch: list[str], model: str) -> list[list[float]]:
    from prompt_opt.llm import get_litellm

    resp = get_litellm().embedding(model=model, input=[t[:_EMBED_MAX_CHARS] for t in batch])
    items = sorted(resp.data, key=lambda item: item["index"] if "index" in item else 0)
    return [item["embedding"] for item in items]


def _embed_texts(texts: list[str], model: str) -> np.ndarray | None:
    """Unit-length embeddings for *texts*, or None if the embedding service is unusable."""
    rows: list[list[float]] = []
    try:
        for i in range(0, len(texts), _EMBED_BATCH):
            rows.extend(_embed_batch(texts[i : i + _EMBED_BATCH], model))
        matrix = np.asarray(rows, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(texts):
            raise ValueError(f"unexpected embedding shape {matrix.shape} for {len(texts)} texts")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Embeddings unavailable (%s: %s)", type(exc).__name__, exc)
        return None
    return _unit_rows(matrix)


def _lexical_vectors(texts: list[str]) -> np.ndarray:
    """Deterministic hashed character n-gram vectors (unit length): the offline fallback."""
    from sklearn.feature_extraction.text import HashingVectorizer

    vectorizer = HashingVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        n_features=_LEXICAL_FEATURES,
        alternate_sign=False,
        norm="l2",
        lowercase=True,
    )
    return vectorizer.transform(texts).toarray().astype(np.float32)


class _Index:
    """Exact inner-product nearest-neighbour index that grows one vector at a time."""

    def __init__(self, dim: int):
        self._n = 0
        try:
            import faiss  # type: ignore[import]

            self._faiss = faiss.IndexFlatIP(dim)
            self._buf = None
        except ImportError:
            self._faiss = None
            self._buf = np.empty((256, dim), dtype=np.float32)

    def add(self, vec: np.ndarray) -> None:
        vec = vec.reshape(1, -1).astype(np.float32)
        if self._faiss is not None:
            self._faiss.add(vec)
        else:
            if self._n == len(self._buf):
                self._buf = np.concatenate([self._buf, np.empty_like(self._buf)])
            self._buf[self._n] = vec[0]
        self._n += 1

    def nearest(self, vec: np.ndarray) -> tuple[float, int]:
        """(similarity, position) of the closest stored vector, or (-1.0, -1) if empty."""
        if self._n == 0:
            return -1.0, -1
        query = vec.reshape(1, -1).astype(np.float32)
        if self._faiss is not None:
            sims, ids = self._faiss.search(query, 1)
            return float(sims[0][0]), int(ids[0][0])
        sims = self._buf[: self._n] @ query[0]
        best = int(np.argmax(sims))
        return float(sims[best]), best


# ─── Public API ──────────────────────────────────────────────────────────────


def deduplicate(
    conversations: list[ConversationRecord],
    *,
    threshold: float | None = None,
    embedding_model: str | None = None,
    lexical_threshold: float = 0.90,
    reference: list[ConversationRecord] | None = None,
) -> DedupResult:
    """
    Remove exact and near-duplicate conversations (first occurrence wins).

    Args:
        conversations:     Input list, in priority order.
        threshold:         Cosine similarity threshold for embeddings (0-1; default
                           DEDUP_COSINE_THRESHOLD). Higher = stricter.
        embedding_model:   litellm model string for embeddings.
        lexical_threshold: Threshold for the offline fallback (character n-gram cosine is on
                           a different scale from embedding cosine).
        reference:         Records that are already accepted elsewhere (e.g. the frozen test
                           split). Conversations duplicating them are removed; they are never
                           returned.

    Returns:
        DedupResult with kept and duplicates lists.
    """
    threshold = threshold or settings.dedup_cosine_threshold
    embedding_model = embedding_model or settings.embedding_model
    reference = reference or []

    if not conversations:
        return DedupResult(kept=[], duplicates=[])

    duplicates: list[tuple[ConversationRecord, str]] = []
    n_exact = n_near = n_vs_ref = 0
    reference_ids = {r.conversation_id for r in reference}

    # ── 1. Exact duplicates (normalised text) ────────────────────────────────
    seen: dict[str, str] = {}
    for ref in reference:
        seen.setdefault(_normalise(_dedup_text(ref)), ref.conversation_id)
    survivors: list[ConversationRecord] = []
    for conv in conversations:
        key = _normalise(_dedup_text(conv))
        if key in seen:
            conv.is_duplicate = True
            duplicates.append((conv, seen[key]))
            n_exact += 1
            n_vs_ref += seen[key] in reference_ids
        else:
            seen[key] = conv.conversation_id
            survivors.append(conv)

    # ── 2. Near duplicates ───────────────────────────────────────────────────
    texts = [_dedup_text(c) for c in [*reference, *survivors]]
    method = "embedding"
    vectors = _embed_texts(texts, embedding_model) if texts else None
    active_threshold = threshold
    if vectors is None:
        method = "lexical"
        active_threshold = lexical_threshold
        logger.warning(
            "Near-duplicate detection is using the offline lexical fallback (threshold %.2f); "
            "configure an embedding model/key for semantic deduplication.",
            active_threshold,
        )
        vectors = _lexical_vectors(texts)

    def comparable(text: str) -> bool:
        return method == "embedding" or len(text) >= _LEXICAL_MIN_CHARS

    kept: list[ConversationRecord] = []
    if len(vectors):
        index = _Index(vectors.shape[1])
        representative_ids: list[str] = []
        for i, ref in enumerate(reference):
            if comparable(texts[i]):
                index.add(vectors[i])
                representative_ids.append(ref.conversation_id)

        for offset, conv in enumerate(survivors):
            text_pos = len(reference) + offset
            if not comparable(texts[text_pos]):
                kept.append(conv)        # too short to compare lexically; exact matches are gone
                continue
            vec = vectors[text_pos]
            sim, pos = index.nearest(vec)
            if pos >= 0 and sim >= active_threshold:
                rep = representative_ids[pos]
                logger.debug("Duplicate: %s ~ %s (sim=%.4f)", conv.conversation_id, rep, sim)
                conv.is_duplicate = True
                duplicates.append((conv, rep))
                n_near += 1
                n_vs_ref += rep in reference_ids
            else:
                index.add(vec)
                representative_ids.append(conv.conversation_id)
                kept.append(conv)

    logger.info(
        "Deduplication complete: %d kept, %d removed (%d exact, %d near via %s, %d of them "
        "duplicates of reference items; threshold=%.2f)",
        len(kept), len(duplicates), n_exact, n_near, method, n_vs_ref, active_threshold,
    )
    return DedupResult(
        kept=kept,
        duplicates=duplicates,
        method=method,
        n_exact=n_exact,
        n_near=n_near,
        n_vs_reference=n_vs_ref,
    )
