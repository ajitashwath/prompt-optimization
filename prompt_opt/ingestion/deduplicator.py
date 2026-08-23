"""
Near-duplicate detection for conversation records.

Uses embedding-based cosine similarity with FAISS ANN search to cluster
conversations, keeping one representative per cluster.

Strategy:
  1. Embed each conversation's concatenated user turns.
  2. Build a FAISS flat-L2 index (exact search; use IVF for >100k records).
  3. For each new conversation, query index; mark as duplicate if nearest
     neighbor has cosine similarity >= threshold.
  4. Return de-duplicated list.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from prompt_opt.config import settings
from prompt_opt.ingestion.schema import ConversationRecord

logger = logging.getLogger(__name__)


@dataclass
class DedupResult:
    kept: list[ConversationRecord]
    duplicates: list[tuple[ConversationRecord, str]]  # (dup, representative_id)

    @property
    def n_kept(self) -> int:
        return len(self.kept)

    @property
    def n_removed(self) -> int:
        return len(self.duplicates)


def _embed_conversation(conv: ConversationRecord, model: str) -> np.ndarray:
    """
    Embed a conversation by concatenating all user turn content.

    Returns a 1-D float32 numpy array.
    """
    text = " ".join(
        m.content or "" for m in conv.user_turns
    ).strip() or f"[empty:{conv.conversation_id}]"

    try:
        from prompt_opt.llm import get_litellm

        resp = get_litellm().embedding(model=model, input=[text])
        vec = np.array(resp.data[0]["embedding"], dtype=np.float32)
    except Exception:
        # Fallback: random vector (for testing without API keys)
        logger.warning(
            "Embedding call failed for %s — using random vector (test mode)",
            conv.conversation_id,
        )
        vec = np.random.default_rng(hash(text) % 2**32).random(1536).astype(np.float32)

    # Normalise to unit length so cosine sim = dot product
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def deduplicate(
    conversations: list[ConversationRecord],
    *,
    threshold: float | None = None,
    embedding_model: str | None = None,
) -> DedupResult:
    """
    Remove near-duplicate conversations.

    Args:
        conversations:   Input list (can be in any order).
        threshold:       Cosine similarity threshold (0–1). Higher = stricter.
        embedding_model: litellm model string for embeddings.

    Returns:
        DedupResult with kept and duplicates lists.
    """
    threshold = threshold or settings.dedup_cosine_threshold
    embedding_model = embedding_model or settings.embedding_model

    if not conversations:
        return DedupResult(kept=[], duplicates=[])

    try:
        import faiss  # type: ignore[import]
    except ImportError:
        logger.warning("faiss not installed — skipping deduplication (all records kept)")
        return DedupResult(kept=conversations, duplicates=[])

    logger.info("Computing embeddings for %d conversations …", len(conversations))
    vecs: list[np.ndarray] = []
    for conv in conversations:
        vecs.append(_embed_conversation(conv, embedding_model))

    matrix = np.stack(vecs)  # (N, D) float32

    # Build flat inner-product index (cosine sim since vecs are normalised)
    dim = matrix.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(matrix)

    kept: list[ConversationRecord] = []
    duplicates: list[tuple[ConversationRecord, str]] = []
    representative_ids: list[str] = []  # ids of already-accepted conversations

    for i, conv in enumerate(conversations):
        if i == 0:
            kept.append(conv)
            representative_ids.append(conv.conversation_id)
            continue

        query = matrix[i : i + 1]
        # Search among all already-accepted vectors only
        # Re-query the full index and check against representative indices
        k = min(i + 1, index.ntotal)
        similarities, indices = index.search(query, k)

        # Find best similarity to any already-accepted representative
        best_sim = -1.0
        best_rep: str | None = None
        for sim, idx in zip(similarities[0], indices[0]):
            if idx < i and conversations[idx].conversation_id in set(representative_ids):
                if sim > best_sim:
                    best_sim = sim
                    best_rep = conversations[idx].conversation_id

        if best_sim >= threshold and best_rep is not None:
            logger.debug(
                "Duplicate detected: %s ≈ %s (sim=%.4f)",
                conv.conversation_id,
                best_rep,
                best_sim,
            )
            conv.is_duplicate = True
            duplicates.append((conv, best_rep))
        else:
            kept.append(conv)
            representative_ids.append(conv.conversation_id)

    logger.info(
        "Deduplication complete: %d kept, %d removed (threshold=%.2f)",
        len(kept),
        len(duplicates),
        threshold,
    )
    return DedupResult(kept=kept, duplicates=duplicates)
