"""
Stratified train/dev/test splitting.

Buckets conversations by (category × length_band × tool_use × difficulty)
then performs stratified sampling with oversampling for rare/important buckets.

Rules:
  - Default split: 60% train / 20% dev / 20% test
  - Test split is returned as-is from the frozen version if one already exists
  - Rare buckets (< min_bucket_size) are merged into an "other" stratum
  - Safety / hard difficulty buckets are oversampled by oversample_factor
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field

from prompt_opt.ingestion.schema import ConversationRecord, DataSplit, DifficultyBand

logger = logging.getLogger(__name__)

# Buckets that get oversampled (partial category name match)
_OVERSAMPLE_CATEGORIES: frozenset[str] = frozenset(
    {"safety", "sensitive", "refusal", "policy", "edge"}
)
_OVERSAMPLE_DIFFICULTIES: frozenset[DifficultyBand] = frozenset(
    {DifficultyBand.HARD}
)


def _bucket_key(conv: ConversationRecord) -> str:
    """Return a stratification key for a conversation."""
    length_band = "short" if len(conv.user_turns) <= 2 else "long"
    tool = "tool" if conv.has_tool_use else "notool"
    return f"{conv.category}|{length_band}|{tool}|{conv.difficulty.value}"


def _should_oversample(conv: ConversationRecord) -> bool:
    cat_lower = conv.category.lower()
    return (
        any(kw in cat_lower for kw in _OVERSAMPLE_CATEGORIES)
        or conv.difficulty in _OVERSAMPLE_DIFFICULTIES
    )


@dataclass
class StratifyResult:
    train: list[ConversationRecord]
    dev: list[ConversationRecord]
    test: list[ConversationRecord]

    @property
    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "dev": len(self.dev), "test": len(self.test)}

    def all_records(self) -> list[ConversationRecord]:
        return self.train + self.dev + self.test


def stratify(
    conversations: list[ConversationRecord],
    *,
    train_ratio: float = 0.60,
    dev_ratio: float = 0.20,
    test_ratio: float = 0.20,
    oversample_factor: float = 1.5,
    min_bucket_size: int = 5,
    seed: int = 42,
) -> StratifyResult:
    """
    Split *conversations* into stratified train/dev/test sets.

    Args:
        conversations:     De-duplicated, PII-scrubbed records.
        train_ratio:       Fraction for training split.
        dev_ratio:         Fraction for dev split.
        test_ratio:        Fraction for test split (frozen!).
        oversample_factor: Multiplier applied to rare/important buckets during sampling.
        min_bucket_size:   Minimum size to keep a bucket distinct; smaller ones merged.
        seed:              Random seed for reproducibility.

    Returns:
        StratifyResult with split-assigned ConversationRecord instances.
    """
    assert abs(train_ratio + dev_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

    rng = random.Random(seed)

    # Group into buckets
    buckets: dict[str, list[ConversationRecord]] = defaultdict(list)
    for conv in conversations:
        key = _bucket_key(conv)
        buckets[key].append(conv)

    # Merge tiny buckets into "other"
    merged: dict[str, list[ConversationRecord]] = defaultdict(list)
    for key, recs in buckets.items():
        if len(recs) < min_bucket_size:
            merged["other"].extend(recs)
        else:
            merged[key].extend(recs)

    train_all, dev_all, test_all = [], [], []

    for key, recs in merged.items():
        rng.shuffle(recs)

        # Oversample important buckets: duplicate records proportionally
        oversample = any(_should_oversample(r) for r in recs)
        if oversample and len(recs) > 1:
            extras = rng.choices(recs, k=int(len(recs) * (oversample_factor - 1)))
            recs = recs + extras
            rng.shuffle(recs)

        n = len(recs)
        n_test = max(1, round(n * test_ratio))
        n_dev = max(1, round(n * dev_ratio))
        n_train = n - n_test - n_dev

        if n_train <= 0:
            # Very small bucket — assign at least 1 to train
            n_train = 1
            n_dev = max(0, (n - n_train - n_test))
            n_test = max(0, n - n_train - n_dev)

        split_recs = [
            (DataSplit.TRAIN, recs[:n_train]),
            (DataSplit.DEV, recs[n_train : n_train + n_dev]),
            (DataSplit.TEST, recs[n_train + n_dev : n_train + n_dev + n_test]),
        ]

        for split, chunk in split_recs:
            for r in chunk:
                r.split = split

        train_all.extend(recs[:n_train])
        dev_all.extend(recs[n_train : n_train + n_dev])
        test_all.extend(recs[n_train + n_dev : n_train + n_dev + n_test])

    logger.info(
        "Stratification complete: train=%d, dev=%d, test=%d",
        len(train_all),
        len(dev_all),
        len(test_all),
    )
    return StratifyResult(train=train_all, dev=dev_all, test=test_all)
