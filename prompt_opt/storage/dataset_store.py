"""
Dataset persistence for stratified splits.

The ingestor produces train/dev/test ConversationRecord splits; this store
persists them as JSONL files so downstream stages (optimizer, gate, shadow,
feedback loop) can reload the exact same data across processes and runs.

Layout::

    <base_path>/
        train.jsonl
        dev.jsonl
        test.jsonl
"""

from __future__ import annotations

import logging
from pathlib import Path

from prompt_opt.ingestion.schema import ConversationRecord, DataSplit
from prompt_opt.ingestion.stratifier import StratifyResult

logger = logging.getLogger(__name__)

VALID_SPLITS = ("train", "dev", "test")


class DatasetStore:
    """Persist and load stratified dataset splits as JSONL files."""

    def __init__(self, base_path: str | Path = "./data/splits"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _split_path(self, split: str) -> Path:
        if split not in VALID_SPLITS:
            raise ValueError(f"Unknown split {split!r}; expected one of {VALID_SPLITS}")
        return self.base_path / f"{split}.jsonl"

    # ── Write ────────────────────────────────────────────────────────────────

    def save(self, result: StratifyResult) -> dict[str, int]:
        """Persist all three splits from a StratifyResult. Returns per-split counts."""
        counts = {}
        for split in VALID_SPLITS:
            records: list[ConversationRecord] = getattr(result, split)
            self.save_split(split, records)
            counts[split] = len(records)
        return counts

    def save_split(self, split: str, records: list[ConversationRecord]) -> None:
        path = self._split_path(split)
        with open(path, "w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(rec.model_dump_json() + "\n")
        logger.info("Saved %d %s records to %s", len(records), split, path)

    # ── Read ─────────────────────────────────────────────────────────────────

    def exists(self, split: str) -> bool:
        return self._split_path(split).exists()

    def load_split(self, split: str) -> list[ConversationRecord]:
        """
        Load a split back into ConversationRecord objects.

        Raises FileNotFoundError if the split was never persisted.
        """
        path = self._split_path(split)
        if not path.exists():
            raise FileNotFoundError(
                f"No persisted {split!r} split at {path}. "
                f"Run `prompt-opt ingest` first."
            )

        records: list[ConversationRecord] = []
        with open(path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = ConversationRecord.model_validate_json(line)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("%s line %d parse error: %s — skipping", path, lineno, exc)
                    continue
                records.append(rec)

        logger.info("Loaded %d %s records from %s", len(records), split, path)
        return records

    def stats(self) -> dict[str, int]:
        """Return record counts per persisted split (0 if missing)."""
        out: dict[str, int] = {}
        for split in VALID_SPLITS:
            try:
                out[split] = len(self.load_split(split))
            except FileNotFoundError:
                out[split] = 0
        return out


def filter_by_split(
    records: list[ConversationRecord], split: str | DataSplit
) -> list[ConversationRecord]:
    """Filter records whose `split` field matches the given value."""
    wanted = split.value if isinstance(split, DataSplit) else split
    return [r for r in records if r.split.value == wanted]
