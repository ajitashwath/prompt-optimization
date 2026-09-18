"""
Dataset persistence for stratified splits.

The ingestor produces train/dev/test ConversationRecord splits; this store
persists them as JSONL files so downstream stages (optimizer, gate, shadow,
feedback loop) can reload the exact same data across processes and runs.

Once a test split has been written it is frozen: `save()` never overwrites it
unless `refresh_test=True` is passed explicitly.

Layout::

    <base_path>/
        train.jsonl
        dev.jsonl
        test.jsonl
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
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

    def save(self, result: StratifyResult, *, refresh_test: bool = False) -> dict[str, int]:
        """
        Persist the splits from a StratifyResult. Returns per-split counts.

        The test split is frozen once it exists: only the champion gate may
        read it, and its value depends on it never changing. So when a test
        split is already on disk (and `refresh_test` is False):

        * the persisted test file is left untouched;
        * conversations already in it are removed from the new train/dev, so
          re-ingesting the same logs cannot leak test items into training;
        * conversations the stratifier newly assigned to test are used as
          training data instead.

        Pass `refresh_test=True` to deliberately rebuild it; the old file is
        archived next to it, but gate results from before the refresh are not
        comparable with later ones.
        """
        train, dev, test = list(result.train), list(result.dev), list(result.test)
        test_path = self._split_path("test")
        counts: dict[str, int] = {}

        if test_path.exists() and not refresh_test:
            frozen_ids = self._read_ids(test_path)
            newly_test = [r for r in test if r.conversation_id not in frozen_ids]
            for rec in newly_test:
                rec.split = DataSplit.TRAIN
            n_train_before, n_dev_before = len(train), len(dev)
            train = [r for r in train if r.conversation_id not in frozen_ids] + newly_test
            dev = [r for r in dev if r.conversation_id not in frozen_ids]
            logger.info(
                "Test split is frozen (%d records, left untouched); removed %d train / "
                "%d dev records that belong to it; %d newly test-assigned records "
                "moved to train",
                len(frozen_ids),
                n_train_before + len(newly_test) - len(train),
                n_dev_before - len(dev),
                len(newly_test),
            )
            counts["test"] = len(frozen_ids)
        else:
            if test_path.exists():
                archive = test_path.with_name(
                    f"test.archived-{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.jsonl"
                )
                test_path.replace(archive)
                logger.warning("Test split refreshed; previous one archived to %s", archive)
            self.save_split("test", test)
            counts["test"] = len(test)

        self.save_split("train", train)
        self.save_split("dev", dev)
        counts["train"], counts["dev"] = len(train), len(dev)
        return {name: counts[name] for name in VALID_SPLITS}

    def save_split(self, split: str, records: list[ConversationRecord]) -> None:
        """Low-level write of a single split (atomic replace). Does not enforce the test freeze."""
        path = self._split_path(split)
        tmp_path = path.with_name(path.name + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(rec.model_dump_json() + "\n")
        os.replace(tmp_path, path)
        logger.info("Saved %d %s records to %s", len(records), split, path)

    @staticmethod
    def _read_ids(path: Path) -> set[str]:
        """Conversation IDs in a JSONL split, without full schema validation."""
        ids: set[str] = set()
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    ids.add(str(json.loads(line)["conversation_id"]))
                except (json.JSONDecodeError, KeyError, TypeError):
                    logger.warning("Unreadable record in %s while reading frozen IDs", path)
        return ids

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

    def find_conversations(
        self, ids: set[str], splits: tuple[str, ...] = ("train", "dev")
    ) -> dict[str, ConversationRecord]:
        """
        Look up conversations by ID (e.g. to show annotators / the judge the context of a
        gold item). The frozen test split is excluded by default: it is reserved for the gate.

        IDs that are not found are simply absent from the result.
        """
        found: dict[str, ConversationRecord] = {}
        wanted = set(ids)
        for split in splits:
            path = self._split_path(split)
            if not path.exists():
                continue
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    if not wanted:
                        return found
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        cid = str(data["conversation_id"])
                        if cid not in wanted:
                            continue
                        found[cid] = ConversationRecord.model_validate(data)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("%s: unreadable record while searching: %s", path, exc)
                        continue
                    wanted.discard(cid)
        return found

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
