"""Tests for DatasetStore split persistence."""

import pytest

from prompt_opt.ingestion.schema import ConversationRecord, Message, MessageRole
from prompt_opt.ingestion.stratifier import StratifyResult
from prompt_opt.storage.dataset_store import DatasetStore


def _conv(cid: str, text: str) -> ConversationRecord:
    return ConversationRecord(
        conversation_id=cid,
        messages=[Message(role=MessageRole.USER, content=text)],
        category="general",
    )


def test_roundtrip_save_load(tmp_path):
    store = DatasetStore(base_path=tmp_path / "splits")
    result = StratifyResult(
        train=[_conv("t1", "hello"), _conv("t2", "world")],
        dev=[_conv("d1", "dev q")],
        test=[_conv("e1", "test q")],
    )

    counts = store.save(result)
    assert counts == {"train": 2, "dev": 1, "test": 1}

    loaded = store.load_split("train")
    assert [r.conversation_id for r in loaded] == ["t1", "t2"]
    assert loaded[0].messages[0].content == "hello"
    assert loaded[0].split.value == "unlabeled"


def test_load_missing_split_raises(tmp_path):
    store = DatasetStore(base_path=tmp_path / "splits")
    with pytest.raises(FileNotFoundError):
        store.load_split("dev")


def test_stats_and_exists(tmp_path):
    store = DatasetStore(base_path=tmp_path / "splits")
    assert store.stats() == {"train": 0, "dev": 0, "test": 0}
    assert not store.exists("test")

    store.save_split("test", [_conv("e1", "x")])
    assert store.exists("test")
    assert store.stats()["test"] == 1


def test_invalid_split_rejected(tmp_path):
    store = DatasetStore(base_path=tmp_path / "splits")
    with pytest.raises(ValueError):
        store.save_split("validation", [])
