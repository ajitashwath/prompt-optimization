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


def _ids(store, split):
    return {r.conversation_id for r in store.load_split(split)}


def test_test_split_is_frozen_across_saves(tmp_path):
    store = DatasetStore(base_path=tmp_path / "splits")
    store.save(
        StratifyResult(
            train=[_conv("t1", "a"), _conv("t2", "b")],
            dev=[_conv("d1", "c")],
            test=[_conv("e1", "d"), _conv("e2", "e")],
        )
    )
    frozen_bytes = (tmp_path / "splits" / "test.jsonl").read_bytes()

    # Re-ingest: e1 is (wrongly) offered as train, e2 as dev, and the stratifier
    # assigns brand-new records n1/n2 to test.
    counts = store.save(
        StratifyResult(
            train=[_conv("t1", "a"), _conv("e1", "d")],
            dev=[_conv("d1", "c"), _conv("e2", "e")],
            test=[_conv("n1", "f"), _conv("n2", "g")],
        )
    )

    assert (tmp_path / "splits" / "test.jsonl").read_bytes() == frozen_bytes
    assert _ids(store, "test") == {"e1", "e2"}
    # Test items never leak into train/dev; new test-assigned records train instead
    assert _ids(store, "train") == {"t1", "n1", "n2"}
    assert _ids(store, "dev") == {"d1"}
    assert counts == {"train": 3, "dev": 1, "test": 2}
    assert not (_ids(store, "train") | _ids(store, "dev")) & _ids(store, "test")


def test_refresh_test_rebuilds_and_archives(tmp_path):
    store = DatasetStore(base_path=tmp_path / "splits")
    store.save(StratifyResult(train=[_conv("t1", "a")], dev=[], test=[_conv("e1", "d")]))

    store.save(
        StratifyResult(train=[_conv("t1", "a")], dev=[], test=[_conv("n1", "f")]),
        refresh_test=True,
    )

    assert _ids(store, "test") == {"n1"}
    archives = list((tmp_path / "splits").glob("test.archived-*.jsonl"))
    assert len(archives) == 1
    assert '"e1"' in archives[0].read_text(encoding="utf-8")


def test_first_save_writes_all_splits_and_no_tmp_files(tmp_path):
    store = DatasetStore(base_path=tmp_path / "splits")
    store.save(
        StratifyResult(train=[_conv("t1", "a")], dev=[_conv("d1", "b")], test=[_conv("e1", "c")])
    )
    assert sorted(p.name for p in (tmp_path / "splits").iterdir()) == [
        "dev.jsonl", "test.jsonl", "train.jsonl",
    ]


def test_find_conversations_searches_train_and_dev_but_not_the_frozen_test_split(tmp_path):
    store = DatasetStore(base_path=tmp_path / "splits")
    store.save(
        StratifyResult(
            train=[_conv("t1", "train question"), _conv("t2", "other")],
            dev=[_conv("d1", "dev question")],
            test=[_conv("e1", "secret test question")],
        )
    )

    found = store.find_conversations({"t1", "d1", "e1", "missing"})

    assert set(found) == {"t1", "d1"}  # test is reserved for the gate; unknown ids are absent
    assert found["t1"].messages[0].content == "train question"
    assert set(store.find_conversations({"e1"}, splits=("test",))) == {"e1"}


def test_find_conversations_without_persisted_splits_is_empty(tmp_path):
    store = DatasetStore(base_path=tmp_path / "splits")
    assert store.find_conversations({"anything"}) == {}


def test_find_conversations_skips_corrupt_lines(tmp_path):
    store = DatasetStore(base_path=tmp_path / "splits")
    store.save_split("train", [_conv("t1", "ok")])
    with open(tmp_path / "splits" / "train.jsonl", "a", encoding="utf-8") as fh:
        fh.write("{not json\n")
    assert set(store.find_conversations({"t1"})) == {"t1"}


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
