"""Studio: annotators see the conversation the responses answer (and it is HTML-escaped)."""

import pytest
from fastapi.testclient import TestClient

from prompt_opt.ingestion.schema import ConversationRecord, Message, MessageRole
from prompt_opt.labeling.gold_store import GoldStore


@pytest.fixture
def studio(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # importing the server creates ./data stores
    (tmp_path / "data").mkdir()
    from prompt_opt.labeling import server

    store = GoldStore(db_url=f"sqlite:///{tmp_path / 'gold.db'}")
    monkeypatch.setattr(server, "gold_store", store)
    return server, store, TestClient(server.app)


def _conversation():
    return ConversationRecord(
        conversation_id="conv_1",
        messages=[
            Message(role=MessageRole.SYSTEM, content="hidden system prompt"),
            Message(role=MessageRole.USER, content="How do I reset my <b>password</b>?"),
            Message(role=MessageRole.ASSISTANT, content="Click 'Forgot password'."),
        ],
    )


def test_annotate_page_shows_the_conversation_context(studio, monkeypatch):
    server, store, client = studio
    monkeypatch.setattr(server, "_conversations_for", lambda ids: {"conv_1": _conversation()})
    item_id = store.add_item("conv_1", "resp a", "resp b", "v1.0")

    page = client.get(f"/annotate/{item_id}")

    assert page.status_code == 200
    assert "How do I reset my &lt;b&gt;password&lt;/b&gt;?" in page.text  # shown, and escaped
    assert "<b>password</b>" not in page.text
    assert "Forgot password" in page.text
    assert "hidden system prompt" not in page.text  # the system prompt is not part of the context


def test_annotate_page_explains_missing_context(studio, monkeypatch):
    server, store, client = studio
    monkeypatch.setattr(server, "_conversations_for", lambda ids: {})
    item_id = store.add_item("unknown_conv", "resp a", "resp b", "v1.0")

    page = client.get(f"/annotate/{item_id}")

    assert page.status_code == 200
    assert "NO CONTEXT" in page.text
    assert "unknown_conv" in page.text


def test_annotate_finds_any_item_by_id_not_just_the_first_hundred(studio, monkeypatch):
    server, store, client = studio
    monkeypatch.setattr(server, "_conversations_for", lambda ids: {})
    ids = [store.add_item(f"c{i}", "a", "b", "v1.0") for i in range(120)]

    assert client.get(f"/annotate/{ids[-1]}").status_code == 200
    assert client.get("/annotate/does-not-exist").status_code == 404
