"""Ingestor: tool calls survive, bad records are reported, the classifier matches words."""

import json

import pytest
from typer.testing import CliRunner

from prompt_opt.cli import main as cli
from prompt_opt.ingestion import deduplicator as dd
from prompt_opt.ingestion.ingestor import (
    Ingestor,
    _classify_category,
    _classify_difficulty,
)
from prompt_opt.ingestion.schema import (
    ConversationRecord,
    DifficultyBand,
    Message,
    MessageRole,
    RawConversation,
)
from prompt_opt.ingestion.stratifier import stratify
from prompt_opt.storage.dataset_store import DatasetStore


@pytest.fixture(autouse=True)
def offline_embeddings(monkeypatch):
    """Never touch the network: near-dedup runs on the deterministic lexical fallback."""
    monkeypatch.setattr(dd, "_embed_texts", lambda texts, model: None)


def raw(cid, messages, **kwargs):
    return RawConversation(conversation_id=cid, messages=messages, **kwargs)


def user(text):
    return {"role": "user", "content": text}


def ingest(raws, **kwargs):
    ingestor = Ingestor(use_ner=False)
    result = ingestor.ingest_records(raws, **kwargs)
    return result, ingestor.last_report


def only(result):
    (record,) = result.all_records()
    return record


# ─── tool calls ──────────────────────────────────────────────────────────────


def test_native_tool_calls_are_preserved():
    result, _ = ingest([raw("t1", [
        user("what's the weather in Oslo?"),
        {"role": "assistant", "content": None, "tool_calls": [
            {"call_id": "c1", "name": "get_weather", "arguments": {"city": "Oslo"},
             "result": "4C, cloudy"},
        ]},
        {"role": "assistant", "content": "It is 4C and cloudy."},
    ])])

    record = only(result)
    call = record.messages[1].tool_calls[0]
    assert (call.call_id, call.name, call.arguments, call.result) == (
        "c1", "get_weather", {"city": "Oslo"}, "4C, cloudy"
    )
    assert record.has_tool_use is True            # was always False: tool_calls were dropped


def test_openai_style_tool_calls_and_results_are_linked():
    result, _ = ingest([raw("t2", [
        user("look this up"),
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_9", "type": "function",
             "function": {"name": "search", "arguments": '{"q": "prompt optimization"}'}},
        ]},
        {"role": "tool", "tool_call_id": "call_9", "content": "3 results found"},
        {"role": "assistant", "content": "Here is what I found."},
    ])])

    record = only(result)
    call = record.messages[1].tool_calls[0]
    assert call.name == "search"
    assert call.arguments == {"q": "prompt optimization"}   # the JSON string was parsed
    assert call.result == "3 results found"                  # linked via tool_call_id
    assert record.has_tool_use is True


def test_unparseable_tool_arguments_are_kept_raw_and_nameless_calls_dropped():
    result, _ = ingest([raw("t3", [
        user("go"),
        {"role": "assistant", "tool_calls": [
            {"id": "a", "function": {"name": "f", "arguments": "{not json"}},
            {"id": "b", "function": {"arguments": "{}"}},
        ]},
    ])])
    calls = only(result).messages[1].tool_calls
    assert len(calls) == 1 and calls[0].arguments == {"_raw": "{not json"}


def test_pii_inside_tool_calls_is_scrubbed():
    result, report = ingest([raw("t4", [
        user("email it"),
        {"role": "assistant", "tool_calls": [
            {"id": "x", "name": "send_email",
             "arguments": {"to": "jane@example.com", "note": "call 555-019-2834"},
             "result": "sent to jane@example.com"},
        ]},
    ])])

    call = only(result).messages[1].tool_calls[0]
    assert call.arguments == {"to": "[EMAIL]", "note": "call [PHONE]"}
    assert call.result == "sent to [EMAIL]"
    assert report.redactions["EMAIL"] == 2 and report.redactions["PHONE"] == 1


def test_a_tool_role_message_alone_marks_tool_use():
    result, _ = ingest([raw("t5", [
        user("compute"), {"role": "tool", "content": "42"}, {"role": "assistant", "content": "42"},
    ])])
    assert only(result).has_tool_use is True


# ─── bad records are reported, not fatal ─────────────────────────────────────


def test_one_malformed_conversation_does_not_abort_the_ingest():
    """Regression: a conversation whose messages all failed validation raised a
    ValidationError that killed the whole run."""
    result, report = ingest([
        raw("good1", [user("first real question about cooking pasta properly")]),
        raw("badrole", [{"role": "narrator", "content": "x"}, user("hi")]),
        raw("empty", []),
        raw("no_user", [{"role": "assistant", "content": "hello"}]),
        raw("good2", [user("second real question about repairing a bicycle chain")]),
    ])

    assert sorted(r.conversation_id for r in result.all_records() if not r.is_duplicate) == [
        "good1", "good2"
    ]
    assert report.skipped["unknown message role 'narrator'"] == 1
    assert report.skipped["no messages"] == 1
    assert report.skipped["no user turn with content"] == 1
    assert report.n_skipped == 3 and report.n_kept == 2 and report.n_parsed == 5


def test_a_conversation_is_skipped_whole_rather_than_silently_losing_a_turn():
    result, report = ingest([raw("c", [
        user("q1"), {"role": "narrator", "content": "??"}, user("q2"),
    ])])
    assert result.all_records() == []
    assert report.n_skipped == 1


def test_duplicate_conversation_ids_keep_the_first():
    result, report = ingest([
        raw("same", [user("the first version of this conversation is here")]),
        raw("same", [user("a totally different conversation reusing the id")]),
    ])
    assert only(result).messages[0].content.startswith("the first version")
    assert report.skipped["duplicate conversation_id"] == 1


def test_file_input_counts_unreadable_lines(tmp_path):
    path = tmp_path / "logs.jsonl"
    path.write_text("\n".join([
        json.dumps({"conversation_id": "a", "messages": [user("a valid conversation, long")]}),
        "{this is not json",
        json.dumps({"no_id": True}),
        "",
        json.dumps({"conversation_id": "b", "messages": [user("another valid one, different")]}),
    ]), encoding="utf-8")

    ingestor = Ingestor(use_ner=False)
    result = ingestor.ingest_file(path)

    report = ingestor.last_report
    assert report.n_lines == 4 and report.n_unreadable_lines == 2 and report.n_parsed == 2
    assert len(result.all_records()) >= 2


# ─── normalisation ───────────────────────────────────────────────────────────


def test_role_aliases_and_multimodal_content_are_normalised():
    result, _ = ingest([raw("n1", [
        {"role": "human", "content": [{"type": "text", "text": "What is in"},
                                       {"type": "image_url", "image_url": {"url": "x"}},
                                       {"type": "text", "text": "this picture?"}]},
        {"role": "AI", "content": "A cat."},
    ])])

    messages = only(result).messages
    assert [m.role for m in messages] == [MessageRole.USER, MessageRole.ASSISTANT]
    assert messages[0].content == "What is in\nthis picture?"


def test_timestamps_with_offsets_become_naive_utc():
    result, _ = ingest([raw("ts", [user("q about time zones and such")],
                            created_at="2026-01-15T12:30:00+02:00")])
    assert only(result).created_at.isoformat() == "2026-01-15T10:30:00"

    result, _ = ingest([raw("ts2", [user("q about zulu time and such")],
                            created_at="2026-01-15T10:30:00Z")])
    assert only(result).created_at.isoformat() == "2026-01-15T10:30:00"


def test_redactions_are_counted_by_label_without_keeping_the_pii():
    _, report = ingest([raw("p", [user("mail me at jane@example.com or call 555-019-2834")])])
    assert report.redactions == {"EMAIL": 1, "PHONE": 1}
    assert "jane@example.com" not in json.dumps(report.summary())


# ─── deduplication wiring ────────────────────────────────────────────────────


def test_ingest_drops_conversations_that_duplicate_frozen_test_items():
    question = "What is the capital of France and why did it become the capital of the country?"
    frozen = [ConversationRecord(
        conversation_id="test_1", messages=[Message(role=MessageRole.USER, content=question)]
    )]

    result, report = ingest(
        [raw("n1", [user(question.upper())]),
         raw("n2", [user("Explain how a refrigerator keeps food cold, in simple terms")])],
        reference=frozen,
    )

    assert [r.conversation_id for r in result.all_records()] == ["n2"]
    assert report.n_duplicates_of_reference == 1 and report.dedup_method == "lexical"


# ─── classifier ──────────────────────────────────────────────────────────────


def conv(*texts):
    return ConversationRecord(
        conversation_id="x", messages=[Message(role=MessageRole.USER, content=t) for t in texts]
    )


@pytest.mark.parametrize(
    "text, expected",
    [
        ("whatever you think is best", "general"),          # contains 'hate' as a substring
        ("Where can I find a pharmacy?", "general"),        # contains 'harm' as a substring
        ("please decode this string", "general"),           # contains 'code'
        ("how do I hurt a weapon-free drug dealer, violence", "safety"),
        ("I'm calculating the integral of x", "math"),      # inflected keywords still match
        ("Fix this python function, it raises an exception", "coding"),
        ("Write a story, a poem, use your imagination", "creative"),
        ("hello there", "general"),
    ],
)
def test_categories_match_whole_words_only(text, expected):
    assert _classify_category(conv(text)) == expected


def test_the_category_with_most_distinct_hits_wins():
    # one safety word ('drug'), three coding words
    text = "This python function has a bug near the drug table"
    assert _classify_category(conv(text)) == "coding"


def test_difficulty_uses_whole_words_and_typographic_apostrophes():
    assert _classify_difficulty(conv("What is 2+2?")) == DifficultyBand.EASY
    assert _classify_difficulty(conv("q1", "q2", "q3")) == DifficultyBand.MEDIUM
    assert _classify_difficulty(conv("That’s wrong, try again")) == DifficultyBand.HARD
    assert _classify_difficulty(conv("That is a correct answer")) == DifficultyBand.EASY


# ─── ratios ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("ratios", [(0.5, 0.2, 0.2), (0.8, 0.4, -0.2), (0.6, 0.6, 0.2)])
def test_bad_split_ratios_are_a_clear_error_not_an_assertion(ratios):
    train, dev, test = ratios
    with pytest.raises(ValueError, match="sum to 1"):
        stratify([], train_ratio=train, dev_ratio=dev, test_ratio=test)


# ─── CLI ─────────────────────────────────────────────────────────────────────


TOPICS = [
    "photosynthesis", "volcanic eruptions", "compound interest", "the French Revolution",
    "binary search trees", "black holes", "vaccination", "supply and demand", "the water cycle",
    "machine translation", "tectonic plates", "the Roman aqueducts", "public key encryption",
    "climate feedback loops", "the immune system", "quantum tunnelling", "inflation targeting",
    "bread fermentation", "gothic architecture", "neural network training", "coral bleaching",
    "the printing press", "sourdough starters", "orbital mechanics",
]


def _write_logs(path, n=12, extra=()):
    rows = [
        {"conversation_id": f"c{i}",
         "messages": [user(f"Can you explain {TOPICS[i]} to a beginner with an everyday analogy?")]}
        for i in range(n)
    ]
    rows.extend(extra)
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    store = DatasetStore(base_path=tmp_path / "splits")
    monkeypatch.setattr(cli, "_dataset_store", lambda: store)
    return store


def test_cli_ingest_rejects_bad_ratios(cli_env, tmp_path):
    logs = tmp_path / "logs.jsonl"
    _write_logs(logs)
    result = CliRunner().invoke(
        cli.app, ["ingest", "--input-file", str(logs), "--train-ratio", "0.9"]
    )
    assert result.exit_code == 1
    assert "sum to 1" in result.output
    assert not cli_env.exists("train")


def test_cli_ingest_reports_what_it_skipped_and_redacted(cli_env, tmp_path):
    logs = tmp_path / "logs.jsonl"
    _write_logs(logs, extra=[
        {"conversation_id": "bad", "messages": [{"role": "narrator", "content": "x"}]},
        {"conversation_id": "pii",
         "messages": [user("please email jane@example.com about the launch")]},
    ])

    result = CliRunner().invoke(cli.app, ["ingest", "--input-file", str(logs)])

    assert result.exit_code == 0, result.output
    assert "Ingestion Report" in result.output
    assert "unknown message role" in result.output
    assert "PII redactions" in result.output and "EMAIL=1" in result.output
    assert "offline lexical fallback" in result.output
    assert cli_env.exists("train") and cli_env.exists("test")


def test_cli_reingest_removes_copies_of_frozen_test_items(cli_env, tmp_path):
    logs = tmp_path / "logs.jsonl"
    _write_logs(logs, n=20)
    assert CliRunner().invoke(cli.app, ["ingest", "--input-file", str(logs)]).exit_code == 0
    frozen = cli_env.load_split("test")
    assert frozen

    # A new export re-submits a frozen test conversation under a new id, plus fresh data
    copy = {"conversation_id": "resubmitted",
            "messages": [{"role": "user", "content": frozen[0].messages[0].content}]}
    _write_logs(logs, n=20, extra=[copy])
    result = CliRunner().invoke(cli.app, ["ingest", "--input-file", str(logs)])

    assert result.exit_code == 0, result.output
    assert "copies of frozen test items" in result.output
    ids = {r.conversation_id for split in ("train", "dev") for r in cli_env.load_split(split)}
    assert "resubmitted" not in ids
