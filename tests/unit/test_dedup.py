"""Deduplication: no missed duplicates, a deterministic offline fallback, a reference set."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from prompt_opt.ingestion import deduplicator as dd
from prompt_opt.ingestion.schema import ConversationRecord, Message, MessageRole


def rec(cid, text=None):
    return ConversationRecord(
        conversation_id=cid,
        messages=[Message(role=MessageRole.USER, content=text if text is not None else cid)],
    )


def unit(*values):
    v = np.array(values, dtype=np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def embeddings(monkeypatch):
    """Install fixed embeddings by text: `embeddings({"text": vector, ...})`."""

    def install(vectors_by_text):
        def fake(texts, model):
            return np.stack([vectors_by_text[t] for t in texts])

        monkeypatch.setattr(dd, "_embed_texts", fake)

    return install


@pytest.fixture
def no_embeddings(monkeypatch):
    monkeypatch.setattr(dd, "_embed_texts", lambda texts, model: None)


@pytest.fixture(params=["faiss", "numpy"])
def index_backend(request, monkeypatch):
    if request.param == "numpy":
        monkeypatch.setitem(sys.modules, "faiss", None)  # `import faiss` -> ImportError
    return request.param


# ─── near-duplicate search ───────────────────────────────────────────────────


def test_an_earlier_duplicate_is_found_even_when_later_items_crowd_the_neighbours(
    embeddings, index_backend
):
    """Regression: the old search asked for the top i+1 neighbours of ALL items, so later
    near-duplicates pushed the earlier true match out of the results and it was missed."""
    embeddings({
        "c0": unit(1, 0, 0, 0),
        "c1": unit(0.96, 0.28, 0, 0),          # cos(c0, c1) = 0.96: a duplicate of c0
        "c2": unit(0.96, 0.28, 0.05, 0),       # closer to c1 than c0 is
        "c3": unit(0.96, 0.28, 0, 0.05),
    })

    result = dd.deduplicate([rec("c0"), rec("c1"), rec("c2"), rec("c3")], threshold=0.95)

    assert [c.conversation_id for c in result.kept] == ["c0"]
    assert {c.conversation_id: rep for c, rep in result.duplicates} == {
        "c1": "c0", "c2": "c0", "c3": "c0",   # all are compared against the KEPT set only
    }
    assert result.n_near == 3 and result.method == "embedding"


def test_distinct_conversations_are_all_kept(embeddings, index_backend):
    embeddings({"a": unit(1, 0, 0), "b": unit(0, 1, 0), "c": unit(0, 0, 1)})
    result = dd.deduplicate([rec("a"), rec("b"), rec("c")], threshold=0.95)
    assert [c.conversation_id for c in result.kept] == ["a", "b", "c"]
    assert result.duplicates == []


def test_first_occurrence_wins_and_duplicates_are_flagged(embeddings):
    embeddings({"x": unit(1, 0), "y": unit(1, 0.01)})
    first, second = rec("x"), rec("y")

    result = dd.deduplicate([first, second], threshold=0.95)

    assert result.kept == [first]
    assert second.is_duplicate is True and first.is_duplicate is False


# ─── exact duplicates ────────────────────────────────────────────────────────


def test_exact_duplicates_are_removed_without_needing_any_embedding(no_embeddings):
    result = dd.deduplicate([
        rec("a", "How do I reset my password?"),
        rec("b", "  how do I  reset my PASSWORD?  "),     # same after case/whitespace normalising
        rec("c", "Something else entirely"),
    ])

    assert [c.conversation_id for c in result.kept] == ["a", "c"]
    assert result.n_exact == 1 and result.n_near == 0
    assert result.duplicates[0][1] == "a"


def test_exact_duplicates_are_found_when_embeddings_work_too(embeddings):
    embeddings({"same text": unit(1, 0)})
    result = dd.deduplicate([rec("a", "same text"), rec("b", "same text")])
    assert result.n_exact == 1 and len(result.kept) == 1


# ─── the offline fallback ────────────────────────────────────────────────────


def test_fallback_is_deterministic_and_never_a_silent_no_op(no_embeddings):
    convs = lambda: [  # noqa: E731
        rec("a", "How do I reset my password on the website?"),
        rec("b", "How do I reset my password on the website ?"),        # near-identical
        rec("c", "What is the capital of France?"),
    ]

    first = dd.deduplicate(convs())
    second = dd.deduplicate(convs())

    assert first.method == "lexical"
    assert [c.conversation_id for c in first.kept] == ["a", "c"]
    assert first.n_near == 1
    assert [c.conversation_id for c in first.kept] == [c.conversation_id for c in second.kept]


def test_fallback_keeps_short_templated_prompts_that_merely_look_alike(no_embeddings):
    """'question number 1' vs '... 2' are 97% similar as character n-grams, but they are
    different prompts: short texts are only ever deduplicated when identical."""
    result = dd.deduplicate([rec(f"c{i}", f"question number {i}") for i in range(6)])
    assert len(result.kept) == 6

    same = dd.deduplicate([rec("a", "question number 1"), rec("b", "Question  number 1")])
    assert len(same.kept) == 1 and same.n_exact == 1


def test_short_reference_items_only_block_identical_text(no_embeddings):
    frozen = [rec("test_1", "What is 2 + 2?")]
    result = dd.deduplicate([rec("n1", "What is 2 + 3?"), rec("n2", "what is 2 + 2?")],
                            reference=frozen)
    assert [c.conversation_id for c in result.kept] == ["n1"]


def test_fallback_does_not_pretend_to_catch_paraphrases(no_embeddings):
    result = dd.deduplicate([
        rec("a", "How do I reset my password?"),
        rec("b", "I forgot my login credentials, how can I get back into my account?"),
    ])
    assert len(result.kept) == 2      # needs real embeddings; documented, and reported via `method`


# ─── embedding service handling ──────────────────────────────────────────────


def test_embeddings_are_requested_in_batches_and_any_failure_falls_back(monkeypatch):
    calls = []

    def flaky(batch, model):
        calls.append(len(batch))
        if len(calls) == 2:
            raise RuntimeError("rate limited")
        return [[1.0, 0.0]] * len(batch)

    monkeypatch.setattr(dd, "_embed_batch", flaky)

    assert dd._embed_texts([f"t{i}" for i in range(150)], "m") is None
    assert calls == [64, 64]           # batches of 64; stopped at the failure

    calls.clear()
    monkeypatch.setattr(dd, "_embed_batch", lambda b, m: [[1.0, 0.0]] * len(b))
    matrix = dd._embed_texts([f"t{i}" for i in range(150)], "m")
    assert matrix.shape == (150, 2)


def test_embed_batch_restores_the_input_order(monkeypatch):
    class FakeLiteLLM:
        def embedding(self, model, input):
            assert input == ["first", "second"]
            data = [{"index": 1, "embedding": [0.0, 1.0]}, {"index": 0, "embedding": [1.0, 0.0]}]
            return SimpleNamespace(data=data)

    monkeypatch.setattr("prompt_opt.llm.get_litellm", lambda: FakeLiteLLM())
    assert dd._embed_batch(["first", "second"], "m") == [[1.0, 0.0], [0.0, 1.0]]


def test_texts_are_truncated_before_embedding(monkeypatch):
    seen = {}

    class FakeLiteLLM:
        def embedding(self, model, input):
            seen["lens"] = [len(t) for t in input]
            return SimpleNamespace(data=[{"embedding": [1.0]} for _ in input])

    monkeypatch.setattr("prompt_opt.llm.get_litellm", lambda: FakeLiteLLM())
    dd._embed_batch(["x" * 50_000], "m")
    assert seen["lens"] == [dd._EMBED_MAX_CHARS]


def test_a_malformed_embedding_response_falls_back(monkeypatch):
    monkeypatch.setattr(dd, "_embed_batch", lambda b, m: [[1.0, 0.0]])  # wrong number of rows
    assert dd._embed_texts(["a", "b", "c"], "m") is None


# ─── reference set (frozen test split) ───────────────────────────────────────


def test_conversations_that_duplicate_reference_items_are_dropped(no_embeddings):
    question = "What is the capital of France and why did it become the capital?"
    frozen = [rec("test_1", question)]
    new = [
        rec("n1", question.replace("?", " ?").replace("and why", "and, why")),  # near-copy
        rec("n2", question.lower()),                       # exact copy (normalised)
        rec("n3", "Explain photosynthesis in simple terms for a ten year old"),
    ]

    result = dd.deduplicate(new, reference=frozen)

    assert [c.conversation_id for c in result.kept] == ["n3"]
    assert {c.conversation_id: rep for c, rep in result.duplicates} == {
        "n1": "test_1", "n2": "test_1",
    }
    assert result.n_vs_reference == 2


def test_reference_items_are_never_returned(no_embeddings):
    frozen = [rec("test_1", "alpha alpha alpha"), rec("test_2", "beta beta beta")]
    result = dd.deduplicate([rec("n1", "gamma gamma gamma")], reference=frozen)
    assert [c.conversation_id for c in result.kept] == ["n1"]


def test_duplicates_within_the_new_batch_still_point_at_the_new_representative(no_embeddings):
    frozen = [rec("test_1", "unrelated frozen item")]
    result = dd.deduplicate(
        [rec("n1", "the first of a pair"), rec("n2", "The first of a pair")], reference=frozen
    )
    assert result.duplicates[0][1] == "n1"
    assert result.n_vs_reference == 0


def test_empty_input():
    assert dd.deduplicate([]).kept == []


def test_conversations_without_user_text_are_never_merged(no_embeddings):
    empty = [
        ConversationRecord(
            conversation_id=f"e{i}",
            messages=[Message(role=MessageRole.ASSISTANT, content="hello")],
        )
        for i in range(3)
    ]
    assert len(dd.deduplicate(empty).kept) == 3
