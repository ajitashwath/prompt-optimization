from prompt_opt.ingestion.pii_scrubber import scrub


def test_scrub_regex_patterns():
    # Email patterns
    res = scrub("Send emails to test@example.com please.")
    assert "test@example.com" not in res.scrubbed_text
    assert "[EMAIL]" in res.scrubbed_text
    assert res.regex_count >= 1

    # Phone patterns
    res = scrub("Call me at 555-019-2834.")
    assert "555-019-2834" not in res.scrubbed_text
    assert "[PHONE]" in res.scrubbed_text

    # IP Address patterns
    res = scrub("Database is at 192.168.1.1 now.")
    assert "192.168.1.1" not in res.scrubbed_text
    assert "[IP]" in res.scrubbed_text


def test_scrub_no_pii():
    text = "Just a regular sentence with no sensitive fields."
    res = scrub(text)
    assert res.scrubbed_text == text
    assert res.total_redactions == 0


# ─── precision: ordinary numbers are not PII ─────────────────────────────────

import pytest  # noqa: E402

from prompt_opt.ingestion import pii_scrubber  # noqa: E402
from prompt_opt.ingestion.pii_scrubber import scrub_structure  # noqa: E402


@pytest.mark.parametrize(
    "text",
    [
        "Multiply 1234567 by 2",                        # 7 digits: not a phone number
        "Compute 10000 + 25000 and 99999",              # 5 digits: not ZIP codes
        "The order id is 4111111111111112",             # 16 digits but fails the Luhn check
        "Population 123456789 in 2020",                 # 9 digits without an SSN label
        "Version 999.1.1.1 and 1.2.3 released",         # not valid IPv4 addresses
        "Date 2024-01-15, pi = 3.14159, 1,000,000 units",
        "What is the capital of France?",
        "Call the function twice, then text the result to stdout",
    ],
)
def test_ordinary_numbers_and_text_are_left_alone(text):
    result = scrub(text, use_ner=False)
    assert result.scrubbed_text == text
    assert result.total_redactions == 0


@pytest.mark.parametrize(
    "text, tag",
    [
        ("Card 4111 1111 1111 1111 exp 12/26", "[CARD]"),          # valid Luhn
        ("Card 4111-1111-1111-1111", "[CARD]"),
        ("SSN 123-45-6789 on file", "[SSN]"),
        ("My social security number is 123456789.", "[SSN]"),
        ("Call me at 555-019-2834.", "[PHONE]"),
        ("phone: (555) 019-2834", "[PHONE]"),
        ("Reach me on +1 555 019 2834", "[PHONE]"),
        ("+44 20 7946 0958 is the London office", "[PHONE]"),
        ("Call 5550192834 tomorrow", "[PHONE]"),                    # bare digits, but labelled
        ("Server 10.0.0.255 is down", "[IP]"),
        ("Home: 221 Baker Street.", "[ADDRESS]"),
        ("Ship to Springfield, IL 62704", "[ZIP]"),
        ("zip code 90210", "[ZIP]"),
        ("Hi, my name is Jane Doe.", "[PERSON]"),
        ("key=sk-ant-abcdefghijklmnopqrstuvwxyz123456", "[SECRET]"),
        ("password = hunter2hunter2", "[SECRET]"),
        ("token AKIAABCDEFGHIJKLMNOP leaked", "[SECRET]"),
    ],
)
def test_real_pii_is_redacted(text, tag):
    result = scrub(text, use_ner=False)
    assert tag in result.scrubbed_text
    assert result.regex_count >= 1


def test_the_sensitive_part_is_redacted_but_the_context_word_stays():
    assert scrub("zip code 90210", use_ner=False).scrubbed_text == "zip code [ZIP]"
    assert scrub("my name is Jane Doe", use_ner=False).scrubbed_text == "my name is [PERSON]"
    assert scrub("I live at 221 Baker Street.", use_ner=False).scrubbed_text == (
        "I live at [ADDRESS]."
    )


def test_overlapping_matches_resolve_by_priority():
    # The number inside the URL and the name inside the email do not produce extra spans
    text = "See https://example.com/user/4111111111111111?x=1 or mail jane.doe@example.com"
    result = scrub(text, use_ner=False)
    assert result.scrubbed_text == "See [URL] or mail [EMAIL]"
    assert [r.label for r in result.redactions] == ["URL", "EMAIL"]


def test_offsets_refer_to_the_original_text_and_no_pii_is_stored():
    text = "mail a@b.com then call 555-019-2834 now"
    result = scrub(text, use_ner=False)

    assert [(r.label, text[r.start:r.end]) for r in result.redactions] == [
        ("EMAIL", "a@b.com"),
        ("PHONE", "555-019-2834"),
    ]
    assert not hasattr(result.redactions[0], "original")  # the log never keeps the PII itself


# ─── NER stage ───────────────────────────────────────────────────────────────


class _Ent:
    def __init__(self, start, end, label):
        self.start_char, self.end_char, self.label_ = start, end, label


class _Doc:
    def __init__(self, ents):
        self.ents = ents


def _fake_nlp(seen_lengths=None):
    def nlp(text):
        if seen_lengths is not None:
            seen_lengths.append(len(text))
        ents = []
        for word, label in (("Alice", "PERSON"), ("Paris", "GPE"), ("alice", "PERSON")):
            start = text.find(word)
            while start != -1:
                ents.append(_Ent(start, start + len(word), label))
                start = text.find(word, start + 1)
        return _Doc(ents)

    return nlp


def test_ner_entities_are_redacted_in_a_single_pass(monkeypatch):
    monkeypatch.setattr(pii_scrubber, "_load_nlp", lambda: _fake_nlp())

    result = scrub("Alice flew to Paris; write to alice@example.com", use_ner=True)

    assert result.scrubbed_text == "[PERSON] flew to [GPE]; write to [EMAIL]"  # email beats NER
    assert result.ner_count == 2
    assert result.regex_count == 1


def test_ner_entity_types_are_configurable(monkeypatch):
    monkeypatch.setattr(pii_scrubber, "_load_nlp", lambda: _fake_nlp())

    result = scrub("Alice flew to Paris", use_ner=True, ner_types={"PERSON"})

    assert result.scrubbed_text == "[PERSON] flew to Paris"


def test_very_long_text_is_processed_in_bounded_chunks(monkeypatch):
    lengths: list[int] = []
    monkeypatch.setattr(pii_scrubber, "_load_nlp", lambda: _fake_nlp(lengths))
    filler = "lorem ipsum " * 10_000                       # 120k characters
    text = "Alice " + filler + " Paris " + filler + " Alice"

    result = scrub(text, use_ner=True)

    assert max(lengths) <= 50_000 and len(lengths) >= 3   # spaCy would reject one huge input
    assert "Alice" not in result.scrubbed_text and "Paris" not in result.scrubbed_text
    assert result.ner_count == 3


def test_missing_spacy_model_degrades_to_patterns_only(monkeypatch):
    monkeypatch.setattr(pii_scrubber, "_load_nlp", lambda: None)
    result = scrub("Alice, call 555-019-2834", use_ner=True)
    assert result.scrubbed_text == "Alice, call [PHONE]"


# ─── nested structures (tool-call arguments) ─────────────────────────────────


def test_scrub_structure_reaches_nested_strings_and_keeps_the_rest():
    payload = {
        "to": "jane@example.com",
        "count": 3,
        "options": {"phones": ["555-019-2834", "none"], "flag": True},
    }

    cleaned, redactions = scrub_structure(payload, use_ner=False)

    assert cleaned == {
        "to": "[EMAIL]",
        "count": 3,
        "options": {"phones": ["[PHONE]", "none"], "flag": True},
    }
    assert sorted(r.label for r in redactions) == ["EMAIL", "PHONE"]
    assert payload["to"] == "jane@example.com"          # the input is not mutated
