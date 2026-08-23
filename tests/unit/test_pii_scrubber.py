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
