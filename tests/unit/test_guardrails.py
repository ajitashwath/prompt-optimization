import pytest
from prompt_opt.eval.guardrails import (
    is_refusal,
    is_valid_json_response,
    check_regression_case,
    RegressionCase,
)


def test_refusal_patterns():
    assert is_refusal("I'm sorry, I cannot perform that task.") is True
    assert is_refusal("Sure, I'd be happy to write a script for you!") is False


def test_json_compliance():
    assert is_valid_json_response('{"status": "ok"}') is True
    assert is_valid_json_response("```json\n{\n  \"status\": \"ok\"\n}\n```") is True
    assert is_valid_json_response("No json here") is False


def test_regression_cases():
    case = RegressionCase(
        conversation_id="reg_01",
        expected_behavior="refuse",
        expected_keywords=["sorry", "cannot"],
    )
    passed, reason = check_regression_case(case, "I'm sorry, I cannot do that.")
    assert passed is True

    passed, reason = check_regression_case(case, "Sure, here's how you do it...")
    assert passed is False
