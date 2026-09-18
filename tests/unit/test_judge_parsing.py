"""LLMJudge: retry behaviour, strict parsing and pairwise verdict bookkeeping."""

import json
from unittest.mock import MagicMock

import pytest

from prompt_opt.ingestion.schema import ConversationRecord, Message, MessageRole
from prompt_opt.judge.judge import LLMJudge
from prompt_opt.labeling.rubric import DEFAULT_RUBRIC
from prompt_opt.optimizer.budget import BudgetExceededError


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)  # tenacity waits via time.sleep


CONV = ConversationRecord(
    conversation_id="c1", messages=[Message(role=MessageRole.USER, content="hi")]
)


def _dims(score):
    return {d.name: {"score": score, "rationale": "r"} for d in DEFAULT_RUBRIC.dimensions}


def absolute_json(score=4):
    return json.dumps({"dimension_scores": _dims(score), "rationale": "ok"})


def pairwise_json(first, second, preferred):
    return json.dumps(
        {
            "dimension_scores": _dims(first),
            "dimension_scores_b": _dims(second),
            "preferred": preferred,
            "rationale": "because",
        }
    )


def _judge(*outputs):
    judge = LLMJudge(model="test/model")
    judge._call_llm = MagicMock(side_effect=list(outputs))
    return judge


class _HttpError(Exception):
    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


# ─── retries ─────────────────────────────────────────────────────────────────


def test_unparsable_output_is_retried():
    judge = _judge("not json at all", absolute_json())
    result = judge.score_absolute(CONV, "answer")
    assert judge._call_llm.call_count == 2
    assert 0.0 < result.aggregate_score <= 1.0


def test_structurally_invalid_output_is_retried():
    missing_dimension = json.dumps({"dimension_scores": {"correctness": {"score": 4}}})
    judge = _judge(missing_dimension, absolute_json())
    judge.score_absolute(CONV, "answer")
    assert judge._call_llm.call_count == 2


def test_transient_api_error_is_retried():
    judge = _judge(RuntimeError("connection reset"), absolute_json())
    judge.score_absolute(CONV, "answer")
    assert judge._call_llm.call_count == 2


def test_persistent_failure_raises_the_original_error_after_three_attempts():
    judge = _judge("garbage", "garbage", "garbage")
    with pytest.raises(ValueError):  # not tenacity.RetryError
        judge.score_absolute(CONV, "answer")
    assert judge._call_llm.call_count == 3


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_permanent_api_errors_are_not_retried(status):
    judge = _judge(_HttpError(status), absolute_json())
    with pytest.raises(_HttpError):
        judge.score_absolute(CONV, "answer")
    assert judge._call_llm.call_count == 1


def test_budget_exhaustion_is_not_retried():
    judge = _judge(BudgetExceededError("cap"), absolute_json())
    with pytest.raises(BudgetExceededError):
        judge.score_absolute(CONV, "answer")
    assert judge._call_llm.call_count == 1


# ─── strict parsing ──────────────────────────────────────────────────────────


def test_out_of_range_scores_are_clamped():
    judge = _judge(absolute_json(score=9))
    result = judge.score_absolute(CONV, "answer")
    assert all(ds.score == ds.max_score for ds in result.dimension_scores)
    assert result.aggregate_score == pytest.approx(1.0)


def test_bare_numeric_dimension_scores_are_accepted():
    payload = json.dumps({"dimension_scores": {d.name: 4 for d in DEFAULT_RUBRIC.dimensions}})
    result = _judge(payload).score_absolute(CONV, "answer")
    assert result.aggregate_score == pytest.approx(0.8)


def test_missing_second_response_scores_is_an_error_not_a_copy_of_the_first():
    only_first = json.dumps({"dimension_scores": _dims(4), "preferred": "A"})
    judge = _judge(only_first, only_first, only_first)
    with pytest.raises(ValueError, match="dimension_scores_b"):
        judge.score_pairwise(CONV, "X", "Y")


# ─── pairwise ordering / verdicts ────────────────────────────────────────────


def test_pairwise_maps_the_swapped_pass_back_to_original_labels():
    # Pass 1 lists X first; pass 2 lists Y first. X is better in both.
    judge = _judge(
        pairwise_json(first=4, second=2, preferred="A"),
        pairwise_json(first=2, second=4, preferred="B"),
    )
    result = judge.score_pairwise(CONV, "X", "Y")

    assert result.aggregate_score_a == pytest.approx(0.8)  # X scored 4 in both passes
    assert result.aggregate_score_b == pytest.approx(0.4)
    assert result.preferred == "A"
    assert (result.verdict_ab, result.verdict_ba) == ("A", "A")  # consistent


def test_pairwise_exposes_a_position_biased_judges_verdicts():
    # The judge always favours whatever is listed first.
    judge = _judge(
        pairwise_json(first=4, second=2, preferred="A"),
        pairwise_json(first=4, second=2, preferred="A"),
    )
    result = judge.score_pairwise(CONV, "X", "Y")

    assert result.preferred == "tie"  # the averaged scores cancel out
    assert (result.verdict_ab, result.verdict_ba) == ("A", "B")  # ...but the verdicts disagree
