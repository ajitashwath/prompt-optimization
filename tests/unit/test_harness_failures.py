"""EvalHarness failure policy: failures are excluded, never scored and never cached."""

from types import SimpleNamespace

import pytest

from prompt_opt import llm
from prompt_opt.eval.cache import EvalCache
from prompt_opt.eval.guardrails import MODEL_CALL_FAILED, RegressionCase, check_regression_case
from prompt_opt.eval.harness import EvalHarness, EvaluationError, PromptCandidate
from prompt_opt.ingestion.schema import ConversationRecord, Message, MessageRole
from prompt_opt.judge.judge import JudgeResult, PairwiseJudgeResult
from prompt_opt.optimizer.budget import BudgetExceededError

MODEL = "test/optimizee"


def _conv(cid: str) -> ConversationRecord:
    return ConversationRecord(
        conversation_id=cid,
        messages=[Message(role=MessageRole.USER, content=f"question for {cid}")],
        category="general",
    )


class FakeJudge:
    def __init__(self, score=0.6, fail_on=()):
        self.score = score
        self.fail_on = set(fail_on)
        self.seen_responses: list[str] = []

    def score_absolute(self, conv, response):
        self.seen_responses.append(response)
        if conv.conversation_id in self.fail_on:
            raise ValueError("judge returned garbage")
        return JudgeResult(
            dimension_scores=[],
            aggregate_score=self.score,
            overall_verdict="score",
            rationale="fine",
        )

    def score_pairwise(self, conv, a, b):
        return PairwiseJudgeResult(
            result_ab=None,
            result_ba=None,
            preferred="A",
            aggregate_score_a=0.9,
            aggregate_score_b=0.4,
            rationale="a wins",
        )


def _completion(failing=(), text="a helpful answer"):
    """A stand-in for prompt_opt.llm.completion failing for selected conversations."""

    def fake(**kwargs):
        user = kwargs["messages"][-1]["content"]
        if any(cid in user for cid in failing):
            raise RuntimeError("429 rate limited")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
        )

    return fake


@pytest.fixture
def make_harness():
    def factory(judge=None, **kwargs):
        judge = judge or FakeJudge()
        return EvalHarness(
            judge,
            judge_version="jv",
            cache=EvalCache("sqlite:///:memory:"),
            optimizee_model=MODEL,
            **kwargs,
        )

    return factory


CAND = PromptCandidate(system_prompt="Be helpful.")


def test_failed_model_call_is_excluded_and_not_cached(monkeypatch, make_harness):
    monkeypatch.setattr(llm, "completion", _completion(failing=("bad",)))
    judge = FakeJudge()
    harness = make_harness(judge, max_failure_rate=0.5)
    convs = [_conv("ok1"), _conv("bad"), _conv("ok2"), _conv("ok3")]

    summary = harness.evaluate(CAND, convs)

    assert summary.n_examples == 3
    assert summary.failed_example_ids == ["bad"]
    assert summary.mean_score == pytest.approx(0.6)
    # The judge never saw an error string, and nothing was cached for the failed example
    assert not any("ERROR" in r for r in judge.seen_responses)
    assert harness.cache.get(CAND.content_hash, "bad", "jv", MODEL) is None
    assert harness.cache.get(CAND.content_hash, "ok1", "jv", MODEL) is not None


def test_a_transient_failure_does_not_poison_later_evaluations(monkeypatch, make_harness):
    harness = make_harness(max_failure_rate=1.0)
    convs = [_conv("c1"), _conv("c2")]

    monkeypatch.setattr(llm, "completion", _completion(failing=("c2",)))
    first = harness.evaluate(CAND, convs)
    assert first.n_examples == 1

    monkeypatch.setattr(llm, "completion", _completion())  # the API recovers
    second = harness.evaluate(CAND, convs)
    assert second.n_examples == 2
    assert second.failed_example_ids == []
    assert second.mean_score == pytest.approx(0.6)  # a real score, not a cached failure


def test_too_many_failures_raise_instead_of_returning_numbers(monkeypatch, make_harness):
    monkeypatch.setattr(llm, "completion", _completion(failing=("c",)))  # every conv fails
    harness = make_harness()  # default limit 20%
    with pytest.raises(EvaluationError, match="3/3 examples failed"):
        harness.evaluate(CAND, [_conv("c1"), _conv("c2"), _conv("c3")])
    assert harness.cache.get(CAND.content_hash, "c1", "jv", MODEL) is None


def test_judge_failure_excludes_the_example(monkeypatch, make_harness):
    monkeypatch.setattr(llm, "completion", _completion())
    harness = make_harness(FakeJudge(fail_on={"c2"}), max_failure_rate=0.5)

    summary = harness.evaluate(CAND, [_conv("c1"), _conv("c2"), _conv("c3")])

    assert summary.failed_example_ids == ["c2"]
    assert summary.n_examples == 2
    assert harness.cache.get(CAND.content_hash, "c2", "jv", MODEL) is None


def test_budget_exhaustion_propagates_and_is_not_a_failure(monkeypatch, make_harness):
    def boom(**kwargs):
        raise BudgetExceededError("cap reached")

    monkeypatch.setattr(llm, "completion", boom)
    harness = make_harness(max_failure_rate=1.0)  # even a permissive limit must not swallow it
    with pytest.raises(BudgetExceededError):
        harness.evaluate(CAND, [_conv("c1")])


def test_paired_mode_skips_examples_without_a_champion_response(monkeypatch, make_harness):
    monkeypatch.setattr(llm, "completion", _completion())
    harness = make_harness()
    convs = [_conv("c1"), _conv("c2"), _conv("c3")]

    summary = harness.evaluate(CAND, convs, champion_responses={"c1": "old", "c3": "old"})

    assert {e.example_id for e in summary.per_example} == {"c1", "c3"}
    assert all(e.outcome == "win" for e in summary.per_example)
    assert summary.failed_example_ids == []


def test_guardrail_baseline_makes_guardrails_active_without_paired_judging(
    monkeypatch, make_harness
):
    # The candidate now refuses everything; the baseline never did.
    monkeypatch.setattr(llm, "completion", _completion(text="I can't help with that."))
    harness = make_harness()
    convs = [_conv(f"c{i}") for i in range(10)]

    without = harness.evaluate(CAND, convs)
    with_baseline = harness.evaluate(
        PromptCandidate(system_prompt="Be terse."),
        convs,
        guardrail_baseline_responses={f"c{i}": "Sure, here you go." for i in range(10)},
    )

    assert without.guardrail_passed is True  # nothing to compare against
    assert with_baseline.guardrail_passed is False
    assert with_baseline.guardrail_details["refusal_rate_delta"] == pytest.approx(1.0)
    assert all(e.outcome is None for e in with_baseline.per_example)  # still absolute scoring


# ─── regression replay ───────────────────────────────────────────────────────


def test_run_regression_uses_demos_and_drops_case_system_messages(monkeypatch, make_harness):
    captured = []

    def fake(**kwargs):
        captured.append(kwargs["messages"])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="No."))])

    monkeypatch.setattr(llm, "completion", fake)
    cand = PromptCandidate(
        system_prompt="SYS", few_shot_demos=[{"user": "demo q", "assistant": "demo a"}]
    )
    case = RegressionCase(
        conversation_id="r1",
        expected_behavior="refuse",
        messages=[
            {"role": "system", "content": "case system"},
            {"role": "user", "content": "do something bad"},
        ],
    )

    pairs = make_harness().run_regression(cand, [case])

    assert pairs == [(case, "No.")]
    assert [m["role"] for m in captured[0]] == ["system", "user", "assistant", "user"]
    assert captured[0][0]["content"] == "SYS"
    assert captured[0][1]["content"] == "demo q"
    assert captured[0][-1]["content"] == "do something bad"


def test_failed_regression_call_always_fails_its_case(monkeypatch, make_harness):
    monkeypatch.setattr(llm, "completion", _completion(failing=("bad",)))
    case = RegressionCase(
        conversation_id="r1",
        expected_behavior="refuse",
        expected_keywords=["error"],  # would match an error string if one leaked through
        messages=[{"role": "user", "content": "bad request"}],
    )

    (_, text), = make_harness().run_regression(CAND, [case])

    assert text == MODEL_CALL_FAILED
    passed, reason = check_regression_case(case, text)
    assert passed is False
    assert "failed" in reason
