"""Budget metering of LLM calls (prompt_opt.llm) and real cost accounting."""

from unittest.mock import MagicMock

import pytest

from prompt_opt import llm
from prompt_opt.optimizer.budget import BudgetController, BudgetExceededError


def _response(prompt_tokens=1000, completion_tokens=500):
    resp = MagicMock()
    resp.usage.prompt_tokens = prompt_tokens
    resp.usage.completion_tokens = completion_tokens
    return resp


@pytest.fixture
def fake_litellm(monkeypatch):
    fake = MagicMock()
    fake.completion.return_value = _response()
    fake.completion_cost.side_effect = Exception("model not in pricing table")
    monkeypatch.setattr(llm, "get_litellm", lambda: fake)
    return fake


def test_calls_and_usage_are_recorded(fake_litellm):
    budget = BudgetController(max_llm_calls=10, max_usd_spend=100.0)
    with llm.track_budget(budget):
        llm.completion(model="openai/gpt-4o-mini", messages=[])
        llm.completion(model="openai/gpt-4o-mini", messages=[])

    assert budget.summary()["llm_calls"] == 2
    # gpt-4o-mini fallback rates: 1k in * 0.00015 + 0.5k out * 0.0006 = 0.00045 per call
    assert budget._usd_spend == pytest.approx(0.0009)


def test_litellm_cost_is_preferred_when_available(fake_litellm):
    fake_litellm.completion_cost.side_effect = None
    fake_litellm.completion_cost.return_value = 0.25
    budget = BudgetController(max_llm_calls=10, max_usd_spend=100.0)
    with llm.track_budget(budget):
        llm.completion(model="anything/unknown", messages=[])
    assert budget._usd_spend == pytest.approx(0.25)


def test_cap_is_enforced_before_the_call(fake_litellm):
    budget = BudgetController(max_llm_calls=1, max_usd_spend=100.0)
    with llm.track_budget(budget):
        llm.completion(model="m", messages=[])
        with pytest.raises(BudgetExceededError):
            llm.completion(model="m", messages=[])
    assert fake_litellm.completion.call_count == 1  # the second call never left


def test_usd_cap_stops_after_spend_is_reached(fake_litellm):
    fake_litellm.completion_cost.side_effect = None
    fake_litellm.completion_cost.return_value = 0.6
    budget = BudgetController(max_llm_calls=100, max_usd_spend=1.0)
    with llm.track_budget(budget):
        llm.completion(model="m", messages=[])
        llm.completion(model="m", messages=[])  # 1.2 spent now
        with pytest.raises(BudgetExceededError):
            llm.completion(model="m", messages=[])


def test_no_budget_means_no_metering(fake_litellm):
    llm.completion(model="m", messages=[])  # must not raise or need a budget
    assert fake_litellm.completion.call_count == 1


def test_tracking_context_is_reset(fake_litellm):
    budget = BudgetController(max_llm_calls=10, max_usd_spend=100.0)
    with llm.track_budget(budget):
        assert llm._active_budget.get() is budget
    assert llm._active_budget.get() is None


def test_responses_without_usage_still_count_as_a_call(monkeypatch):
    fake = MagicMock()
    fake.completion.return_value = MagicMock()  # usage attributes are MagicMocks, not ints
    fake.completion_cost.return_value = MagicMock()
    monkeypatch.setattr(llm, "get_litellm", lambda: fake)
    budget = BudgetController(max_llm_calls=10, max_usd_spend=100.0)
    with llm.track_budget(budget):
        llm.completion(model="unknown/model", messages=[])
    assert budget.summary()["llm_calls"] == 1
    assert budget._usd_spend == 0.0
