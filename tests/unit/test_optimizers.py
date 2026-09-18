"""Optimizer contract: fair comparison against the initial prompt, real metering, no dupes."""

import pytest

from prompt_opt import llm
from prompt_opt.eval.harness import EvaluationError, PromptCandidate
from prompt_opt.eval.metrics import PerExampleResult, compute_eval_summary
from prompt_opt.ingestion.schema import ConversationRecord, Message, MessageRole
from prompt_opt.optimizer import gepa as gepa_mod
from prompt_opt.optimizer import miprov2 as mipro_mod
from prompt_opt.optimizer.base import OptimizationError
from prompt_opt.optimizer.budget import BudgetController, BudgetExceededError
from prompt_opt.optimizer.gepa import GEPA
from prompt_opt.optimizer.miprov2 import MIPROv2

INITIAL = PromptCandidate(system_prompt="initial")


def _conv(cid, text=None):
    return ConversationRecord(
        conversation_id=cid,
        messages=[
            Message(role=MessageRole.USER, content=text or f"question {cid}"),
            Message(role=MessageRole.ASSISTANT, content=f"historical reply {cid}"),
        ],
        category="general",
    )


DEV = [_conv(f"d{i}") for i in range(12)]
TRAIN = [_conv(f"t{i}") for i in range(12)]


class FakeHarness:
    """Scores candidates by system prompt; records every evaluate() call."""

    judge_version = "fake"

    def __init__(self, scores, guardrail_fail=(), fail_from_call=None, budget_error_on=None):
        self.scores = scores
        self.guardrail_fail = set(guardrail_fail)
        self.fail_from_call = fail_from_call      # raise EvaluationError from this call on
        self.budget_error_on = budget_error_on    # raise BudgetExceededError on this call
        self.calls = []
        self.active_budgets = []

    def evaluate(self, candidate, convs, *, split="dev", **kwargs):
        self.calls.append(
            {"cand": candidate, "ids": [c.conversation_id for c in convs], "split": split, **kwargs}
        )
        self.active_budgets.append(llm._active_budget.get())
        n = len(self.calls)
        if self.budget_error_on == n:
            raise BudgetExceededError("cap")
        if self.fail_from_call is not None and n >= self.fail_from_call:
            raise EvaluationError("too many failures")
        score = self.scores.get(candidate.system_prompt, 0.4)
        per_example = [
            PerExampleResult(
                example_id=c.conversation_id,
                category="general",
                challenger_score=score,
                champion_score=None,
                outcome=None,
                judge_rationale="rationale",
                response_text=f"GENERATED for {c.conversation_id}",
            )
            for c in convs
        ]
        summary = compute_eval_summary(
            prompt_hash=candidate.content_hash,
            split=split,
            judge_version="fake",
            per_example=per_example,
        )
        summary.guardrail_passed = candidate.system_prompt not in self.guardrail_fail
        return summary

    def run_regression(self, candidate, cases):
        return [(case, "ok") for case in cases]


@pytest.fixture
def mipro_proposals(monkeypatch):
    def install(variants):
        monkeypatch.setattr(mipro_mod, "_call_proposer", lambda system, user, model=None: variants)

    return install


def _budget():
    return BudgetController(max_llm_calls=10_000, max_usd_spend=1_000.0)


def _mipro(**kwargs):
    kwargs.setdefault("n_trials", 12)
    kwargs.setdefault("minibatch_size", 4)
    kwargs.setdefault("n_demo_subsets", 2)
    return MIPROv2(**kwargs)


# ─── MIPROv2 ─────────────────────────────────────────────────────────────────


def test_mipro_returns_the_initial_prompt_when_nothing_beats_it(mipro_proposals):
    mipro_proposals(["worse a", "worse b"])
    harness = FakeHarness({"initial": 0.7, "worse a": 0.5, "worse b": 0.6})

    results = _mipro().optimize(INITIAL, TRAIN, DEV, harness, _budget())

    best, summary = results[0]
    assert best.content_hash == INITIAL.content_hash
    assert summary.n_examples == len(DEV)


def test_mipro_ties_go_to_the_initial_prompt(mipro_proposals):
    mipro_proposals(["same score"])
    harness = FakeHarness({"initial": 0.7, "same score": 0.7})

    best, _ = _mipro().optimize(INITIAL, TRAIN, DEV, harness, _budget())[0]

    assert best.system_prompt == "initial"


def test_mipro_a_better_candidate_wins_on_the_full_dev_split(mipro_proposals):
    mipro_proposals(["better", "worse"])
    harness = FakeHarness({"initial": 0.5, "better": 0.9, "worse": 0.3})

    results = _mipro().optimize(INITIAL, TRAIN, DEV, harness, _budget())

    assert results[0][0].system_prompt == "better"
    assert INITIAL.content_hash in {c.content_hash for c, _ in results}
    # Every returned candidate was scored on the full dev split, i.e. under the same conditions
    assert all(s.n_examples == len(DEV) for _, s in results)


def test_mipro_trials_share_one_fixed_minibatch(mipro_proposals):
    mipro_proposals(["a", "b", "c"])
    harness = FakeHarness({"initial": 0.5})

    _mipro(minibatch_size=4).optimize(INITIAL, TRAIN, DEV, harness, _budget())

    dev_ids = {c.conversation_id for c in DEV}
    minibatch_calls = [
        tuple(call["ids"]) for call in harness.calls
        if call["split"] == "dev" and len(call["ids"]) == 4
    ]
    assert len(minibatch_calls) >= 3  # baseline + trials
    assert len(set(minibatch_calls)) == 1  # ... all on exactly the same examples
    assert set(minibatch_calls[0]) <= dev_ids


def test_mipro_guardrails_are_checked_against_the_initial_prompts_responses(mipro_proposals):
    mipro_proposals(["variant"])
    harness = FakeHarness({"initial": 0.5, "variant": 0.6})

    _mipro().optimize(INITIAL, TRAIN, DEV, harness, _budget())

    candidate_calls = [c for c in harness.calls if c["cand"].system_prompt == "variant"]
    assert candidate_calls
    for call in candidate_calls:
        baseline = call["guardrail_baseline_responses"]
        assert baseline and all(text.startswith("GENERATED") for text in baseline.values())
    initial_dev_call = next(
        c for c in harness.calls if c["cand"].system_prompt == "initial" and c["split"] == "dev"
    )
    assert initial_dev_call.get("guardrail_baseline_responses") is None  # nothing to compare to


def test_mipro_a_guardrail_failing_candidate_never_beats_a_passing_one(mipro_proposals):
    mipro_proposals(["risky", "safe"])
    harness = FakeHarness(
        {"initial": 0.4, "risky": 0.95, "safe": 0.7}, guardrail_fail={"risky"}
    )

    results = _mipro().optimize(INITIAL, TRAIN, DEV, harness, _budget())

    assert results[0][0].system_prompt == "safe"
    assert results[-1][0].system_prompt == "risky"


def test_mipro_regression_suite_is_replayed_for_each_finalist(mipro_proposals):
    mipro_proposals(["variant"])
    harness = FakeHarness({"initial": 0.5, "variant": 0.6})
    seen = []
    original = harness.run_regression

    def recording_regression(cand, cases):
        seen.append(cand.system_prompt)
        return original(cand, cases)

    harness.run_regression = recording_regression

    _mipro(regression_cases=["case"]).optimize(INITIAL, TRAIN, DEV, harness, _budget())

    assert "initial" in seen and "variant" in seen
    regression_evals = [c for c in harness.calls if c.get("regression_data")]
    assert regression_evals  # passed through to the harness for the guardrail check


def test_mipro_demos_are_generated_responses_not_log_replies(mipro_proposals):
    mipro_proposals([])
    harness = FakeHarness({"initial": 0.9})  # scores above the demo threshold

    _mipro(n_demo_subsets=3).optimize(INITIAL, TRAIN, DEV, harness, _budget())

    demos = [d for c in harness.calls for d in (c["cand"].few_shot_demos or [])]
    assert demos, "expected bootstrapped demos to be tried"
    assert all(d["assistant"].startswith("GENERATED") for d in demos)
    assert not any("historical reply" in d["assistant"] for d in demos)


def test_mipro_keeps_the_initial_prompts_own_demos_in_the_search_space(mipro_proposals):
    mipro_proposals([])
    initial = PromptCandidate("initial", few_shot_demos=[{"user": "q", "assistant": "a"}])
    harness = FakeHarness({"initial": 0.5})

    results = _mipro().optimize(initial, TRAIN, DEV, harness, _budget())

    assert results[0][0].content_hash == initial.content_hash  # demos were not silently dropped


def test_mipro_every_call_is_metered_against_the_budget(mipro_proposals):
    mipro_proposals(["variant"])
    harness = FakeHarness({"initial": 0.5})
    budget = _budget()

    _mipro().optimize(INITIAL, TRAIN, DEV, harness, budget)

    assert harness.active_budgets and all(b is budget for b in harness.active_budgets)
    assert llm._active_budget.get() is None  # and tracking ends with the run


def test_mipro_budget_exhausted_before_the_baseline_returns_an_empty_evaluation(mipro_proposals):
    mipro_proposals(["variant"])
    harness = FakeHarness({}, budget_error_on=1)

    results = _mipro().optimize(INITIAL, TRAIN, DEV, harness, _budget())

    assert results[0][0].content_hash == INITIAL.content_hash
    assert results[0][1].n_examples == 0


def test_mipro_raises_when_every_trial_fails(mipro_proposals):
    mipro_proposals(["variant"])
    # call 1 = baseline, call 2 = demo bootstrap; every trial after that fails
    harness = FakeHarness({"initial": 0.5}, fail_from_call=3)

    with pytest.raises(OptimizationError, match="All .* search trials failed"):
        _mipro().optimize(INITIAL, TRAIN, DEV, harness, _budget())


def test_mipro_top_k_finalists_are_distinct(mipro_proposals):
    mipro_proposals(["only variant"])  # tiny search space -> trials revisit the same candidates
    harness = FakeHarness({"initial": 0.5, "only variant": 0.9})

    results = _mipro(n_trials=15, top_k=3).optimize(INITIAL, TRAIN, DEV, harness, _budget())

    hashes = [c.content_hash for c, _ in results]
    assert len(hashes) == len(set(hashes))


# ─── GEPA ────────────────────────────────────────────────────────────────────


@pytest.fixture
def gepa_proposals(monkeypatch):
    counter = {"n": 0}

    def install(variants, mutant_scores=None):
        def text(system, user, model=None):
            counter["n"] += 1
            return f"mutant {counter['n']}"

        monkeypatch.setattr(
            gepa_mod, "_call_proposer_json_array", lambda s, u, model=None: variants
        )
        monkeypatch.setattr(gepa_mod, "_call_proposer_text", text)

    return install


def _gepa(**kwargs):
    kwargs.setdefault("max_generations", 4)
    kwargs.setdefault("minibatch_size", 4)
    kwargs.setdefault("stagnation_limit", 99)
    kwargs.setdefault("crossover_prob", 0.0)
    return GEPA(**kwargs)


def test_gepa_results_contain_no_duplicate_candidates(gepa_proposals):
    gepa_proposals(["v1", "v2"])
    # The seed stays strongest, so it sits on the Pareto front through every generation
    harness = FakeHarness({"initial": 0.9, "v1": 0.5, "v2": 0.4})

    results = _gepa().optimize(INITIAL, TRAIN, DEV, harness, _budget())

    hashes = [c.content_hash for c, _ in results]
    assert len(hashes) == len(set(hashes))
    assert hashes[0] == INITIAL.content_hash


def test_gepa_final_candidates_are_scored_on_the_full_dev_split(gepa_proposals):
    gepa_proposals(["v1"])
    harness = FakeHarness({"initial": 0.5, "v1": 0.8, "mutant 1": 0.95})

    results = _gepa().optimize(INITIAL, TRAIN, DEV, harness, _budget())

    assert results[0][0].system_prompt == "mutant 1"
    assert all(s.n_examples == len(DEV) for _, s in results)


def test_gepa_returns_the_initial_prompt_when_nothing_beats_it(gepa_proposals):
    gepa_proposals(["v1"])
    harness = FakeHarness({"initial": 0.8, "v1": 0.5})

    results = _gepa().optimize(INITIAL, TRAIN, DEV, harness, _budget())

    assert results[0][0].content_hash == INITIAL.content_hash


def test_gepa_reflects_on_train_but_selects_on_dev(gepa_proposals):
    gepa_proposals(["v1"])
    harness = FakeHarness({"initial": 0.5, "v1": 0.6})

    _gepa().optimize(INITIAL, TRAIN, DEV, harness, _budget())

    train_calls = [c for c in harness.calls if c["split"] == "train"]
    assert train_calls, "mutation must reflect on train-split failures"
    train_ids = {c.conversation_id for c in TRAIN}
    assert all(set(c["ids"]) <= train_ids for c in train_calls)
    dev_ids = {c.conversation_id for c in DEV}
    assert all(set(c["ids"]) <= dev_ids for c in harness.calls if c["split"] == "dev")


def test_gepa_candidates_share_one_dev_minibatch(gepa_proposals):
    gepa_proposals(["v1", "v2"])
    harness = FakeHarness({"initial": 0.5})

    _gepa(minibatch_size=4).optimize(INITIAL, TRAIN, DEV, harness, _budget())

    batches = {
        tuple(c["ids"]) for c in harness.calls if c["split"] == "dev" and len(c["ids"]) == 4
    }
    assert len(batches) == 1


def test_gepa_never_reevaluates_the_same_prompt(gepa_proposals, monkeypatch):
    gepa_proposals(["v1", "v1", "initial"])  # duplicate / identical-to-seed variants
    harness = FakeHarness({"initial": 0.5, "v1": 0.6})
    monkeypatch.setattr(gepa_mod, "_call_proposer_text", lambda s, u, model=None: "v1")

    _gepa().optimize(INITIAL, TRAIN, DEV, harness, _budget())

    minibatch_evals = [
        c["cand"].content_hash for c in harness.calls
        if c["split"] == "dev" and len(c["ids"]) == 4
    ]
    assert len(minibatch_evals) == len(set(minibatch_evals))


def test_gepa_budget_exhausted_before_the_baseline(gepa_proposals):
    gepa_proposals(["v1"])
    harness = FakeHarness({}, budget_error_on=1)

    results = _gepa().optimize(INITIAL, TRAIN, DEV, harness, _budget())

    assert results[0][0].content_hash == INITIAL.content_hash
    assert results[0][1].n_examples == 0


def test_gepa_uses_the_metered_proposer(gepa_proposals):
    gepa_proposals(["v1"])
    harness = FakeHarness({"initial": 0.5})
    budget = _budget()

    _gepa().optimize(INITIAL, TRAIN, DEV, harness, budget)

    assert all(b is budget for b in harness.active_budgets)
