"""Tests for the champion/challenger gate's paired-comparison path."""

from unittest.mock import MagicMock

from prompt_opt.eval.harness import EvalHarness, PromptCandidate
from prompt_opt.eval.metrics import PerExampleResult, compute_eval_summary
from prompt_opt.gate.champion_gate import ChampionGate, GateCriteria
from prompt_opt.ingestion.schema import ConversationRecord, Message, MessageRole

N = 10


def _conv(i: int) -> ConversationRecord:
    return ConversationRecord(
        conversation_id=f"test_{i}",
        messages=[Message(role=MessageRole.USER, content=f"question {i}")],
        category="general",
    )


def _summary(cand: PromptCandidate, *, scores, champ_scores=None, outcomes=None,
             texts=None, champ_texts=None):
    per_example = []
    for i in range(N):
        cs = scores[i] if scores else 0.5
        ms = champ_scores[i] if champ_scores else None
        oc = outcomes[i] if outcomes else None
        rt = texts[i] if texts else f"resp_{i}"
        ct = champ_texts[i] if champ_texts else None
        per_example.append(
            PerExampleResult(
                example_id=f"test_{i}",
                category="general",
                challenger_score=cs,
                champion_score=ms,
                outcome=oc,
                judge_rationale="r",
                response_text=rt,
                champion_response_text=ct,
            )
        )
    return compute_eval_summary(
        prompt_hash=cand.content_hash,
        split="test",
        judge_version="v0",
        per_example=per_example,
    )


def test_gate_builds_paired_comparison_without_precomputed_responses():
    """When no champion_responses are passed, the gate must evaluate the champion
    first and feed its response texts into the challenger eval (paired mode)."""
    challenger = PromptCandidate(system_prompt="Challenger")
    champion = PromptCandidate(system_prompt="Champion")

    harness = MagicMock(spec=EvalHarness)

    def evaluate_side_effect(candidate, convs, **kwargs):
        if candidate is champion:
            return _summary(champion, scores=[0.5] * N, texts=[f"champ_{i}" for i in range(N)])
        assert kwargs.get("champion_responses") == {
            f"test_{i}": f"champ_{i}" for i in range(N)
        }, "gate must pass the champion's response texts for paired scoring"
        return _summary(
            challenger,
            scores=[0.9] * N,
            champ_scores=[0.5] * N,
            outcomes=["win"] * N,
            texts=[f"chal_{i}" for i in range(N)],
            champ_texts=[f"champ_{i}" for i in range(N)],
        )

    harness.evaluate.side_effect = evaluate_side_effect

    # 10 examples and no regression suite: fine for this wiring test, but the defaults
    # (>= 30 examples, suite must run) are relaxed explicitly rather than passed vacuously.
    criteria = GateCriteria(min_test_examples=5, require_regression_suite=False)
    gate = ChampionGate(harness, criteria=criteria)
    result = gate.evaluate(challenger, champion, [_conv(i) for i in range(N)])

    # Challenger won every paired comparison
    assert result.stat_test is not None
    assert result.stat_test.win_rate == 1.0
    assert result.stat_test.ci_excludes_null is True

    # All criteria pass → pending human spot-check (default criteria)
    assert not result.failure_reasons
    assert result.decision == "PENDING_HUMAN"


def test_gate_fails_when_challenger_loses():
    challenger = PromptCandidate(system_prompt="Challenger")
    champion = PromptCandidate(system_prompt="Champion")

    harness = MagicMock(spec=EvalHarness)
    harness.evaluate.return_value = _summary(
        challenger,
        scores=[0.3] * N,
        champ_scores=[0.8] * N,
        outcomes=["loss"] * N,
    )
    # First call returns this summary regardless; gate passes it as champion
    # responses map built from... nothing — pre-computed responses supplied:
    champ_map = {f"test_{i}": f"champ_{i}" for i in range(N)}
    gate = ChampionGate(harness)
    result = gate.evaluate(challenger, champion, [_conv(i) for i in range(N)],
                           champion_responses=champ_map)

    assert result.decision == "FAIL"
    assert any("Win rate" in r for r in result.failure_reasons)


def test_gate_result_record_human_review():
    from prompt_opt.gate.champion_gate import GateResult

    # Approve path: PENDING_HUMAN -> PASS
    result = GateResult(decision="PENDING_HUMAN", challenger_hash="a", champion_hash="b")
    returned = result.record_human_review(reviewer="alice", notes="ok", approved=True)
    assert returned is result
    assert result.decision == "PASS"
    assert result.human_approved is True
    assert result.human_reviewer == "alice"

    # Reject path: PENDING_HUMAN -> FAIL with recorded reason
    result = GateResult(decision="PENDING_HUMAN", challenger_hash="a", champion_hash="b")
    result.record_human_review(reviewer="bob", notes="risky", approved=False)
    assert result.decision == "FAIL"
    assert any("bob" in r for r in result.failure_reasons)

    # Recording review on a non-pending gate raises
    import pytest

    with pytest.raises(ValueError):
        result.record_human_review(reviewer="carol", approved=True)
