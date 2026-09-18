"""Tests for PromptRegistry content-hash lookup and rollback succession semantics."""

import pytest

from prompt_opt.registry.prompt_registry import PromptRegistry


def _registry() -> PromptRegistry:
    return PromptRegistry(db_url="sqlite:///:memory:")


def test_get_by_content_hash():
    reg = _registry()
    vid = reg.register(system_prompt="Hello prompt")
    version = reg.get(vid)
    assert version is not None

    found = reg.get_by_content_hash(version.content_hash)
    assert found is not None
    assert found.version_id == vid
    assert reg.get_by_content_hash("nonexistent") is None


def test_rollback_chain_semantics():
    reg = _registry()
    v1 = reg.register(system_prompt="V1")
    v2 = reg.register(system_prompt="V2")

    # No champion yet
    assert reg.rollback() is None

    reg.promote(v1)
    # v1 is the first champion — no predecessor to roll back to
    assert reg.rollback() is None

    reg.promote(v2)
    assert reg.rollback() == v1
    assert reg.get_champion().version_id == v1

    # Rollback does not ping-pong: v1 was promoted first, no earlier version exists
    assert reg.rollback() is None


def test_rollback_after_repromote():
    reg = _registry()
    ids = [reg.register(system_prompt=f"V{i}") for i in range(1, 4)]
    v1, v2, v3 = ids

    reg.promote(v1)
    reg.promote(v2)
    reg.promote(v3)

    assert reg.rollback() == v2

    # Re-promote v3 and roll back again — chain still intact despite the rollback event
    reg.promote(v3)
    assert reg.get_champion().version_id == v3
    assert reg.rollback() == v2


def test_rollback_never_returns_to_a_rolled_back_version():
    reg = _registry()
    a = reg.register(system_prompt="A")
    b = reg.register(system_prompt="B")
    c = reg.register(system_prompt="C")

    reg.promote(a)
    reg.promote(b)
    assert reg.rollback() == a  # B rejected

    reg.promote(c)
    # B was already rolled back from; the predecessor of C is A, not B
    assert reg.rollback() == a
    assert reg.get_champion().version_id == a
    assert reg.rollback() is None


def test_rollback_chain_after_repeated_rollbacks_and_promotions():
    reg = _registry()
    v1, v2, v3, v4 = (reg.register(system_prompt=f"V{i}") for i in range(1, 5))

    reg.promote(v1)
    reg.promote(v2)
    reg.promote(v3)
    assert reg.rollback() == v2  # v3 rejected
    assert reg.rollback() == v1  # v2 rejected too
    reg.promote(v4)
    assert reg.rollback() == v1  # neither v2 nor v3 come back


def test_register_existing_hash_keeps_gate_result():
    reg = _registry()
    vid = reg.register(system_prompt="Z", eval_scores={"mean_score": 0.9})

    again = reg.register(
        system_prompt="Z",
        eval_scores={"mean_score": 0.1},
        gate_result_dict={"decision": "PASS"},
    )

    assert again == vid
    version = reg.get(vid)
    assert version.gate_result == {"decision": "PASS"}
    # First-stored eval scores are not overwritten
    assert version.eval_scores == {"mean_score": 0.9}


def test_register_existing_hash_fills_missing_eval_scores():
    reg = _registry()
    vid = reg.register(system_prompt="Z")
    reg.register(system_prompt="Z", eval_scores={"mean_score": 0.7})
    assert reg.get(vid).eval_scores == {"mean_score": 0.7}


def test_attach_gate_result():
    reg = _registry()
    vid = reg.register(system_prompt="P")
    assert reg.get(vid).gate_result == {}

    reg.attach_gate_result(vid, {"decision": "PENDING_HUMAN"})
    assert reg.get(vid).gate_result == {"decision": "PENDING_HUMAN"}
    reg.attach_gate_result(vid, {"decision": "PASS"})
    assert reg.get(vid).gate_result["decision"] == "PASS"

    with pytest.raises(KeyError):
        reg.attach_gate_result("prompt_v999", {})


def test_champion_history_audit_log():
    reg = _registry()
    v1 = reg.register(system_prompt="V1")
    v2 = reg.register(system_prompt="V2")

    reg.promote(v1, approver="alice")
    reg.promote(v2, approver="bob")
    reg.rollback(actor="monitor")

    history = reg.champion_history()
    actions = [(h["action"], h["version_id"]) for h in history]

    # Newest first
    assert actions[0] == ("rollback", v1)
    assert ("promote", v2) in actions
    assert ("promote", v1) in actions

    promote_events = [h for h in history if h["action"] == "promote"]
    actors = {h["actor"] for h in promote_events}
    assert actors == {"alice", "bob"}
