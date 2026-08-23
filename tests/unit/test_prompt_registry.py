"""Tests for PromptRegistry content-hash lookup and rollback succession semantics."""

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
