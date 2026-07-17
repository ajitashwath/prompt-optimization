"""
Eval harness: replay user turns with a candidate system prompt and score responses.

Flow for each conversation:
  1. Replace the system message with the candidate system prompt.
  2. Replay all user turns using the configured optimizee model.
  3. Score the final assistant response with the calibrated judge.
  4. Compute guardrail metrics.
  5. Cache results.

Supports optional champion_responses baseline for paired comparison.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Literal

from prompt_opt.config import settings
from prompt_opt.eval.cache import EvalCache
from prompt_opt.eval.guardrails import GuardrailChecker, GuardrailResult, RegressionCase
from prompt_opt.eval.metrics import EvalSummary, PerExampleResult, compute_eval_summary, _outcome
from prompt_opt.ingestion.schema import ConversationRecord, Message, MessageRole
from prompt_opt.judge.judge import LLMJudge

logger = logging.getLogger(__name__)


# ─── PromptCandidate ─────────────────────────────────────────────────────────


@dataclass
class PromptCandidate:
    """A candidate system prompt with optional few-shot demonstrations."""

    system_prompt: str
    few_shot_demos: list[dict] | None = None  # list of {"user": ..., "assistant": ...} dicts
    metadata: dict | None = None

    @property
    def content_hash(self) -> str:
        payload = json.dumps(
            {"system": self.system_prompt, "demos": self.few_shot_demos or []},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ─── LLM response generator ─────────────────────────────────────────────────


def _replay_conversation(
    conv: ConversationRecord,
    system_prompt: str,
    few_shot_demos: list[dict] | None,
    model: str,
) -> str:
    """
    Replay a conversation with a new system prompt and return the final response.

    Inserts few-shot demos (if any) after the system message before the first user turn.
    """
    messages: list[dict] = [{"role": "system", "content": system_prompt}]

    # Insert few-shot demonstrations
    if few_shot_demos:
        for demo in few_shot_demos:
            messages.append({"role": "user", "content": demo["user"]})
            messages.append({"role": "assistant", "content": demo["assistant"]})

    # Append actual conversation (without its original system message)
    for msg in conv.without_system_prompt():
        if msg.role in (MessageRole.USER, MessageRole.ASSISTANT):
            messages.append({"role": msg.role.value, "content": msg.content or ""})

    # Remove trailing assistant message if any (we're generating a new one)
    if messages and messages[-1]["role"] == "assistant":
        messages.pop()

    try:
        import litellm  # type: ignore[import]

        resp = litellm.completion(
            model=model,
            messages=messages,
            temperature=0.0,
        )
        return resp.choices[0].message.content or ""
    except Exception as exc:
        logger.warning("LLM call failed for conv %s: %s", conv.conversation_id, exc)
        return "[ERROR: model call failed]"


# ─── EvalHarness ─────────────────────────────────────────────────────────────


class EvalHarness:
    """
    Score a PromptCandidate against a dataset split using the calibrated judge.

    The harness is the single authoritative scorer used by:
      - The optimizer's internal search (dev split)
      - The champion/challenger gate (test split)

    Args:
        judge:              Calibrated LLMJudge instance.
        judge_version:      Version string for cache keying.
        cache:              EvalCache instance (shared across optimizer runs).
        guardrail_checker:  GuardrailChecker instance.
        optimizee_model:    Model being optimized (for replaying responses).
    """

    def __init__(
        self,
        judge: LLMJudge,
        judge_version: str,
        cache: EvalCache | None = None,
        guardrail_checker: GuardrailChecker | None = None,
        optimizee_model: str | None = None,
    ):
        self.judge = judge
        self.judge_version = judge_version
        self.cache = cache or EvalCache()
        self.guardrail_checker = guardrail_checker or GuardrailChecker()
        self.optimizee_model = optimizee_model or settings.optimizee_model

    def evaluate(
        self,
        candidate: PromptCandidate,
        conversations: list[ConversationRecord],
        *,
        split: str = "dev",
        champion_responses: dict[str, str] | None = None,
        regression_data: list[tuple[RegressionCase, str]] | None = None,
        sample_size: int | None = None,
    ) -> EvalSummary:
        """
        Score *candidate* against *conversations*.

        Args:
            candidate:           Prompt candidate to evaluate.
            conversations:       List of ConversationRecord objects for this split.
            split:               Dataset split name (for logging/reporting).
            champion_responses:  Map from conversation_id → champion's response text,
                                 for paired comparison. Pass None for absolute scoring.
            regression_data:     (RegressionCase, response_text) pairs from regression suite.
            sample_size:         If set, randomly sample this many conversations (minibatch mode).
        """
        if sample_size and sample_size < len(conversations):
            import random
            conversations = random.sample(conversations, sample_size)

        logger.info(
            "Evaluating prompt %s on %d %s examples …",
            candidate.content_hash,
            len(conversations),
            split,
        )

        per_example: list[PerExampleResult] = []
        challenger_texts: list[str] = []
        champion_texts: list[str] = []

        for conv in conversations:
            cid = conv.conversation_id
            cached = self.cache.get(
                candidate.content_hash, cid, self.judge_version, self.optimizee_model
            )

            if cached:
                challenger_score = cached.score
                challenger_text = cached.score_breakdown.get("response_text", "")
                rationale = cached.rationale
            else:
                challenger_text = _replay_conversation(
                    conv,
                    candidate.system_prompt,
                    candidate.few_shot_demos,
                    self.optimizee_model,
                )
                # Score (absolute mode here; pairwise handled separately if champion provided)
                judge_result = self.judge.score_absolute(conv, challenger_text)
                challenger_score = judge_result.aggregate_score
                rationale = judge_result.rationale

                self.cache.put(
                    candidate.content_hash,
                    cid,
                    self.judge_version,
                    self.optimizee_model,
                    challenger_score,
                    score_breakdown={"response_text": challenger_text, **judge_result.scores_dict()},
                    rationale=rationale,
                )

            challenger_texts.append(challenger_text)

            champ_score: float | None = None
            champ_text: str | None = None
            outcome = None

            if champion_responses and cid in champion_responses:
                champ_text = champion_responses[cid]
                champion_texts.append(champ_text)
                # Pairwise score for paired comparison
                pair_result = self.judge.score_pairwise(conv, challenger_text, champ_text)
                challenger_score = pair_result.aggregate_score_a
                champ_score = pair_result.aggregate_score_b
                outcome = _outcome(challenger_score, champ_score)

            per_example.append(
                PerExampleResult(
                    example_id=cid,
                    category=conv.category,
                    challenger_score=challenger_score,
                    champion_score=champ_score,
                    outcome=outcome,
                    judge_rationale=rationale,
                )
            )

        # Guardrail check
        guardrail_result = self.guardrail_checker.check(
            challenger_responses=challenger_texts,
            champion_responses=champion_texts if champion_texts else None,
            regression_responses=regression_data,
        )

        return compute_eval_summary(
            prompt_hash=candidate.content_hash,
            split=split,
            judge_version=self.judge_version,
            per_example=per_example,
            guardrail_result=guardrail_result,
        )
