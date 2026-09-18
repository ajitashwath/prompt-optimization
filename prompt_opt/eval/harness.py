"""
Eval harness: replay user turns with a candidate system prompt and score responses.

Flow for each conversation:
  1. Replace the system message with the candidate system prompt.
  2. Replay all user turns using the configured optimizee model.
  3. Score the final assistant response with the calibrated judge.
  4. Compute guardrail metrics.
  5. Cache results.

Supports optional champion_responses baseline for paired comparison.

Failure policy: a model or judge failure never becomes a score. The example is
excluded from the summary (and never cached); if too many examples fail the
evaluation raises `EvaluationError` instead of returning numbers that mostly
measure API outages. Budget exhaustion always propagates.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from dataclasses import dataclass

from prompt_opt import llm
from prompt_opt.config import settings
from prompt_opt.eval.cache import EvalCache
from prompt_opt.eval.guardrails import MODEL_CALL_FAILED, GuardrailChecker, RegressionCase
from prompt_opt.eval.metrics import EvalSummary, PerExampleResult, _outcome, compute_eval_summary
from prompt_opt.ingestion.schema import ConversationRecord, MessageRole
from prompt_opt.judge.judge import LLMJudge
from prompt_opt.optimizer.budget import BudgetExceededError

logger = logging.getLogger(__name__)


class ModelCallError(RuntimeError):
    """The optimizee model call failed (after litellm's own retries)."""


class EvaluationError(RuntimeError):
    """Too many examples failed for the evaluation to be meaningful."""


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


def build_messages(
    system_prompt: str,
    few_shot_demos: list[dict] | None,
    turns: list[dict],
) -> list[dict]:
    """System prompt, then few-shot demos as user/assistant turns, then the real turns."""
    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    for demo in few_shot_demos or []:
        messages.append({"role": "user", "content": demo["user"]})
        messages.append({"role": "assistant", "content": demo["assistant"]})
    messages.extend(turns)
    return messages


def _complete(model: str, messages: list[dict]) -> str:
    """One metered optimizee call. Raises ModelCallError on failure."""
    try:
        resp = llm.completion(model=model, messages=messages, temperature=0.0, num_retries=2)
        return resp.choices[0].message.content or ""
    except BudgetExceededError:
        raise
    except Exception as exc:
        raise ModelCallError(f"{type(exc).__name__}: {exc}") from exc


def replay_turns(conv: ConversationRecord) -> list[dict]:
    """
    The turns to send when replaying *conv* under a new prompt: user/assistant messages
    only (no system or tool messages), minus a trailing assistant reply, because that
    reply is exactly what the new prompt is asked to produce.
    """
    turns = [
        {"role": msg.role.value, "content": msg.content or ""}
        for msg in conv.without_system_prompt()
        if msg.role in (MessageRole.USER, MessageRole.ASSISTANT)
    ]
    if turns and turns[-1]["role"] == "assistant":
        turns.pop()
    return turns


def _replay_conversation(
    conv: ConversationRecord,
    system_prompt: str,
    few_shot_demos: list[dict] | None,
    model: str,
) -> str:
    """
    Replay a conversation with a new system prompt and return the final response.

    Inserts few-shot demos (if any) after the system message before the first user turn.
    Raises ModelCallError if the model call fails.
    """
    return _complete(model, build_messages(system_prompt, few_shot_demos, replay_turns(conv)))


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
        max_failure_rate:   Raise EvaluationError if more than this fraction of the
                            attempted examples fail (model/judge errors).
    """

    def __init__(
        self,
        judge: LLMJudge,
        judge_version: str,
        cache: EvalCache | None = None,
        guardrail_checker: GuardrailChecker | None = None,
        optimizee_model: str | None = None,
        max_failure_rate: float = 0.2,
    ):
        self.judge = judge
        self.judge_version = judge_version
        self.cache = cache or EvalCache()
        self.guardrail_checker = guardrail_checker or GuardrailChecker()
        self.optimizee_model = optimizee_model or settings.optimizee_model
        self.max_failure_rate = max_failure_rate

    def run_regression(
        self, candidate: PromptCandidate, cases: list[RegressionCase]
    ) -> list[tuple[RegressionCase, str]]:
        """
        Replay each regression case through the candidate (system prompt AND few-shot
        demos, exactly as `evaluate` does). A failed call yields MODEL_CALL_FAILED,
        which always fails its case.
        """
        pairs: list[tuple[RegressionCase, str]] = []
        for case in cases:
            turns = [
                {"role": m.get("role", "user"), "content": m.get("content") or ""}
                for m in case.messages
                if m.get("role") != "system"
            ]
            messages = build_messages(candidate.system_prompt, candidate.few_shot_demos, turns)
            try:
                text = _complete(self.optimizee_model, messages)
            except ModelCallError as exc:
                logger.warning("Regression call failed for %s: %s", case.conversation_id, exc)
                text = MODEL_CALL_FAILED
            pairs.append((case, text))
        return pairs

    def evaluate(
        self,
        candidate: PromptCandidate,
        conversations: list[ConversationRecord],
        *,
        split: str = "dev",
        champion_responses: dict[str, str] | None = None,
        regression_data: list[tuple[RegressionCase, str]] | None = None,
        sample_size: int | None = None,
        sample_seed: int | None = None,
        guardrail_baseline_responses: dict[str, str] | None = None,
    ) -> EvalSummary:
        """
        Score *candidate* against *conversations*.

        Args:
            candidate:           Prompt candidate to evaluate.
            conversations:       List of ConversationRecord objects for this split.
            split:               Dataset split name (for logging/reporting).
            champion_responses:  Map from conversation_id → champion's response text,
                                 for paired comparison. Pass None for absolute scoring.
                                 In paired mode, conversations without a champion
                                 response are skipped.
            regression_data:     (RegressionCase, response_text) pairs from regression suite.
            sample_size:         If set, randomly sample this many conversations
                                 (minibatch mode); `sample_seed` makes it reproducible.
            guardrail_baseline_responses:
                                 Map from conversation_id → baseline response text used
                                 ONLY for the guardrail comparison (refusal/length/token
                                 drift) without switching to paired judging.

        Raises:
            EvaluationError:     more than `max_failure_rate` of the attempted examples
                                 failed.
            BudgetExceededError: an active budget cap was hit.
        """
        if sample_size and sample_size < len(conversations):
            conversations = random.Random(sample_seed).sample(conversations, sample_size)

        logger.info(
            "Evaluating prompt %s on %d %s examples …",
            candidate.content_hash,
            len(conversations),
            split,
        )

        paired = bool(champion_responses)
        per_example: list[PerExampleResult] = []
        challenger_texts: list[str] = []
        champion_texts: list[str] = []
        failed_ids: list[str] = []
        last_error: Exception | None = None

        for conv in conversations:
            cid = conv.conversation_id
            if paired and cid not in champion_responses:  # type: ignore[operator]
                logger.debug("Skipping %s: no champion response to pair with", cid)
                continue

            try:
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
                    judge_result = self.judge.score_absolute(conv, challenger_text)
                    challenger_score = judge_result.aggregate_score
                    rationale = judge_result.rationale

                    # Only successfully generated AND judged results reach the cache.
                    self.cache.put(
                        candidate.content_hash,
                        cid,
                        self.judge_version,
                        self.optimizee_model,
                        challenger_score,
                        score_breakdown={
                            "response_text": challenger_text,
                            **judge_result.scores_dict(),
                        },
                        rationale=rationale,
                    )

                champ_score: float | None = None
                champ_text: str | None = None
                outcome = None

                if paired:
                    champ_text = champion_responses[cid]  # type: ignore[index]
                    pair_result = self.judge.score_pairwise(conv, challenger_text, champ_text)
                    challenger_score = pair_result.aggregate_score_a
                    champ_score = pair_result.aggregate_score_b
                    outcome = _outcome(challenger_score, champ_score)
            except BudgetExceededError:
                raise
            except Exception as exc:  # noqa: BLE001 - model/judge failure: exclude, don't score
                failed_ids.append(cid)
                last_error = exc
                logger.warning("Example %s excluded from evaluation: %s", cid, exc)
                continue

            challenger_texts.append(challenger_text)
            if champ_text is not None:
                champion_texts.append(champ_text)
            per_example.append(
                PerExampleResult(
                    example_id=cid,
                    category=conv.category,
                    challenger_score=challenger_score,
                    champion_score=champ_score,
                    outcome=outcome,
                    judge_rationale=rationale,
                    response_text=challenger_text,
                    champion_response_text=champ_text,
                )
            )

        attempted = len(per_example) + len(failed_ids)
        if failed_ids and len(failed_ids) / attempted > self.max_failure_rate:
            raise EvaluationError(
                f"{len(failed_ids)}/{attempted} examples failed while evaluating prompt "
                f"{candidate.content_hash} on the {split} split (limit "
                f"{self.max_failure_rate:.0%}); last error: {last_error}"
            ) from last_error

        # Guardrail baseline: paired champion responses, else the explicit baseline map
        baseline_texts: list[str] | None = champion_texts or None
        if baseline_texts is None and guardrail_baseline_responses:
            baseline_texts = [
                guardrail_baseline_responses[e.example_id]
                for e in per_example
                if e.example_id in guardrail_baseline_responses
            ] or None

        guardrail_result = self.guardrail_checker.check(
            challenger_responses=challenger_texts,
            champion_responses=baseline_texts,
            regression_responses=regression_data,
        )

        summary = compute_eval_summary(
            prompt_hash=candidate.content_hash,
            split=split,
            judge_version=self.judge_version,
            per_example=per_example,
            guardrail_result=guardrail_result,
        )
        summary.failed_example_ids = failed_ids
        return summary
