"""
LLM-as-Judge: scores candidate responses against a rubric.

Key design decisions:
  - Pairwise: runs A→B and B→A orderings, averages to cancel position bias.
  - Structured output: JSON with per-dimension scores + rationale.
  - Retries transient API errors and unparsable/invalid judge output (up to 3
    attempts); permanent errors (auth, bad request) and budget caps fail fast.
  - Separable from the model being optimized (use a different model family).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Literal

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from prompt_opt.config import settings
from prompt_opt.ingestion.schema import ConversationRecord
from prompt_opt.labeling.rubric import Rubric

logger = logging.getLogger(__name__)


# ─── Output types ────────────────────────────────────────────────────────────


@dataclass
class DimensionScore:
    dimension: str
    score: float       # 1 to rubric dimension scale
    max_score: float
    rationale: str = ""


@dataclass
class JudgeResult:
    """Result of a single judge call (one ordering)."""
    dimension_scores: list[DimensionScore]
    aggregate_score: float          # normalised 0–1
    overall_verdict: str            # "A" | "B" | "tie" (pairwise) or "score" (absolute)
    rationale: str                  # free-text explanation
    raw_response: str = ""

    def scores_dict(self) -> dict[str, float]:
        return {ds.dimension: ds.score for ds in self.dimension_scores}


@dataclass
class PairwiseJudgeResult:
    """Debiased result from running both A→B and B→A orderings."""
    result_ab: JudgeResult      # A listed first
    result_ba: JudgeResult      # B listed first (A/B labels swapped in prompt)
    preferred: Literal["A", "B", "tie"]
    aggregate_score_a: float    # average score for response A across both orderings
    aggregate_score_b: float
    rationale: str
    # Raw verdicts of the two orderings, both in ORIGINAL labels (A = response_a):
    #   verdict_ab: verdict when response_a was listed first
    #   verdict_ba: verdict when response_b was listed first, mapped back
    # Used by the calibrator to measure position bias. None if unavailable.
    verdict_ab: str | None = None
    verdict_ba: str | None = None


# ─── Prompt templates ─────────────────────────────────────────────────────────

_PAIRWISE_SYSTEM = """\
You are an expert evaluator of AI assistant responses.
Your task is to compare two responses to a user conversation and decide
which is better, using the provided rubric.

Be objective. Focus on quality, not style preference.
Your evaluation must be grounded in the rubric dimensions provided.

Respond ONLY with a JSON object matching this exact schema:
{
  "dimension_scores": {
    "<dimension_name>": {"score": <int>, "rationale": "<one sentence>"}
    // repeat for each dimension, scoring RESPONSE A
  },
  "dimension_scores_b": {
    "<dimension_name>": {"score": <int>, "rationale": "<one sentence>"}
    // repeat for each dimension, scoring RESPONSE B
  },
  "preferred": "A" | "B" | "tie",
  "rationale": "<2–4 sentence explanation of your overall verdict>"
}
"""

_PAIRWISE_USER = """\
## Conversation History
{conversation}

## Response A
{response_a}

## Response B
{response_b}

## Rubric
{rubric_text}

Evaluate both responses on each rubric dimension (1 = worst, {max_scale} = best).
Then give your overall preference: A, B, or tie.
"""

_ABSOLUTE_SYSTEM = """\
You are an expert evaluator of AI assistant responses. Score the response using the provided rubric.
Respond ONLY with JSON matching this schema:
{
  "dimension_scores": {
    "<dimension_name>": {"score": <int>, "rationale": "<one sentence>"}
  },
  "overall_score": <float between 0 and 1>,
  "rationale": "<2–4 sentence summary>"
}
"""

_ABSOLUTE_USER = """\
## Conversation History
{conversation}

## Response to Evaluate
{response}

## Rubric
{rubric_text}

Score the response on each dimension (1 = worst, {max_scale} = best).
"""


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _format_conversation(conv: ConversationRecord) -> str:
    """Format conversation messages as readable text (excluding system prompt)."""
    lines = []
    for msg in conv.without_system_prompt():
        role = msg.role.value.upper()
        content = msg.content or "[no content]"
        lines.append(f"**{role}:** {content}")
    return "\n\n".join(lines)


def _format_rubric(rubric: Rubric) -> str:
    lines = [f"Mode: {rubric.mode}"]
    for dim in rubric.dimensions:
        lines.append(
            f"- **{dim.name}** (weight={dim.weight:.1f}, scale=1–{dim.scale}): "
            f"{dim.description}"
        )
    return "\n".join(lines)


def _extract_json(text: str) -> dict:
    """Extract the first JSON object from a text that may contain prose."""
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try extracting from markdown code block
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # Try extracting first {...} block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError(f"No valid JSON found in response: {text[:300]!r}")


def _parse_dimension_scores(raw_dims: object, rubric: Rubric) -> list[DimensionScore]:
    """
    Validate and convert a `{dimension: {"score": n, "rationale": ...}}` object.

    Raises ValueError (retryable) if a rubric dimension is missing or malformed,
    instead of silently substituting a default score. Out-of-range scores are
    clamped to [1, scale].
    """
    if not isinstance(raw_dims, dict):
        raise ValueError(f"dimension scores must be an object, got {type(raw_dims).__name__}")
    dim_scores = []
    for dim in rubric.dimensions:
        raw = raw_dims.get(dim.name)
        if isinstance(raw, int | float) and not isinstance(raw, bool):
            raw = {"score": raw}
        if not isinstance(raw, dict) or "score" not in raw:
            raise ValueError(f"judge output is missing a score for dimension {dim.name!r}")
        try:
            score = float(raw["score"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"non-numeric score for dimension {dim.name!r}") from exc
        score = min(max(score, 1.0), float(dim.scale))
        dim_scores.append(
            DimensionScore(
                dimension=dim.name,
                score=score,
                max_score=float(dim.scale),
                rationale=str(raw.get("rationale", "")),
            )
        )
    return dim_scores


def _parse_pairwise_side(data: dict, rubric: Rubric, key: str) -> JudgeResult:
    """Parse one response's scores (`dimension_scores` = first-listed, `_b` = second-listed)."""
    if key not in data:
        raise ValueError(f"judge output is missing {key!r}")
    dim_scores = _parse_dimension_scores(data[key], rubric)
    aggregate = rubric.aggregate_scores({ds.dimension: ds.score for ds in dim_scores})
    return JudgeResult(
        dimension_scores=dim_scores,
        aggregate_score=aggregate,
        overall_verdict=_normalise_verdict(data.get("preferred")),
        rationale=str(data.get("rationale", "")),
        raw_response=json.dumps(data),
    )


def _normalise_verdict(value: object) -> str:
    return value if value in ("A", "B", "tie") else "tie"


def _flip_verdict(verdict: str) -> str:
    return {"A": "B", "B": "A"}.get(verdict, "tie")


def _is_retryable(exc: BaseException) -> bool:
    """Retry transient API failures and unparsable output; never permanent errors."""
    from prompt_opt.optimizer.budget import BudgetExceededError

    if isinstance(exc, BudgetExceededError):
        return False
    if isinstance(exc, ValueError):  # includes json.JSONDecodeError
        return True
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and status in (400, 401, 403, 404, 422):
        return False  # bad key / unknown model / unsupported params: retrying cannot help
    return True


# ─── LLMJudge ────────────────────────────────────────────────────────────────


class LLMJudge:
    """
    Evaluate LLM responses using another LLM as a judge.

    Supports both pairwise (A vs B, position-bias corrected) and absolute
    (single-response Likert) scoring.
    """

    def __init__(
        self,
        model: str | None = None,
        rubric: Rubric | None = None,
    ):
        self.model = model or settings.judge_model
        from prompt_opt.labeling.rubric import DEFAULT_RUBRIC

        self.rubric = rubric or DEFAULT_RUBRIC

    def _call_llm(self, system: str, user: str) -> str:
        """Make a single metered LLM call via litellm (no retry)."""
        from prompt_opt import llm

        resp = llm.completion(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content or ""

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=10),
        retry=retry_if_exception(_is_retryable),
        reraise=True,
    )
    def _call_json(self, system: str, user: str, validate=None) -> dict:
        """
        Call the judge and return its parsed JSON output.

        The whole call + parse + validation is retried, so unparsable or
        structurally invalid output gets a fresh attempt just like a transient
        API error. After the last attempt the original exception is re-raised.
        """
        data = _extract_json(self._call_llm(system, user))
        if not isinstance(data, dict):
            raise ValueError(f"judge output must be a JSON object, got {type(data).__name__}")
        if validate is not None:
            validate(data)
        return data

    def _validate_pairwise(self, data: dict) -> None:
        _parse_pairwise_side(data, self.rubric, "dimension_scores")
        _parse_pairwise_side(data, self.rubric, "dimension_scores_b")

    def score_pairwise(
        self,
        conversation: ConversationRecord,
        response_a: str,
        response_b: str,
    ) -> PairwiseJudgeResult:
        """
        Score two responses against the rubric with position-bias correction.

        Runs both A→B and B→A orderings; averages per-response scores. The raw
        verdict of each ordering is returned (in original labels) so callers can
        measure the judge's position bias.
        """
        conv_text = _format_conversation(conversation)
        rubric_text = _format_rubric(self.rubric)
        max_scale = max(d.scale for d in self.rubric.dimensions)

        # ── Pass 1: A listed first ───────────────────────────────────────────
        user_ab = _PAIRWISE_USER.format(
            conversation=conv_text,
            response_a=response_a,
            response_b=response_b,
            rubric_text=rubric_text,
            max_scale=max_scale,
        )
        data_ab = self._call_json(_PAIRWISE_SYSTEM, user_ab, self._validate_pairwise)
        result_ab_a = _parse_pairwise_side(data_ab, self.rubric, "dimension_scores")
        result_ab_b = _parse_pairwise_side(data_ab, self.rubric, "dimension_scores_b")

        # ── Pass 2: B listed first (swap labels) ────────────────────────────
        user_ba = _PAIRWISE_USER.format(
            conversation=conv_text,
            response_a=response_b,  # intentionally swapped
            response_b=response_a,
            rubric_text=rubric_text,
            max_scale=max_scale,
        )
        data_ba = self._call_json(_PAIRWISE_SYSTEM, user_ba, self._validate_pairwise)
        # In the B→A pass, "A" in the JSON refers to the original B and vice versa
        result_ba_b = _parse_pairwise_side(data_ba, self.rubric, "dimension_scores")    # orig B
        result_ba_a = _parse_pairwise_side(data_ba, self.rubric, "dimension_scores_b")  # orig A

        # Average each response's scores over both orderings
        score_a = (result_ab_a.aggregate_score + result_ba_a.aggregate_score) / 2
        score_b = (result_ab_b.aggregate_score + result_ba_b.aggregate_score) / 2

        if score_a > score_b + 0.05:
            preferred: Literal["A", "B", "tie"] = "A"
        elif score_b > score_a + 0.05:
            preferred = "B"
        else:
            preferred = "tie"

        rationale = (
            f"[Pass A→B] {data_ab.get('rationale', '')}\n"
            f"[Pass B→A] {data_ba.get('rationale', '')}"
        )

        return PairwiseJudgeResult(
            result_ab=result_ab_a,
            result_ba=result_ba_a,
            preferred=preferred,
            aggregate_score_a=score_a,
            aggregate_score_b=score_b,
            rationale=rationale,
            verdict_ab=_normalise_verdict(data_ab.get("preferred")),
            verdict_ba=_flip_verdict(_normalise_verdict(data_ba.get("preferred"))),
        )

    def score_absolute(
        self,
        conversation: ConversationRecord,
        response: str,
    ) -> JudgeResult:
        """Score a single response in absolute mode."""
        conv_text = _format_conversation(conversation)
        rubric_text = _format_rubric(self.rubric)
        max_scale = max(d.scale for d in self.rubric.dimensions)

        user = _ABSOLUTE_USER.format(
            conversation=conv_text,
            response=response,
            rubric_text=rubric_text,
            max_scale=max_scale,
        )
        data = self._call_json(
            _ABSOLUTE_SYSTEM,
            user,
            lambda d: _parse_dimension_scores(d.get("dimension_scores"), self.rubric),
        )
        dim_scores = _parse_dimension_scores(data.get("dimension_scores"), self.rubric)

        aggregate = self.rubric.aggregate_scores(
            {ds.dimension: ds.score for ds in dim_scores}
        )

        return JudgeResult(
            dimension_scores=dim_scores,
            aggregate_score=aggregate,
            overall_verdict="score",
            rationale=str(data.get("rationale", "")),
            raw_response=json.dumps(data),
        )
