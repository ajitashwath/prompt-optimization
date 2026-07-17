"""
LLM-as-Judge: scores candidate responses against a rubric.

Key design decisions:
  - Pairwise: runs A→B and B→A orderings, averages to cancel position bias.
  - Structured output: JSON with per-dimension scores + rationale.
  - Retries on parse failure (up to 3 attempts with tenacity).
  - Separable from the model being optimized (use a different model family).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Literal

from tenacity import retry, stop_after_attempt, wait_exponential

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


# ─── Prompt templates ─────────────────────────────────────────────────────────

_PAIRWISE_SYSTEM = """\
You are an expert evaluator of AI assistant responses. Your task is to compare two responses to a user conversation and decide which is better, using the provided rubric.

Be objective. Focus on quality, not style preference. Your evaluation must be grounded in the rubric dimensions provided.

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
        lines.append(f"- **{dim.name}** (weight={dim.weight:.1f}, scale=1–{dim.scale}): {dim.description}")
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


def _parse_pairwise_response(data: dict, rubric: Rubric) -> JudgeResult:
    """Parse the A-perspective scores from a pairwise JSON response."""
    dim_scores = []
    raw_dims = data.get("dimension_scores", {})
    for dim in rubric.dimensions:
        raw = raw_dims.get(dim.name, {})
        score = float(raw.get("score", dim.scale // 2))
        dim_scores.append(
            DimensionScore(
                dimension=dim.name,
                score=score,
                max_score=float(dim.scale),
                rationale=raw.get("rationale", ""),
            )
        )
    aggregate = rubric.aggregate_scores({ds.dimension: ds.score for ds in dim_scores})
    return JudgeResult(
        dimension_scores=dim_scores,
        aggregate_score=aggregate,
        overall_verdict=data.get("preferred", "tie"),
        rationale=data.get("rationale", ""),
        raw_response=json.dumps(data),
    )


def _parse_pairwise_response_b(data: dict, rubric: Rubric) -> JudgeResult:
    """Parse the B-perspective scores from a pairwise JSON response."""
    dim_scores = []
    raw_dims = data.get("dimension_scores_b", data.get("dimension_scores", {}))
    for dim in rubric.dimensions:
        raw = raw_dims.get(dim.name, {})
        score = float(raw.get("score", dim.scale // 2))
        dim_scores.append(
            DimensionScore(
                dimension=dim.name,
                score=score,
                max_score=float(dim.scale),
                rationale=raw.get("rationale", ""),
            )
        )
    aggregate = rubric.aggregate_scores({ds.dimension: ds.score for ds in dim_scores})
    return JudgeResult(
        dimension_scores=dim_scores,
        aggregate_score=aggregate,
        overall_verdict=data.get("preferred", "tie"),
        rationale=data.get("rationale", ""),
        raw_response=json.dumps(data),
    )


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

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def _call_llm(self, system: str, user: str) -> str:
        """Make a single LLM call via litellm with retry on transient failures."""
        try:
            import litellm  # type: ignore[import]

            resp = litellm.completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            return resp.choices[0].message.content or ""
        except Exception:
            raise

    def score_pairwise(
        self,
        conversation: ConversationRecord,
        response_a: str,
        response_b: str,
    ) -> PairwiseJudgeResult:
        """
        Score two responses against the rubric with position-bias correction.

        Runs both A→B and B→A orderings; averages per-response scores.
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
        raw_ab = self._call_llm(_PAIRWISE_SYSTEM, user_ab)
        data_ab = _extract_json(raw_ab)
        result_ab_a = _parse_pairwise_response(data_ab, self.rubric)
        result_ab_b = _parse_pairwise_response_b(data_ab, self.rubric)

        # ── Pass 2: B listed first (swap labels) ────────────────────────────
        user_ba = _PAIRWISE_USER.format(
            conversation=conv_text,
            response_a=response_b,  # intentionally swapped
            response_b=response_a,
            rubric_text=rubric_text,
            max_scale=max_scale,
        )
        raw_ba = self._call_llm(_PAIRWISE_SYSTEM, user_ba)
        data_ba = _extract_json(raw_ba)
        # In B→A pass: "A" in JSON refers to original B; "B" refers to original A
        result_ba_b = _parse_pairwise_response(data_ba, self.rubric)   # original B score
        result_ba_a = _parse_pairwise_response_b(data_ba, self.rubric) # original A score

        # Average A scores from both passes
        score_a = (result_ab_a.aggregate_score + result_ba_a.aggregate_score) / 2
        score_b = (result_ab_b.aggregate_score + result_ba_b.aggregate_score) / 2

        # Determine consensus preference
        votes: dict[str, int] = {"A": 0, "B": 0, "tie": 0}
        ab_pref = data_ab.get("preferred", "tie")
        ba_pref = data_ba.get("preferred", "tie")
        votes[ab_pref] = votes.get(ab_pref, 0) + 1
        # In B→A pass the labels are flipped
        if ba_pref == "A":
            votes["B"] += 1
        elif ba_pref == "B":
            votes["A"] += 1
        else:
            votes["tie"] += 1

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
        raw = self._call_llm(_ABSOLUTE_SYSTEM, user)
        data = _extract_json(raw)

        dim_scores = []
        raw_dims = data.get("dimension_scores", {})
        for dim in self.rubric.dimensions:
            raw_dim = raw_dims.get(dim.name, {})
            score = float(raw_dim.get("score", dim.scale // 2))
            dim_scores.append(
                DimensionScore(
                    dimension=dim.name,
                    score=score,
                    max_score=float(dim.scale),
                    rationale=raw_dim.get("rationale", ""),
                )
            )

        aggregate = rubric.aggregate_scores({ds.dimension: ds.score for ds in dim_scores}) \
            if False else self.rubric.aggregate_scores(
                {ds.dimension: ds.score for ds in dim_scores}
            )

        return JudgeResult(
            dimension_scores=dim_scores,
            aggregate_score=aggregate,
            overall_verdict="score",
            rationale=data.get("rationale", ""),
            raw_response=raw,
        )
