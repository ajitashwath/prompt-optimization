"""
Shadow mode deployment.

Runs the challenger prompt alongside the champion on live traffic,
logs responses without serving them to users, and scores asynchronously.

Usage::

    shadow = ShadowRunner(challenger_prompt, champion_prompt, log_path="./data/shadow_logs")
    result = shadow.run_request(messages)
    # result.champion_response is served; challenger_response is logged only
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from prompt_opt.config import settings
from prompt_opt.eval.harness import PromptCandidate

logger = logging.getLogger(__name__)


@dataclass
class ShadowResult:
    request_id: str
    champion_response: str       # served to the user
    challenger_response: str     # logged only, NOT served
    timestamp: str
    latency_champion_ms: float
    latency_challenger_ms: float
    error: str | None = None


class ShadowRunner:
    """
    Run champion and challenger in parallel (for logging/scoring), serve only champion.

    In production this would run as an async sidecar or thread pool; here we
    run sequentially with timing measurements to keep dependencies minimal.
    """

    def __init__(
        self,
        challenger: PromptCandidate,
        champion: PromptCandidate,
        model: str | None = None,
        log_path: str | None = None,
    ):
        self.challenger = challenger
        self.champion = champion
        self.model = model or settings.optimizee_model
        self._log_path = Path(log_path or settings.shadow_log_path)
        self._log_path.mkdir(parents=True, exist_ok=True)
        self._log_file = self._log_path / "shadow_log.jsonl"

    def _call_model(self, system_prompt: str, messages: list[dict]) -> tuple[str, float]:
        """Call the model; returns (response_text, latency_ms)."""
        import time

        import litellm  # type: ignore[import]

        full_messages = [{"role": "system", "content": system_prompt}] + messages
        t0 = time.perf_counter()
        try:
            resp = litellm.completion(
                model=self.model,
                messages=full_messages,
                temperature=0.0,
            )
            latency_ms = (time.perf_counter() - t0) * 1000
            return resp.choices[0].message.content or "", latency_ms
        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000
            return f"[ERROR: {exc}]", latency_ms

    def run_request(self, messages: list[dict]) -> ShadowResult:
        """
        Process a single request through both champion and challenger.

        Args:
            messages: List of {"role": ..., "content": ...} dicts (without system message).

        Returns:
            ShadowResult — serve champion_response to users; log challenger_response.
        """
        request_id = str(uuid.uuid4())[:8]

        champ_text, champ_latency = self._call_model(self.champion.system_prompt, messages)
        chal_text, chal_latency = self._call_model(self.challenger.system_prompt, messages)

        result = ShadowResult(
            request_id=request_id,
            champion_response=champ_text,
            challenger_response=chal_text,
            timestamp=datetime.utcnow().isoformat(),
            latency_champion_ms=champ_latency,
            latency_challenger_ms=chal_latency,
        )

        self._log(result, messages)
        return result

    def _log(self, result: ShadowResult, messages: list[dict]) -> None:
        record = {
            "request_id": result.request_id,
            "timestamp": result.timestamp,
            "champion_hash": self.champion.content_hash,
            "challenger_hash": self.challenger.content_hash,
            "messages": messages,
            "champion_response": result.champion_response,
            "challenger_response": result.challenger_response,
            "latency_champion_ms": result.latency_champion_ms,
            "latency_challenger_ms": result.latency_challenger_ms,
        }
        with open(self._log_file, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
