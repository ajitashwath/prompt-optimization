"""
Global configuration via pydantic-settings.

All values can be overridden by environment variables or a .env file.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROVIDER_KEY_VARS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY")


def load_env_file(path: str | Path = ".env") -> None:
    """
    Export the entries of a `.env` file (in the current directory) to `os.environ`.

    pydantic-settings reads `.env` into `Settings`, but provider API keys are not
    settings fields and litellm reads them from the process environment, so they
    must be exported. Variables already set in the shell take precedence.
    """
    env_path = Path(path)
    if env_path.is_file():
        load_dotenv(env_path, override=False)


def has_provider_key() -> bool:
    """True if at least one provider key is set to something other than a placeholder."""
    for var in PROVIDER_KEY_VARS:
        value = os.environ.get(var, "").strip()
        # .env.example ships placeholders such as `sk-...`
        if value and not value.endswith("..."):
            return True
    return False


load_env_file()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM Models ──────────────────────────────────────────────────────────
    optimizee_model: str = Field("openai/gpt-4o-mini", description="Model being optimized")
    judge_model: str = Field(
        "anthropic/claude-3-5-sonnet-20241022", description="Model used as judge"
    )
    proposer_model: str = Field(
        "openai/gpt-4o", description="Model for instruction proposals/mutations"
    )
    embedding_model: str = Field(
        "openai/text-embedding-3-small", description="Model for embeddings"
    )

    # ── Storage ─────────────────────────────────────────────────────────────
    database_url: str = Field(
        "sqlite+aiosqlite:///./data/prompt_opt.db", description="SQLAlchemy async URL"
    )
    blob_store_path: str = Field("./data/blobs", description="Path for raw log blobs")
    shadow_log_path: str = Field("./data/shadow_logs")

    # ── Budget ───────────────────────────────────────────────────────────────
    max_llm_calls_per_run: int = Field(5000)
    max_wall_seconds_per_run: int = Field(3600)
    max_usd_spend_per_run: float = Field(10.0)

    # ── Gate ─────────────────────────────────────────────────────────────────
    gate_min_win_rate: float = Field(0.55)
    gate_ci_level: float = Field(0.95)
    # Max tolerated shift in refusal rate (absolute, e.g. 0.10 = 10 points) between
    # challenger and champion; shared by the eval guardrails and the champion gate.
    gate_max_guardrail_regression_pct: float = Field(0.10)
    # Minimum number of paired test examples for a gate verdict to mean anything.
    gate_min_test_examples: int = Field(30)

    # ── Ingestion ────────────────────────────────────────────────────────────
    dedup_cosine_threshold: float = Field(0.95)
    # Add the spaCy NER pass (names, organisations, places) to PII scrubbing. Off by default:
    # it needs `python -m spacy download en_core_web_sm` and redacts entities such as countries
    # that are not always personal data.
    pii_use_ner: bool = Field(False)

    # ── Judge Calibration ────────────────────────────────────────────────────
    calibration_min_kappa: float = Field(0.6)
    calibration_min_correlation: float = Field(0.7)

    # ── Deployment ───────────────────────────────────────────────────────────
    canary_initial_pct: int = Field(1)
    canary_soak_seconds: int = Field(3600)

    # ── Feedback Loop ────────────────────────────────────────────────────────
    drift_kl_threshold: float = Field(0.1)
    reingest_cron: str = Field("0 2 * * *")
    reoptimize_cron: str = Field("0 3 * * 1")
    # Hourly: compare data/recent_scores.json with the saved drift baseline
    drift_cron: str = Field("30 * * * *")

    # ── Logging ──────────────────────────────────────────────────────────────
    log_level: str = Field("INFO")

    @field_validator("gate_min_win_rate", "gate_ci_level", "dedup_cosine_threshold")
    @classmethod
    def _between_zero_one(cls, v: float) -> float:
        if not 0 < v < 1:
            raise ValueError(f"Value {v} must be strictly between 0 and 1")
        return v


# Singleton — import this everywhere
settings = Settings()
