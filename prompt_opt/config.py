"""
Global configuration via pydantic-settings.

All values can be overridden by environment variables or a .env file.
"""

from __future__ import annotations

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    max_llm_calls_per_run: int = Field(500)
    max_wall_seconds_per_run: int = Field(3600)
    max_usd_spend_per_run: float = Field(10.0)

    # ── Gate ─────────────────────────────────────────────────────────────────
    gate_min_win_rate: float = Field(0.55)
    gate_ci_level: float = Field(0.95)
    gate_max_guardrail_regression_pct: float = Field(0.05)

    # ── Dedup ────────────────────────────────────────────────────────────────
    dedup_cosine_threshold: float = Field(0.95)

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
