"""Centralized configuration for Generative Claims.

Uses pydantic-settings for type-safe, validated configuration with
environment-variable overrides and sensible defaults.

Design Decisions:
    - All paths are relative to PROJECT_ROOT for portability.
    - Frozen models prevent accidental mutation after initialization.
    - Nested models keep concerns (profiler, validator, etc.) separated.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

# ── Project Root ─────────────────────────────────────────────────────────
# Two levels up from src/utils/config.py  →  generative_claims/
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent


class ProfilerConfig(BaseModel):
    """Configuration for the statistical profiler."""

    model_config = {"frozen": True}

    # Input
    raw_data_filename: str = "Insurance claims data.csv"

    # Output paths (relative to PROJECT_ROOT)
    statistics_output: str = "data/processed/statistics.json"
    fd_output: str = "data/processed/functional_dependencies.json"

    # Distribution fitting
    candidate_distributions: List[str] = Field(
        default=["norm", "gamma", "beta", "lognorm", "expon", "uniform"],
        description="scipy.stats distribution names to try during fitting.",
    )
    fit_timeout_seconds: float = 30.0

    # Functional dependency discovery
    fd_confidence_threshold: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for probabilistic FDs.",
    )
    fd_max_lhs_size: int = Field(
        default=1,
        ge=1,
        le=4,
        description="Maximum number of columns on the LHS of an FD.",
    )
    fd_sample_size: int = Field(
        default=50_000,
        ge=100,
        description="Sample size for FD discovery (performance guard).",
    )


class ValidatorConfig(BaseModel):
    """Configuration for schema validation."""

    model_config = {"frozen": True}

    schema_output: str = "configs/schema.yaml"
    max_null_fraction: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Maximum allowed null fraction per column.",
    )


class GeneratorConfig(BaseModel):
    """Configuration for synthetic data generation."""

    model_config = {"frozen": True}

    target_claim_rate: float = Field(
        default=0.06,
        ge=0.0,
        le=1.0,
        description="Target claim rate in synthetic data.",
    )
    batch_size: int = Field(default=1_000, ge=1)
    random_seed: int = 42


class Settings(BaseSettings):
    """Top-level application settings.

    Values can be overridden via environment variables prefixed with
    ``GC_`` (e.g., ``GC_LOG_LEVEL=DEBUG``).
    """

    model_config = {
        "env_prefix": "GC_",
        "env_nested_delimiter": "__",
        "frozen": True,
    }

    # General
    project_root: Path = PROJECT_ROOT
    log_level: str = "INFO"
    log_dir: str = "logs"

    # Module configs
    profiler: ProfilerConfig = ProfilerConfig()
    validator: ValidatorConfig = ValidatorConfig()
    generator: GeneratorConfig = GeneratorConfig()

    # ── Derived helpers ──────────────────────────────────────────────

    @property
    def raw_data_path(self) -> Path:
        """Absolute path to the raw CSV file."""
        # Check project root itself first (for test fixtures)
        candidate_root = self.project_root / self.profiler.raw_data_filename
        if candidate_root.exists():
            return candidate_root
        # Look in the parent directory (pbl 1 project) where the CSV lives
        candidate = self.project_root.parent / self.profiler.raw_data_filename
        if candidate.exists():
            return candidate
        # Fallback: check data/raw inside project
        return self.project_root / "data" / "raw" / self.profiler.raw_data_filename

    @property
    def statistics_path(self) -> Path:
        return self.project_root / self.profiler.statistics_output

    @property
    def fd_path(self) -> Path:
        return self.project_root / self.profiler.fd_output

    @property
    def schema_path(self) -> Path:
        return self.project_root / self.validator.schema_output

    @property
    def log_path(self) -> Path:
        return self.project_root / self.log_dir


def get_settings() -> Settings:
    """Factory that returns a singleton-like Settings instance."""
    return Settings()
