"""Comprehensive unit tests for the Profiler module.

Covers:
    - StatisticalProfiler: loading, numerical/categorical profiling,
      distribution fitting, JSON serialization.
    - FDDiscovery: exact FDs, probabilistic FDs, edge cases, subsampling.
    - SchemaValidator: schema generation, validation pass/fail, YAML I/O.

Uses synthetic fixture data so tests are fast, deterministic, and
independent of the real 58k-row CSV.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# ── Ensure project root is on sys.path ───────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.profiler.statistical_profiler import StatisticalProfiler, _params_to_dict
from src.profiler.fd_discovery import FDDiscovery
from src.validator.schema_validator import SchemaValidator
from src.utils.config import Settings, ProfilerConfig, ValidatorConfig
from src.utils.exceptions import (
    DataLoadError,
    FunctionalDependencyError,
    SchemaValidationError,
    StatisticsComputationError,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create a small, deterministic DataFrame mimicking claims data."""
    np.random.seed(42)
    n = 200

    models = ["M1", "M2", "M3", "M4"]
    engine_map = {"M1": "1.5L Turbo", "M2": "2.0L NA", "M3": "1.5L Turbo", "M4": "3.0L V6"}
    fuel_map = {"M1": "Petrol", "M2": "Diesel", "M3": "Petrol", "M4": "Diesel"}

    model_col = np.random.choice(models, size=n)
    engine_col = [engine_map[m] for m in model_col]
    fuel_col = [fuel_map[m] for m in model_col]

    df = pd.DataFrame({
        "policy_id": [f"POL{i:06d}" for i in range(n)],
        "subscription_length": np.round(np.random.uniform(1, 12, n), 1),
        "customer_age": np.random.randint(18, 80, n),
        "model": model_col,
        "engine_type": engine_col,
        "fuel_type": fuel_col,
        "airbags": np.random.choice([2, 4, 6], size=n),
        "is_esc": np.random.choice(["Yes", "No"], size=n),
        "displacement": np.random.choice([1198, 1493, 1998, 2993], size=n),
        "claim_status": np.random.choice([0, 1], size=n, p=[0.94, 0.06]),
    })
    return df


@pytest.fixture
def tmp_csv(sample_df: pd.DataFrame, tmp_path: Path) -> Path:
    """Write sample_df to a temporary CSV and return its path."""
    csv_path = tmp_path / "Insurance claims data.csv"
    sample_df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def test_settings(tmp_csv: Path, tmp_path: Path) -> Settings:
    """Settings that point to temp directories."""
    return Settings(
        project_root=tmp_path,
        profiler=ProfilerConfig(
            raw_data_filename=tmp_csv.name,
            statistics_output="data/processed/statistics.json",
            fd_output="data/processed/functional_dependencies.json",
            fd_max_lhs_size=1,         # keep tests fast
            fd_sample_size=200,
            fd_confidence_threshold=0.95,
        ),
        validator=ValidatorConfig(
            schema_output="configs/schema.yaml",
        ),
    )


# ============================================================================
# StatisticalProfiler Tests
# ============================================================================


class TestStatisticalProfiler:
    """Tests for StatisticalProfiler."""

    def test_load_data_success(
        self, test_settings: Settings, sample_df: pd.DataFrame
    ) -> None:
        """Profiler loads CSV and returns correct shape."""
        profiler = StatisticalProfiler(settings=test_settings)
        df = profiler.load_data()
        assert df.shape == sample_df.shape
        assert list(df.columns) == list(sample_df.columns)

    def test_load_data_missing_file(self, tmp_path: Path) -> None:
        """Profiler raises DataLoadError for missing CSV."""
        settings = Settings(
            project_root=tmp_path,
            profiler=ProfilerConfig(raw_data_filename="nonexistent.csv"),
        )
        profiler = StatisticalProfiler(settings=settings)
        with pytest.raises(DataLoadError):
            profiler.load_data()

    def test_profile_numerical_column(
        self, test_settings: Settings
    ) -> None:
        """Numerical columns get min, max, mean, std, percentiles."""
        profiler = StatisticalProfiler(settings=test_settings)
        profiler.load_data()
        profile = profiler.profile_all_columns()

        col_stats = profile["columns"]["customer_age"]
        assert col_stats["type"] == "numerical"
        assert "min" in col_stats
        assert "max" in col_stats
        assert "mean" in col_stats
        assert "std" in col_stats
        assert "percentiles" in col_stats
        assert "p50" in col_stats["percentiles"]

    def test_profile_categorical_column(
        self, test_settings: Settings
    ) -> None:
        """Categorical columns get value_counts, cardinality, mode."""
        profiler = StatisticalProfiler(settings=test_settings)
        profiler.load_data()
        profile = profiler.profile_all_columns()

        col_stats = profile["columns"]["model"]
        assert col_stats["type"] == "categorical"
        assert col_stats["cardinality"] == 4
        assert "value_counts" in col_stats
        assert "value_frequencies" in col_stats

    def test_binary_detection(self, test_settings: Settings) -> None:
        """Binary columns (Yes/No) are flagged with positive_rate."""
        profiler = StatisticalProfiler(settings=test_settings)
        profiler.load_data()
        profile = profiler.profile_all_columns()

        col_stats = profile["columns"]["is_esc"]
        assert col_stats["is_binary"] is True
        assert 0.0 <= col_stats["positive_rate"] <= 1.0

    def test_distribution_fitting(self, test_settings: Settings) -> None:
        """Distribution fit returns a named distribution with params."""
        profiler = StatisticalProfiler(settings=test_settings)
        profiler.load_data()
        profile = profiler.profile_all_columns()

        dist_info = profile["columns"]["subscription_length"]["distribution"]
        assert "name" in dist_info
        assert "params" in dist_info

    def test_save_creates_json(self, test_settings: Settings) -> None:
        """Profile is saved to the configured JSON path."""
        profiler = StatisticalProfiler(settings=test_settings)
        profiler.run()

        out_path = test_settings.statistics_path
        assert out_path.exists()

        with open(out_path) as fh:
            data = json.load(fh)
        assert "_meta" in data
        assert "columns" in data

    def test_full_run_pipeline(self, test_settings: Settings) -> None:
        """End-to-end run returns a valid profile."""
        profiler = StatisticalProfiler(settings=test_settings)
        result = profiler.run()

        assert result["_meta"]["n_rows"] == 200
        assert result["_meta"]["n_columns"] == 10
        assert len(result["columns"]) == 10

    def test_meta_has_timestamp(self, test_settings: Settings) -> None:
        """Meta section includes profiling timestamp."""
        profiler = StatisticalProfiler(settings=test_settings)
        result = profiler.run()
        assert "profiled_at" in result["_meta"]

    def test_params_to_dict(self) -> None:
        """_params_to_dict correctly labels scipy fit params."""
        result = _params_to_dict("norm", (5.0, 2.0))
        assert "loc" in result
        assert "scale" in result
        assert result["loc"] == 5.0


# ============================================================================
# FDDiscovery Tests
# ============================================================================


class TestFDDiscovery:
    """Tests for FDDiscovery."""

    def test_exact_fd_model_to_engine(
        self, test_settings: Settings, sample_df: pd.DataFrame
    ) -> None:
        """model → engine_type is an exact FD in our fixture data."""
        fd = FDDiscovery(settings=test_settings)
        result = fd.run(df=sample_df)

        exact_lhs_rhs = {
            (tuple(f["lhs"]), f["rhs"]) for f in result["exact"]
        }
        assert (("model",), "engine_type") in exact_lhs_rhs

    def test_exact_fd_model_to_fuel(
        self, test_settings: Settings, sample_df: pd.DataFrame
    ) -> None:
        """model → fuel_type is an exact FD in our fixture data."""
        fd = FDDiscovery(settings=test_settings)
        result = fd.run(df=sample_df)

        exact_lhs_rhs = {
            (tuple(f["lhs"]), f["rhs"]) for f in result["exact"]
        }
        assert (("model",), "fuel_type") in exact_lhs_rhs

    def test_non_fd_not_in_exact(
        self, test_settings: Settings, sample_df: pd.DataFrame
    ) -> None:
        """is_esc → customer_age should NOT be an exact FD."""
        fd = FDDiscovery(settings=test_settings)
        result = fd.run(df=sample_df)

        exact_lhs_rhs = {
            (tuple(f["lhs"]), f["rhs"]) for f in result["exact"]
        }
        assert (("is_esc",), "customer_age") not in exact_lhs_rhs

    def test_skips_unique_columns(
        self, test_settings: Settings, sample_df: pd.DataFrame
    ) -> None:
        """policy_id (unique per row) should be excluded from candidates."""
        fd = FDDiscovery(settings=test_settings)
        fd.df = sample_df
        candidates = fd._select_candidate_columns()
        assert "policy_id" not in candidates

    def test_confidence_computation(
        self, test_settings: Settings, sample_df: pd.DataFrame
    ) -> None:
        """Confidence of an exact FD is 1.0."""
        fd = FDDiscovery(settings=test_settings)
        fd.df = sample_df
        conf = fd._compute_fd_confidence(["model"], "engine_type")
        assert conf is not None
        assert abs(conf - 1.0) < 1e-9

    def test_result_structure(
        self, test_settings: Settings, sample_df: pd.DataFrame
    ) -> None:
        """Result has the expected top-level keys."""
        fd = FDDiscovery(settings=test_settings)
        result = fd.run(df=sample_df)

        assert "_meta" in result
        assert "exact" in result
        assert "probabilistic" in result
        assert "n_exact_fds" in result["_meta"]

    def test_save_creates_json(
        self, test_settings: Settings, sample_df: pd.DataFrame
    ) -> None:
        """FD result is persisted to JSON."""
        fd = FDDiscovery(settings=test_settings)
        fd.run(df=sample_df)

        out_path = test_settings.fd_path
        assert out_path.exists()

        with open(out_path) as fh:
            data = json.load(fh)
        assert "exact" in data

    def test_summarize(
        self, test_settings: Settings, sample_df: pd.DataFrame
    ) -> None:
        """Summarize produces non-empty readable output."""
        fd = FDDiscovery(settings=test_settings)
        result = fd.run(df=sample_df)
        summary = FDDiscovery.summarize(result)
        assert "Exact FDs" in summary
        assert len(summary) > 50


# ============================================================================
# SchemaValidator Tests
# ============================================================================


class TestSchemaValidator:
    """Tests for SchemaValidator."""

    def test_generate_schema_produces_columns(
        self, test_settings: Settings, sample_df: pd.DataFrame
    ) -> None:
        """Schema generation returns a schema with all columns."""
        sv = SchemaValidator(settings=test_settings)
        schema = sv.generate_schema(df=sample_df)
        assert schema is not None
        assert len(schema.columns) == len(sample_df.columns)

    def test_validate_good_data_passes(
        self, test_settings: Settings, sample_df: pd.DataFrame
    ) -> None:
        """Real data validates against its own schema with zero errors."""
        sv = SchemaValidator(settings=test_settings)
        sv.generate_schema(df=sample_df)
        errors = sv.validate(sample_df)
        assert len(errors) == 0

    def test_validate_bad_range_fails(
        self, test_settings: Settings, sample_df: pd.DataFrame
    ) -> None:
        """Synthetic data with out-of-range values triggers violations."""
        sv = SchemaValidator(settings=test_settings)
        sv.generate_schema(df=sample_df)

        bad_df = sample_df.copy()
        bad_df.loc[0, "customer_age"] = 999  # way out of range
        errors = sv.validate(bad_df)
        assert len(errors) > 0

    def test_validate_bad_category_fails(
        self, test_settings: Settings, sample_df: pd.DataFrame
    ) -> None:
        """Invalid categorical value triggers a schema violation."""
        sv = SchemaValidator(settings=test_settings)
        sv.generate_schema(df=sample_df)

        bad_df = sample_df.copy()
        bad_df.loc[0, "model"] = "NONEXISTENT_MODEL"
        errors = sv.validate(bad_df)
        assert len(errors) > 0

    def test_save_load_roundtrip(
        self, test_settings: Settings, sample_df: pd.DataFrame
    ) -> None:
        """Schema survives a save → load round-trip."""
        sv = SchemaValidator(settings=test_settings)
        sv.generate_schema(df=sample_df)

        # Load into a fresh instance
        sv2 = SchemaValidator(settings=test_settings)
        loaded = sv2.load_schema()
        assert loaded is not None
        assert len(loaded.columns) == len(sample_df.columns)

    def test_yaml_file_created(
        self, test_settings: Settings, sample_df: pd.DataFrame
    ) -> None:
        """Schema YAML file exists after generation."""
        sv = SchemaValidator(settings=test_settings)
        sv.generate_schema(df=sample_df)
        assert test_settings.schema_path.exists()

    def test_format_errors_empty(self) -> None:
        """format_errors returns success message for empty list."""
        msg = SchemaValidator.format_errors([])
        assert "passed" in msg.lower()

    def test_format_errors_nonempty(self) -> None:
        """format_errors formats error list."""
        errors = [{"column": "age", "check": "in_range", "failure_case": "999", "index": "0"}]
        msg = SchemaValidator.format_errors(errors)
        assert "1 validation error" in msg
        assert "age" in msg


# ============================================================================
# Integration-style Tests
# ============================================================================


class TestIntegration:
    """Light integration tests combining multiple modules."""

    def test_profiler_then_validator(
        self, test_settings: Settings, sample_df: pd.DataFrame
    ) -> None:
        """Profile first, then validate the same data — zero errors."""
        profiler = StatisticalProfiler(settings=test_settings)
        profiler.df = sample_df
        profiler.profile_all_columns()

        sv = SchemaValidator(settings=test_settings)
        sv.generate_schema(df=sample_df)
        errors = sv.validate(sample_df)
        assert len(errors) == 0

    def test_fd_and_profiler_concurrent(
        self, test_settings: Settings, sample_df: pd.DataFrame
    ) -> None:
        """Both profiler and FD discovery can run on the same data."""
        profiler = StatisticalProfiler(settings=test_settings)
        profiler.df = sample_df
        stats = profiler.profile_all_columns()

        fd = FDDiscovery(settings=test_settings)
        fds = fd.run(df=sample_df)

        assert len(stats["columns"]) > 0
        assert fds["_meta"]["n_exact_fds"] >= 0
