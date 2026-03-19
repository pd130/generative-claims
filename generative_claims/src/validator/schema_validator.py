"""Schema Validator for Insurance Claims Data.

Auto-generates a pandera ``DataFrameSchema`` from the real CSV, persists
it to YAML, and provides a ``validate()`` method for checking synthetic
data against the learned schema.

Design Decisions:
    1. **pandera** gives us declarative, composable column checks with
       rich error messages — much better than hand-rolled if/else.
    2. **Auto-generation from data** means the schema tracks the real
       distribution: ranges, cardinalities, and null-fractions are
       learned, not guessed.
    3. **YAML persistence** makes the schema human-editable (a data
       engineer can tighten constraints without touching Python).
    4. **Tolerance margins** (±10 % on numeric ranges) prevent false
       positives when the generator intentionally explores boundaries.
    5. **Two-level validation**: ``strict`` (reject on first failure) and
       ``report`` (collect all violations for analysis).

Usage:
    >>> from src.validator.schema_validator import SchemaValidator
    >>> sv = SchemaValidator()
    >>> sv.generate_schema()          # learn from real data
    >>> errors = sv.validate(synth_df)  # check synthetic data
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema, Check
import yaml

from src.utils.config import Settings, get_settings
from src.utils.exceptions import (
    DataLoadError,
    SchemaGenerationError,
    SchemaValidationError,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)

# ── Type aliases ─────────────────────────────────────────────────────────
SchemaDict = Dict[str, Any]


class SchemaValidator:
    """Generate, persist, and apply a pandera schema for claims data.

    Attributes:
        settings: Application configuration.
        schema: The pandera DataFrameSchema (populated after generation).
    """

    # Tolerance factor applied to numeric min/max when building checks
    RANGE_TOLERANCE: float = 0.10  # ±10 %

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings: Settings = settings or get_settings()
        self.schema: Optional[DataFrameSchema] = None
        self._schema_dict: Optional[SchemaDict] = None
        self._df: Optional[pd.DataFrame] = None

    # ── Public API ───────────────────────────────────────────────────

    def generate_schema(
        self, df: Optional[pd.DataFrame] = None
    ) -> DataFrameSchema:
        """Learn a schema from the real dataset.

        Args:
            df: Optional pre-loaded DataFrame. If ``None``, loads CSV.

        Returns:
            A pandera ``DataFrameSchema`` encoding types, ranges,
            allowed values, and null-fraction constraints.
        """
        t0 = time.perf_counter()
        logger.info("Schema generation started")

        if df is not None:
            self._df = df
        else:
            self._load_data()

        assert self._df is not None
        columns: Dict[str, Column] = {}
        schema_dict: SchemaDict = {"columns": {}}

        for col in self._df.columns:
            col_schema, col_dict = self._build_column_schema(col)
            columns[col] = col_schema
            schema_dict["columns"][col] = col_dict

        self.schema = DataFrameSchema(
            columns=columns,
            coerce=True,
            strict=False,  # allow extra columns in synthetic data
        )

        schema_dict["_meta"] = {
            "n_columns": len(columns),
            "generated_from_rows": int(len(self._df)),
            "generated_at": pd.Timestamp.now().isoformat(),
            "range_tolerance": self.RANGE_TOLERANCE,
        }
        self._schema_dict = schema_dict

        self.save_schema()

        elapsed = time.perf_counter() - t0
        logger.info(f"Schema generated for {len(columns)} columns in {elapsed:.2f}s")
        return self.schema

    def validate(
        self,
        df: pd.DataFrame,
        lazy: bool = True,
    ) -> List[Dict[str, Any]]:
        """Validate a DataFrame against the learned schema.

        Args:
            df: DataFrame to validate (typically synthetic data).
            lazy: If ``True``, collect all errors instead of failing fast.

        Returns:
            List of error dicts. Empty list means full pass.

        Raises:
            SchemaValidationError: If schema has not been generated yet.
        """
        if self.schema is None:
            # Try to load from YAML
            self.load_schema()
            if self.schema is None:
                raise SchemaValidationError(
                    "No schema available. Call generate_schema() first."
                )

        errors: List[Dict[str, Any]] = []
        try:
            self.schema.validate(df, lazy=lazy)
            logger.info("Validation PASSED — no schema violations.")
        except pa.errors.SchemaErrors as exc:
            for _, row in exc.failure_cases.iterrows():
                errors.append(
                    {
                        "column": str(row.get("column", "unknown")),
                        "check": str(row.get("check", "unknown")),
                        "failure_case": str(row.get("failure_case", "")),
                        "index": str(row.get("index", "")),
                    }
                )
            logger.warning(f"Validation FAILED — {len(errors)} violations found.")
        except pa.errors.SchemaError as exc:
            errors.append({"column": "N/A", "check": str(exc)})
            logger.warning(f"Validation FAILED — {exc}")

        return errors

    def save_schema(self) -> Path:
        """Persist the schema dict to YAML.

        Returns:
            Path to the written YAML file.
        """
        if self._schema_dict is None:
            raise SchemaGenerationError("No schema to save.")

        out_path = self.settings.schema_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w", encoding="utf-8") as fh:
            yaml.dump(
                self._schema_dict,
                fh,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )

        logger.info(f"Schema saved to {out_path}")
        return out_path

    def load_schema(self) -> Optional[DataFrameSchema]:
        """Load schema from YAML and reconstruct pandera schema.

        Returns:
            Reconstructed ``DataFrameSchema``, or ``None`` if file missing.
        """
        yaml_path = self.settings.schema_path
        if not yaml_path.exists():
            logger.warning(f"Schema file not found at {yaml_path}")
            return None

        with open(yaml_path, "r", encoding="utf-8") as fh:
            schema_dict = yaml.safe_load(fh)

        self._schema_dict = schema_dict
        columns: Dict[str, Column] = {}

        for col_name, col_def in schema_dict.get("columns", {}).items():
            columns[col_name] = self._reconstruct_column(col_name, col_def)

        self.schema = DataFrameSchema(columns=columns, coerce=True, strict=False)
        logger.info(f"Schema loaded from {yaml_path} ({len(columns)} columns)")
        return self.schema

    # ── Column schema builders ───────────────────────────────────────

    def _build_column_schema(
        self, col: str
    ) -> Tuple[Column, Dict[str, Any]]:
        """Build a pandera Column and a serializable dict for one column.

        Args:
            col: Column name.

        Returns:
            Tuple of (pandera Column, dict representation).
        """
        assert self._df is not None
        series = self._df[col]
        col_dict: Dict[str, Any] = {}

        null_frac = float(series.isna().mean())
        nullable = null_frac > 0
        col_dict["nullable"] = nullable
        col_dict["null_fraction"] = round(null_frac, 6)

        checks: List[Check] = []

        if pd.api.types.is_numeric_dtype(series):
            col_dict["dtype"] = "numeric"
            clean = series.dropna()
            vmin = float(clean.min())
            vmax = float(clean.max())

            # Apply tolerance
            margin = max(abs(vmin), abs(vmax)) * self.RANGE_TOLERANCE
            lower = vmin - margin
            upper = vmax + margin

            col_dict["min"] = round(vmin, 6)
            col_dict["max"] = round(vmax, 6)
            col_dict["check_lower"] = round(lower, 6)
            col_dict["check_upper"] = round(upper, 6)

            checks.append(Check.in_range(lower, upper))

            # Integer check
            if pd.api.types.is_integer_dtype(series):
                col_dict["is_integer"] = True

            return (
                Column(
                    dtype=float if not pd.api.types.is_integer_dtype(series) else int,
                    checks=checks,
                    nullable=nullable,
                    coerce=True,
                    required=True,
                ),
                col_dict,
            )
        else:
            col_dict["dtype"] = "categorical"
            clean = series.dropna()
            allowed = sorted(clean.unique().astype(str).tolist())
            col_dict["allowed_values"] = allowed
            col_dict["cardinality"] = len(allowed)

            checks.append(Check.isin(allowed))

            return (
                Column(
                    dtype=str,
                    checks=checks,
                    nullable=nullable,
                    coerce=True,
                    required=True,
                ),
                col_dict,
            )

    def _reconstruct_column(
        self, col_name: str, col_def: Dict[str, Any]
    ) -> Column:
        """Rebuild a pandera Column from a YAML-loaded dict.

        Args:
            col_name: Column name.
            col_def: Dict from YAML.

        Returns:
            pandera Column instance.
        """
        checks: List[Check] = []
        dtype_str = col_def.get("dtype", "categorical")
        nullable = col_def.get("nullable", False)

        if dtype_str == "numeric":
            lower = col_def.get("check_lower")
            upper = col_def.get("check_upper")
            if lower is not None and upper is not None:
                checks.append(Check.in_range(lower, upper))
            dtype = int if col_def.get("is_integer", False) else float
        else:
            allowed = col_def.get("allowed_values", [])
            if allowed:
                checks.append(Check.isin(allowed))
            dtype = str

        return Column(
            dtype=dtype,
            checks=checks,
            nullable=nullable,
            coerce=True,
            required=True,
        )

    # ── Helpers ──────────────────────────────────────────────────────

    def _load_data(self) -> None:
        """Load the raw CSV."""
        csv_path = self.settings.raw_data_path
        if not csv_path.exists():
            raise DataLoadError(f"CSV not found at {csv_path}")
        try:
            self._df = pd.read_csv(csv_path)
            logger.info(f"Loaded {self._df.shape[0]:,} rows for schema generation")
        except Exception as exc:
            raise DataLoadError(f"Failed to read CSV: {exc}") from exc

    @staticmethod
    def format_errors(errors: List[Dict[str, Any]]) -> str:
        """Format validation errors into a readable report.

        Args:
            errors: List of error dicts from ``validate()``.

        Returns:
            Multi-line formatted string.
        """
        if not errors:
            return "✓ All checks passed."

        lines = [f"✗ {len(errors)} validation error(s):", ""]
        for i, err in enumerate(errors[:50], 1):
            lines.append(
                f"  {i:3d}. Column: {err['column']:<30s}  "
                f"Check: {err['check']}"
            )
        if len(errors) > 50:
            lines.append(f"  ... and {len(errors) - 50} more.")
        return "\n".join(lines)


# ── CLI entry point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    sv = SchemaValidator()
    sv.generate_schema()
    print(f"Schema generated and saved to {sv.settings.schema_path}")
