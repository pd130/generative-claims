"""Synthetic Insurance Claims Data Generator – Dynamic Prompt Edition.

Uses the profiled statistics, functional dependencies, and schema to
generate high-fidelity synthetic insurance claims data.  The key insight
from FD discovery is that ``model`` determines 33+ columns, so we use a
lookup-table approach for those and sample the remaining columns from
their fitted distributions.

The **dynamic prompt parser** turns natural-language requests into
precise generation plans:
    • Model allocation   – "34 M3 cars, rest divided among others"
    • Percentage splits   – "50 % M3, 30 % M4, 20 % M1"
    • Exclusive selection – "only M3", "all diesel"
    • Fuel / age / risk   – "diesel high-risk young drivers"
    • Exact claim rate    – "10 % claim rate"
    • Exclusions          – "no M5", "without M7"

Architecture:
    1. Load FD lookup tables from the real data (model → attributes).
    2. Parse the user prompt into a structured allocation / filter dict.
    3. Allocate or sample the ``model`` column.
    4. Look up model-determined columns and perturb them.
    5. Sample independent columns (region, customer_age, …).
    6. Calibrate ``claim_status`` to the target rate.

Usage:
    >>> from src.generator.synthetic_generator import SyntheticGenerator
    >>> gen = SyntheticGenerator()
    >>> df = gen.generate(n_rows=1000)
    >>> df = gen.generate(n_rows=2000, prompt="34 M3 cars, rest divided among others")
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from src.utils.config import Settings, get_settings
from src.utils.exceptions import GeneratorError
from src.utils.logging import get_logger

logger = get_logger(__name__)
warnings.filterwarnings("ignore", category=RuntimeWarning)


class SyntheticGenerator:
    """Generate synthetic insurance claims data respecting FDs and distributions.

    The generator uses a multi-phase approach:
        Phase A: Allocate / sample the ``model`` column (supports exact
                 counts, percentages, exclusive selection, and exclusions).
        Phase B: Look up all model-determined columns from real data.
        Phase B2: Perturb numerical FD columns for uniqueness.
        Phase C: Sample remaining independent columns from fitted distributions.
        Phase D: Generate policy IDs.
        Phase E: Calibrate ``claim_status`` to the target rate.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings: Settings = settings or get_settings()
        self._statistics: Optional[Dict[str, Any]] = None
        self._fd_data: Optional[Dict[str, Any]] = None
        self._real_df: Optional[pd.DataFrame] = None
        self._model_lookup: Optional[pd.DataFrame] = None
        self._region_lookup: Optional[pd.DataFrame] = None

    # ── Public API ───────────────────────────────────────────────────

    def generate(
        self,
        n_rows: int = 1000,
        claim_rate: Optional[float] = None,
        seed: Optional[int] = None,
        prompt: Optional[str] = None,
    ) -> pd.DataFrame:
        """Generate synthetic insurance claims data.

        Args:
            n_rows: Number of rows to generate.
            claim_rate: Target claim rate (0-1). Defaults to config value.
            seed: Random seed for reproducibility.
            prompt: Natural language prompt to customize generation.

        Returns:
            DataFrame with the same schema as the original data.
        """
        if seed is None:
            seed = self.settings.generator.random_seed
        rng = np.random.default_rng(seed)

        if claim_rate is None:
            claim_rate = self.settings.generator.target_claim_rate

        logger.info(f"Generating {n_rows:,} synthetic rows (claim_rate={claim_rate})")

        # Load artifacts
        self._load_artifacts()

        # Parse prompt for filters
        filters = self._parse_prompt(prompt) if prompt else {}

        # ── Phase A: Allocate or sample models ────────────────────
        if "model_allocation" in filters:
            models = self._build_allocated_models(filters, n_rows, rng)
        else:
            model_probs = self._get_model_probabilities(filters)
            models = rng.choice(
                list(model_probs.keys()),
                size=n_rows,
                p=list(model_probs.values()),
            )

        # Phase B: Look up model-determined columns
        df = self._lookup_model_attributes(models)

        # Phase B2: Perturb numerical FD columns so data is truly synthetic
        df = self._perturb_numerical_fd_columns(df, rng)

        # Phase C: Sample independent columns
        df = self._sample_independent_columns(df, rng, n_rows, filters)

        # Phase D: Generate policy IDs
        df["policy_id"] = [f"SYN{i:06d}" for i in range(n_rows)]

        # Phase E: Calibrate claim status
        df["claim_status"] = self._generate_claims(n_rows, claim_rate, rng)

        # Reorder columns to match original
        original_cols = self._statistics["_meta"]["columns"]
        df = df[[c for c in original_cols if c in df.columns]]

        logger.info(f"Generated {len(df):,} rows, claim_rate={df['claim_status'].mean():.4f}")
        return df

    # ── Artifact loading ─────────────────────────────────────────────

    def _load_artifacts(self) -> None:
        """Load statistics, FDs, and real data for lookups."""
        if self._statistics is not None:
            return  # already loaded

        # Load statistics
        stats_path = self.settings.statistics_path
        if not stats_path.exists():
            raise GeneratorError(f"Statistics not found at {stats_path}. Run profiler first.")
        with open(stats_path) as f:
            self._statistics = json.load(f)

        # Load FDs
        fd_path = self.settings.fd_path
        if not fd_path.exists():
            raise GeneratorError(f"FDs not found at {fd_path}. Run FD discovery first.")
        with open(fd_path) as f:
            self._fd_data = json.load(f)

        # Load real data for lookup tables
        csv_path = self.settings.raw_data_path
        if not csv_path.exists():
            raise GeneratorError(f"CSV not found at {csv_path}")
        self._real_df = pd.read_csv(csv_path)

        # Build model lookup: one row per unique model with all determined cols
        self._build_model_lookup()
        self._build_region_lookup()

        logger.info("Artifacts loaded successfully")

    def _build_model_lookup(self) -> None:
        """Build a lookup table: model → all model-determined columns."""
        assert self._real_df is not None and self._fd_data is not None

        # Find all columns determined by 'model' from exact FDs
        model_determined: set[str] = set()
        for fd in self._fd_data.get("exact", []):
            if fd["lhs"] == ["model"]:
                model_determined.add(fd["rhs"])

        # Always include model itself
        model_determined.add("model")

        # Get columns that exist in the dataframe
        valid_cols = [c for c in model_determined if c in self._real_df.columns]

        # Take one representative row per model
        self._model_lookup = (
            self._real_df[valid_cols]
            .drop_duplicates(subset=["model"])
            .reset_index(drop=True)
        )
        self._model_determined_cols = valid_cols

        logger.info(
            f"Model lookup: {len(self._model_lookup)} models, "
            f"{len(valid_cols)} determined columns"
        )

    def _build_region_lookup(self) -> None:
        """Build lookup for region_code ↔ region_density."""
        assert self._real_df is not None
        self._region_lookup = (
            self._real_df[["region_code", "region_density"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )

    # ── Generation phases ────────────────────────────────────────────

    def _get_model_probabilities(
        self, filters: Dict[str, Any]
    ) -> Dict[str, float]:
        """Get sampling probabilities for each model.

        Handles simple filters (fuel_type, single model, exclusions).
        For advanced model-count allocation use ``_build_allocated_models``.
        """
        col_stats = self._statistics["columns"]["model"]  # type: ignore
        freqs: Dict[str, float] = dict(col_stats["value_frequencies"])

        # Apply fuel-type filter (single or multi)
        if "fuel_type" in filters or "fuel_types" in filters:
            desired_fuels = filters.get("fuel_types") or [filters["fuel_type"]]
            assert self._model_lookup is not None
            valid_models = set(
                self._model_lookup[
                    self._model_lookup["fuel_type"].str.lower().isin(
                        [f.lower() for f in desired_fuels]
                    )
                ]["model"].values
            )
            freqs = {k: v for k, v in freqs.items() if k in valid_models}

        # Segment filter
        if "segment" in filters:
            desired_seg = filters["segment"]
            assert self._model_lookup is not None
            if "segment" in self._model_lookup.columns:
                valid_models = set(
                    self._model_lookup[
                        self._model_lookup["segment"].str.lower() == desired_seg.lower()
                    ]["model"].values
                )
                freqs = {k: v for k, v in freqs.items() if k in valid_models}

        # Transmission filter
        if "transmission_type" in filters:
            desired_t = filters["transmission_type"]
            assert self._model_lookup is not None
            if "transmission_type" in self._model_lookup.columns:
                valid_models = set(
                    self._model_lookup[
                        self._model_lookup["transmission_type"].str.lower() == desired_t.lower()
                    ]["model"].values
                )
                freqs = {k: v for k, v in freqs.items() if k in valid_models}

        # Steering filter
        if "steering_type" in filters:
            desired_s = filters["steering_type"]
            assert self._model_lookup is not None
            if "steering_type" in self._model_lookup.columns:
                valid_models = set(
                    self._model_lookup[
                        self._model_lookup["steering_type"].str.lower() == desired_s.lower()
                    ]["model"].values
                )
                freqs = {k: v for k, v in freqs.items() if k in valid_models}

        # Brake type filter
        if "rear_brakes_type" in filters:
            desired_b = filters["rear_brakes_type"]
            assert self._model_lookup is not None
            if "rear_brakes_type" in self._model_lookup.columns:
                valid_models = set(
                    self._model_lookup[
                        self._model_lookup["rear_brakes_type"].str.lower() == desired_b.lower()
                    ]["model"].values
                )
                freqs = {k: v for k, v in freqs.items() if k in valid_models}

        # NCAP min filter
        if "ncap_min" in filters:
            min_ncap = filters["ncap_min"]
            assert self._model_lookup is not None
            if "ncap_rating" in self._model_lookup.columns:
                valid_models = set(
                    self._model_lookup[
                        self._model_lookup["ncap_rating"] >= min_ncap
                    ]["model"].values
                )
                freqs = {k: v for k, v in freqs.items() if k in valid_models}

        # Safety-feature ON/OFF filters
        for col_name in filters.get("safety_on", []):
            assert self._model_lookup is not None
            if col_name in self._model_lookup.columns:
                valid_models = set(
                    self._model_lookup[
                        self._model_lookup[col_name] == 1
                    ]["model"].values
                )
                freqs = {k: v for k, v in freqs.items() if k in valid_models}
        for col_name in filters.get("safety_off", []):
            assert self._model_lookup is not None
            if col_name in self._model_lookup.columns:
                valid_models = set(
                    self._model_lookup[
                        self._model_lookup[col_name] == 0
                    ]["model"].values
                )
                freqs = {k: v for k, v in freqs.items() if k in valid_models}

        # Single-model filter
        if "model" in filters:
            desired = filters["model"]
            freqs = {k: v for k, v in freqs.items() if k.lower() == desired.lower()}

        # Exclusions
        excluded: Set[str] = filters.get("exclude_models", set())
        if excluded:
            freqs = {k: v for k, v in freqs.items() if k not in excluded}

        if not freqs:
            # Fallback to all models
            freqs = dict(col_stats["value_frequencies"])

        # Normalize
        total = sum(freqs.values())
        return {k: v / total for k, v in freqs.items()}

    def _build_allocated_models(
        self,
        filters: Dict[str, Any],
        n_rows: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Build model array respecting exact count / percentage allocations.

        Handles:
            - Exact counts per model   (e.g., 34 M3 cars).
            - Percentage allocation     (e.g., 50 % M3).
            - Exclusive selection       (e.g., only M3).
            - Rest distribution         (proportional, equal, or to a named model).
            - Model exclusions          (e.g., no M5).

        Args:
            filters: Parsed prompt filters containing ``model_allocation``.
            n_rows: Total rows to generate.
            rng: Random generator.

        Returns:
            Array of model names with length *n_rows*.
        """
        allocation = filters["model_allocation"]
        rest_strategy = filters.get("rest_strategy", "distribute")
        excluded: Set[str] = filters.get("exclude_models", set())

        # All valid model names from statistics
        all_probs = self._get_model_probabilities({})
        all_models = [m for m in all_probs if m not in excluded]

        models = np.empty(n_rows, dtype=object)
        assigned = 0
        specified_models: set[str] = set()

        for model_name, spec in allocation.items():
            # Validate model exists
            if model_name not in all_probs:
                logger.warning(f"Model '{model_name}' not found in data — skipping.")
                continue

            specified_models.add(model_name)

            # Exclusive: every row is this model
            if spec.get("type") == "exclusive":
                models[:] = model_name
                logger.info(f"Exclusive allocation: all {n_rows} rows → {model_name}")
                rng.shuffle(models)
                return models

            # Count or percent
            if spec["type"] == "count":
                count = min(spec["value"], n_rows - assigned)
            elif spec["type"] == "percent":
                count = int(round(n_rows * spec["value"] / 100.0))
                count = min(count, n_rows - assigned)
            else:
                continue

            models[assigned : assigned + count] = model_name
            assigned += count
            logger.info(f"Allocated {count} rows → {model_name}")

        # ── Handle remaining rows ────────────────────────────────
        remaining = n_rows - assigned
        if remaining > 0:
            if (
                rest_strategy not in ("distribute", "equal")
                and rest_strategy in all_probs
            ):
                # Rest goes to a specific model
                models[assigned:] = rest_strategy
                logger.info(f"Rest ({remaining} rows) → {rest_strategy}")

            elif rest_strategy == "equal":
                other_models = [m for m in all_models if m not in specified_models]
                if not other_models:
                    other_models = all_models
                per_model = remaining // len(other_models)
                leftover = remaining % len(other_models)
                idx = assigned
                for i, model in enumerate(other_models):
                    cnt = per_model + (1 if i < leftover else 0)
                    models[idx : idx + cnt] = model
                    idx += cnt
                logger.info(f"Equal distribution ({remaining} rows) among {len(other_models)} models")

            else:
                # Distribute proportionally among non-specified models
                other_models = [m for m in all_models if m not in specified_models]
                if not other_models:
                    other_models = all_models

                other_probs = {k: all_probs[k] for k in other_models if k in all_probs}
                total = sum(other_probs.values())
                if total > 0:
                    other_probs = {k: v / total for k, v in other_probs.items()}
                else:
                    other_probs = {k: 1.0 / len(other_models) for k in other_models}

                rest_models = rng.choice(
                    list(other_probs.keys()),
                    size=remaining,
                    p=list(other_probs.values()),
                )
                models[assigned:] = rest_models
                logger.info(
                    f"Proportional distribution ({remaining} rows) "
                    f"among {len(other_models)} models"
                )

        # Shuffle so specified models are interspersed, not contiguous
        rng.shuffle(models)
        return models

    def _lookup_model_attributes(self, models: np.ndarray) -> pd.DataFrame:
        """Look up all model-determined columns for sampled models."""
        assert self._model_lookup is not None
        df = pd.DataFrame({"model": models})
        df = df.merge(self._model_lookup, on="model", how="left")
        return df

    def _sample_independent_columns(
        self,
        df: pd.DataFrame,
        rng: np.random.Generator,
        n_rows: int,
        filters: Dict[str, Any],
    ) -> pd.DataFrame:
        """Sample columns NOT determined by model.

        Independent columns: subscription_length, vehicle_age, customer_age,
        region_code / region_density.
        """
        assert self._statistics is not None

        # ── subscription_length ───────────────────────────────────
        sub_vals = self._sample_numerical("subscription_length", n_rows, rng)
        sub_min = filters.get("subscription_min")
        sub_max = filters.get("subscription_max")
        if sub_min is not None:
            sub_vals = np.clip(sub_vals, sub_min, None)
        if sub_max is not None:
            sub_vals = np.clip(sub_vals, None, sub_max)
        df["subscription_length"] = sub_vals

        # ── vehicle_age ───────────────────────────────────────────
        # The fitted distribution for vehicle_age is degenerate (scale=0),
        # so we resample directly from the real data, then convert to months.
        # Prompt filters (vehicle_age_min / _max) are specified in years.
        assert self._real_df is not None
        real_va = self._real_df["vehicle_age"].dropna().values
        v_ages_years = rng.choice(real_va, size=n_rows, replace=True)

        va_min = filters.get("vehicle_age_min")
        va_max = filters.get("vehicle_age_max")
        if va_min is not None or va_max is not None:
            lo = va_min if va_min is not None else real_va.min()
            hi = va_max if va_max is not None else real_va.max()
            pool = real_va[(real_va >= lo) & (real_va <= hi)]
            if len(pool) == 0:
                pool = np.array([lo])  # fallback
            v_ages_years = rng.choice(pool, size=n_rows, replace=True)

        # Add small jitter (±0.1 yr) so synthetic rows aren't exact copies
        jitter = rng.uniform(-0.1, 0.1, size=n_rows)
        v_ages_years = np.clip(v_ages_years + jitter, 0, real_va.max())

        # Convert years → months (integer)
        v_ages_months = np.round(v_ages_years * 12).astype(int)
        df["vehicle_age"] = v_ages_months

        # ── customer_age ──────────────────────────────────────────
        age_min = filters.get("age_min")
        age_max = filters.get("age_max")
        ages = self._sample_numerical("customer_age", n_rows, rng)
        if age_min is not None:
            ages = np.clip(ages, age_min, None)
        if age_max is not None:
            ages = np.clip(ages, None, age_max)
        df["customer_age"] = ages.astype(int)

        # ── region_code + region_density ──────────────────────────
        assert self._region_lookup is not None
        region_stats = self._statistics["columns"]["region_code"]
        region_codes = list(region_stats["value_frequencies"].keys())
        region_probs = list(region_stats["value_frequencies"].values())
        total = sum(region_probs)
        region_probs = [p / total for p in region_probs]

        fixed_region = filters.get("region_code")
        if fixed_region:
            # All rows get the requested region
            sampled_regions = np.full(n_rows, fixed_region)
        else:
            sampled_regions = rng.choice(region_codes, size=n_rows, p=region_probs)
        df["region_code"] = sampled_regions

        # Lookup region_density from region_code
        region_map = dict(
            zip(
                self._region_lookup["region_code"].astype(str),
                self._region_lookup["region_density"],
            )
        )
        df["region_density"] = df["region_code"].map(region_map).astype(int)

        return df

    def _sample_numerical(
        self, col: str, n: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Sample *n* values for a numerical column from its fitted distribution."""
        col_stats = self._statistics["columns"][col]  # type: ignore
        dist_info = col_stats.get("distribution", {})
        vmin = col_stats["min"]
        vmax = col_stats["max"]
        col_mean = col_stats.get("mean", (vmin + vmax) / 2)
        col_std = col_stats.get("std", (vmax - vmin) / 4)

        dist_name = dist_info.get("name", "empirical")
        params = dist_info.get("params", {})

        # Detect degenerate fits (scale ≈ 0 → all samples collapse to loc)
        scale_val = params.get("scale", 1.0)
        is_degenerate = (
            isinstance(scale_val, (int, float)) and abs(scale_val) < 1e-6
        )

        if dist_name in ("empirical", "constant") or is_degenerate:
            if is_degenerate:
                logger.warning(
                    f"Degenerate distribution for '{col}' (scale≈0). "
                    f"Falling back to normal(mean={col_mean:.2f}, std={col_std:.2f})."
                )
            # Use normal distribution anchored to real mean/std, clipped to range
            if col_std > 1e-9:
                values = rng.normal(col_mean, col_std, size=n)
            else:
                values = rng.uniform(vmin, vmax, size=n)
        else:
            try:
                dist = getattr(sp_stats, dist_name)
                shape_names = list(dist.shapes.split(", ")) if dist.shapes else []
                all_names = shape_names + ["loc", "scale"]
                param_values = tuple(params.get(name, 0) for name in all_names)
                values = dist.rvs(
                    *param_values, size=n,
                    random_state=rng.integers(0, 2**31),
                )
            except Exception as exc:
                logger.warning(
                    f"Distribution sampling failed for '{col}': {exc}. Using uniform."
                )
                values = rng.uniform(vmin, vmax, size=n)

        values = np.clip(values, max(vmin, 0), vmax)
        values = values + 0.0  # eliminate -0.0

        if col_stats.get("dtype", "").startswith("int"):
            values = np.round(values).astype(int)
        else:
            values = np.round(values, 1)
        return values

    def _generate_claims(
        self, n: int, claim_rate: float, rng: np.random.Generator
    ) -> np.ndarray:
        """Generate ``claim_status`` with EXACT calibrated rate.

        Places exactly ``round(n * claim_rate)`` ones and shuffles.
        """
        n_claims = int(round(n * claim_rate))
        n_claims = max(0, min(n_claims, n))
        claims = np.zeros(n, dtype=int)
        claims[:n_claims] = 1
        rng.shuffle(claims)
        return claims

    def _perturb_numerical_fd_columns(
        self, df: pd.DataFrame, rng: np.random.Generator
    ) -> pd.DataFrame:
        """Add ±2 % noise to numerical FD-determined columns."""
        assert self._statistics is not None
        PERTURB_FRAC = 0.02

        for col in df.columns:
            if col == "model":
                continue
            col_stats = self._statistics["columns"].get(col, {})
            if col_stats.get("type") != "numerical":
                continue

            vmin = col_stats["min"]
            vmax = col_stats["max"]
            col_range = vmax - vmin
            if col_range < 1e-9:
                continue

            noise = rng.uniform(
                -PERTURB_FRAC * col_range,
                PERTURB_FRAC * col_range,
                size=len(df),
            )
            perturbed = df[col].astype(float).values + noise
            perturbed = np.clip(perturbed, max(vmin, 0), vmax)

            if col_stats.get("dtype", "").startswith("int"):
                df[col] = np.round(perturbed).astype(int)
            else:
                df[col] = np.round(perturbed, 1)

        return df

    # ── Helpers ───────────────────────────────────────────────────────

    def get_available_models(self) -> List[str]:
        """Return sorted list of valid model names."""
        self._load_artifacts()
        assert self._model_lookup is not None
        return sorted(self._model_lookup["model"].tolist())

    def get_available_segments(self) -> List[str]:
        """Return sorted list of valid segments."""
        self._load_artifacts()
        return sorted(self._statistics["columns"]["segment"]["value_frequencies"].keys())  # type: ignore

    def get_available_fuel_types(self) -> List[str]:
        """Return list of valid fuel types."""
        self._load_artifacts()
        return sorted(self._statistics["columns"]["fuel_type"]["value_frequencies"].keys())  # type: ignore

    # ── Prompt parsing ───────────────────────────────────────────────

    def _parse_prompt(self, prompt: str) -> Dict[str, Any]:
        """Parse a natural-language prompt into structured generation filters.

        Capabilities (all case-insensitive):

        **Model allocation** (advanced):
            ``"34 M3 cars"``               → exactly 34 M3 rows
            ``"500 M3 and 300 M4"``        → exact counts for two models
            ``"50% M3, 30% M4, 20% M1"``  → percentage-based split
            ``"only M3"`` / ``"all M3"``   → every row is M3
            ``"no M5"`` / ``"without M7"`` → exclude specific models
            ``"rest divided among others"``→ proportional distribution
            ``"rest equally"``             → equal distribution
            ``"rest M1"``                  → remainder → M1

        **Filtering**:
            ``"diesel"`` / ``"petrol"`` / ``"CNG"`` → fuel type
            ``"young drivers"`` / ``"age 18-25"``   → customer age range
            ``"middle aged"``                       → age 35-50
            ``"vehicle age 2-5"``                   → vehicle age range
            ``"new cars"`` / ``"old cars"``          → vehicle age presets
            ``"subscription 10-30"``                → subscription length
            ``"region C5"``                         → region code
            ``"segment B2"`` / ``"utility"``        → segment filter
            ``"manual"`` / ``"automatic"``          → transmission type
            ``"power steering"`` / ``"electric steering"`` → steering type
            ``"disc brakes"`` / ``"drum brakes"``   → brake type
            ``"with parking camera"``               → safety feature ON
            ``"without esc"``                       → safety feature OFF
            ``"ncap 4+"`` / ``"5 star"``            → NCAP rating
            ``"high risk"`` / ``"low risk"``        → preset claim rates
            ``"10% claim rate"``                    → exact claim rate
            ``"no claims"`` / ``"all claims"``      → 0% / 100% claim rate

        **Row count**:
            ``"2000 rows"`` ``"1k rows"`` ``"generate 5000"``

        Args:
            prompt: Natural language instruction.

        Returns:
            Dict of parsed filters.
        """
        filters: Dict[str, Any] = {}
        p = prompt.lower().strip()

        # ── 1. Row count (inc. shorthand like 1k, 2.5k) ──────────
        m = re.search(
            r"\b(\d+(?:\.\d+)?)\s*k\s*(?:rows?|records?|entries|samples?|data\s*points?)\b",
            p,
        )
        if m:
            filters["n_rows"] = int(float(m.group(1)) * 1000)
        else:
            m = re.search(
                r"\b(\d+)\s*(?:rows?|records?|entries|samples?|data\s*points?)\b", p
            )
            if m:
                filters["n_rows"] = int(m.group(1))
            else:
                m = re.search(r"(?:generate|create|make|produce)\s+(\d+)\b", p)
                if m:
                    filters["n_rows"] = int(m.group(1))

        n_rows_val = filters.get("n_rows")

        # ── 2. Model allocation ───────────────────────────────────
        model_allocation: Dict[str, Dict[str, Any]] = {}

        #  2a  "<model>=<count>" / "<model>:<count>"  e.g. "M3=200", "M4:300"
        #  Process these FIRST so pattern 2b doesn't steal their numbers.
        for mdl, cnt_s in re.findall(r"\b(m\d+)\s*[=:x×]\s*(\d+)\b", p):
            model_allocation[mdl.upper()] = {"type": "count", "value": int(cnt_s)}

        #  2b  "<count> <model>"  e.g. "34 M3", "500 M4 cars"
        #  Skip numbers that are part of an =/:  assignment (negative look-behind).
        for cnt_s, mdl in re.findall(r"(?<![=:x×])\b(\d+)\s+(m\d+)\b", p):
            cnt = int(cnt_s)
            if cnt == n_rows_val:       # skip the row-count number
                continue
            k = mdl.upper()
            if k not in model_allocation:
                model_allocation[k] = {"type": "count", "value": cnt}

        #  2c  "<pct>% <model>"  e.g. "50% M3"
        for pct_s, mdl in re.findall(r"(\d+(?:\.\d+)?)\s*%\s*(m\d+)", p):
            model_allocation[mdl.upper()] = {
                "type": "percent",
                "value": float(pct_s),
            }

        #  2d  "only <model>" / "<model> only" / "all <model>"
        m_only = re.search(
            r"(?:only|all|exclusively|just)\s+(m\d+)\b|\b(m\d+)\s+only\b", p
        )
        if m_only and not model_allocation:
            mdl = (m_only.group(1) or m_only.group(2)).upper()
            model_allocation[mdl] = {"type": "exclusive"}

        #  2e  Exclusions: "no M5", "without M7", "exclude M3"
        excluded: Set[str] = set()
        for mdl in re.findall(
            r"(?:no|exclude|without|except|not)\s+(m\d+)\b", p
        ):
            excluded.add(mdl.upper())
        if excluded:
            filters["exclude_models"] = excluded

        #  2f  Rest strategy
        rest_strategy = "distribute"
        m_rest = re.search(r"(?:rest|remaining|other).*?(m\d+)", p)
        if m_rest:
            rest_strategy = m_rest.group(1).upper()
        elif "equal" in p:
            rest_strategy = "equal"

        if model_allocation:
            filters["model_allocation"] = model_allocation
            filters["rest_strategy"] = rest_strategy

        # ── 3. Simple model filter (backward compat) ──────────────
        if "model_allocation" not in filters:
            for m_single in re.finditer(r"\b(m\d+)\b", p):
                mdl = m_single.group(1).upper()
                if mdl not in excluded:
                    filters["model"] = mdl
                    break

        # ── 4. Fuel type (supports multi: "petrol or diesel") ─────
        found_fuels: List[str] = []
        for fuel in ("diesel", "petrol", "cng"):
            if re.search(r"\b" + fuel + r"\b", p):
                found_fuels.append(fuel.upper() if fuel == "cng" else fuel.capitalize())
        if len(found_fuels) == 1:
            filters["fuel_type"] = found_fuels[0]
        elif len(found_fuels) > 1:
            filters["fuel_types"] = found_fuels  # plural

        # ── 5. Segment filter ─────────────────────────────────────
        #  e.g. "segment B2", "utility vehicles", "A segment"
        m_seg = re.search(
            r"\b(?:segment|seg)\s*(a|b1|b2|c1|c2|utility)\b"
            r"|\b(a|b1|b2|c1|c2|utility)\s*segment\b"
            r"|\b(utility)\s*(?:vehicles?)?\b",
            p,
        )
        if m_seg:
            seg = (m_seg.group(1) or m_seg.group(2) or m_seg.group(3)).upper()
            if seg == "UTILITY":
                seg = "Utility"  # exact casing in data
            filters["segment"] = seg

        # ── 6. Transmission type ──────────────────────────────────
        if re.search(r"\b(?:automatic|auto)\b", p):
            filters["transmission_type"] = "Automatic"
        elif re.search(r"\bmanual\b", p):
            filters["transmission_type"] = "Manual"

        # ── 7. Steering type ──────────────────────────────────────
        m_steer = re.search(r"\b(power|electric|manual)\s*steering\b", p)
        if m_steer:
            filters["steering_type"] = m_steer.group(1).capitalize()

        # ── 8. Brake type ─────────────────────────────────────────
        if re.search(r"\bdisc\s*brakes?\b", p):
            filters["rear_brakes_type"] = "Disc"
        elif re.search(r"\bdrum\s*brakes?\b", p):
            filters["rear_brakes_type"] = "Drum"

        # ── 9. Safety-feature toggles ─────────────────────────────
        #  "with parking camera", "without esc", "has tpms"
        safety_map = {
            "esc": "is_esc",
            "tpms": "is_tpms",
            "parking sensor": "is_parking_sensors",
            "parking camera": "is_parking_camera",
            "fog light": "is_front_fog_lights",
            "rear wiper": "is_rear_window_wiper",
            "brake assist": "is_brake_assist",
            "power lock": "is_power_door_locks",
            "central lock": "is_central_locking",
            "speed alert": "is_speed_alert",
            "ecw": "is_ecw",
        }
        for keyword, col_name in safety_map.items():
            m_on = re.search(r"(?:with|has|enable)\s+" + keyword, p)
            m_off = re.search(r"(?:without|no|disable)\s+" + keyword, p)
            if m_on:
                filters.setdefault("safety_on", []).append(col_name)
            elif m_off:
                filters.setdefault("safety_off", []).append(col_name)

        # ── 10. NCAP rating ───────────────────────────────────────
        m_ncap = re.search(r"(?:ncap|star)\s*(\d)\+?", p)
        if not m_ncap:
            m_ncap = re.search(r"(\d)\s*star", p)
        if m_ncap:
            filters["ncap_min"] = int(m_ncap.group(1))

        # ── 11. Customer age range ────────────────────────────────
        m_age = re.search(r"(?:customer\s*)?age\s*(\d+)\s*[-–to]+\s*(\d+)", p)
        if m_age:
            filters["age_min"] = int(m_age.group(1))
            filters["age_max"] = int(m_age.group(2))
        elif "young" in p:
            filters["age_min"], filters["age_max"] = 18, 35
        elif re.search(r"\bmiddle\s*aged?\b", p):
            filters["age_min"], filters["age_max"] = 35, 50
        elif re.search(r"\b(?:old|senior|elderly)\b", p):
            filters["age_min"], filters["age_max"] = 55, 75

        # ── 12. Vehicle age presets ────────────────────────────────
        m_va = re.search(r"vehicle\s*age\s*(\d+)\s*[-–to]+\s*(\d+)", p)
        if m_va:
            filters["vehicle_age_min"] = int(m_va.group(1))
            filters["vehicle_age_max"] = int(m_va.group(2))
        elif re.search(r"\b(?:brand\s*new|new\s*cars?|new\s*vehicles?)\b", p):
            filters["vehicle_age_min"] = 0
            filters["vehicle_age_max"] = 2
        elif re.search(r"\b(?:old\s*cars?|old\s*vehicles?|used\s*cars?)\b", p):
            filters["vehicle_age_min"] = 8
            filters["vehicle_age_max"] = 20

        # ── 13. Subscription length range ─────────────────────────
        m_sub = re.search(
            r"subscription\s*(?:length)?\s*(\d+)\s*[-–to]+\s*(\d+)", p
        )
        if m_sub:
            filters["subscription_min"] = int(m_sub.group(1))
            filters["subscription_max"] = int(m_sub.group(2))

        # ── 14. Region filter ─────────────────────────────────────
        m_reg = re.search(r"region\s*(?:code)?\s*(c\d+)", p)
        if m_reg:
            filters["region_code"] = m_reg.group(1).upper()

        # ── 15. Claim rate ────────────────────────────────────────
        if re.search(r"\bno\s*claims?\b|\bzero\s*claims?\b|\b0\s*%?\s*claim", p):
            filters["claim_rate_override"] = 0.0
        elif re.search(r"\ball\s*claims?\b|\b100\s*%\s*claim", p):
            filters["claim_rate_override"] = 1.0
        elif "high risk" in p or "risky" in p:
            filters["claim_rate_override"] = 0.15
        elif "low risk" in p or "safe" in p:
            filters["claim_rate_override"] = 0.02

        m_claim = re.search(r"(\d+(?:\.\d+)?)\s*%\s*claim", p)
        if m_claim:
            filters["claim_rate_override"] = float(m_claim.group(1)) / 100.0

        logger.info(f"Parsed prompt → {filters}")
        return filters

    # ── Convenience wrapper ──────────────────────────────────────────

    def generate_from_prompt(
        self, prompt: str, seed: Optional[int] = None
    ) -> pd.DataFrame:
        """Generate synthetic data from a natural language prompt.

        Args:
            prompt: E.g., "2000 rows with 34 M3 cars and rest divided among others"
            seed: Random seed.

        Returns:
            Synthetic DataFrame.
        """
        filters = self._parse_prompt(prompt)
        n_rows = filters.pop("n_rows", 1000)
        claim_rate = filters.pop(
            "claim_rate_override",
            self.settings.generator.target_claim_rate,
        )
        return self.generate(
            n_rows=n_rows,
            claim_rate=claim_rate,
            seed=seed,
            prompt=prompt,
        )


# ── CLI entry point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    gen = SyntheticGenerator()
    df = gen.generate(n_rows=100)
    print(df.head(10).to_string())
    print(f"\nShape: {df.shape}")
    print(f"Claim rate: {df['claim_status'].mean():.4f}")
