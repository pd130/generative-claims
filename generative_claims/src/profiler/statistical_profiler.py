"""Statistical Profiler for Insurance Claims Data.

Generates per-column statistics, fits parametric distributions, and
persists everything to ``data/processed/statistics.json``.

Design Decisions:
    1. **Vectorized NumPy/pandas** for speed on 200k+ rows.
    2. **scipy.stats** distribution fitting with AIC-based model selection
       so downstream generators use the best-fit family + params.
    3. **Categorical profiling** captures value counts *and* relative
       frequencies, enabling faithful discrete sampling later.
    4. **Deterministic output** — JSON is human-readable and diffable,
       enabling CI regression checks on schema drift.
    5. **Graceful degradation** — if a distribution fit fails for a column,
       it logs a warning and falls back to empirical percentiles only.

Usage:
    >>> from src.profiler.statistical_profiler import StatisticalProfiler
    >>> profiler = StatisticalProfiler()
    >>> stats = profiler.run()  # loads CSV, profiles, saves JSON
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from src.utils.config import Settings, get_settings
from src.utils.exceptions import (
    DataLoadError,
    DistributionFitError,
    StatisticsComputationError,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Silence scipy optimisation chatter during fitting
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── Type aliases ─────────────────────────────────────────────────────────
ColumnStats = Dict[str, Any]
ProfileResult = Dict[str, Any]


class StatisticalProfiler:
    """Compute and persist per-column statistics for the claims dataset.

    Attributes:
        settings: Application configuration.
        df: Loaded dataframe (populated after ``load_data``).
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings: Settings = settings or get_settings()
        self.df: Optional[pd.DataFrame] = None
        self._profile: Optional[ProfileResult] = None

    # ── Public API ───────────────────────────────────────────────────

    def run(self) -> ProfileResult:
        """Execute the full profiling pipeline.

        Returns:
            A nested dict containing per-column statistics.

        Raises:
            DataLoadError: If the CSV cannot be read.
            StatisticsComputationError: If profiling fails critically.
        """
        t0 = time.perf_counter()
        logger.info("Statistical profiling started")

        self.load_data()
        profile = self.profile_all_columns()
        self.save(profile)

        elapsed = time.perf_counter() - t0
        logger.info(f"Statistical profiling completed in {elapsed:.2f}s")
        return profile

    def load_data(self) -> pd.DataFrame:
        """Load the raw CSV into ``self.df``.

        Returns:
            The loaded DataFrame.

        Raises:
            DataLoadError: If the file is missing or unreadable.
        """
        csv_path = self.settings.raw_data_path
        logger.info(f"Loading data from {csv_path}")

        if not csv_path.exists():
            raise DataLoadError(f"CSV not found at {csv_path}")

        try:
            self.df = pd.read_csv(csv_path)
            logger.info(
                f"Loaded {self.df.shape[0]:,} rows × {self.df.shape[1]} columns"
            )
            return self.df
        except Exception as exc:
            raise DataLoadError(f"Failed to read CSV: {exc}") from exc

    def profile_all_columns(self) -> ProfileResult:
        """Profile every column in the loaded DataFrame.

        Returns:
            A dict keyed by column name, each value a ``ColumnStats`` dict.

        Raises:
            StatisticsComputationError: On critical failure.
        """
        if self.df is None:
            raise StatisticsComputationError("No data loaded. Call load_data() first.")

        profile: ProfileResult = {
            "_meta": {
                "n_rows": int(self.df.shape[0]),
                "n_columns": int(self.df.shape[1]),
                "columns": list(self.df.columns),
                "profiled_at": pd.Timestamp.now().isoformat(),
            },
            "columns": {},
        }

        for col in self.df.columns:
            try:
                profile["columns"][col] = self._profile_column(col)
            except Exception as exc:
                logger.warning(f"Failed to profile column '{col}': {exc}")
                profile["columns"][col] = {"error": str(exc)}

        self._profile = profile
        return profile

    def save(self, profile: Optional[ProfileResult] = None) -> Path:
        """Persist the profile to JSON.

        Args:
            profile: Profile dict; uses cached result if ``None``.

        Returns:
            Path to the written JSON file.
        """
        profile = profile or self._profile
        if profile is None:
            raise StatisticsComputationError("No profile to save. Run profiling first.")

        out_path = self.settings.statistics_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(profile, fh, indent=2, default=_json_serializer)

        logger.info(f"Statistics saved to {out_path}")
        return out_path

    # ── Private helpers ──────────────────────────────────────────────

    def _profile_column(self, col: str) -> ColumnStats:
        """Profile a single column, dispatching by dtype.

        Args:
            col: Column name.

        Returns:
            Statistics dict for the column.
        """
        series = self.df[col]  # type: ignore[index]
        base: ColumnStats = {
            "dtype": str(series.dtype),
            "n_total": int(len(series)),
            "n_missing": int(series.isna().sum()),
            "null_fraction": round(float(series.isna().mean()), 6),
            "n_unique": int(series.nunique()),
        }

        if pd.api.types.is_numeric_dtype(series):
            base["type"] = "numerical"
            base.update(self._profile_numerical(series))
        else:
            base["type"] = "categorical"
            base.update(self._profile_categorical(series))

        return base

    def _profile_numerical(self, series: pd.Series) -> ColumnStats:
        """Compute statistics for a numerical column.

        Includes descriptive stats, percentiles, and best-fit distribution.

        Args:
            series: Numeric pandas Series.

        Returns:
            Dict of numerical statistics.
        """
        clean = series.dropna().astype(float)
        result: ColumnStats = {}

        # Descriptive statistics (vectorized)
        result["min"] = float(clean.min())
        result["max"] = float(clean.max())
        result["mean"] = float(clean.mean())
        result["median"] = float(clean.median())
        result["std"] = float(clean.std())
        result["skewness"] = float(clean.skew())
        result["kurtosis"] = float(clean.kurt())

        # Percentiles
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        pct_values = np.percentile(clean.values, percentiles)
        result["percentiles"] = {
            f"p{p}": round(float(v), 6) for p, v in zip(percentiles, pct_values)
        }

        # Distribution fitting
        result["distribution"] = self._fit_best_distribution(clean)

        return result

    def _profile_categorical(self, series: pd.Series) -> ColumnStats:
        """Compute statistics for a categorical column.

        Args:
            series: Object/categorical pandas Series.

        Returns:
            Dict of categorical statistics (value counts, frequencies).
        """
        clean = series.dropna()
        vc = clean.value_counts()
        freq = clean.value_counts(normalize=True)

        result: ColumnStats = {
            "cardinality": int(clean.nunique()),
            "mode": str(vc.index[0]) if len(vc) > 0 else None,
            "mode_frequency": round(float(freq.iloc[0]), 6) if len(freq) > 0 else None,
        }

        # Full value-count table (capped at 100 for sanity)
        top_n = min(100, len(vc))
        result["value_counts"] = {
            str(k): int(v) for k, v in vc.head(top_n).items()
        }
        result["value_frequencies"] = {
            str(k): round(float(v), 6) for k, v in freq.head(top_n).items()
        }

        # Binary detection
        unique_vals = set(clean.unique())
        if unique_vals <= {"Yes", "No"} or unique_vals <= {0, 1}:
            result["is_binary"] = True
            positive = clean.isin(["Yes", 1]).mean()
            result["positive_rate"] = round(float(positive), 6)
        else:
            result["is_binary"] = False

        return result

    def _fit_best_distribution(
        self, data: pd.Series
    ) -> Dict[str, Any]:
        """Fit candidate distributions and select the best by AIC.

        Uses Maximum Likelihood Estimation via ``scipy.stats.fit`` and
        selects the distribution with the lowest Akaike Information
        Criterion.

        Args:
            data: Cleaned numeric Series (no NaNs).

        Returns:
            Dict with keys ``name``, ``params``, ``aic``, ``ks_statistic``,
            ``ks_pvalue``.
        """
        arr = data.values.astype(float)

        # Skip fitting for constant or near-constant columns
        if np.std(arr) < 1e-10:
            return {"name": "constant", "params": {"value": float(arr[0])}, "aic": None}

        candidates: List[Tuple[str, float, Dict[str, float], float, float]] = []

        for dist_name in self.settings.profiler.candidate_distributions:
            try:
                dist = getattr(sp_stats, dist_name)
                params = dist.fit(arr)

                # Log-likelihood → AIC
                log_lik = np.sum(dist.logpdf(arr, *params))
                if not np.isfinite(log_lik):
                    continue
                k = len(params)
                aic = 2 * k - 2 * log_lik

                # Kolmogorov-Smirnov test
                ks_stat, ks_p = sp_stats.kstest(arr, dist_name, args=params)

                candidates.append((dist_name, aic, _params_to_dict(dist_name, params), ks_stat, ks_p))
            except Exception as exc:
                logger.debug(f"Distribution '{dist_name}' failed for column: {exc}")
                continue

        if not candidates:
            logger.warning("No distribution fit succeeded; using empirical fallback")
            return {"name": "empirical", "params": {}, "aic": None}

        # Select by lowest AIC
        best = min(candidates, key=lambda c: c[1])
        return {
            "name": best[0],
            "params": best[2],
            "aic": round(best[1], 4),
            "ks_statistic": round(best[3], 6),
            "ks_pvalue": round(best[4], 6),
        }


# ── Module-level helpers ─────────────────────────────────────────────────


def _params_to_dict(dist_name: str, params: tuple) -> Dict[str, float]:
    """Convert scipy fit params tuple to a named dict.

    scipy.stats distributions return ``(*shape, loc, scale)``; this
    function labels them properly per distribution.

    Args:
        dist_name: Name of the scipy distribution.
        params: Fitted parameter tuple.

    Returns:
        Dict mapping parameter names to values.
    """
    dist = getattr(sp_stats, dist_name)
    shape_names: List[str] = list(dist.shapes.split(", ")) if dist.shapes else []
    names = shape_names + ["loc", "scale"]

    return {n: round(float(v), 8) for n, v in zip(names, params)}


def _json_serializer(obj: Any) -> Any:
    """Fallback JSON serializer for numpy/pandas types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ── CLI entry point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    profiler = StatisticalProfiler()
    result = profiler.run()
    print(f"Profiled {len(result['columns'])} columns.")
