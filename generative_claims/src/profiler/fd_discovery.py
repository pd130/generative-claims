"""Functional Dependency (FD) Discovery for Insurance Claims Data.

Discovers exact and probabilistic functional dependencies between
columns, enabling the Generator to maintain realistic co-occurrence
patterns (e.g., ``model → engine_type``, ``model → fuel_type``).

Design Decisions:
    1. **Brute-force exact FD check**: For each candidate ``X → Y``,
       verify that every distinct ``X`` value maps to exactly one ``Y``
       value. This is O(n) per pair using pandas ``groupby().nunique()``.
    2. **Probabilistic FDs**: Relax the requirement — accept ``X → Y``
       if ≥ *threshold* (default 95 %) of ``X`` groups are deterministic.
       This handles real-world noise (typos, versioning) gracefully.
    3. **LHS combinations**: Support composite keys (e.g.,
       ``(model, fuel_type) → engine_type``), capped at
       ``fd_max_lhs_size`` columns to keep runtime tractable.
    4. **Sampling guard**: For very large datasets, subsample to
       ``fd_sample_size`` rows to keep discovery under 60 s.
    5. **Pruning**: Skip identity FDs and superkey FDs (X is unique →
       trivially determines everything).

Output schema (``functional_dependencies.json``):
    {
        "exact": [{"lhs": ["model"], "rhs": "engine_type", "confidence": 1.0}],
        "probabilistic": [{"lhs": ["model"], "rhs": "displacement", "confidence": 0.97}],
        "_meta": { ... }
    }

Usage:
    >>> from src.profiler.fd_discovery import FDDiscovery
    >>> fd = FDDiscovery()
    >>> result = fd.run()
"""

from __future__ import annotations

import itertools
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from src.utils.config import Settings, get_settings
from src.utils.exceptions import DataLoadError, FunctionalDependencyError
from src.utils.logging import get_logger

logger = get_logger(__name__)

# ── Type aliases ─────────────────────────────────────────────────────────
FDRecord = Dict[str, Any]
FDResult = Dict[str, Any]


class FDDiscovery:
    """Discover functional dependencies in the insurance claims dataset.

    Attributes:
        settings: Application configuration.
        df: Loaded (and possibly sampled) DataFrame.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings: Settings = settings or get_settings()
        self.df: Optional[pd.DataFrame] = None
        self._result: Optional[FDResult] = None

    # ── Public API ───────────────────────────────────────────────────

    def run(self, df: Optional[pd.DataFrame] = None) -> FDResult:
        """Execute the full FD discovery pipeline.

        Args:
            df: Optional pre-loaded DataFrame. If ``None``, loads from CSV.

        Returns:
            Dict with ``exact``, ``probabilistic``, and ``_meta`` keys.
        """
        t0 = time.perf_counter()
        logger.info("Functional dependency discovery started")

        if df is not None:
            self.df = df
        else:
            self._load_data()

        # Subsample for performance
        self._maybe_subsample()

        # Identify candidate columns (exclude unique IDs)
        candidates = self._select_candidate_columns()
        logger.info(f"Candidate columns for FD discovery: {len(candidates)}")

        exact_fds: List[FDRecord] = []
        prob_fds: List[FDRecord] = []

        # Enumerate LHS combinations
        max_lhs = self.settings.profiler.fd_max_lhs_size
        threshold = self.settings.profiler.fd_confidence_threshold

        for lhs_size in range(1, max_lhs + 1):
            for lhs_combo in itertools.combinations(candidates, lhs_size):
                lhs = list(lhs_combo)
                for rhs in candidates:
                    if rhs in lhs:
                        continue
                    confidence = self._compute_fd_confidence(lhs, rhs)
                    if confidence is None:
                        continue

                    record: FDRecord = {
                        "lhs": lhs,
                        "rhs": rhs,
                        "confidence": round(confidence, 6),
                    }

                    if confidence >= 1.0 - 1e-9:
                        exact_fds.append(record)
                    elif confidence >= threshold:
                        prob_fds.append(record)

        # Sort by confidence descending
        exact_fds.sort(key=lambda r: r["confidence"], reverse=True)
        prob_fds.sort(key=lambda r: r["confidence"], reverse=True)

        elapsed = time.perf_counter() - t0
        result: FDResult = {
            "_meta": {
                "n_exact_fds": len(exact_fds),
                "n_probabilistic_fds": len(prob_fds),
                "confidence_threshold": threshold,
                "max_lhs_size": max_lhs,
                "n_rows_analysed": int(len(self.df)),  # type: ignore[arg-type]
                "elapsed_seconds": round(elapsed, 2),
                "discovered_at": pd.Timestamp.now().isoformat(),
            },
            "exact": exact_fds,
            "probabilistic": prob_fds,
        }

        self._result = result
        self.save(result)

        logger.info(
            f"FD discovery completed: {len(exact_fds)} exact, "
            f"{len(prob_fds)} probabilistic in {elapsed:.2f}s"
        )
        return result

    def save(self, result: Optional[FDResult] = None) -> Path:
        """Persist FD results to JSON.

        Args:
            result: FD result dict; uses cached if ``None``.

        Returns:
            Path to the written file.
        """
        result = result or self._result
        if result is None:
            raise FunctionalDependencyError("No FD result to save.")

        out_path = self.settings.fd_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)

        logger.info(f"Functional dependencies saved to {out_path}")
        return out_path

    # ── Confidence computation ───────────────────────────────────────

    def _compute_fd_confidence(
        self, lhs: List[str], rhs: str
    ) -> Optional[float]:
        """Compute the confidence of the FD ``lhs → rhs``.

        Confidence = fraction of LHS groups where the RHS has exactly
        one distinct value.

        Args:
            lhs: Left-hand-side column(s).
            rhs: Right-hand-side column.

        Returns:
            Confidence in [0, 1], or ``None`` if computation fails.
        """
        assert self.df is not None
        try:
            grouped = self.df.groupby(lhs, observed=True)[rhs].nunique()
            n_deterministic = int((grouped == 1).sum())
            n_groups = len(grouped)
            if n_groups == 0:
                return None
            return n_deterministic / n_groups
        except Exception as exc:
            logger.debug(f"FD confidence failed for {lhs} → {rhs}: {exc}")
            return None

    # ── Helpers ──────────────────────────────────────────────────────

    def _load_data(self) -> None:
        """Load raw CSV into ``self.df``."""
        csv_path = self.settings.raw_data_path
        if not csv_path.exists():
            raise DataLoadError(f"CSV not found at {csv_path}")

        try:
            self.df = pd.read_csv(csv_path)
            logger.info(f"Loaded {self.df.shape[0]:,} rows for FD discovery")
        except Exception as exc:
            raise DataLoadError(f"Failed to read CSV: {exc}") from exc

    def _maybe_subsample(self) -> None:
        """Subsample the DataFrame if it exceeds the configured limit."""
        assert self.df is not None
        max_rows = self.settings.profiler.fd_sample_size
        if len(self.df) > max_rows:
            logger.info(
                f"Subsampling from {len(self.df):,} to {max_rows:,} rows "
                f"for FD discovery"
            )
            self.df = self.df.sample(n=max_rows, random_state=42).reset_index(
                drop=True
            )

    def _select_candidate_columns(self) -> List[str]:
        """Select columns suitable for FD discovery.

        Excludes:
            - Columns with unique-per-row values (IDs, primary keys).
            - Columns with cardinality == 1 (constant).
            - High-cardinality numerical columns (> 500 unique values)
              which almost never participate in meaningful FDs.

        Returns:
            List of candidate column names.
        """
        assert self.df is not None
        candidates: List[str] = []
        n_rows = len(self.df)

        for col in self.df.columns:
            nunique = self.df[col].nunique()
            # Skip unique IDs (cardinality ≥ 95% of rows)
            if nunique >= 0.95 * n_rows:
                logger.debug(f"Skipping '{col}' (near-unique, {nunique} / {n_rows})")
                continue
            # Skip constant columns
            if nunique <= 1:
                logger.debug(f"Skipping '{col}' (constant)")
                continue
            # Skip high-cardinality numerical columns (they create
            # massive groupby operations and rarely yield interesting FDs)
            if nunique > 500:
                logger.debug(
                    f"Skipping '{col}' (high cardinality: {nunique})"
                )
                continue
            candidates.append(col)

        return candidates

    # ── Utilities ────────────────────────────────────────────────────

    @staticmethod
    def summarize(result: FDResult) -> str:
        """Return a human-readable summary of FD discovery results.

        Args:
            result: Output of ``run()``.

        Returns:
            Formatted multi-line summary string.
        """
        lines = [
            "=" * 60,
            "Functional Dependency Discovery Summary",
            "=" * 60,
            f"Exact FDs:         {result['_meta']['n_exact_fds']}",
            f"Probabilistic FDs: {result['_meta']['n_probabilistic_fds']}",
            f"Threshold:         {result['_meta']['confidence_threshold']}",
            f"Rows analysed:     {result['_meta']['n_rows_analysed']:,}",
            f"Time:              {result['_meta']['elapsed_seconds']:.1f}s",
            "-" * 60,
        ]

        if result["exact"]:
            lines.append("\nTop exact FDs:")
            for fd in result["exact"][:15]:
                lhs_str = ", ".join(fd["lhs"])
                lines.append(f"  {lhs_str} → {fd['rhs']}  (conf={fd['confidence']:.4f})")

        if result["probabilistic"]:
            lines.append("\nTop probabilistic FDs:")
            for fd in result["probabilistic"][:15]:
                lhs_str = ", ".join(fd["lhs"])
                lines.append(f"  {lhs_str} → {fd['rhs']}  (conf={fd['confidence']:.4f})")

        lines.append("=" * 60)
        return "\n".join(lines)


# ── CLI entry point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    fd = FDDiscovery()
    result = fd.run()
    print(FDDiscovery.summarize(result))
