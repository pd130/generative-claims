"""Custom exception hierarchy for Generative Claims.

Design Decision:
    All exceptions inherit from GenerativeClaimsError to allow callers
    to catch broad or specific failures. Each module has its own
    exception subtree for granular error handling.
"""

from __future__ import annotations


class GenerativeClaimsError(Exception):
    """Base exception for the entire Generative Claims system."""


# ── Profiler Exceptions ─────────────────────────────────────────────────


class ProfilerError(GenerativeClaimsError):
    """Base exception for the Profiler module."""


class DataLoadError(ProfilerError):
    """Raised when the source CSV cannot be loaded or parsed."""


class StatisticsComputationError(ProfilerError):
    """Raised when per-column statistics cannot be computed."""


class DistributionFitError(ProfilerError):
    """Raised when scipy distribution fitting fails for a column."""


class FunctionalDependencyError(ProfilerError):
    """Raised when FD discovery encounters an irrecoverable error."""


# ── Validator Exceptions ─────────────────────────────────────────────────


class ValidatorError(GenerativeClaimsError):
    """Base exception for the Validator module."""


class SchemaValidationError(ValidatorError):
    """Raised when synthetic data fails schema validation."""


class SchemaGenerationError(ValidatorError):
    """Raised when automatic schema generation fails."""


# ── Generator Exceptions ─────────────────────────────────────────────────


class GeneratorError(GenerativeClaimsError):
    """Base exception for the Generator module."""


# ── Controller Exceptions ────────────────────────────────────────────────


class ControllerError(GenerativeClaimsError):
    """Base exception for the Controller module."""


# ── Retrieval Exceptions ─────────────────────────────────────────────────


class RetrievalError(GenerativeClaimsError):
    """Base exception for the Retrieval module."""
