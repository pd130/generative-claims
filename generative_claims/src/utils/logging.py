"""Logging setup for Generative Claims.

Uses loguru for structured, colourful, rotated logging.

Usage:
    from src.utils.logging import get_logger
    logger = get_logger(__name__)
    logger.info("Profiling started", extra={"rows": 58592})
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

from src.utils.config import get_settings


_CONFIGURED = False


def _configure_once() -> None:
    """Initialize loguru sinks exactly once (idempotent)."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    settings = get_settings()
    log_dir = settings.log_path
    log_dir.mkdir(parents=True, exist_ok=True)

    # Remove default sink
    logger.remove()

    # Console sink  – coloured, human-friendly
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # File sink – JSON, rotated, for post-mortem analysis
    logger.add(
        log_dir / "generative_claims.log",
        level="DEBUG",
        rotation="10 MB",
        retention="30 days",
        compression="gz",
        serialize=True,
    )

    _CONFIGURED = True


def get_logger(name: str) -> "logger":  # type: ignore[type-arg]
    """Return a contextual logger bound to *name*.

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        A loguru logger instance with the module name bound.
    """
    _configure_once()
    return logger.bind(module=name)
