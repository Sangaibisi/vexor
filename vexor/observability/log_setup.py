"""Logging configuration — no mixin, just a plain function that returns a logger."""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING

from loguru import logger

from vexor._helpers import LOG_FILE_NAME

if TYPE_CHECKING:
    from vexor.config.observability import LogSpec

# ---------------------------------------------------------------------------
# Format strings
# ---------------------------------------------------------------------------

_DEV_FMT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{module}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)


# ---------------------------------------------------------------------------
# Production-mode JSON serialiser
# ---------------------------------------------------------------------------


def _json_serialise(record: dict) -> str:
    subset = {
        "timestamp": record["time"].strftime("%Y-%m-%d %H:%M:%S"),
        "message": record["message"],
        "level": record["level"].name,
        "function": record["function"],
        "file": record["file"].name if record["file"] else None,
        "line": record["line"],
        "elapsed": str(record["elapsed"]),
    }
    return json.dumps(subset, default=str)


def _prod_patcher(record: dict) -> None:
    record["extra"]["serialized"] = _json_serialise(record)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def configure_logging(spec: "LogSpec") -> "logger":
    """Set up loguru handlers based on *spec* and return the configured logger."""
    logger.remove()

    if not spec.enabled:
        return logger

    if spec.production_mode:
        logger = logger.patch(_prod_patcher)
        logger.add(
            sys.stderr,
            format="{extra[serialized]}",
            level="WARNING",
        )
        logger.add(
            sys.stdout,
            format="{extra[serialized]}",
            level="DEBUG",
            filter=lambda r: r["level"].no < 30,
        )
    else:
        logger.add(sys.stdout, format=_DEV_FMT, level="INFO", colorize=True)
        logger.add(sys.stderr, format=_DEV_FMT, level="ERROR", colorize=True)

    if spec.log_to_file:
        logger.add(
            LOG_FILE_NAME,
            format=_DEV_FMT,
            level="INFO",
            rotation="25 MB",
            backtrace=True,
            diagnose=True,
        )

    return logger
