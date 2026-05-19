"""
Centralized logging configuration using Loguru.

Provides:
- Console output with color formatting
- Rotating file output (daily rotation, 7-day retention)
- Structured log format
- Single logger instance imported everywhere

Usage:
    from backend.logger import logger
    logger.info("Something happened")
    logger.error("Something broke: {error}", error=str(e))
"""

import sys
from pathlib import Path
from loguru import logger


def setup_logger(log_level: str = "INFO", log_dir: str = "./logs") -> None:
    """
    Configure Loguru logger with console and file handlers.

    Args:
        log_level: Logging level string (DEBUG, INFO, WARNING, ERROR)
        log_dir: Directory path where log files will be stored
    """
    # ─── Ensure log directory exists ──────────────────────────────
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # ─── Remove default Loguru handler ────────────────────────────
    logger.remove()

    # ─── Console Handler (colorized, human-readable) ──────────────
    logger.add(
        sys.stdout,
        level=log_level,
        colorize=True,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
    )

    # ─── File Handler (JSON-like, rotating daily, 7-day retention) ─
    logger.add(
        sink=f"{log_dir}/app_{{time:YYYY-MM-DD}}.log",
        level=log_level,
        rotation="00:00",        # New file every midnight
        retention="7 days",      # Keep logs for 7 days
        compression="zip",       # Compress old logs
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{message}"
        ),
        enqueue=True,            # Thread-safe async logging
    )

    logger.info(
        "Logger initialized | level={level} | log_dir={log_dir}",
        level=log_level,
        log_dir=log_dir,
    )


# ─── Export the configured logger ─────────────────────────────────
# Other modules import this directly:
# from backend.logger import logger
__all__ = ["logger", "setup_logger"]