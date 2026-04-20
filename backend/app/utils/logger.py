"""
Juris AI — Loguru Configuration
Structured logging to both stdout (colorized) and rotating log files.
"""

import sys
from pathlib import Path
from loguru import logger

from app.config import LOG_LEVEL, LOG_DIR


def setup_logger() -> None:
    """
    Configure Loguru for the entire application.
    
    Outputs:
        - stdout: colorized, human-readable logs
        - File: structured logs with daily rotation, 7-day retention
    """
    # Remove default Loguru handler
    logger.remove()

    # Format string matching the spec requirement
    log_format = (
        "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
        "{module}:{function} | {message}"
    )

    # Console handler — colorized
    logger.add(
        sys.stdout,
        format=log_format,
        level=LOG_LEVEL,
        colorize=True,
        backtrace=True,
        diagnose=False,
    )

    # File handler — daily rotation, 7-day retention
    log_dir = Path(LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.add(
        str(log_dir / "juris_{time:YYYY-MM-DD}.log"),
        format=log_format,
        level=LOG_LEVEL,
        rotation="00:00",     # New file at midnight
        retention="7 days",   # Keep logs for 7 days
        compression=None,     # No compression for easy reading
        backtrace=True,
        diagnose=False,
        encoding="utf-8",
    )

    logger.info("Logger initialized | level={} | log_dir={}", LOG_LEVEL, LOG_DIR)
