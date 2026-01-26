"""Logging setup for the forecasting pipeline."""

import logging
import sys
from pathlib import Path
from datetime import datetime

_logger: logging.Logger | None = None


def setup_logger(
    name: str = "forecast",
    level: str = "INFO",
    log_dir: str = "logs",
) -> logging.Logger:
    """Set up and configure the logger with console and file handlers.

    Args:
        name: Logger name.
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
        log_dir: Directory for log files.

    Returns:
        Configured logger instance.
    """
    global _logger

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(
        log_path / f"forecast_{timestamp}.log",
        encoding="utf-8",
    )
    file_handler.setLevel(getattr(logging, level.upper()))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    _logger = logger
    return logger


def get_logger() -> logging.Logger:
    """Get the configured logger instance.

    Returns:
        Logger instance, creates default if not set up.
    """
    global _logger
    if _logger is None:
        _logger = setup_logger()
    return _logger
