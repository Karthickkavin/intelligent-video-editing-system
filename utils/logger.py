"""
Logging setup for the Intelligent Video Editing System.

Call ``setup_logger()`` once at startup.  Every module then just does::

    import logging
    logger = logging.getLogger(__name__)
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional


def setup_logger(
    name: str = "video_editor",
    log_dir: str = "logs",
    log_level: int = logging.INFO,
    max_bytes: int = 5 * 1024 * 1024,  # 5 MB per file
    backup_count: int = 3,
    console: bool = True,
) -> logging.Logger:
    """Configure and return a logger with both file and console handlers.

    Parameters
    ----------
    name:
        Logger name (also used as the log filename).
    log_dir:
        Directory where log files are stored.
    log_level:
        Minimum log level to record.
    max_bytes:
        Maximum size of a single log file before rotation.
    backup_count:
        Number of rotated log files to keep.
    console:
        Whether to also log to stdout.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{name}.log")

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Avoid duplicate handlers when called multiple times (e.g. in notebooks)
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Rotating file handler
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Optional console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return the logger for *name*, falling back to the root editor logger."""
    return logging.getLogger(name or "video_editor")
