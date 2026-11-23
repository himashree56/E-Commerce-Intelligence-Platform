"""Structured logging configuration for E-Commerce Intelligence Platform."""
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_object: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
        }
        if hasattr(record, "extra_fields") and isinstance(record.extra_fields, dict):
            log_object.update(record.extra_fields)
        if record.exc_info:
            log_object["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_object)


def setup_logging(
    log_dir: Optional[Path] = None,
    log_file_name: str = "ingestion.log",
    log_level: int = logging.INFO,
    json_format: bool = False,
) -> logging.Logger:
    """Configures root logger with both console and file handlers.

    Args:
        log_dir: Directory where log files are stored. Defaults to ./logs.
        log_file_name: Name of the log file.
        log_level: Logging severity level.
        json_format: If True, uses JSON format for file logging.

    Returns:
        Configured root logger instance.
    """
    if log_dir is None:
        log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / log_file_name

    logger = logging.getLogger("ecommerce_platform")
    logger.setLevel(log_level)

    # Avoid duplicate handlers if setup_logging is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console Handler (Human-readable)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s:%(funcName)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File Handler (Structured JSON or standard text)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(log_level)
    if json_format:
        file_handler.setFormatter(JSONFormatter())
    else:
        file_handler.setFormatter(console_formatter)
    logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "ecommerce_platform") -> logging.Logger:
    """Retrieves a logger instance by name.

    Args:
        name: Logger name.

    Returns:
        Logger instance.
    """
    return logging.getLogger(name)
