"""
Logging configuration for the application.
Uses Python's standard logging with a clean, structured format.
"""
import logging
import sys


def setup_logging(level: int = logging.INFO) -> None:
    """Configures the root logger for the application."""
    log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    # Avoid duplicate handlers on reload
    if not root_logger.handlers:
        root_logger.addHandler(handler)

    # Quieten noisy third-party loggers
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("easyocr").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
