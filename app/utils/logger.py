"""
Centralized logging setup.

Usage in any module:
    from app.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("message")
"""

import logging
import sys

from app.config.settings import settings


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.handlers:
        # Avoid adding duplicate handlers if this is called multiple times
        return logger

    logger.setLevel(settings.log_level.upper())
    logger.propagate = False  # prevent double-logging via the root logger

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if settings.app_env == "production" and settings.log_level.upper() == "DEBUG":
        logger.warning(
            "LOG_LEVEL=DEBUG in production is discouraged; verify no handler "
            "logs request/response content before enabling this in production."
        )

    return logger