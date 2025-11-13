import logging
import sys


def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Set up structured logger with console output."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    if logger.handlers:
        logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level.upper()))

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.propagate = False

    return logger
