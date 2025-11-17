import logging
import sys

_root_configured = False


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored output."""

    GREY = "\x1b[38;5;245m"
    LIGHT_GREY = "\x1b[38;5;250m"
    YELLOW = "\x1b[33m"
    RED = "\x1b[31m"
    BOLD_RED = "\x1b[31;1m"
    RESET = "\x1b[0m"

    LEVEL_COLORS = {
        logging.DEBUG: GREY,
        logging.INFO: GREY,
        logging.WARNING: YELLOW,
        logging.ERROR: RED,
        logging.CRITICAL: BOLD_RED,
    }

    def format(self, record):
        formatted = super().format(record)
        log_color = self.LEVEL_COLORS.get(record.levelno, self.RESET)
        return f"{log_color}{formatted}{self.RESET}"


def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Set up structured logger with console output."""
    global _root_configured

    # Configure root logger once to handle all third-party libraries
    if not _root_configured:
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.setLevel(logging.INFO)

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = ColoredFormatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
        _root_configured = True

    # Configure specific logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    return logger
