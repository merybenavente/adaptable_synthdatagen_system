import logging
import os
import sys

_root_configured = False


class Colors:
    """ANSI color codes for terminal output."""

    # Reset
    RESET = "\033[0m"

    # Text colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright colors
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"

    # Styles
    BOLD = "\033[1m"
    DIM = "\033[2m"
    UNDERLINE = "\033[4m"

    # Legacy aliases for backward compatibility
    GREY = "\x1b[38;5;245m"
    LIGHT_GREY = "\x1b[38;5;250m"
    BOLD_RED = "\x1b[31;1m"
    PURPLE = "\033[95m"

    @staticmethod
    def disable():
        """Disable colors (for non-terminal output)."""
        for attr in dir(Colors):
            if not attr.startswith("_") and attr not in ("disable", "enabled"):
                setattr(Colors, attr, "")

    @staticmethod
    def enabled():
        """Check if colors are enabled."""
        return hasattr(sys.stdout, "isatty") and sys.stdout.isatty() and os.getenv("TERM") != "dumb"


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored output."""

    LEVEL_COLORS = {
        logging.DEBUG: Colors.GREY,
        logging.INFO: Colors.GREY,
        logging.WARNING: Colors.YELLOW,
        logging.ERROR: Colors.RED,
        logging.CRITICAL: Colors.BOLD_RED,
    }

    def format(self, record):
        formatted = super().format(record)
        log_color = self.LEVEL_COLORS.get(record.levelno, Colors.RESET)
        return f"{log_color}{formatted}{Colors.RESET}"


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


# Auto-disable colors if not in terminal
if not Colors.enabled():
    Colors.disable()
