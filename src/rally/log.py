"""Centralized logging configuration for market-rally."""

import logging
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"


def setup_logging(
    level: int = logging.INFO,
    log_to_file: bool = True,
    name: str = "rally",
) -> logging.Logger:
    """Configure root 'rally' logger with console + optional file handler.

    Call once at process startup (e.g. in orchestrator or CLI entry point).
    All rally.* modules then use ``logging.getLogger(__name__)``.
    """
    LOGS_DIR.mkdir(exist_ok=True)

    root = logging.getLogger("rally")
    if root.handlers:
        return root  # already configured
    root.setLevel(level)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    root.addHandler(console)

    # File handler
    if log_to_file:
        today = datetime.now().strftime("%Y-%m-%d")
        fh = logging.FileHandler(LOGS_DIR / f"{name}_{today}.log")
        fh.setFormatter(fmt)
        root.addHandler(fh)

    return root
