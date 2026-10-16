"""Logging to console and to logs/<script>.log."""

from __future__ import annotations

import logging
import os
from pathlib import Path


def setup_logging(name: str, log_dir: Path, level: str | None = None) -> logging.Logger:
    level = (level or os.environ.get("LOG_LEVEL") or "INFO").upper()
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)
    file_handler = logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)
    for noisy in ("urllib3", "requests"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logging.getLogger(f"gccvt.{name}")
