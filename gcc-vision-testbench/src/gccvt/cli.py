"""Shared CLI plumbing for the scripts."""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Callable

from .config import Settings
from .errors import GccvtError
from .log import setup_logging


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config-dir", help="config folder (default: ./config or $GCCVT_CONFIG_DIR)")
    p.add_argument("--log-level", help="DEBUG, INFO, WARNING, ERROR (default: $LOG_LEVEL or INFO)")
    p.add_argument("--dry-run", action="store_true", help="show what would be done, change nothing")
    return p


def init(name: str, args: argparse.Namespace) -> tuple[Settings, logging.Logger]:
    settings = Settings(config_dir=args.config_dir)
    log = setup_logging(name, settings.path("logs"), args.log_level)
    return settings, log


def run_main(main: Callable[[list[str] | None], int]) -> None:
    try:
        sys.exit(main(None))
    except GccvtError as e:
        logging.getLogger("gccvt").error("%s", e)
        sys.exit(2)
    except KeyboardInterrupt:
        logging.getLogger("gccvt").warning("Interrupted - re-run the same command to resume.")
        sys.exit(130)
