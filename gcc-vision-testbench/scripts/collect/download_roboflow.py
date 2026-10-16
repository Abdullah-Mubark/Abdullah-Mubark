#!/usr/bin/env python3
"""Thin wrapper - see src/gccvt/collect/roboflow.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from gccvt.cli import run_main  # noqa: E402
from gccvt.collect.roboflow import main  # noqa: E402

if __name__ == "__main__":
    run_main(main)
