"""Shared helpers for the prep stage: clip selection and per-clip frame index."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

from ..config import Settings
from ..manifest import Manifest

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".ts", ".mts", ".webm"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".pgm"}
INDEX = "index.csv"
INDEX_FIELDS = ["frame", "t_s", "source_frame", "hash", "status", "dup_of"]
DUP_DIR = "_duplicates"


def add_filter_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--clip", nargs="+", metavar="CLIP_ID", help="only these clip ids")
    p.add_argument("--task", choices=["plate", "face"])
    p.add_argument("--country")
    p.add_argument("--condition")
    p.add_argument("--force", action="store_true", help="redo even if already done")


def select_clips(settings: Settings, args: argparse.Namespace, tasks: list[str] | None = None) -> list[dict]:
    """Manifest rows that are videos or image sequences (Roboflow image datasets are skipped)."""
    rows = []
    country = settings.normalize_country(args.country) if args.country else None
    for row in Manifest(settings.path("manifest")).rows.values():
        if row["source"] == "roboflow":
            continue
        if args.clip and row["clip_id"] not in args.clip:
            continue
        if args.task and row["task"] != args.task:
            continue
        if tasks and row["task"] not in tasks:
            continue
        if country and row["country"] != country:
            continue
        if args.condition and row["condition"] != args.condition.lower():
            continue
        rows.append(row)
    return rows


def clip_source(settings: Settings, row: dict) -> Path:
    p = Path(row["path"])
    return p if p.is_absolute() else settings.root / p


def frames_dir(settings: Settings, clip_id: str) -> Path:
    return settings.path("frames") / clip_id


def read_index(folder: Path) -> list[dict[str, str]]:
    path = folder / INDEX
    if not path.is_file():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return [{k: r.get(k, "") or "" for k in INDEX_FIELDS} for r in csv.DictReader(f)]


def write_index(folder: Path, rows: list[dict[str, Any]]) -> None:
    tmp = folder / (INDEX + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in INDEX_FIELDS})
    os.replace(tmp, folder / INDEX)


def kept_frames(folder: Path) -> list[Path]:
    """Frames that survived de-duplication (those still in the clip folder root)."""
    rows = read_index(folder)
    if rows:
        return [folder / r["frame"] for r in rows if r["status"] != "duplicate" and (folder / r["frame"]).is_file()]
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def read_state(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def write_state(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True))
