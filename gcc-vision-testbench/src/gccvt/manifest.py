"""manifest.csv: one row per clip (or image dataset), keyed by clip_id."""

from __future__ import annotations

import csv
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

FIELDS = [
    "clip_id", "task", "source", "dataset", "country", "condition", "path",
    "duration_s", "fps", "width", "height", "frame_count",
    "source_url", "source_key", "license", "gt_path", "meta_path", "sha256",
    "added_at", "notes",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class Manifest:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.rows: dict[str, dict[str, str]] = {}
        if self.path.is_file():
            with open(self.path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row.get("clip_id"):
                        self.rows[row["clip_id"]] = {k: row.get(k, "") or "" for k in FIELDS}

    def get(self, clip_id: str) -> dict[str, str] | None:
        return self.rows.get(clip_id)

    def find_by_source_key(self, key: str) -> dict[str, str] | None:
        for row in self.rows.values():
            if row.get("source_key") == key:
                return row
        return None

    def clip_path(self, folder: Path, pattern: str, ext: str, id_prefix: str, source_key: str) -> Path:
        """Where a new clip goes. clip_id is f"{id_prefix}-{file stem}".

        A source the manifest already lists keeps its file name, so clip ids (and the labels and
        results keyed by them) stay the same when manifest.csv is copied to another machine and the
        clips are collected again there. Otherwise the next number that is free both in the folder
        and among the manifest's clip ids is used, so a new clip never takes a listed one's id.
        """
        lead = id_prefix + "-"
        row = self.find_by_source_key(source_key)
        if row and row["clip_id"].startswith(lead):
            return folder / (row["clip_id"][len(lead):] + ext)
        reserved = [cid[len(lead):] for cid in self.rows if cid.startswith(lead)]
        return next_clip_path(folder, pattern, ext, reserved)[0]

    def upsert(self, row: dict[str, Any]) -> None:
        unknown = set(row) - set(FIELDS)
        if unknown:
            raise ValueError(f"unknown manifest fields: {sorted(unknown)}")
        clip_id = row["clip_id"]
        current = self.rows.get(clip_id, {k: "" for k in FIELDS})
        if not current.get("added_at"):
            current["added_at"] = now_iso()
        for k, v in row.items():
            current[k] = "" if v is None else _fmt(v)
        self.rows[clip_id] = current

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            for cid in sorted(self.rows):
                w.writerow(self.rows[cid])
        os.replace(tmp, self.path)


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.3f}".rstrip("0").rstrip(".")
    return str(v)


def next_clip_path(folder: Path, pattern: str, ext: str, reserved: Iterable[str] = ()) -> tuple[Path, int]:
    """Next free clip_NNNN path in folder (numbering continues after the highest existing).

    reserved: file stems that count as taken even if not on disk (clips listed in the manifest).
    """
    prefix = pattern.split("{", 1)[0]
    rx = re.compile(rf"^{re.escape(prefix)}(\d+)(\.|$)")
    names = [p.name for p in folder.glob(f"{prefix}*")] if folder.is_dir() else []
    nums = [int(m.group(1)) for name in [*names, *reserved] if (m := rx.match(name))]
    n = max(nums, default=0) + 1
    return folder / f"{pattern.format(n=n)}{ext}", n
