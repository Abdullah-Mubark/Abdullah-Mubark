"""Show disk use under data/ and delete chosen categories, keeping manifest.csv in step.

Categories:
  clips    plate videos (YouTube, local) + their frames and draft labels
  images   Roboflow image datasets
  faces    face videos / ChokePoint sequences + their frames and draft labels
  frames   extracted frames (make extract recreates them)
  drafts   draft labels from pre-annotation (make preannotate recreates them); gt/ is kept
  results  evaluation runs
  raw      downloads as received: archives, full YouTube videos, yt-dlp leftovers
  gt       your corrected ground truth - never part of "all", only deleted when named
  all      everything above except gt

Without --yes nothing is deleted: the command only shows what would go. Clips whose
annotations/{clip_id}/gt/ exists are kept (with their frames) unless --include-labelled.
"""

from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from .cli import base_parser, init
from .config import Settings
from .manifest import Manifest

ROW_CATS = {"clips", "images", "faces"}
DIR_CATS = {"frames": "frames", "drafts": "annotations", "results": "results", "raw": "raw", "gt": "annotations"}
ROOTS = {"clips": "plates", "images": "plates_images", "faces": "faces", **DIR_CATS}
ORDER = ["clips", "images", "faces", "frames", "drafts", "results", "raw", "gt"]
ALL = [c for c in ORDER if c != "gt"]


@dataclass
class Plan:
    paths: dict[str, list[Path]] = field(default_factory=lambda: {c: [] for c in ORDER})
    rows: set[str] = field(default_factory=set)            # manifest clip_ids to drop
    kept: dict[str, list[str]] = field(default_factory=lambda: {c: [] for c in ORDER})

    def add(self, cat: str, p: Path) -> None:
        if (p.exists() or p.is_symlink()) and p not in self.paths[cat]:
            self.paths[cat].append(p)


def size_of(p: Path) -> tuple[int, int]:
    """(bytes, files) under p, not following symlinks."""
    if p.is_symlink() or p.is_file():
        return p.lstat().st_size, 1
    total = n = 0
    for dirpath, _, files in os.walk(p):
        for f in files:
            try:
                total += os.lstat(os.path.join(dirpath, f)).st_size
                n += 1
            except OSError:
                pass
    return total, n


def human(b: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if b < 1024 or unit == "TB":
            return f"{b:.0f} {unit}" if unit in ("B", "KB") else f"{b:.1f} {unit}"
        b /= 1024
    return ""


def row_category(row: dict) -> str:
    if row["source"] == "roboflow":
        return "images"
    return "faces" if row["task"] == "face" else "clips"


def build_plan(settings: Settings, manifest: Manifest, cats: list[str], args) -> Plan:
    plan = Plan()
    ann, frames = settings.path("annotations"), settings.path("frames")
    country = settings.normalize_country(args.country) if args.country else None
    filtered = bool(args.clip or args.source or country or args.condition)

    def wanted(row: dict) -> bool:
        return not ((args.clip and row["clip_id"] not in args.clip)
                    or (args.source and row["source"] != args.source)
                    or (country and row["country"] != country)
                    or (args.condition and row["condition"] != args.condition.lower()))

    def labelled(clip_id: str) -> bool:
        return (ann / clip_id / "gt").is_dir()

    def abs_(p: str) -> Path:
        return Path(p) if Path(p).is_absolute() else settings.root / p

    def drafts(clip_id: str) -> list[Path]:
        d = ann / clip_id
        return [p for p in d.iterdir() if p.name != "gt"] if d.is_dir() else []

    # Manifest-backed categories: the clip itself, its metadata, and everything derived from it.
    for cid, row in manifest.rows.items():
        cat = row_category(row)
        if cat not in cats or not wanted(row):
            continue
        if labelled(cid) and not args.include_labelled:
            plan.kept[cat].append(cid)
            continue
        plan.rows.add(cid)
        for p in (row["path"], row["meta_path"]):
            if p:
                plan.add(cat, abs_(p))
        plan.add(cat, frames / cid)
        for ext in (".mp4", ".csv"):                              # make face-clips output
            plan.add(cat, settings.path("faces") / "clips" / f"{cid}{ext}")
        for p in drafts(cid):
            plan.add(cat, p)
    # Nothing of a row category left in the manifest: sweep its folder too (temp files, markers).
    if not filtered:
        for cat in ROW_CATS & set(cats):
            if not any(row_category(r) == cat and cid not in plan.rows for cid, r in manifest.rows.items()):
                root = settings.path(ROOTS[cat])
                for p in sorted(root.iterdir()) if root.is_dir() else []:
                    plan.add(cat, p)

    # Folder categories. frames/drafts/gt are per clip, so the clip filters apply to them.
    def clip_dirs(root: Path):
        for d in sorted(root.iterdir()) if root.is_dir() else []:
            row = manifest.get(d.name)
            if filtered and not (row and wanted(row)):
                continue
            yield d

    if "frames" in cats:
        for d in clip_dirs(frames):
            if labelled(d.name) and not args.include_labelled:
                plan.kept["frames"].append(d.name)
            else:
                plan.add("frames", d)
    if "drafts" in cats:
        for d in clip_dirs(ann):
            for p in drafts(d.name):
                plan.add("drafts", p)
    if "gt" in cats:
        for d in clip_dirs(ann):
            plan.add("gt", d / "gt")
    for cat in ("results", "raw"):
        root = settings.path(ROOTS[cat])
        if cat in cats and root.is_dir():
            for p in sorted(root.iterdir()):
                plan.add(cat, p)
    return plan


def remove(p: Path) -> None:
    if p.is_symlink() or p.is_file():
        p.unlink()
    elif p.is_dir():
        shutil.rmtree(p)


def prune_empty(root: Path) -> None:
    for dirpath, _, _ in sorted(os.walk(root), key=lambda t: -len(t[0])):
        d = Path(dirpath)
        if d != root and not any(d.iterdir()):
            d.rmdir()


def overview(settings: Settings, manifest: Manifest) -> None:
    print(f"Disk use under {settings.rel(settings.path('data_root'))}/:\n")
    print(f"  {'category':9} {'size':>10} {'files':>8}  folder")
    total = 0
    for cat in ORDER:
        root = settings.path(ROOTS[cat])
        if cat in ("drafts", "gt"):
            sub = [d / "gt" if cat == "gt" else p
                   for d in (sorted(root.iterdir()) if root.is_dir() else [])
                   for p in ([d] if cat == "gt" else [q for q in d.iterdir() if q.name != "gt"])]
            sub = [p for p in sub if p.exists()]
            b, n = map(sum, zip(*[size_of(p) for p in sub])) if sub else (0, 0)
            where = f"{settings.rel(root)}/*/{'gt' if cat == 'gt' else '(not gt)'}"
        else:
            b, n = size_of(root) if root.exists() else (0, 0)
            where = settings.rel(root)
        if cat != "gt":
            total += b
        note = "  <- your labels, never in 'all'" if cat == "gt" else ""
        print(f"  {cat:9} {human(b):>10} {n:8d}  {where}{note}")
    print(f"\n  total (excluding gt): {human(total)}   manifest rows: {len(manifest.rows)}")
    print("\nDelete with:  make clean-data ARGS=\"CATEGORY... [--source youtube] [--country ksa] --yes\""
          "\n  e.g. ARGS=\"raw\" to preview, ARGS=\"raw frames --yes\" to delete. See --help.")


def main(argv: list[str] | None = None) -> int:
    p = base_parser(__doc__.splitlines()[0])
    p.description = __doc__
    p.formatter_class = argparse.RawDescriptionHelpFormatter
    p.add_argument("categories", nargs="*", choices=ORDER + ["all"], metavar="CATEGORY",
                   help=f"one or more of: {', '.join(ORDER)}, all")
    p.add_argument("--source", help="only manifest rows from this source (youtube, local, roboflow, chokepoint)")
    p.add_argument("--country")
    p.add_argument("--condition")
    p.add_argument("--clip", nargs="+", metavar="CLIP_ID")
    p.add_argument("--include-labelled", action="store_true",
                   help="also delete clips (and frames) that have gt/ labels; gt/ itself is kept unless 'gt' is named")
    p.add_argument("--yes", action="store_true", help="actually delete (default: only show what would be deleted)")
    args = p.parse_args(argv)
    settings, log = init("clean_data", args)
    manifest = Manifest(settings.path("manifest"))

    if not args.categories:
        overview(settings, manifest)
        return 0
    cats = sorted({c for c in args.categories for c in (ALL if c == "all" else [c])}, key=ORDER.index)
    plan = build_plan(settings, manifest, cats, args)

    total = 0
    for cat in cats:
        sizes = [size_of(q) for q in plan.paths[cat]]
        b, n = (sum(s[0] for s in sizes), sum(s[1] for s in sizes)) if sizes else (0, 0)
        total += b
        print(f"{cat:9} {human(b):>10} {n:8d} files in {len(plan.paths[cat])} item(s)")
        for q in plan.paths[cat][:8]:
            print(f"            {settings.rel(q)}")
        if len(plan.paths[cat]) > 8:
            print(f"            ... and {len(plan.paths[cat]) - 8} more")
        if plan.kept[cat]:
            print(f"            kept (labelled, see --include-labelled): {', '.join(plan.kept[cat])}")
    rows_note = f", {len(plan.rows)} manifest row(s)" if plan.rows else ""
    if not args.yes:
        print(f"\nWould free {human(total)}{rows_note}. Nothing deleted - add --yes to delete.")
        return 0

    for cat in cats:
        for q in plan.paths[cat]:
            remove(q)
        root = settings.path(ROOTS[cat])
        if root.is_dir() and cat in ROW_CATS:
            prune_empty(root)
    if settings.path("annotations").is_dir():
        prune_empty(settings.path("annotations"))
    if plan.rows:
        for cid in plan.rows:
            manifest.remove(cid)
        manifest.save()
    log.info("Deleted %s in %s%s", human(total), ", ".join(cats), rows_note)
    return 0
