"""Set up the data on a new machine in one command, or pack what can't be downloaded again.

  make pack    -> gccvt-transfer.tgz: manifest.csv, your gt/ labels and eval results (no videos)
  make fetch   -> unpacks gccvt-transfer.tgz if present, then collects YouTube clips, ChokePoint and
                  Roboflow (skipped without ROBOFLOW_API_KEY) and encodes the face MP4s

The manifest is unpacked before collecting, so every clip gets the same clip_id as on the machine
it came from and the labels and results still match. A failed step doesn't stop the others; the
summary at the end lists what to re-run (finished downloads are skipped on re-runs).
"""

from __future__ import annotations

import argparse
import os
import tarfile
from pathlib import Path
from typing import Callable

from . import archives
from .cli import base_parser, init
from .collect import chokepoint, roboflow, summary, youtube
from .config import Settings
from .errors import GccvtError
from .prep import sequence_clips

DEFAULT_BUNDLE = "gccvt-transfer.tgz"


def pack_members(data_root: Path) -> list[Path]:
    """Files worth carrying over: everything that can't be downloaded or recomputed cheaply."""
    out = []
    if (data_root / "manifest.csv").is_file():
        out.append(data_root / "manifest.csv")
    out += sorted(p for p in data_root.glob("annotations/*/gt/**/*") if p.is_file())
    out += sorted(p for p in (data_root / "results").rglob("*") if p.is_file() and p.suffix != ".mp4")
    return out


def pack(settings: Settings, bundle: Path, log) -> int:
    root = settings.path("data_root")
    members = pack_members(root)
    if not members:
        log.error("Nothing to pack under %s (no manifest, labels or results yet)", root)
        return 1
    with tarfile.open(bundle, "w:gz") as tf:
        for p in members:
            tf.add(p, arcname=p.relative_to(root).as_posix())
    size = bundle.stat().st_size / 2**20
    log.info("Packed %d files (%.1f MiB) -> %s", len(members), size, bundle)
    log.info("Copy it to the other machine's gcc-vision-testbench/ folder and run: make fetch")
    return 0


def unpack(settings: Settings, bundle: Path, log) -> None:
    """Unpack into the data folder, then rename the bundle so re-runs don't overwrite the manifest."""
    root = settings.path("data_root")
    archives.extract(bundle, root)  # refuses absolute paths, '..' and links outside root
    done = bundle.with_name(bundle.name + ".unpacked")
    bundle.replace(done)
    log.info("Unpacked %s into %s (renamed to %s)", bundle.name, settings.rel(root), done.name)


def steps(settings: Settings, cfg_args: list[str]) -> list[tuple[str, Callable[[], int] | None, str]]:
    """(name, run, why it's skipped) in order. run is None for a skipped step."""
    has_key = bool(os.environ.get("ROBOFLOW_API_KEY", "").strip())
    return [
        ("YouTube clips", lambda: youtube.main(cfg_args), ""),
        ("ChokePoint faces", lambda: chokepoint.main(cfg_args), ""),
        ("Roboflow datasets", (lambda: roboflow.main(cfg_args)) if has_key else None,
         "no ROBOFLOW_API_KEY in .env"),
        ("Face MP4s", lambda: sequence_clips.main(cfg_args), ""),
    ]


def main(argv: list[str] | None = None) -> int:
    p = base_parser(__doc__.splitlines()[0])
    p.description, p.formatter_class = __doc__, argparse.RawDescriptionHelpFormatter
    p.add_argument("--pack", nargs="?", const=DEFAULT_BUNDLE, metavar="FILE",
                   help=f"write the transfer bundle (default {DEFAULT_BUNDLE}) and exit")
    p.add_argument("--from", dest="bundle", metavar="FILE",
                   help=f"bundle to unpack first (default: {DEFAULT_BUNDLE} if it exists)")
    args = p.parse_args(argv)
    settings, log = init("fetch_data", args)
    cfg_args = ["--config-dir", str(settings.config_dir)]
    if args.log_level:
        cfg_args += ["--log-level", args.log_level]

    if args.pack:
        return pack(settings, settings.root / args.pack, log)

    bundle = settings.root / (args.bundle or DEFAULT_BUNDLE)
    if args.bundle and not bundle.is_file():
        raise GccvtError(f"bundle not found: {bundle}")
    if bundle.is_file():
        unpack(settings, bundle, log)
    else:
        log.info("No %s here: fetching without labels/results from another machine", bundle.name)
    if args.dry_run:
        for name, run, why in steps(settings, cfg_args):
            print(f"{name:18} {'skip: ' + why if run is None else 'run'}")
        return 0

    results: list[tuple[str, str]] = []
    for name, run, why in steps(settings, cfg_args):
        if run is None:
            results.append((name, f"skipped ({why})"))
            continue
        log.info("==== %s ====", name)
        try:
            code = run()
        except GccvtError as e:
            log.error("%s: %s", name, e)
            code = 2
        results.append((name, "ok" if code == 0 else f"FAILED (exit {code})"))

    summary.main(cfg_args)
    log.info("Fetch summary:")
    for name, status in results:
        log.info("  %-18s %s", name, status)
    failed = [n for n, s in results if s.startswith("FAILED")]
    if failed:
        log.error("Re-run `make fetch` to retry %s (finished items are skipped)", ", ".join(failed))
        return 1
    log.info("Done. Next: make prep (frames for labelling/eval), make eval")
    return 0
