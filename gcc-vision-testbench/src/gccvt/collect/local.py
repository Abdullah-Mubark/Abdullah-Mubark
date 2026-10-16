"""Import your own recorded footage from a folder into the data layout and manifest.

Plates -> data/plates/{country}/{condition}/clip_NNNN.<ext>
Faces  -> data/faces/{dataset}/clip_NNNN.<ext>
Files are de-duplicated by SHA-256, so re-running is safe.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from ..cli import base_parser, init
from ..config import Settings
from ..errors import ConfigError, GccvtError
from ..manifest import Manifest, now_iso
from ..media import probe, sha256_file, transcode, write_json

MODES = ("copy", "link", "transcode")
CONSENT_MSG = (
    "Footage of people and vehicles is personal data under Saudi PDPL / UAE PDPL. Confirm you have "
    "consent and a lawful basis by setting consent_ack: true in config/local_sources.yaml or passing "
    "--consent-ack. See DATA_NOTICE.md."
)


def normalize_source(src: dict[str, Any], idx: int | str) -> dict[str, Any]:
    where = f"local_sources.yaml sources[{idx}]: " if isinstance(idx, int) else ""
    if not src.get("path"):
        raise ConfigError(f"{where}path is required")
    path = Path(str(src["path"])).expanduser()
    if not path.exists():
        raise ConfigError(f"{where}path does not exist: {path}")
    task = str(src.get("task") or "plate").lower()
    if task not in ("plate", "face"):
        raise ConfigError(f"{where}task must be 'plate' or 'face', got {task!r}")
    mode = str(src.get("mode") or "copy").lower()
    if mode not in MODES:
        raise ConfigError(f"{where}mode must be one of {MODES}, got {mode!r}")
    if task == "face" and not src.get("dataset"):
        raise ConfigError(f"{where}face sources need a dataset name (-> data/faces/<dataset>/)")
    return {**src, "path": path, "task": task, "mode": mode, "where": where,
            "recursive": src.get("recursive", True)}


def find_videos(path: Path, exts: set[str], recursive: bool) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in exts else []
    it = path.rglob("*") if recursive else path.glob("*")
    return sorted(p for p in it if p.is_file() and p.suffix.lower() in exts and not p.name.startswith("."))


def place(src: Path, dst: Path, mode: str, video_cfg: dict) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "transcode":
        transcode(src, dst, video_cfg)
    elif mode == "link":
        try:
            os.link(src, dst)
        except OSError:
            dst.symlink_to(src.resolve())
    else:
        tmp = dst.with_name(dst.name + ".tmp")
        shutil.copy2(src, tmp)
        tmp.replace(dst)


def import_file(f: Path, src: dict, settings: Settings, manifest: Manifest, log, dry_run: bool) -> str:
    """Returns 'added', 'skipped' or 'dry'."""
    rel_parts = [p for p in f.relative_to(src["path"]).parts[:-1]] if src["path"].is_dir() else []
    parts = rel_parts + list(src["path"].parts[-3:])  # folder names can carry country/condition
    task = src["task"]
    if task == "plate":
        country = (settings.normalize_country(src["country"], where=src["where"]) if src.get("country")
                   else settings.guess_country(parts))
        condition = (settings.normalize_condition(src["condition"], where=src["where"]) if src.get("condition")
                     else settings.guess_condition(parts))
        if not country or not condition:
            raise ConfigError(f"{f}: cannot tell country/condition - set them in the source config "
                              f"or put files under folders like .../ksa/night/")
        folder = settings.path("plates") / country / condition
    else:
        country = settings.normalize_country(src["country"]) if src.get("country") else ""
        condition = settings.normalize_condition(src["condition"]) if src.get("condition") else ""
        folder = settings.path("faces") / str(src["dataset"])

    digest = sha256_file(f)
    key = f"local:{digest}"
    row = manifest.find_by_source_key(key)
    if row and (settings.root / row["path"]).exists():
        log.info("Already imported: %s -> %s", f, row["path"])
        return "skipped"
    ext = ".mp4" if src["mode"] == "transcode" else f.suffix.lower()
    id_prefix = f"plate-{country}-{condition}" if task == "plate" else f"face-{src['dataset']}"
    dst = manifest.clip_path(folder, settings.section("clips").get("name_pattern", "clip_{n:04d}"), ext,
                             id_prefix, key)
    if dry_run:
        print(f"{f} -> {settings.rel(dst)} ({src['mode']})")
        return "dry"

    place(f, dst, src["mode"], settings.section("video"))
    info = probe(dst)
    meta_path = dst.with_suffix(".json")
    write_json(meta_path, {
        "original_path": str(f.resolve()), "sha256": digest, "task": task,
        "country": country, "condition": condition, "dataset": src.get("dataset", "local"),
        "import_mode": src["mode"], "duration_s": info.get("duration_s"), "width": info.get("width"),
        "height": info.get("height"), "fps": info.get("fps"), "notes": src.get("notes", ""),
        "imported_at": now_iso(), "consent_ack": True,
    })
    clip_id = f"{id_prefix}-{dst.stem}"
    manifest.upsert({
        "clip_id": clip_id, "task": task, "source": "local", "dataset": src.get("dataset", "local"),
        "country": country, "condition": condition, "path": settings.rel(dst),
        "duration_s": info.get("duration_s"), "fps": info.get("fps"), "width": info.get("width"),
        "height": info.get("height"), "frame_count": info.get("frame_count"),
        "source_key": key, "sha256": digest, "license": "self-recorded (consent required)",
        "meta_path": settings.rel(meta_path), "notes": src.get("notes", ""),
    })
    manifest.save()
    log.info("Imported %s -> %s", f.name, settings.rel(dst))
    return "added"


def main(argv: list[str] | None = None) -> int:
    p = base_parser(__doc__.splitlines()[0])
    p.add_argument("--src", help="folder or file to import (instead of local_sources.yaml)")
    p.add_argument("--task", choices=["plate", "face"], default="plate")
    p.add_argument("--country")
    p.add_argument("--condition")
    p.add_argument("--dataset", help="face dataset name")
    p.add_argument("--mode", choices=MODES, default="copy")
    p.add_argument("--no-recursive", action="store_true")
    p.add_argument("--consent-ack", action="store_true", help="confirm consent / lawful basis (see DATA_NOTICE.md)")
    args = p.parse_args(argv)
    settings, log = init("local_import", args)
    cfg = settings.load("local_sources.yaml")
    exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in (cfg.get("extensions") or [".mp4"])}

    if args.src:
        sources = [normalize_source({
            "path": args.src, "task": args.task, "country": args.country, "condition": args.condition,
            "dataset": args.dataset, "mode": args.mode, "recursive": not args.no_recursive,
        }, "cli")]
    else:
        sources = [normalize_source(s, i) for i, s in enumerate(cfg.get("sources") or [])]
    if not sources:
        log.info("No sources in local_sources.yaml and no --src given - nothing to do.")
        return 0
    if not (args.consent_ack or cfg.get("consent_ack") is True):
        raise ConfigError(CONSENT_MSG)

    manifest = Manifest(settings.path("manifest"))
    counts = {"added": 0, "skipped": 0, "dry": 0, "failed": 0}
    for src in sources:
        files = find_videos(src["path"], exts, src["recursive"])
        log.info("%s: %d video file(s)", src["path"], len(files))
        for f in files:
            try:
                counts[import_file(f, src, settings, manifest, log, args.dry_run)] += 1
            except GccvtError as e:
                log.error("%s: %s", f, e)
                counts["failed"] += 1
    log.info("Local import: %(added)d added, %(skipped)d already present, %(failed)d failed", counts)
    return 1 if counts["failed"] else 0
