"""Extract frames from every clip in the manifest at a fixed rate (default 2 fps; extract.fps_by_task
can set a rate per task, e.g. fewer plate frames to label).

Output: data/frames/{clip_id}/000000.jpg ... plus index.csv (frame, timestamp, source frame).
"""

from __future__ import annotations

import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from ..cli import base_parser, init
from ..config import Settings
from ..errors import ConfigError, GccvtError, MediaError
from ..media import probe, require_tool, run
from .common import (IMAGE_EXTS, VIDEO_EXTS, add_filter_args, clip_source, frames_dir, read_state,
                     select_clips, write_index, write_state)

STATE = ".extract.json"


def extract_video(src: Path, out: Path, fps: float, fmt: str, quality: int) -> list[dict[str, Any]]:
    info = probe(src)
    src_fps = info.get("fps") or fps
    rate = min(fps, src_fps)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), "-vf", f"fps={rate}"]
    if fmt == "jpg":
        cmd += ["-qscale:v", str(quality)]
    cmd += ["-start_number", "0", str(out / f"%06d.{fmt}")]
    run(cmd, what=f"ffmpeg extract frames from {src.name}")
    frames = sorted(out.glob(f"*.{fmt}"))
    return [{"frame": f.name, "t_s": f"{i / rate:.3f}", "source_frame": round(i / rate * src_fps)}
            for i, f in enumerate(frames)]


def extract_sequence(src: Path, out: Path, fps: float, src_fps: float, mode: str) -> list[dict[str, Any]]:
    images = sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    step = max(1, round(src_fps / fps)) if fps else 1
    rows = []
    for i, img in enumerate(images[::step]):
        dst = out / img.name  # keep original names: ChokePoint ground truth refers to them
        if mode == "link":
            try:
                os.link(img, dst)
            except OSError:
                shutil.copy2(img, dst)
        else:
            shutil.copy2(img, dst)
        rows.append({"frame": dst.name, "t_s": f"{i * step / src_fps:.3f}", "source_frame": img.stem})
    return rows


def task_fps(cfg: dict, task: str) -> float:
    """extract.fps_by_task[task] if set, else extract.fps."""
    return float((cfg.get("fps_by_task") or {}).get(task, cfg.get("fps", 2)))


def extract_clip(settings: Settings, row: dict, cfg: dict, force: bool, log) -> str:
    """Returns 'done', 'skipped'."""
    src = clip_source(settings, row)
    out = frames_dir(settings, row["clip_id"])
    fps = task_fps(cfg, row["task"])
    fmt = str(cfg.get("image_format", "jpg")).lower()
    if fmt not in ("jpg", "png"):
        raise ConfigError(f"prep.yaml extract.image_format must be jpg or png, got {fmt!r}")
    if not src.exists():
        raise MediaError(f"source missing: {src} (re-run the collector?)")
    st = src.stat()
    want = {"fps": fps, "format": fmt, "source": str(src), "source_size": st.st_size,
            "source_mtime": int(st.st_mtime), "quality": cfg.get("jpeg_quality", 2),
            "sequences": cfg.get("sequences", "link")}
    if not force and read_state(out / STATE) == want:
        log.debug("Up to date: %s", row["clip_id"])
        return "skipped"

    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        if src.is_dir():
            src_fps = float(row.get("fps") or 0) or 30.0
            rows = extract_sequence(src, tmp, fps, src_fps, str(cfg.get("sequences", "link")))
        elif src.suffix.lower() in VIDEO_EXTS:
            rows = extract_video(src, tmp, fps, fmt, int(cfg.get("jpeg_quality", 2)))
        else:
            raise MediaError(f"{src}: not a video or image folder")
        if not rows:
            raise MediaError(f"{src}: no frames extracted")
        write_index(tmp, [{**r, "status": "kept"} for r in rows])
        write_state(tmp / STATE, want)
        shutil.rmtree(out, ignore_errors=True)  # previous extraction with other settings
        tmp.rename(out)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    log.info("%s: %d frames", row["clip_id"], len(rows))
    return "done"


def main(argv: list[str] | None = None) -> int:
    p = base_parser(__doc__.splitlines()[0])
    add_filter_args(p)
    p.add_argument("--fps", type=float, help="override extract.fps")
    p.add_argument("--workers", type=int, help="override extract.workers")
    args = p.parse_args(argv)
    settings, log = init("extract_frames", args)
    cfg = dict(settings.load("prep.yaml").get("extract") or {})
    if args.fps:
        cfg["fps"], cfg["fps_by_task"] = args.fps, {}  # --fps wins over the per-task rates too
    rates = {t: task_fps(cfg, t) for t in ("plate", "face")}
    if min(rates.values()) <= 0:
        raise ConfigError("extract.fps and extract.fps_by_task must be > 0")

    clips = select_clips(settings, args, cfg.get("tasks"))
    if not clips:
        log.info("No clips in the manifest match - run 'make collect' first.")
        return 0
    log.info("%d clip(s) selected (fps: %s)", len(clips), ", ".join(f"{t} {r:g}" for t, r in rates.items()))
    if args.dry_run:
        for r in clips:
            print(f"{r['clip_id']:45} {r['path']}")
        return 0
    require_tool("ffmpeg")

    counts = {"done": 0, "skipped": 0, "failed": 0}
    workers = max(1, int(args.workers or cfg.get("workers", 2)))
    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(extract_clip, settings, r, cfg, args.force, log): r for r in clips}
        for fut in as_completed(futures):
            try:
                counts[fut.result()] += 1
            except GccvtError as e:
                log.error("%s: %s", futures[fut]["clip_id"], e)
                counts["failed"] += 1
    log.info("Frames: %(done)d extracted, %(skipped)d up to date, %(failed)d failed", counts)
    return 1 if counts["failed"] else 0
