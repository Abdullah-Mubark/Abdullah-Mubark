"""Encode image-sequence clips (e.g. ChokePoint) as MP4 videos you can play or feed to a video model.

Output: data/faces/clips/{clip_id}.mp4 plus {clip_id}.csv (video frame -> original image name), so
ground truth keyed by image name (ChokePoint) can be lined up with the video. The image sequence
stays the evaluation source; the MP4s are not added to the manifest.
"""

from __future__ import annotations

import csv
from pathlib import Path

from ..cli import base_parser, init
from ..config import Settings
from ..errors import GccvtError
from ..media import encoder_args, require_tool, run
from .common import IMAGE_EXTS, add_filter_args, clip_source, select_clips


def clips_dir(settings: Settings) -> Path:
    return settings.path("faces") / "clips"


def encode_sequence(images: list[Path], dst: Path, fps: float, video_cfg: dict) -> None:
    """Images (in order) -> H.264 MP4 at fps, native resolution. Written via a temp file."""
    listing = dst.with_suffix(".ffconcat")
    lines = ["ffconcat version 1.0"]
    for img in images:
        lines += [f"file '{img.resolve().as_posix()}'", f"duration {1 / fps:.6f}"]
    lines.append(f"file '{images[-1].resolve().as_posix()}'")  # concat demuxer drops the last duration
    listing.write_text("\n".join(lines) + "\n")
    tmp = dst.with_name(dst.stem + ".tmp" + dst.suffix)
    try:
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0",
             "-i", str(listing), "-fps_mode", "cfr", "-r", f"{fps:g}", "-frames:v", str(len(images)),
             "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", *encoder_args(video_cfg), "-an", str(tmp)],
            what=f"ffmpeg encode {dst.name}")
        tmp.replace(dst)
    finally:
        tmp.unlink(missing_ok=True)
        listing.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    p = base_parser(__doc__.splitlines()[0])
    add_filter_args(p)
    args = p.parse_args(argv)
    settings, log = init("sequence_clips", args)
    out_dir = clips_dir(settings)
    rows = [r for r in select_clips(settings, args) if clip_source(settings, r).is_dir()]
    if not rows:
        log.info("No image-sequence clips in the manifest (run make collect-faces first).")
        return 0
    if not args.dry_run:
        require_tool("ffmpeg")
    done = skipped = failed = 0
    for i, row in enumerate(rows, 1):
        src = clip_source(settings, row)
        dst = out_dir / f"{row['clip_id']}.mp4"
        images = sorted(q for q in src.iterdir() if q.suffix.lower() in IMAGE_EXTS)
        if not images:
            continue
        index = dst.with_suffix(".csv")
        if dst.is_file() and index.is_file() and not args.force and \
                sum(1 for _ in open(index)) - 1 == len(images):
            skipped += 1
            continue
        fps = float(row["fps"] or 30)
        if args.dry_run:
            print(f"{row['clip_id']}: {len(images)} frames @ {fps:g} fps -> {settings.rel(dst)}")
            continue
        log.info("[%d/%d] %s: %d frames", i, len(rows), row["clip_id"], len(images))
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            encode_sequence(images, dst, fps, settings.section("video"))
        except GccvtError as e:
            log.error("%s: %s", row["clip_id"], e)
            failed += 1
            continue
        with open(index, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "t_s", "image"])
            w.writerows([n, f"{n / fps:.3f}", img.name] for n, img in enumerate(images))
        done += 1
    log.info("Face clips: %d encoded, %d up to date, %d failed -> %s", done, skipped, failed, settings.rel(out_dir))
    return 1 if failed else 0
