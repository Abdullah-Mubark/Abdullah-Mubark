"""Re-encode collected clips in place with the current video settings (codec, crf, max_fps).

Use it after changing video: in config/settings.yaml, e.g. to shrink existing H.264 clips to HEVC.
Clip ids, paths and labels stay the same; manifest.csv and each clip's metadata JSON get the new
fps, frame count and size. Clips already in the target codec and frame rate are skipped. Only
YouTube clips by default: local footage in "copy" mode is your original, so it needs --source local.
"""

from __future__ import annotations

import json
from pathlib import Path

from .cli import base_parser, init
from .errors import GccvtError
from .manifest import Manifest
from .media import CODEC_NAMES, fps_cap, probe, require_tool, transcode
from .prep.common import add_filter_args, clip_source


def main(argv: list[str] | None = None) -> int:
    p = base_parser(__doc__.splitlines()[0])
    add_filter_args(p)
    p.add_argument("--source", default="youtube", help="manifest source to re-encode (default: youtube)")
    args = p.parse_args(argv)
    settings, log = init("recompress", args)
    video_cfg = settings.section("video")
    want_codec = CODEC_NAMES.get(str(video_cfg.get("codec", "libx264")))
    manifest = Manifest(settings.path("manifest"))
    country = settings.normalize_country(args.country) if args.country else None
    rows = [r for r in manifest.rows.values()
            if r["source"] == args.source and clip_source(settings, r).is_file()
            and not (args.clip and r["clip_id"] not in args.clip)
            and not (args.task and r["task"] != args.task)
            and not (country and r["country"] != country)
            and not (args.condition and r["condition"] != args.condition.lower())]
    if not rows:
        log.info("No %s clips to re-encode.", args.source)
        return 0
    if not args.dry_run:
        require_tool("ffmpeg")

    before = after = 0
    done = skipped = failed = 0
    for i, row in enumerate(sorted(rows, key=lambda r: r["clip_id"]), 1):
        path = clip_source(settings, row)
        info = probe(path)
        if not args.force and info.get("codec") == want_codec and not fps_cap(info.get("fps"), video_cfg):
            skipped += 1
            continue
        size = path.stat().st_size
        if args.dry_run:
            print(f"{row['clip_id']}: {info.get('codec')} {info.get('fps')} fps, {size / 2**20:.0f} MiB")
            continue
        log.info("[%d/%d] %s (%s, %.0f MiB)", i, len(rows), row["clip_id"], info.get("codec"), size / 2**20)
        try:
            new = transcode(path, path, video_cfg)
        except GccvtError as e:
            log.error("%s: %s", row["clip_id"], e)
            failed += 1
            continue
        fields = {k: new.get(k) for k in ("duration_s", "fps", "width", "height", "frame_count")}
        manifest.upsert({"clip_id": row["clip_id"], **fields})
        manifest.save()
        meta_path = Path(row["meta_path"]) if row.get("meta_path") else None
        if meta_path and not meta_path.is_absolute():
            meta_path = settings.root / meta_path
        if meta_path and meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta.update({k: v for k, v in fields.items() if k != "frame_count"}, codec=new.get("codec"))
            meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        before += size
        after += path.stat().st_size
        done += 1
    if done:
        log.info("Re-encoded %d clip(s): %.2f GiB -> %.2f GiB (%.1fx smaller)",
                 done, before / 2**30, after / 2**30, before / after if after else 0)
    log.info("%d re-encoded, %d already up to date, %d failed", done, skipped, failed)
    return 1 if failed else 0
