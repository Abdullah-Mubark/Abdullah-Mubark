"""Download YouTube segments with yt-dlp and trim/encode them to MP4 clips (video.target_height) with ffmpeg.

Clips go to data/plates/{country}/{condition}/clip_NNNN.mp4 with a metadata JSON beside each.
Already-collected segments (same video + start/end) are skipped. Several segments can run at once
(defaults.jobs / --jobs) so downloading overlaps with encoding; clip numbers are still handed out
in config order before anything starts.
"""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..cli import base_parser, init
from ..config import Settings
from ..errors import ConfigError, DownloadError, GccvtError, MediaError, ToolMissingError
from ..manifest import Manifest, now_iso
from ..media import parse_timestamp, require_tool, transcode, write_json
from ..net import with_retries

YT_ID_RX = re.compile(r"(?:v=|youtu\.be/|shorts/|live/|embed/)([A-Za-z0-9_-]{11})")
# yt-dlp errors that retrying will not fix
PERMANENT_ERRORS = ("Video unavailable", "Private video", "This video has been removed",
                    "members-only", "copyright", "not available in your country",
                    "Unsupported URL", "is not a valid URL", "Incomplete YouTube ID")
SIGN_IN_ERRORS = ("Sign in to confirm", "confirm your age", "cookies")


@dataclass
class Job:
    url: str
    video_key: str
    country: str
    condition: str
    start: float
    end: float
    notes: str
    where: str

    @property
    def source_key(self) -> str:
        return f"youtube:{self.video_key}:{self.start:.3f}-{self.end:.3f}"


def video_key(url: str) -> str:
    m = YT_ID_RX.search(url)
    return m.group(1) if m else url.strip()


def build_jobs(cfg: dict[str, Any], settings: Settings) -> list[Job]:
    jobs: list[Job] = []
    for i, src in enumerate(cfg.get("sources") or []):
        where = f"youtube_sources.yaml sources[{i}]: "
        if not isinstance(src, dict):
            raise ConfigError(f"{where}must be a mapping")
        if not src.get("enabled", True):
            continue
        url = str(src.get("url") or "").strip()
        if not url:
            raise ConfigError(f"{where}url is required")
        country = settings.normalize_country(src.get("country"), where=where)
        segments = src.get("segments") or [{"start": src.get("start"), "end": src.get("end")}]
        for j, seg in enumerate(segments):
            w = f"{where}segment {j}: " if src.get("segments") else where
            if seg.get("start") is None or seg.get("end") is None:
                raise ConfigError(f"{w}start and end are required")
            try:
                start, end = parse_timestamp(seg["start"]), parse_timestamp(seg["end"])
            except ValueError as e:
                raise ConfigError(f"{w}{e}") from None
            if end <= start:
                raise ConfigError(f"{w}end ({seg['end']}) must be after start ({seg['start']})")
            condition = settings.normalize_condition(seg.get("condition", src.get("condition")), where=w)
            jobs.append(Job(url, video_key(url), country, condition, start, end,
                            str(seg.get("notes", src.get("notes", "")) or ""), w))
    return jobs


class YtDlp:
    def __init__(self, defaults: dict[str, Any], net: dict[str, Any]):
        if importlib.util.find_spec("yt_dlp"):
            self.base = [sys.executable, "-m", "yt_dlp"]
        elif shutil.which("yt-dlp"):
            self.base = ["yt-dlp"]
        else:
            raise ToolMissingError("yt-dlp not found. Install it: pip install -r requirements.txt")
        self.base += ["--no-playlist", "--no-progress", "--retries", "10", "--fragment-retries", "10"]
        if defaults.get("cookies_from_browser"):
            self.base += ["--cookies-from-browser", str(defaults["cookies_from_browser"])]
        if defaults.get("rate_limit"):
            self.base += ["--limit-rate", str(defaults["rate_limit"])]
        self.format = str(defaults.get("format") or "bv*[height<=1080]+ba/b")
        self.attempts = int(defaults.get("attempts", 3))
        self.net = net

    def _run(self, args: list[str], what: str) -> str:
        def call() -> str:
            proc = subprocess.run(self.base + args, capture_output=True, text=True)
            if proc.returncode == 0:
                return proc.stdout
            err = "\n".join((proc.stderr or "").strip().splitlines()[-8:])
            if any(s.lower() in err.lower() for s in SIGN_IN_ERRORS):
                raise DownloadError(f"{what}: YouTube requires sign-in. Set defaults.cookies_from_browser "
                                    f"in youtube_sources.yaml (e.g. chrome).\n{err}")
            if any(s.lower() in err.lower() for s in PERMANENT_ERRORS):
                raise DownloadError(f"{what}: {err}")
            raise _Retry(err)

        return with_retries(call, what=what, attempts=self.attempts, retry_on=(_Retry,),
                            backoff_s=float(self.net.get("backoff_s", 2)),
                            backoff_max_s=float(self.net.get("backoff_max_s", 60)))

    def info(self, url: str) -> dict[str, Any]:
        out = self._run(["-J", "--skip-download", url], f"yt-dlp metadata {url}")
        try:
            return json.loads(out)
        except ValueError as e:
            raise DownloadError(f"yt-dlp returned invalid metadata for {url}") from e

    def download(self, url: str, out_template: Path, section: tuple[float, float] | None) -> Path:
        args = ["-f", self.format, "--merge-output-format", "mkv", "-o", str(out_template)]
        if section:
            args += ["--download-sections", f"*{section[0]:.3f}-{section[1]:.3f}", "--force-keyframes-at-cuts"]
        self._run(args + [url], f"yt-dlp download {url}")
        stem = out_template.name.split(".%(ext)s")[0]
        found = [p for p in out_template.parent.glob(f"{stem}.*") if is_final_download(p)]
        if not found:
            raise DownloadError(f"yt-dlp reported success but no file found for {url}")
        return max(found, key=lambda p: p.stat().st_size)


class _Retry(Exception):
    pass


def is_final_download(p: Path) -> bool:
    """Skip yt-dlp leftovers: .part/.ytdl files and per-format pieces like name.f137.mp4."""
    if p.name.endswith((".part", ".ytdl", ".json", ".temp")) or ".part-" in p.name:
        return False
    return not any(s.startswith(".f") and s[2:].isdigit() for s in p.suffixes[:-1])


_full_video_locks: defaultdict[str, threading.Lock] = defaultdict(threading.Lock)  # one full download per video


def process(job: Job, dst: Path, ytdlp: YtDlp, settings: Settings, defaults: dict, manifest: Manifest, log) -> Path:
    raw = settings.path("raw") / "youtube"
    video_cfg = settings.section("video")
    info = ytdlp.info(job.url)
    vid = info.get("id") or job.video_key
    duration = job.end - job.start
    mode = defaults.get("download_mode", "section")

    meta_out: dict[str, Any] = {}
    if mode == "section":
        tmp_dir = raw / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        template = tmp_dir / f"{vid}_{int(job.start * 1000)}_{int(job.end * 1000)}.%(ext)s"
        try:
            src = ytdlp.download(job.url, template, (job.start, job.end))
            meta_out = transcode(src, dst, video_cfg, duration=duration)
            src.unlink(missing_ok=True)
        except GccvtError as e:
            if not defaults.get("fallback_to_full", True):
                raise
            log.warning("Section download failed (%s); falling back to full video download", str(e).splitlines()[0])
            mode = "full"
    if mode == "full":
        full_dir = raw / "full"
        full_dir.mkdir(parents=True, exist_ok=True)
        with _full_video_locks[vid]:
            cached = [p for p in full_dir.glob(f"{vid}.*") if is_final_download(p)]
            src = cached[0] if cached else ytdlp.download(job.url, full_dir / f"{vid}.%(ext)s", None)
        if cached:
            log.info("Using cached full video %s", src.name)
        meta_out = transcode(src, dst, video_cfg, start=job.start, duration=duration)
    elif mode != "section":
        raise ConfigError(f"youtube_sources.yaml defaults.download_mode must be 'section' or 'full', got {mode!r}")

    meta_path = dst.with_suffix(".json")
    meta = {
        "source_url": job.url, "video_id": vid, "title": info.get("title"),
        "uploader": info.get("uploader"), "upload_date": info.get("upload_date"),
        "country": job.country, "condition": job.condition,
        "start_s": job.start, "end_s": job.end,
        "duration_s": meta_out.get("duration_s"), "width": meta_out.get("width"),
        "height": meta_out.get("height"), "fps": meta_out.get("fps"),
        "source_height": info.get("height"), "notes": job.notes,
        "collected_at": now_iso(), "license": "YouTube - internal testing only, do not redistribute",
    }
    write_json(meta_path, meta)
    manifest.upsert({
        "clip_id": f"plate-{job.country}-{job.condition}-{dst.stem}",
        "task": "plate", "source": "youtube", "dataset": "youtube",
        "country": job.country, "condition": job.condition, "path": settings.rel(dst),
        "duration_s": meta_out.get("duration_s"), "fps": meta_out.get("fps"),
        "width": meta_out.get("width"), "height": meta_out.get("height"),
        "frame_count": meta_out.get("frame_count"), "source_url": job.url,
        "source_key": job.source_key, "license": meta["license"],
        "meta_path": settings.rel(meta_path), "notes": job.notes,
    })
    manifest.save()
    return dst


def main(argv: list[str] | None = None) -> int:
    p = base_parser(__doc__.splitlines()[0])
    p.add_argument("--country", help="only sources for this country")
    p.add_argument("--condition", help="only day or night")
    p.add_argument("--limit", type=int, help="process at most N segments")
    p.add_argument("--jobs", type=int, help="segments to process at once (default: defaults.jobs, else 1)")
    args = p.parse_args(argv)
    settings, log = init("download_youtube", args)
    cfg = settings.load("youtube_sources.yaml")
    defaults = cfg.get("defaults") or {}

    jobs = build_jobs(cfg, settings)
    if args.country:
        c = settings.normalize_country(args.country)
        jobs = [j for j in jobs if j.country == c]
    if args.condition:
        c = settings.normalize_condition(args.condition)
        jobs = [j for j in jobs if j.condition == c]
    if not jobs:
        log.info("No enabled sources in youtube_sources.yaml - nothing to do.")
        return 0

    manifest = Manifest(settings.path("manifest"))
    todo = []
    for job in jobs:
        row = manifest.find_by_source_key(job.source_key)
        if row and (settings.root / row["path"]).is_file():
            log.info("Already collected: %s [%s] -> %s", job.url, _span(job), row["path"])
        else:
            todo.append(job)
    if args.limit:
        todo = todo[: args.limit]
    log.info("%d segment(s) to download, %d already collected", len(todo), len(jobs) - len(todo))

    if args.dry_run:
        for j in todo:
            print(f"{j.country:8} {j.condition:6} {_span(j):>22}  {j.url}")
        return 0
    if not todo:
        return 0

    require_tool("ffmpeg")
    require_tool("ffprobe")
    ytdlp = YtDlp(defaults, settings.section("network"))
    # Hand out clip paths up front, in config order, so ids don't depend on which download finishes first.
    pattern = settings.section("clips").get("name_pattern", "clip_{n:04d}")
    handed_out: defaultdict[str, list[str]] = defaultdict(list)
    plan = []
    for job in todo:
        prefix = f"plate-{job.country}-{job.condition}"
        dst = manifest.clip_path(settings.path("plates") / job.country / job.condition, pattern, ".mp4",
                                 prefix, job.source_key, handed_out[prefix])
        handed_out[prefix].append(dst.stem)
        plan.append((job, dst))

    def run_one(i: int, job: Job, dst: Path) -> Job | None:
        log.info("[%d/%d] %s %s/%s [%s]", i, len(todo), job.url, job.country, job.condition, _span(job))
        try:
            process(job, dst, ytdlp, settings, defaults, manifest, log)
            log.info("Saved %s", settings.rel(dst))
            return None
        except (DownloadError, MediaError, ToolMissingError) as e:
            log.error("%s%s: %s", job.where, job.url, e)
            return job

    workers = max(1, args.jobs or int(defaults.get("jobs") or 1))
    with ThreadPoolExecutor(workers) as ex:
        failed = [j for j in ex.map(lambda a: run_one(*a), ((i, j, d) for i, (j, d) in enumerate(plan, 1))) if j]
    if failed:
        log.error("%d/%d segment(s) failed - fix the errors above and re-run (finished ones are skipped)",
                  len(failed), len(todo))
        return 1
    log.info("YouTube done.")
    return 0


def _span(job: Job) -> str:
    return f"{_hms(job.start)}-{_hms(job.end)}"


def _hms(s: float) -> str:
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{sec:06.3f}".rstrip("0").rstrip(".")
