"""Download the ChokePoint face video dataset from Zenodo, one file at a time.

Resumes partial downloads, verifies Zenodo's md5 checksums, skips files already verified,
extracts archives and registers each camera sequence in manifest.csv.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Any

from .. import archives
from ..cli import base_parser, init
from ..config import Settings
from ..errors import ConfigError, GccvtError
from ..manifest import Manifest
from ..media import probe
from ..net import HttpClient

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".pgm"}


def zenodo_files(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalise the Zenodo API file list (legacy and InvenioRDM shapes)."""
    files = record.get("files") or []
    if isinstance(files, dict):
        files = files.get("entries") or []
        files = list(files.values()) if isinstance(files, dict) else files
    out = []
    for f in files:
        name = f.get("key") or f.get("filename")
        links = f.get("links") or {}
        url = links.get("content") or links.get("self") or links.get("download")
        if not name or not url:
            continue
        size = f.get("size", f.get("filesize"))
        out.append({"name": name, "url": url, "size": int(size) if size is not None else None,
                    "checksum": f.get("checksum")})
    return out


def select_files(files: list[dict], include: list[str], exclude: list[str]) -> list[dict]:
    return [
        f for f in files
        if any(fnmatch.fnmatch(f["name"], p) for p in include)
        and not any(fnmatch.fnmatch(f["name"], p) for p in exclude)
    ]


def index_sequences(settings: Settings, manifest: Manifest, faces_dir: Path, cfg: dict, record_url: str) -> int:
    """Register every folder that directly contains image frames as one clip."""
    fps = float(cfg.get("fps", 30))
    xmls = {p.stem: p for p in faces_dir.rglob("*.xml")}
    count = 0
    for d in sorted(p for p in faces_dir.rglob("*") if p.is_dir()):
        frames = sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not frames:
            continue
        info: dict[str, Any] = {}
        try:
            info = probe(frames[0])
        except GccvtError:
            pass
        gt = xmls.get(d.name)
        manifest.upsert({
            "clip_id": f"face-chokepoint-{d.relative_to(faces_dir).as_posix().replace('/', '__')}",
            "task": "face", "source": "chokepoint", "dataset": "chokepoint",
            "path": settings.rel(d), "fps": fps, "frame_count": len(frames),
            "duration_s": len(frames) / fps if fps else None,
            "width": info.get("width"), "height": info.get("height"),
            "source_url": record_url, "source_key": f"chokepoint:{d.relative_to(faces_dir).as_posix()}",
            "license": cfg.get("license", ""), "gt_path": settings.rel(gt) if gt else "",
        })
        count += 1
    return count


def main(argv: list[str] | None = None) -> int:
    p = base_parser(__doc__.splitlines()[0])
    p.add_argument("--list", action="store_true", help="list files in the Zenodo record and exit")
    p.add_argument("--only", nargs="+", metavar="PATTERN", help="override files.include patterns")
    p.add_argument("--no-extract", action="store_true", help="download only")
    args = p.parse_args(argv)
    settings, log = init("download_chokepoint", args)
    cfg = settings.load("chokepoint.yaml")

    zen = cfg.get("zenodo") or {}
    if not zen.get("api_url") or not zen.get("record_id"):
        raise ConfigError("chokepoint.yaml needs zenodo.api_url and zenodo.record_id")
    record_url = zen["api_url"].format(record_id=zen["record_id"])
    client = HttpClient(settings.section("network"))

    log.info("Fetching Zenodo record %s", record_url)
    record = client.get_json(record_url, what="Zenodo record lookup")
    files = zenodo_files(record)
    if not files:
        raise GccvtError(f"Zenodo record {zen['record_id']} lists no downloadable files")
    fcfg = cfg.get("files") or {}
    chosen = select_files(files, args.only or fcfg.get("include") or ["*"], fcfg.get("exclude") or [])
    total = sum(f["size"] or 0 for f in chosen)
    log.info("%d/%d files selected (%.2f GiB)", len(chosen), len(files), total / 2**30)

    if args.list or args.dry_run:
        for f in files:
            mark = "*" if f in chosen else " "
            size = f"{f['size'] / 2**20:10.1f} MiB" if f["size"] is not None else "         ? MiB"
            print(f"{mark} {size}  {f['name']}")
        return 0

    raw_dir = settings.path("raw") / "chokepoint"
    faces_dir = settings.path("faces") / "chokepoint"
    marker_dir = faces_dir / ".extracted"
    extract = cfg.get("extract", True) and not args.no_extract
    failed: list[str] = []

    for i, f in enumerate(chosen, 1):
        log.info("[%d/%d] %s", i, len(chosen), f["name"])
        dest = raw_dir / f["name"]
        marker = marker_dir / f"{f['name']}.done"
        try:
            if extract and marker.is_file() and marker.read_text().strip() == (f["checksum"] or ""):
                log.info("Already downloaded and extracted: %s", f["name"])
                continue
            client.download(f["url"], dest, size=f["size"], checksum=f["checksum"], what=f["name"])
            if extract and archives.is_archive(dest):
                log.info("Extracting %s -> %s", f["name"], faces_dir)
                archives.extract(dest, faces_dir)
                marker_dir.mkdir(parents=True, exist_ok=True)
                marker.write_text(f["checksum"] or "")
                if cfg.get("delete_archives_after_extract"):
                    dest.unlink(missing_ok=True)
                    log.info("Deleted archive %s (marker kept, so it won't be re-downloaded)", f["name"])
        except GccvtError as e:
            log.error("%s: %s", f["name"], e)
            failed.append(f["name"])

    if faces_dir.is_dir():
        manifest = Manifest(settings.path("manifest"))
        n = index_sequences(settings, manifest, faces_dir, cfg, record_url)
        manifest.save()
        log.info("Manifest: %d ChokePoint sequences registered", n)

    if failed:
        log.error("%d file(s) failed: %s - re-run to resume", len(failed), ", ".join(failed))
        return 1
    log.info("ChokePoint done.")
    return 0
