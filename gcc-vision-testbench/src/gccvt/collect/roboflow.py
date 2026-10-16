"""Download plate datasets from Roboflow Universe (YOLOv8 export) via the Roboflow REST API."""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

from .. import archives
from ..cli import base_parser, init
from ..config import Settings, env_required
from ..errors import ConfigError, DownloadError, GccvtError
from ..manifest import Manifest
from ..net import HttpClient

URL_RX = re.compile(r"universe\.roboflow\.com/([^/?#]+)/([^/?#]+)(?:/(?:dataset|model)/(\d+))?")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_entry(entry: dict[str, Any], settings: Settings, idx: int) -> dict[str, Any]:
    where = f"roboflow_datasets.yaml datasets[{idx}]: "
    if entry.get("url"):
        m = URL_RX.search(str(entry["url"]))
        if not m:
            raise ConfigError(f"{where}cannot parse url {entry['url']!r}; expected "
                              "https://universe.roboflow.com/<workspace>/<project>[/dataset/<version>]")
        ws, proj, ver = m.groups()
    elif entry.get("slug"):
        parts = str(entry["slug"]).strip("/").split("/")
        if len(parts) not in (2, 3):
            raise ConfigError(f"{where}slug must be 'workspace/project[/version]', got {entry['slug']!r}")
        ws, proj, ver = (parts + [None])[:3]
    else:
        ws, proj, ver = entry.get("workspace"), entry.get("project"), entry.get("version")
        if not (ws and proj):
            raise ConfigError(f"{where}needs url, slug, or workspace+project[+version]")
    if ver is not None and not str(ver).isdigit():
        raise ConfigError(f"{where}version must be a number, got {ver!r}")
    return {
        "workspace": ws, "project": proj, "version": int(ver) if ver is not None else None,
        "country": settings.normalize_country(entry.get("country"), where=where),
        "notes": entry.get("notes", ""),
    }


def get_export_link(client: HttpClient, api: str, ds: dict, fmt: str, key: str,
                    poll_s: float, timeout_s: float, log) -> str:
    """Ask Roboflow for the export link; the first request may start export generation."""
    url = f"{api}/{ds['workspace']}/{ds['project']}/{ds['version']}/{fmt}"
    what = f"Roboflow export {ds['workspace']}/{ds['project']}/{ds['version']} ({fmt})"
    deadline = time.monotonic() + timeout_s
    while True:
        data = client.get_json(url, params={"api_key": key}, what=what)
        link = (data.get("export") or {}).get("link") if isinstance(data, dict) else None
        if link:
            return link
        if isinstance(data, dict) and data.get("error"):
            raise DownloadError(f"{what}: {data['error']}")
        if time.monotonic() > deadline:
            raise DownloadError(f"{what}: export not ready after {timeout_s:.0f}s - try again later")
        progress = data.get("progress") if isinstance(data, dict) else None
        log.info("%s: export being generated%s - checking again in %.0fs", what,
                 f" ({float(progress) * 100:.0f}%)" if isinstance(progress, (int, float)) else "", poll_s)
        time.sleep(poll_s)


def project_info(client: HttpClient, api: str, ds: dict, key: str) -> dict:
    return client.get_json(f"{api}/{ds['workspace']}/{ds['project']}", params={"api_key": key},
                           what=f"Roboflow project info {ds['workspace']}/{ds['project']}")


def latest_version(info: dict) -> int | None:
    """Highest version number in a Roboflow project-info response."""
    nums = []
    for v in info.get("versions") or []:
        vid = str(v.get("id", "")) if isinstance(v, dict) else str(v)
        tail = vid.rstrip("/").rsplit("/", 1)[-1]
        if tail.isdigit():
            nums.append(int(tail))
    if nums:
        return max(nums)
    count = (info.get("project") or {}).get("versions")
    return int(count) if isinstance(count, int) and count > 0 else None


def count_images(folder: Path) -> int:
    return sum(1 for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def main(argv: list[str] | None = None) -> int:
    p = base_parser(__doc__.splitlines()[0])
    p.add_argument("--force", action="store_true", help="re-download even if already extracted")
    args = p.parse_args(argv)
    settings, log = init("download_roboflow", args)
    cfg = settings.load("roboflow_datasets.yaml")
    api = str(cfg.get("api_url") or "").rstrip("/")
    fmt = str(cfg.get("format") or "yolov8")
    if not api:
        raise ConfigError("roboflow_datasets.yaml needs api_url")

    entries = [(i, e) for i, e in enumerate(cfg.get("datasets") or []) if e.get("enabled", True)]
    datasets = [parse_entry(e, settings, i) for i, e in entries]
    if not datasets:
        log.info("No enabled datasets in roboflow_datasets.yaml - nothing to do.")
        return 0
    if args.dry_run:
        for ds in datasets:
            print(f"{ds['country']:8} {ds['workspace']}/{ds['project']}/{ds['version'] or 'latest'} ({fmt})")
        return 0

    key = env_required("ROBOFLOW_API_KEY", "Add it to .env (see .env.example); get one at "
                       "https://app.roboflow.com/settings/api")
    client = HttpClient(settings.section("network"))
    manifest = Manifest(settings.path("manifest"))
    failed: list[str] = []

    for ds in datasets:
        try:
            info: dict = {}
            try:
                info = project_info(client, api, ds, key)
            except GccvtError as e:
                if ds["version"] is None:
                    raise
                log.warning("%s", e)
            if ds["version"] is None:
                ds["version"] = latest_version(info)
                if ds["version"] is None:
                    raise DownloadError(f"{ds['workspace']}/{ds['project']}: no dataset versions found - the project "
                                        "may have no generated version; set one explicitly (url .../dataset/<N>)")
                log.info("%s/%s: using latest version %d", ds["workspace"], ds["project"], ds["version"])
            name = f"{ds['workspace']}__{ds['project']}__v{ds['version']}"
            out_dir = settings.path("plates_images") / ds["country"] / name
            done_marker = out_dir / ".complete"
            if done_marker.is_file() and not args.force:
                log.info("Already downloaded: %s", name)
            else:
                link = get_export_link(client, api, ds, fmt, key, float(cfg.get("export_poll_s", 10)),
                                       float(cfg.get("export_timeout_s", 900)), log)
                zip_path = settings.path("raw") / "roboflow" / f"{name}__{fmt}.zip"
                if args.force:
                    zip_path.unlink(missing_ok=True)
                client.download(link, zip_path, what=f"{name}.zip")
                log.info("Extracting -> %s", out_dir)
                archives.extract(zip_path, out_dir)
                done_marker.write_text(fmt)
            manifest.upsert({
                "clip_id": f"plateimg-{ds['country']}-{name}",
                "task": "plate", "source": "roboflow", "dataset": name,
                "country": ds["country"], "path": settings.rel(out_dir),
                "frame_count": count_images(out_dir),
                "source_url": f"https://universe.roboflow.com/{ds['workspace']}/{ds['project']}/dataset/{ds['version']}",
                "source_key": f"roboflow:{ds['workspace']}/{ds['project']}/{ds['version']}:{fmt}",
                "license": str((info.get("project") or {}).get("license") or ""),
                "gt_path": settings.rel(out_dir), "notes": ds["notes"],
            })
            manifest.save()
        except GccvtError as e:
            label = f"{ds['workspace']}/{ds['project']}"
            log.error("%s: %s", label, e)
            failed.append(label)

    if failed:
        log.error("%d dataset(s) failed: %s", len(failed), ", ".join(failed))
        return 1
    log.info("Roboflow done.")
    return 0
