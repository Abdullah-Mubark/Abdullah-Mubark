"""Provenance: everything needed to tell two runs apart.

Recorded in each eval run's run.json (and the scoring environment in metrics.json):
code version, Python/package/ffmpeg versions, compute device, checksums of model weight files,
config files and the labelled evaluation data.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

from .config import Settings
from .media import sha256_file

PACKAGES = [
    "numpy", "Pillow", "PyYAML", "requests", "python-dotenv", "yt-dlp",
    "ultralytics", "torch", "torchvision", "onnxruntime", "onnxruntime-gpu", "insightface",
    "opencv-python", "opencv-python-headless",
]
CONFIG_FILES = ["settings.yaml", "prep.yaml", "eval.yaml", "models.yaml"]


def _run(cmd: list[str]) -> str | None:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def git_info(root: Path) -> dict[str, Any]:
    """Commit of the project checkout, and whether the project folder has uncommitted changes."""
    if not shutil.which("git"):
        return {"available": False}
    commit = _run(["git", "-C", str(root), "rev-parse", "HEAD"])
    if commit is None:
        return {"available": False}
    status = _run(["git", "-C", str(root), "status", "--porcelain", "--", "."]) or ""
    changed = [line[3:] for line in status.splitlines() if line.strip()]
    return {
        "commit": commit,
        "branch": _run(["git", "-C", str(root), "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(changed),
        "changed_files": changed[:50],
    }


def package_versions() -> dict[str, str]:
    out = {}
    for name in PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return out


def ffmpeg_version() -> str | None:
    if not shutil.which("ffmpeg"):
        return None
    first = (_run(["ffmpeg", "-hide_banner", "-version"]) or "").splitlines()
    return first[0] if first else None


def device_info() -> dict[str, Any]:
    """GPU details if torch / onnxruntime are already loaded (never imports them itself)."""
    info: dict[str, Any] = {"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            info["torch_cuda"] = bool(torch.cuda.is_available())
            if info["torch_cuda"]:
                info["torch_device"] = torch.cuda.get_device_name(0)
                info["cudnn_deterministic"] = bool(torch.backends.cudnn.deterministic)
        except Exception:  # noqa: BLE001 - informational only
            pass
    ort = sys.modules.get("onnxruntime")
    if ort is not None:
        try:
            info["onnxruntime_providers"] = list(ort.get_available_providers())
        except Exception:  # noqa: BLE001
            pass
    return info


def environment(settings: Settings) -> dict[str, Any]:
    """Code + software environment (cheap; used for runs and for re-scoring)."""
    return {
        "git": git_info(settings.root),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": package_versions(),
        "ffmpeg": ffmpeg_version(),
    }


def file_record(path: Path, settings: Settings) -> dict[str, Any]:
    return {"path": settings.rel(path), "size": path.stat().st_size, "sha256": sha256_file(path)}


def model_files(settings: Settings, resolved: dict[str, dict]) -> dict[str, dict]:
    """sha256 of every adapter param that points at an existing file (weights, model paths, tables...).

    resolved: {role: params as passed to the adapter} - so relative paths match what was loaded.
    """
    out: dict[str, dict] = {}
    for role, params in resolved.items():
        for key, val in params.items():
            if isinstance(val, str) and val and Path(val).is_file():
                out[f"{role}.{key}"] = file_record(Path(val), settings)
    return out


def config_hashes(settings: Settings) -> dict[str, str]:
    out = {}
    for name in CONFIG_FILES:
        p = settings.config_dir / name
        if p.is_file():
            out[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def data_fingerprint(settings: Settings, gts: dict[str, dict]) -> dict[str, Any]:
    """Fingerprint of the evaluation set: every labelled frame (path + size) and its ground truth.

    Changes when frames are re-extracted differently, labels are corrected, or clips are added.
    """
    h = hashlib.sha256()
    frames = 0
    for clip_id in sorted(gts):
        for frame in sorted(gts[clip_id], key=str):
            p = Path(frame)
            size = p.stat().st_size if p.is_file() else -1
            h.update(f"{clip_id}|{settings.rel(p)}|{size}|".encode())
            h.update(json.dumps(gts[clip_id][frame], sort_keys=True, default=str).encode())
            frames += 1
    return {"clips": len(gts), "frames": frames, "sha256": h.hexdigest()}
