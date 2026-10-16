"""ffmpeg / ffprobe helpers."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .errors import MediaError, ToolMissingError

log = logging.getLogger(__name__)

INSTALL_HINT = {
    "ffmpeg": "Install ffmpeg (Ubuntu: sudo apt install ffmpeg | macOS: brew install ffmpeg | Windows: winget install ffmpeg)",
    "ffprobe": "ffprobe ships with ffmpeg - install ffmpeg",
}


def require_tool(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise ToolMissingError(f"'{name}' not found on PATH. {INSTALL_HINT.get(name, '')}")
    return path


def run(cmd: list[str], *, what: str) -> subprocess.CompletedProcess:
    log.debug("run: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or proc.stdout or "").strip().splitlines()[-15:])
        raise MediaError(f"{what} failed (exit {proc.returncode}):\n{tail}")
    return proc


def parse_timestamp(value: Any) -> float:
    """'01:02:03.5', '02:03', '123', 123 -> seconds."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        secs = float(value)
    else:
        text = str(value).strip()
        try:
            parts = [float(p) for p in text.split(":")]
        except ValueError:
            raise ValueError(f"invalid timestamp {value!r}; use HH:MM:SS, MM:SS or seconds") from None
        if not 1 <= len(parts) <= 3:
            raise ValueError(f"invalid timestamp {value!r}; use HH:MM:SS, MM:SS or seconds")
        secs = 0.0
        for p in parts:
            secs = secs * 60 + p
    if secs < 0:
        raise ValueError(f"timestamp must be >= 0, got {value!r}")
    return secs


def probe(path: Path) -> dict[str, Any]:
    """Duration, resolution, fps and frame count of a video (or first image of a sequence)."""
    require_tool("ffprobe")
    proc = run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        what=f"ffprobe {Path(path).name}",
    )
    info = json.loads(proc.stdout or "{}")
    video = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
    if video is None:
        raise MediaError(f"{path}: no video stream found")
    fps = _ratio(video.get("avg_frame_rate")) or _ratio(video.get("r_frame_rate"))
    duration = _float(info.get("format", {}).get("duration")) or _float(video.get("duration"))
    frames = video.get("nb_frames")
    frame_count = int(frames) if frames and str(frames).isdigit() else (
        round(duration * fps) if duration and fps else None
    )
    return {
        "duration_s": duration,
        "fps": round(fps, 3) if fps else None,
        "width": video.get("width"),
        "height": video.get("height"),
        "frame_count": frame_count,
        "codec": video.get("codec_name"),
    }


def _ratio(v: Any) -> float | None:
    try:
        num, den = str(v).split("/")
        return float(num) / float(den) if float(den) else None
    except (ValueError, TypeError):
        return _float(v)


def _float(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def transcode(
    src: Path,
    dst: Path,
    video_cfg: dict[str, Any],
    *,
    start: float | None = None,
    duration: float | None = None,
) -> dict[str, Any]:
    """Re-encode src -> dst (H.264 mp4), optionally trimming, scaled to the target height.

    Writes to a temp file first so an interrupted run never leaves a half-written clip.
    Returns probe() of the result.
    """
    require_tool("ffmpeg")
    target = int(video_cfg.get("target_height", 1080))
    src_info = probe(src)
    src_h = int(src_info.get("height") or 0)
    if src_h and src_h < target and not video_cfg.get("allow_upscale", False):
        log.warning("%s is %dp (< %dp); keeping source resolution (video.allow_upscale=false)", src.name, src_h, target)
        vf = "scale=trunc(iw/2)*2:trunc(ih/2)*2"
    else:
        vf = f"scale=-2:{target}"

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    if start:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(src)]
    if duration:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += [
        "-vf", vf,
        "-c:v", str(video_cfg.get("codec", "libx264")),
        "-crf", str(video_cfg.get("crf", 18)),
        "-preset", str(video_cfg.get("preset", "medium")),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
    ]
    cmd += ["-c:a", "aac", "-b:a", "128k"] if video_cfg.get("keep_audio") else ["-an"]
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.stem + ".tmp" + dst.suffix)
    cmd.append(str(tmp))
    try:
        run(cmd, what=f"ffmpeg encode {dst.name}")
        tmp.replace(dst)
    finally:
        tmp.unlink(missing_ok=True)
    return probe(dst)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)
