"""Render annotated MP4 clips from evaluated frames.

Colours (status palette, always paired with a text tag):
  green  = ground truth that was detected     blue   = detection matched to ground truth
  red    = MISSED ground truth                orange = FALSE + detection (no ground truth)
  yellow = the failure a failure clip is about
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..config import Settings
from ..eval import metrics as M
from ..media import require_tool, run
from ..text_norm import normalize_plate

log = logging.getLogger(__name__)

GOOD, MATCH, MISS, FALSE_POS, HILITE = "#0ca30c", "#2a78d6", "#d03b3b", "#ec835a", "#fab219"
LEGEND = [(GOOD, "ground truth (found)"), (MATCH, "detection"), (MISS, "MISSED"), (FALSE_POS, "FALSE +")]


def load_font(candidates: list[str], size: int) -> ImageFont.ImageFont:
    for c in candidates:
        try:
            return ImageFont.truetype(c, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


class Annotator:
    def __init__(self, settings: Settings, cfg: dict, report_cfg: dict):
        self.settings = settings
        self.thr = float(cfg["detection"].get("score_threshold", 0.5))
        norm = (cfg.get("plates") or {}).get("normalize") or {}
        self.norm_kw = {"ksa_arabic_to_latin": norm.get("ksa_arabic_to_latin", True),
                        "canonical_order": norm.get("canonical_order", True), "fold": norm.get("fold")}
        self.max_h = int((report_cfg.get("clips") or {}).get("max_height", 720))
        self.fonts = list(report_cfg.get("fonts") or [])
        self._font_cache: dict[int, Any] = {}

    def font(self, size: int):
        if size not in self._font_cache:
            self._font_cache[size] = load_font(self.fonts, size)
        return self._font_cache[size]

    def render(self, rec: dict, highlight: list[float] | None = None, title: str = "") -> Image.Image:
        src = Path(rec["frame"])
        src = src if src.is_absolute() else self.settings.root / src
        with Image.open(src) as im:
            im = im.convert("RGB")
        scale = min(1.0, self.max_h / im.height)
        if scale < 1:
            im = im.resize((round(im.width * scale), round(im.height * scale)), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(im)
        lw = max(2, round(im.height / 240))
        fs = max(12, round(im.height / 36))
        font = self.font(fs)

        def sc(xyxy):
            return [v * scale for v in xyxy]

        kept = [p for p in rec["preds"] if p["score"] >= self.thr]
        p2g, g2p = M.match(kept, rec["gt"], rec["iou"])
        for gi, g in enumerate(rec["gt"]):
            found = g2p[gi] >= 0
            box = sc(g["xyxy"])
            draw.rectangle(box, outline=GOOD if found else MISS, width=lw if found else lw + 1)
            tag = (g.get("text") or g.get("identity") or "") if found else f"MISSED {g.get('text') or ''}".strip()
            if tag:
                self._label(draw, box, tag, GOOD if found else MISS, font, below=True)
        for pi, p in enumerate(kept):
            box = sc(p["xyxy"])
            matched = p2g[pi] >= 0
            color = MATCH if matched else FALSE_POS
            draw.rectangle(box, outline=color, width=lw)
            tag = f"{p['score']:.2f}" if matched else f"FALSE + {p['score']:.2f}"
            if "text" in p:
                ok = matched and normalize_plate(rec["gt"][p2g[pi]].get("text") or "", country=rec["country"],
                                                 **self.norm_kw) == normalize_plate(p["text"], country=rec["country"],
                                                                                    **self.norm_kw)
                tag += f"  {p['text']}  {'✓' if ok else '✗'}" if matched and rec["gt"][p2g[pi]].get("text") \
                    else f"  {p['text']}"
            self._label(draw, box, tag, color, font, below=False)
        if highlight:
            box = sc(highlight)
            pad = lw * 3
            draw.rectangle([box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad], outline=HILITE, width=lw + 2)
        self._header(draw, im, title or f"{rec['clip_id']}  ·  {Path(rec['frame']).name}", font, fs)
        return im

    def _label(self, draw, box, text, color, font, below: bool) -> None:
        x1, y1, x2, y2 = box
        tw, th = _text_size(draw, text, font)
        pad = 3
        y = y2 + 2 if below else y1 - th - 2 * pad - 2
        if y < 0:
            y = y2 + 2
        if y + th + 2 * pad > draw.im.size[1]:
            y = max(0, y1 - th - 2 * pad - 2)
        x = min(max(0, x1), max(0, draw.im.size[0] - tw - 2 * pad))
        draw.rectangle([x, y, x + tw + 2 * pad, y + th + 2 * pad], fill=color)
        draw.text((x + pad, y + pad), text, fill="#ffffff" if color != HILITE else "#0b0b0b", font=font)

    def _header(self, draw, im, title, font, fs) -> None:
        h = fs + 12
        overlay = Image.new("RGBA", (im.width, h), (11, 11, 11, 170))
        im.paste(overlay, (0, 0), overlay)
        draw.text((8, 6), title, fill="#ffffff", font=font)
        # legend at the bottom, shrunk until it fits the frame width (portrait / narrow video)
        lfs = self.legend_size(draw, im.width, fs)
        lfont = self.font(lfs)
        h = lfs + 12
        x, y = 8, im.height - h
        im.paste(Image.new("RGBA", (im.width, h), (11, 11, 11, 150)), (0, y), Image.new("L", (im.width, h), 150))
        for color, label in LEGEND:
            draw.rectangle([x, y + 6, x + lfs, y + 6 + lfs], outline=color, width=max(2, lfs // 6))
            draw.text((x + lfs + 6, y + 6), label, fill="#ffffff", font=lfont)
            x += lfs + 18 + _text_size(draw, label, lfont)[0]

    def legend_size(self, draw, width: int, fs: int) -> int:
        """Largest font size <= fs whose legend fits in `width` (never below 8)."""
        while fs > 8 and self._legend_width(draw, fs) > width - 16:
            fs -= 1
        return fs

    def _legend_width(self, draw, size: int) -> int:
        font = self.font(size)
        return sum(size + 18 + _text_size(draw, label, font)[0] for _, label in LEGEND) - 18


def _text_size(draw, text, font) -> tuple[int, int]:
    l, t, r, b = draw.textbbox((0, 0), text, font=font)
    return r - l, b - t


def encode(frames_dir: Path, out: Path, fps: float, crf: int) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp.mp4")
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", str(fps),
         "-i", str(frames_dir / "%06d.jpg"), "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
         "-c:v", "libx264", "-crf", str(crf), "-preset", "veryfast", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", str(tmp)], what=f"ffmpeg encode {out.name}")
    tmp.replace(out)


def write_clip(annotator: Annotator, recs: list[dict], out: Path, fps: float, crf: int,
               highlight: dict[str, list[float]] | None = None, titles: dict[str, str] | None = None,
               poster_index: int = 0) -> Path:
    """Annotate each record's frame and encode them in order into an MP4.

    Also writes <name>.jpg next to it (the frame at poster_index) for the video poster. Returns its path.
    """
    require_tool("ffmpeg")
    tmp = Path(tempfile.mkdtemp(prefix="gccvt_clip_"))
    poster = out.with_suffix(".jpg")
    try:
        size = None
        for i, rec in enumerate(recs):
            im = annotator.render(rec, (highlight or {}).get(rec["frame"]), (titles or {}).get(rec["frame"], ""))
            if size is None:
                size = im.size
            elif im.size != size:  # mixed frame sizes in one clip: keep the first size
                im = im.resize(size)
            im.save(tmp / f"{i:06d}.jpg", quality=88)
        encode(tmp, out, fps, crf)
        shutil.copyfile(tmp / f"{min(poster_index, len(recs) - 1):06d}.jpg", poster)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return poster


def sample(recs: list[dict], max_frames: int) -> list[dict]:
    if max_frames <= 0 or len(recs) <= max_frames:
        return recs
    idx = np.linspace(0, len(recs) - 1, max_frames).round().astype(int)
    return [recs[i] for i in sorted(set(idx))]


def render_run_clips(settings: Settings, frames: list[dict], failures: dict, cfg: dict, run_dir: Path) -> dict:
    """Write clips/{clip_id}.mp4 and clips/failures/*.mp4 (+ .jpg posters).

    Returns {"clips": {clip_id: {"video", "poster"}}, "failure_clips": {"kind:i": {"video", "poster"}}}.
    """
    rcfg = cfg.get("report") or {}
    ccfg, fcfg = rcfg.get("clips") or {}, rcfg.get("failure_clips") or {}
    out: dict[str, dict] = {"clips": {}, "failure_clips": {}}
    if not ccfg.get("enabled", True):
        return out
    ann = Annotator(settings, cfg, rcfg)
    fps, crf = float(ccfg.get("playback_fps", 2)), int(ccfg.get("crf", 23))
    by_clip: dict[str, list[dict]] = {}
    for rec in frames:
        by_clip.setdefault(rec["clip_id"], []).append(rec)
    for recs in by_clip.values():
        recs.sort(key=lambda r: r["frame"])

    for clip_id, recs in by_clip.items():
        if recs[0]["source"] == "roboflow":
            continue  # unrelated still images - the failure gallery covers them
        rel = f"clips/{_safe(clip_id)}.mp4"
        try:
            chosen = sample(recs, int(ccfg.get("max_frames", 600)))
            # poster: the frame with the most going on (objects + detections)
            busiest = max(range(len(chosen)), key=lambda k: len(chosen[k]["gt"]) + len(chosen[k]["preds"]))
            poster = write_clip(ann, chosen, run_dir / rel, fps, crf, poster_index=busiest)
            out["clips"][clip_id] = {"video": rel, "poster": poster.relative_to(run_dir).as_posix()}
            log.info("Clip: %s", rel)
        except Exception as e:  # one bad clip must not stop the report
            log.error("clip %s failed: %s", clip_id, e)

    per_type, window = int(fcfg.get("per_type", 5)), int(fcfg.get("window", 3))
    for kind in ("missed_detections", "false_positives", "plate_misreads"):
        for i, item in enumerate((failures.get(kind) or [])[:per_type]):
            recs = by_clip.get(item["clip_id"], [])
            idx = next((k for k, r in enumerate(recs) if r["frame"] == item["frame"]), None)
            if idx is None or recs[idx]["source"] == "roboflow":
                continue
            win = recs[max(0, idx - window): idx + window + 1]
            rel = f"clips/failures/{kind}_{i:03d}.mp4"
            label = {"missed_detections": "missed", "false_positives": "false positive",
                     "plate_misreads": "misread"}[kind]
            try:
                poster = write_clip(ann, win, run_dir / rel, max(1.0, fps / 2), crf,
                                    highlight={item["frame"]: item["xyxy"]},
                                    titles={item["frame"]: f"{label}  ·  {item['clip_id']}  ·  {Path(item['frame']).name}"},
                                    poster_index=idx - max(0, idx - window))
                out["failure_clips"][f"{kind}:{i}"] = {"video": rel, "poster": poster.relative_to(run_dir).as_posix()}
            except Exception as e:
                log.error("failure clip %s failed: %s", rel, e)
    return out


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
