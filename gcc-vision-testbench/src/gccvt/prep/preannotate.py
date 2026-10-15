"""Run detectors on extracted frames to create draft labels for manual correction.

Output per clip in data/annotations/{clip_id}/:
  yolo/labels/<frame>.txt, yolo/classes.txt, yolo/images.txt   (YOLO; class ids follow preannotate.classes)
  cvat/annotations.xml                                          (CVAT for images 1.1, boxes marked source="auto")
  predictions.jsonl                                             (raw boxes with scores)
Only frames kept by dedupe are labelled.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ..adapters import BaseAdapter, Box, create_adapter
from ..cli import base_parser, init
from ..config import Settings
from ..errors import ConfigError, GccvtError
from .common import add_filter_args, frames_dir, kept_frames, read_state, select_clips, write_state

STATE = ".preannotate.json"
FORMATS = ("yolo", "cvat")


def load_bgr(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.ascontiguousarray(np.asarray(img.convert("RGB"))[:, :, ::-1])


def resolve_params(settings: Settings, params: dict[str, Any]) -> dict[str, Any]:
    """Make relative weight paths relative to the project root (if the file exists there)."""
    out = dict(params)
    for key in ("weights", "model_path"):
        val = out.get(key)
        if isinstance(val, str) and not Path(val).is_absolute() and (settings.root / val).exists():
            out[key] = str(settings.root / val)
    return out


def yolo_line(box: Box, cls_id: int, w: int, h: int) -> str:
    cx, cy = (box.x1 + box.x2) / 2 / w, (box.y1 + box.y2) / 2 / h
    bw, bh = (box.x2 - box.x1) / w, (box.y2 - box.y1) / h
    return f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def write_cvat(path: Path, task_name: str, classes: list[str], images: list[dict[str, Any]]) -> None:
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"
    task = ET.SubElement(ET.SubElement(root, "meta"), "task")
    ET.SubElement(task, "name").text = task_name
    ET.SubElement(task, "size").text = str(len(images))
    labels = ET.SubElement(task, "labels")
    for c in classes:
        lab = ET.SubElement(labels, "label")
        ET.SubElement(lab, "name").text = c
        ET.SubElement(lab, "attributes")
    for i, im in enumerate(images):
        el = ET.SubElement(root, "image", id=str(i), name=im["name"], width=str(im["width"]), height=str(im["height"]))
        for b in im["boxes"]:
            ET.SubElement(el, "box", label=b.label, source="auto", occluded="0", z_order="0",
                          xtl=f"{b.x1:.2f}", ytl=f"{b.y1:.2f}", xbr=f"{b.x2:.2f}", ybr=f"{b.y2:.2f}")
    ET.indent(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def annotate_clip(settings: Settings, row: dict, detectors: list[tuple[dict, BaseAdapter]],
                  classes: list[str], formats: list[str], state_key: dict, force: bool, log) -> str:
    clip_id = row["clip_id"]
    folder = frames_dir(settings, clip_id)
    frames = kept_frames(folder)
    if not frames:
        raise GccvtError(f"{clip_id}: no frames - run extract_frames first")
    out = settings.path("annotations") / clip_id
    want = {**state_key, "frames": hashlib.sha1("\n".join(f.name for f in frames).encode()).hexdigest()}
    if not force and read_state(out / STATE) == want:
        return "skipped"

    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / "yolo" / "labels").mkdir(parents=True)
    images: list[dict[str, Any]] = []
    n_boxes = 0
    try:
        with open(tmp / "predictions.jsonl", "w", encoding="utf-8") as pred:
            for f in frames:
                frame = load_bgr(f)
                h, w = frame.shape[:2]
                boxes: list[Box] = []
                for det_cfg, adapter in detectors:
                    for b in adapter.detect(frame):
                        b = Box(b.x1, b.y1, b.x2, b.y2, b.score, det_cfg["label"], b.meta).clipped(w, h)
                        if b is not None:
                            boxes.append(b)
                n_boxes += len(boxes)
                images.append({"name": f.name, "width": w, "height": h, "boxes": boxes})
                pred.write(json.dumps({"frame": f.name, "width": w, "height": h, "boxes": [
                    {"label": b.label, "score": round(b.score, 4),
                     "xyxy": [round(v, 2) for v in (b.x1, b.y1, b.x2, b.y2)]} for b in boxes]}) + "\n")
                if "yolo" in formats:
                    lines = [yolo_line(b, classes.index(b.label), w, h) for b in boxes]
                    (tmp / "yolo" / "labels" / f"{f.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        if "yolo" in formats:
            (tmp / "yolo" / "classes.txt").write_text("\n".join(classes) + "\n")
            (tmp / "yolo" / "images.txt").write_text("\n".join(str(f.resolve()) for f in frames) + "\n")
        else:
            shutil.rmtree(tmp / "yolo")
        if "cvat" in formats:
            write_cvat(tmp / "cvat" / "annotations.xml", clip_id, classes, images)
        write_state(tmp / STATE, want)
        shutil.rmtree(out, ignore_errors=True)
        tmp.rename(out)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    log.info("%s: %d frames, %d draft boxes", clip_id, len(frames), n_boxes)
    return "done"


def main(argv: list[str] | None = None) -> int:
    p = base_parser(__doc__.splitlines()[0])
    add_filter_args(p)
    p.add_argument("--detector", nargs="+", help="only these detector names from prep.yaml")
    args = p.parse_args(argv)
    settings, log = init("preannotate", args)
    cfg = settings.load("prep.yaml").get("preannotate") or {}
    classes = list(cfg.get("classes") or ["plate", "face"])
    formats = list(cfg.get("formats") or FORMATS)
    if bad := set(formats) - set(FORMATS):
        raise ConfigError(f"preannotate.formats: unknown {sorted(bad)}; use {FORMATS}")

    det_cfgs = []
    for name, d in (cfg.get("detectors") or {}).items():
        if not d or not d.get("enabled", False) or (args.detector and name not in args.detector):
            continue
        if not d.get("adapter"):
            raise ConfigError(f"prep.yaml preannotate.detectors.{name}: adapter is required")
        label = d.get("label", name)
        if label not in classes:
            raise ConfigError(f"preannotate.detectors.{name}.label {label!r} is not in preannotate.classes {classes}")
        det_cfgs.append({"name": name, "adapter": d["adapter"], "label": label,
                         "tasks": list(d.get("tasks") or ["plate", "face"]),
                         "params": resolve_params(settings, {"label": label, **(d.get("params") or {})})})
    if not det_cfgs:
        log.info("No detectors enabled in prep.yaml (preannotate.detectors.*.enabled) - skipping pre-annotation.")
        return 0

    clips = [r for r in select_clips(settings, args)
             if frames_dir(settings, r["clip_id"]).is_dir() and any(r["task"] in d["tasks"] for d in det_cfgs)]
    if not clips:
        log.info("No clips with extracted frames match the enabled detectors.")
        return 0
    if args.dry_run:
        for r in clips:
            print(f"{r['clip_id']:45} {[d['name'] for d in det_cfgs if r['task'] in d['tasks']]}")
        return 0

    adapters: dict[str, BaseAdapter] = {}
    for d in det_cfgs:
        log.info("Loading detector %s (%s)", d["name"], d["adapter"])
        adapters[d["name"]] = create_adapter(d["adapter"], d["params"])

    counts = {"done": 0, "skipped": 0, "failed": 0}
    try:
        for r in clips:
            active = [(d, adapters[d["name"]]) for d in det_cfgs if r["task"] in d["tasks"]]
            state_key = {"classes": classes, "formats": formats,
                         "detectors": [{k: d[k] for k in ("name", "adapter", "label", "params")} for d, _ in active]}
            try:
                counts[annotate_clip(settings, r, active, classes, formats, state_key, args.force, log)] += 1
            except GccvtError as e:
                log.error("%s", e)
                counts["failed"] += 1
    finally:
        for a in adapters.values():
            a.close()
    log.info("Pre-annotation: %(done)d clips labelled, %(skipped)d up to date, %(failed)d failed", counts)
    return 1 if counts["failed"] else 0
