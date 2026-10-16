"""Ground-truth loading. Every loader returns {frame_path: [gt boxes]} for labelled frames only.

GT box: {"xyxy": [x1, y1, x2, y2], "label": str, "text": str|None, "identity": str|None}

Sources:
- ChokePoint: groundtruth XML (eye positions per person) -> face boxes derived from the eyes
- Roboflow: YOLO labels in the export's split folders (boxes only)
- Clips: data/annotations/{clip_id}/gt/  - CVAT for images 1.1 XML (annotations.xml, plate text and
  face identity as box attributes) or YOLO (labels/*.txt + classes.txt)
"""

from __future__ import annotations

import fnmatch
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from ..config import Settings
from ..errors import GccvtError
from ..prep.common import IMAGE_EXTS, clip_source, frames_dir, kept_frames

log = logging.getLogger(__name__)
GT = dict[Path, list[dict[str, Any]]]


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as im:
        return im.size


def _yolo_file(path: Path, names: list[str], w: int, h: int, class_map: dict[str, str]) -> list[dict]:
    boxes = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cid = int(float(parts[0]))
        cx, cy, bw, bh = (float(v) for v in parts[1:5])
        name = names[cid] if cid < len(names) else str(cid)
        label = class_map.get(name, class_map.get("*", name))
        if not label:
            continue
        boxes.append({"xyxy": [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h],
                      "label": label, "text": None, "identity": None})
    return boxes


def load_cvat(xml_path: Path, frames: dict[str, Path], text_attr: str, id_attr: str) -> GT:
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError as e:
        raise GccvtError(f"{xml_path}: invalid XML: {e}") from e
    out: GT = {}
    for img in root.iter("image"):
        name = Path(img.get("name", "")).name
        frame = frames.get(name)
        if frame is None:
            continue
        boxes = []
        for b in img.iter("box"):
            attrs = {a.get("name"): (a.text or "").strip() for a in b.iter("attribute")}
            boxes.append({"xyxy": [float(b.get(k)) for k in ("xtl", "ytl", "xbr", "ybr")],
                          "label": b.get("label"), "text": attrs.get(text_attr) or None,
                          "identity": attrs.get(id_attr) or None})
        out[frame] = boxes
    return out


def load_yolo_dir(gt_dir: Path, frames: dict[str, Path], class_map: dict[str, str]) -> GT:
    classes_file = gt_dir / "classes.txt"
    names = classes_file.read_text().split() if classes_file.is_file() else []
    out: GT = {}
    for stem_frame in frames.values():
        lbl = gt_dir / "labels" / f"{stem_frame.stem}.txt"
        if lbl.is_file():  # a missing label file means "not labelled", an empty one means "no objects"
            out[stem_frame] = _yolo_file(lbl, names, *image_size(stem_frame), class_map)
    return out


def load_clip_gt(settings: Settings, row: dict, cfg: dict) -> GT:
    gt_dir = settings.path("annotations") / row["clip_id"] / "gt"
    folder = frames_dir(settings, row["clip_id"])
    if not folder.is_dir():
        return {}
    frames = {p.name: p for p in kept_frames(folder)}
    if (gt_dir / "annotations.xml").is_file():
        return load_cvat(gt_dir / "annotations.xml", frames, cfg.get("text_attribute", "text"),
                         cfg.get("identity_attribute", "identity"))
    if (gt_dir / "labels").is_dir():
        return load_yolo_dir(gt_dir, frames, {})
    return {}


def eyes_to_box(lx: float, ly: float, rx: float, ry: float, geom: dict) -> list[float]:
    d = max(((rx - lx) ** 2 + (ry - ly) ** 2) ** 0.5, 1.0)
    w, h = float(geom.get("width", 2.0)) * d, float(geom.get("height", 2.4)) * d
    cx, ey = (lx + rx) / 2, (ly + ry) / 2
    top = ey - float(geom.get("eye_line", 0.4)) * h
    return [cx - w / 2, top, cx + w / 2, top + h]


def load_chokepoint(settings: Settings, row: dict, cfg: dict) -> GT:
    if not row.get("gt_path"):
        return {}
    xml_path = Path(row["gt_path"])
    xml_path = xml_path if xml_path.is_absolute() else settings.root / xml_path
    folder = frames_dir(settings, row["clip_id"])
    if not (xml_path.is_file() and folder.is_dir()):
        return {}
    frames = {p.stem: p for p in kept_frames(folder)}
    geom = cfg.get("box_from_eyes") or {}
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError as e:
        raise GccvtError(f"{xml_path}: invalid XML: {e}") from e
    out: GT = {}
    for fr in root.iter("frame"):
        frame = frames.get(str(fr.get("number", "")))
        if frame is None:
            continue
        boxes = []
        for person in fr.iter("person"):
            le, re_ = person.find("leftEye"), person.find("rightEye")
            if le is None or re_ is None:
                continue
            try:
                pts = [float(le.get("x")), float(le.get("y")), float(re_.get("x")), float(re_.get("y"))]
            except (TypeError, ValueError):
                continue
            boxes.append({"xyxy": eyes_to_box(*pts, geom), "label": "face", "text": None,
                          "identity": str(person.get("id"))})
        out[frame] = boxes
    return out


def load_roboflow(settings: Settings, row: dict, cfg: dict) -> GT:
    root = clip_source(settings, row)
    names: list[str] = []
    data_yaml = root / "data.yaml"
    if data_yaml.is_file():
        raw = (yaml.safe_load(data_yaml.read_text()) or {}).get("names") or []
        names = list(raw.values()) if isinstance(raw, dict) else list(raw)
    class_map = cfg.get("class_map") or {"*": "plate"}
    out: GT = {}
    for split in cfg.get("splits") or ["test"]:
        img_dir, lbl_dir = root / split / "images", root / split / "labels"
        if not img_dir.is_dir():
            continue
        for img in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS):
            lbl = lbl_dir / f"{img.stem}.txt"
            if lbl.is_file():
                out[img] = _yolo_file(lbl, names, *image_size(img), class_map)
        if out:
            break  # first split that exists wins (e.g. test, then valid)
    return out


def load_gt(settings: Settings, row: dict, eval_cfg: dict) -> GT:
    if row["source"] == "chokepoint":
        return load_chokepoint(settings, row, (eval_cfg.get("faces") or {}).get("chokepoint") or {})
    if row["source"] == "roboflow":
        return load_roboflow(settings, row, eval_cfg.get("roboflow") or {})
    return load_clip_gt(settings, row, eval_cfg.get("gt") or {})


def sequence_name(row: dict) -> str:
    """ChokePoint sequence name (e.g. P1E_S1_C1) from the clip path."""
    return Path(row["path"]).name


def in_patterns(name: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(name, p) for p in patterns)
