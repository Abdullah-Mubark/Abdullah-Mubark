"""Train a YOLOv8 plate detector on the Roboflow datasets in the manifest -> models/plate_yolov8.pt.

Builds data/train/plate/ (image symlinks + labels remapped to one class, "plate") from each dataset's
train/valid splits, never from the split `make eval` scores on (eval.yaml roboflow.splits: the
first that exists, normally test). Classes go through eval.yaml roboflow.class_map, so the model
learns exactly what evaluation counts as a plate. Settings: config/train.yaml.
"""

from __future__ import annotations

import json
import os
import random
import shutil
from pathlib import Path
from typing import Any

import yaml

from .cli import base_parser, init
from .config import Settings
from .errors import ConfigError, ToolMissingError
from .manifest import Manifest, now_iso
from .prep.common import IMAGE_EXTS, clip_source


def class_names(root: Path) -> list[str]:
    raw = (yaml.safe_load((root / "data.yaml").read_text()) or {}).get("names") or []
    return list(raw.values()) if isinstance(raw, dict) else list(raw)


def plate_lines(label_file: Path, names: list[str], class_map: dict[str, str]) -> list[str]:
    """YOLO label lines -> class-0 'plate' boxes. Polygon rows become their bounding box."""
    out = []
    for line in label_file.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cid = int(float(parts[0]))
        name = names[cid] if cid < len(names) else str(cid)
        if class_map.get(name, class_map.get("*", name)) != "plate":
            continue
        vals = [float(v) for v in parts[1:]]
        if len(vals) == 4:
            cx, cy, w, h = vals
        else:  # polygon x1 y1 x2 y2 ...
            xs, ys = vals[0::2], vals[1::2]
            cx, cy, w, h = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, max(xs) - min(xs), max(ys) - min(ys)
        if w > 0 and h > 0:
            out.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return out


def build_dataset(settings: Settings, cfg: dict, eval_cfg: dict, out: Path, log) -> dict[str, Any]:
    """Write out/{images,labels}/{train,val} and out/data.yaml. Returns per-dataset counts."""
    rcfg = eval_cfg.get("roboflow") or {}
    class_map = rcfg.get("class_map") or {"*": "plate"}
    eval_splits = rcfg.get("splits") or ["test"]
    caps = {"train": int(cfg.get("max_per_dataset", 1500)), "val": int(cfg.get("max_val_per_dataset", 200))}
    rng = random.Random(int(cfg.get("seed", 0)))
    rows = [r for r in Manifest(settings.path("manifest")).rows.values()
            if r["source"] == "roboflow" and r["task"] == "plate" and clip_source(settings, r).is_dir()]
    if not rows:
        raise ConfigError("No Roboflow plate datasets in the manifest: run make collect-roboflow first")
    if out.exists():
        shutil.rmtree(out)
    counts: dict[str, Any] = {}
    for row in sorted(rows, key=lambda r: r["clip_id"]):
        root = clip_source(settings, row)
        names = class_names(root)
        held_out = next((s for s in eval_splits if (root / s / "images").is_dir()), None)
        stats = {"held_out_for_eval": held_out, "train": 0, "val": 0, "plates": 0}
        for split, dest in (("train", "train"), ("valid", "val")):
            if split == held_out or not (root / split / "images").is_dir():
                continue
            imgs = sorted(p for p in (root / split / "images").iterdir() if p.suffix.lower() in IMAGE_EXTS)
            rng.shuffle(imgs)
            for img in imgs[: caps[dest]]:
                lbl = root / split / "labels" / f"{img.stem}.txt"
                lines = plate_lines(lbl, names, class_map) if lbl.is_file() else []
                name = f"{row['dataset']}__{img.name}"
                (out / "images" / dest).mkdir(parents=True, exist_ok=True)
                (out / "labels" / dest).mkdir(parents=True, exist_ok=True)
                os.symlink(img.resolve(), out / "images" / dest / name)
                (out / "labels" / dest / f"{Path(name).stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
                stats[dest] += 1
                stats["plates"] += len(lines)
        counts[row["dataset"]] = stats
        log.info("%s: %d train, %d val images, %d plate boxes (held out for eval: %s)",
                 row["dataset"], stats["train"], stats["val"], stats["plates"], held_out)
    if not (out / "images" / "val").is_dir():
        raise ConfigError("No validation images: none of the datasets has a valid split outside the eval split")
    (out / "data.yaml").write_text(yaml.safe_dump(
        {"path": str(out.resolve()), "train": "images/train", "val": "images/val", "names": {0: "plate"}}))
    return counts


def main(argv: list[str] | None = None) -> int:
    p = base_parser(__doc__.splitlines()[0])
    p.add_argument("--epochs", type=int, help="override plate.epochs")
    p.add_argument("--imgsz", type=int, help="override plate.imgsz")
    p.add_argument("--max-per-dataset", type=int, help="override plate.max_per_dataset (train images)")
    args = p.parse_args(argv)
    settings, log = init("train_plate", args)
    cfg = dict((settings.load("train.yaml").get("plate") or {}))
    for key in ("epochs", "imgsz", "max_per_dataset"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    data_dir = settings.path("data_root") / "train" / "plate"
    counts = build_dataset(settings, cfg, settings.load("eval.yaml"), data_dir, log)
    n_train = sum(c["train"] for c in counts.values())
    log.info("Dataset: %d train, %d val images -> %s", n_train, sum(c["val"] for c in counts.values()),
             settings.rel(data_dir))
    if args.dry_run:
        return 0
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise ToolMissingError("training needs ultralytics: make install-models") from e

    model = YOLO(str(cfg.get("base_model", "yolov8n.pt")))
    run_dir = data_dir / "runs"
    model.train(data=str(data_dir / "data.yaml"), epochs=int(cfg.get("epochs", 20)),
                imgsz=int(cfg.get("imgsz", 640)), batch=int(cfg.get("batch", 16)),
                patience=int(cfg.get("patience", 8)), workers=int(cfg.get("workers", 4)),
                device=str(cfg.get("device", "cpu")), seed=int(cfg.get("seed", 0)),
                project=str(run_dir), name="last", exist_ok=True, plots=False, verbose=False)
    best = Path(model.trainer.best)
    dst = Path(str(cfg.get("output", "models/plate_yolov8.pt")))
    dst = dst if dst.is_absolute() else settings.root / dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, dst)
    metrics = {k: round(float(v), 4) for k, v in (model.trainer.metrics or {}).items()}
    card = {"trained_at": now_iso(), "base_model": cfg.get("base_model", "yolov8n.pt"), "settings": cfg,
            "datasets": counts, "val_metrics": metrics, "weights": settings.rel(dst),
            "licence_note": "Ultralytics YOLOv8 is AGPL-3.0; check each Roboflow dataset's licence"}
    dst.with_suffix(".json").write_text(json.dumps(card, indent=2) + "\n")
    log.info("Saved %s (val %s)", settings.rel(dst),
             ", ".join(f"{k.split('/')[-1]} {v:.3f}" for k, v in metrics.items() if "metrics/" in k))
    return 0
