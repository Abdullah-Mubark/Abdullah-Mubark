"""Run a model over every labelled frame and compute metrics.

Writes data/results/{run_id}/:
  run.json           model + settings snapshot
  predictions.jsonl  per frame: ground truth, detections, plate reads (raw model output)
  embeddings.npz     face embeddings for the ChokePoint recognition protocol
  metrics.json       overall + breakdowns by country / condition / source / clip
  failures.json      worst failures, with crops in failures/
Re-score an existing run with different thresholds: --rescore RUN_ID (no inference).
"""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from ..adapters import BaseAdapter, Box, create_adapter
from ..cli import base_parser, init
from ..config import Settings
from ..errors import ConfigError, GccvtError
from ..manifest import Manifest
from ..prep.common import add_filter_args
from ..prep.preannotate import load_bgr, resolve_params
from ..text_norm import cer, normalize_plate
from . import metrics as M
from .gt import in_patterns, load_gt, sequence_name

ROLES = ("plate_detector", "plate_reader", "face_detector", "face_recognizer")
TASK_ROLES = {"plate": ("plate_detector", "plate_reader"), "face": ("face_detector", "face_recognizer")}


# ---------------------------------------------------------------- helpers
def crop(frame: np.ndarray, xyxy: list[float], margin: float = 0.0) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = xyxy
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    x1, y1 = max(0, int(x1 - mx)), max(0, int(y1 - my))
    x2, y2 = min(w, int(math.ceil(x2 + mx))), min(h, int(math.ceil(y2 + my)))
    return np.ascontiguousarray(frame[y1:max(y2, y1 + 1), x1:max(x2, x1 + 1)])


def clean(obj: Any) -> Any:
    """NaN -> None so metrics.json is valid JSON."""
    if isinstance(obj, float):
        return None if math.isnan(obj) else round(obj, 5)
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.generic):
        return clean(obj.item())
    return obj


def load_models(settings: Settings, spec: dict, needed: set[str], log) -> dict[str, BaseAdapter]:
    loaded: dict[str, BaseAdapter] = {}
    cache: dict[str, BaseAdapter] = {}
    for role in ROLES:
        if role not in needed or not spec.get(role):
            continue
        r = spec[role]
        if not r.get("adapter"):
            raise ConfigError(f"models.yaml: {role} needs an adapter")
        params = resolve_params(settings, r.get("params") or {})
        key = json.dumps([r["adapter"], params], sort_keys=True, default=str)
        if key not in cache:
            log.info("Loading %s: %s", role, r["adapter"])
            cache[key] = create_adapter(r["adapter"], params)
        loaded[role] = cache[key]
    return loaded


# ---------------------------------------------------------------- inference
def evenly(items: list, k: int) -> list:
    if k <= 0 or len(items) <= k:
        return items
    idx = np.linspace(0, len(items) - 1, k).round().astype(int)
    return [items[i] for i in sorted(set(idx))]


def recognition_plan(row: dict, gt: dict, rec_cfg: dict) -> dict[tuple[str, int], str]:
    """Which (frame, gt_index) faces to embed for the ChokePoint protocol -> 'gallery' | 'probe'."""
    if row["source"] != "chokepoint" or not rec_cfg.get("enabled", True):
        return {}
    seq = sequence_name(row)
    if in_patterns(seq, rec_cfg.get("gallery") or []):
        role, cap = "gallery", int(rec_cfg.get("gallery_frames_per_identity", 10))
    elif in_patterns(seq, rec_cfg.get("probe") or []):
        role, cap = "probe", int(rec_cfg.get("probe_frames_per_track", 20))
    else:
        return {}
    by_id: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for frame, boxes in sorted(gt.items()):
        for i, b in enumerate(boxes):
            if b.get("identity"):
                by_id[b["identity"]].append((str(frame), i))
    return {k: role for items in by_id.values() for k in evenly(items, cap)}


def infer(settings: Settings, rows: list[dict], gts: dict[str, dict], models: dict[str, BaseAdapter],
          cfg: dict, run_dir: Path, limit: int | None, log) -> None:
    det_iou = cfg["detection"]["iou"]
    plates_cfg = cfg.get("plates") or {}
    ocr_on = set(plates_cfg.get("ocr_on") or ["pred", "gt"])
    min_read = float(plates_cfg.get("min_det_score", 0.3))
    choke = (cfg.get("faces") or {}).get("chokepoint") or {}
    rec_cfg = choke.get("recognition") or {}
    margin = float(rec_cfg.get("crop_margin", 0.2))
    emb_vecs: list[np.ndarray] = []
    emb_meta: list[dict] = []
    n_frames, t0 = 0, time.monotonic()

    with open(run_dir / "predictions.jsonl", "w", encoding="utf-8") as out:
        for row in rows:
            task = row["task"]
            det_role, aux_role = TASK_ROLES[task]
            detector, aux = models.get(det_role), models.get(aux_role)
            gt = gts[row["clip_id"]]
            plan = recognition_plan(row, gt, rec_cfg) if task == "face" and aux else {}
            frames = sorted(gt)
            if limit:
                frames = frames[:limit]
            log.info("%s: %d labelled frames", row["clip_id"], len(frames))
            for fpath in frames:
                img = load_bgr(fpath)
                h, w = img.shape[:2]
                preds: list[dict] = []
                if detector is not None:
                    for b in detector.detect(img):
                        b = Box(b.x1, b.y1, b.x2, b.y2, b.score, task, b.meta).clipped(w, h)
                        if b is not None:
                            preds.append({"xyxy": [round(float(v), 2) for v in (b.x1, b.y1, b.x2, b.y2)],
                                          "score": round(float(b.score), 5), "label": task})
                gt_reads: dict[int, dict] = {}
                if task == "plate" and aux is not None:
                    if "pred" in ocr_on:
                        for p in preds:
                            if p["score"] >= min_read:
                                r = aux.read_plate(crop(img, p["xyxy"]))
                                p["text"], p["text_score"] = r.text, round(float(r.score), 4)
                    if "gt" in ocr_on:
                        for i, g in enumerate(gt[fpath]):
                            if g["label"] == "plate" and g.get("text"):
                                r = aux.read_plate(crop(img, g["xyxy"]))
                                gt_reads[i] = {"text": r.text, "score": round(float(r.score), 4)}
                for i, g in enumerate(gt[fpath]):
                    role = plan.get((str(fpath), i))
                    if role:
                        try:
                            emb_vecs.append(np.asarray(aux.embed_face(crop(img, g["xyxy"], margin)), np.float32))
                            ok = True
                        except (ValueError, RuntimeError) as e:
                            log.debug("embed failed %s #%d: %s", fpath.name, i, e)
                            ok = False
                        emb_meta.append({"clip_id": row["clip_id"], "frame": settings.rel(fpath), "gt_index": i,
                                         "identity": g["identity"], "role": role, "ok": ok})
                rec = {
                    "clip_id": row["clip_id"], "task": task, "source": row["source"],
                    "country": row["country"] or "-", "condition": row["condition"] or "-",
                    "frame": settings.rel(fpath), "width": w, "height": h,
                    "iou": float(choke.get("iou", det_iou[task])) if row["source"] == "chokepoint"
                    else float(det_iou[task]),
                    "gt": [{**g, "xyxy": [round(v, 2) for v in g["xyxy"]]} for g in gt[fpath]],
                    "preds": preds,
                    "gt_reads": {str(k): v for k, v in gt_reads.items()},
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_frames += 1
                if n_frames % 200 == 0:
                    log.info("%d frames (%.1f fps)", n_frames, n_frames / (time.monotonic() - t0))
    ok_vecs = [v for v, m in zip(emb_vecs, emb_meta) if m["ok"]]
    np.savez_compressed(run_dir / "embeddings.npz",
                        vectors=np.stack(ok_vecs) if ok_vecs else np.zeros((0, 0), np.float32),
                        meta=np.array(json.dumps(emb_meta)))
    log.info("Inference done: %d frames in %.0fs", n_frames, time.monotonic() - t0)


# ---------------------------------------------------------------- scoring
def load_predictions(run_dir: Path) -> list[dict]:
    path = run_dir / "predictions.jsonl"
    if not path.is_file():
        raise GccvtError(f"{path} not found")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def plate_pairs(frames: list[dict], cfg: dict) -> tuple[list[dict], list[dict]]:
    """End-to-end and recognition-only OCR records: {ref, hyp, raw_ref, raw_hyp, frame...}."""
    norm = (cfg.get("plates") or {}).get("normalize") or {}
    min_read = float((cfg.get("plates") or {}).get("min_det_score", 0.3))
    kw = {"ksa_arabic_to_latin": norm.get("ksa_arabic_to_latin", True),
          "canonical_order": norm.get("canonical_order", True), "fold": norm.get("fold")}
    e2e, rec_only = [], []
    for f in frames:
        if f["task"] != "plate":
            continue
        read_preds = [p for p in f["preds"] if p["score"] >= min_read and "text" in p]
        _, g2p = M.match(read_preds, f["gt"], f["iou"])
        has_reader = any("text" in p for p in f["preds"]) or bool(f["gt_reads"])
        for i, g in enumerate(f["gt"]):
            if g["label"] != "plate" or not g.get("text"):
                continue
            ref = normalize_plate(g["text"], country=f["country"], **kw)
            base = {"frame": f["frame"], "clip_id": f["clip_id"], "country": f["country"],
                    "condition": f["condition"], "source": f["source"], "gt_xyxy": g["xyxy"],
                    "raw_ref": g["text"], "ref": ref}
            if has_reader:
                p = read_preds[g2p[i]] if g2p[i] >= 0 else None
                raw = p["text"] if p else ""
                e2e.append({**base, "raw_hyp": raw, "hyp": normalize_plate(raw, country=f["country"], **kw),
                            "detected": p is not None})
            r = f["gt_reads"].get(str(i))
            if r is not None:
                rec_only.append({**base, "raw_hyp": r["text"],
                                 "hyp": normalize_plate(r["text"], country=f["country"], **kw)})
    return e2e, rec_only


def grouped(items: list[dict], fields: list[str]) -> dict[str, dict[str, list[dict]]]:
    out: dict[str, dict[str, list[dict]]] = {}
    for field in fields:
        groups: dict[str, list[dict]] = defaultdict(list)
        for it in items:
            groups[str(it.get(field) or "-")].append(it)
        out[field] = dict(sorted(groups.items()))
    return out


def recognition_metrics(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "embeddings.npz"
    if not path.is_file():
        return {}
    data = np.load(path, allow_pickle=False)
    meta = json.loads(str(data["meta"]))
    if not meta:
        return {}
    vecs = iter(data["vectors"])
    rows = [(m, next(vecs) if m["ok"] else None) for m in meta]
    by_id: dict[str, list[np.ndarray]] = defaultdict(list)
    for m, v in rows:
        if m["role"] == "gallery" and v is not None:
            by_id[m["identity"]].append(v)
    gallery = {pid: np.mean(vs, axis=0) for pid, vs in by_id.items()}
    probes = [(m["identity"], v) for m, v in rows if m["role"] == "probe" and v is not None]
    tracks: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    for m, v in rows:
        if m["role"] == "probe" and v is not None:
            tracks[(m["clip_id"], m["identity"])].append(v)
    return {
        "gallery_identities": len(gallery),
        "embed_failures": sum(1 for m, v in rows if v is None),
        "frame_level": M.identification(gallery, probes),
        "track_level": M.identification(gallery, [(pid, np.mean(vs, axis=0)) for (_, pid), vs in tracks.items()]),
    }


def compute_metrics(frames: list[dict], cfg: dict, run_dir: Path) -> tuple[dict, dict]:
    det = cfg["detection"]
    lo, hi, step = det.get("map_iou", [0.5, 0.95, 0.05])
    map_ious = [round(float(x), 3) for x in np.arange(lo, hi + 1e-9, step)]
    fields = list(cfg.get("breakdown") or ["country", "condition", "source", "clip_id"])
    result: dict[str, Any] = {"frames": len(frames), "detection": {}, "plate_ocr": {}}
    for task in ("plate", "face"):
        tf = [f for f in frames if f["task"] == task]
        if not tf:
            continue
        args = (task, float(det["iou"][task]), float(det.get("score_threshold", 0.5)), map_ious)
        result["detection"][task] = {
            "overall": M.detection_metrics(tf, *args),
            "by": {fld: {k: M.detection_metrics(v, *args) for k, v in g.items()} for fld, g in grouped(tf, fields).items()},
        }
    e2e, rec_only = plate_pairs(frames, cfg)
    for name, recs in (("end_to_end", e2e), ("recognition_only", rec_only)):
        if recs:
            result["plate_ocr"][name] = {
                "overall": M.ocr_metrics([(r["ref"], r["hyp"]) for r in recs]),
                "by": {fld: {k: M.ocr_metrics([(r["ref"], r["hyp"]) for r in v]) for k, v in g.items()}
                       for fld, g in grouped(recs, fields).items()},
            }
    rec = recognition_metrics(run_dir)
    if rec:
        result["face_recognition"] = rec
    return result, {"e2e": e2e, "rec_only": rec_only}


# ---------------------------------------------------------------- failures
def save_crop(settings: Settings, frame_rel: str, xyxy: list[float], out: Path, context: float,
              color: str, extra: list[float] | None = None) -> None:
    src = Path(frame_rel)
    src = src if src.is_absolute() else settings.root / src
    with Image.open(src) as im:
        im = im.convert("RGB")
        draw = ImageDraw.Draw(im)
        width = max(2, im.width // 400)
        if extra:
            draw.rectangle(extra, outline="#2a9d8f", width=width)
        draw.rectangle(xyxy, outline=color, width=width)
        x1, y1, x2, y2 = xyxy
        mx, my = (x2 - x1) * context + 16, (y2 - y1) * context + 16
        box = (max(0, int(x1 - mx)), max(0, int(y1 - my)), min(im.width, int(x2 + mx)), min(im.height, int(y2 + my)))
        im = im.crop(box)
        im.thumbnail((480, 480))
        out.parent.mkdir(parents=True, exist_ok=True)
        im.save(out, quality=85)


def find_failures(settings: Settings, frames: list[dict], ocr: dict, cfg: dict, run_dir: Path, log) -> dict:
    fcfg = cfg.get("failures") or {}
    n = int(fcfg.get("per_type", 25))
    ctx = float(fcfg.get("context", 0.6))
    thr = float(cfg["detection"].get("score_threshold", 0.5))
    missed, false_pos = [], []
    for f in frames:
        kept = [p for p in f["preds"] if p["score"] >= thr]
        p2g, g2p = M.match(kept, f["gt"], f["iou"])
        for gi, pi in enumerate(g2p):
            if pi < 0:
                g = f["gt"][gi]
                area = (g["xyxy"][2] - g["xyxy"][0]) * (g["xyxy"][3] - g["xyxy"][1])
                missed.append({**_ctx(f), "xyxy": g["xyxy"], "label": g["label"], "area": round(area)})
        for pi, gi in enumerate(p2g):
            if gi < 0:
                false_pos.append({**_ctx(f), "xyxy": kept[pi]["xyxy"], "label": kept[pi]["label"],
                                  "score": kept[pi]["score"]})
    wrong = [{**_ctx(r), "xyxy": r["gt_xyxy"], "raw_ref": r["raw_ref"], "raw_hyp": r["raw_hyp"],
              "ref": r["ref"], "hyp": r["hyp"], "cer": round(cer(r["ref"], r["hyp"]), 4),
              "detected": r.get("detected", True)} for r in ocr["e2e"] if r["ref"] != r["hyp"]]
    out = {
        "missed_detections": sorted(missed, key=lambda x: -x["area"])[:n],
        "false_positives": sorted(false_pos, key=lambda x: -x["score"])[:n],
        "plate_misreads": sorted(wrong, key=lambda x: (-x["cer"], x["frame"]))[:n],
        "counts": {"missed_detections": len(missed), "false_positives": len(false_pos), "plate_misreads": len(wrong)},
    }
    colors = {"missed_detections": "#e63946", "false_positives": "#f4a261", "plate_misreads": "#e63946"}
    for kind in ("missed_detections", "false_positives", "plate_misreads"):
        for i, item in enumerate(out[kind]):
            rel = f"failures/{kind}_{i:03d}.jpg"
            try:
                save_crop(settings, item["frame"], item["xyxy"], run_dir / rel, ctx, colors[kind])
                item["crop"] = rel
            except OSError as e:
                log.warning("could not save crop for %s: %s", item["frame"], e)
    return out


def _ctx(f: dict) -> dict:
    return {k: f[k] for k in ("clip_id", "frame", "country", "condition", "source")}


# ---------------------------------------------------------------- console summary
def summary_lines(m: dict) -> list[str]:
    def pct(v):
        return "  n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{100 * v:5.1f}"
    lines = []
    for task, d in m.get("detection", {}).items():
        o = d["overall"]
        lines.append(f"{task:5} detection  P {pct(o['precision'])}  R {pct(o['recall'])}  AP {pct(o['ap'])}  "
                     f"mAP[.5:.95] {pct(o['map@[.5:.95]'])}  ({o['gt']} GT in {o['frames']} frames)")
    for name, d in m.get("plate_ocr", {}).items():
        o = d["overall"]
        lines.append(f"plate OCR {name:16} exact {pct(o['exact_match'])}  CER {pct(o['cer'])}  "
                     f"digits {pct(o['digits_exact_match'])}  ({o['plates']} plates)")
    rec = m.get("face_recognition", {})
    for lvl in ("frame_level", "track_level"):
        if rec.get(lvl, {}).get("probes"):
            o = rec[lvl]
            lines.append(f"face recognition {lvl:11} rank-1 {pct(o['rank1'])}  EER {pct(o.get('eer'))}  "
                         f"({o['probes']} probes)")
    return lines


# ---------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    p = base_parser(__doc__.splitlines()[0])
    add_filter_args(p)
    p.add_argument("--model", help="model name from config/models.yaml (default: its 'default')")
    p.add_argument("--run-id", help="results folder name (default: <model>_<timestamp>)")
    p.add_argument("--rescore", metavar="RUN_ID", help="recompute metrics for an existing run (no inference)")
    p.add_argument("--limit-frames", type=int, help="at most N labelled frames per clip (quick check)")
    args = p.parse_args(argv)
    settings, log = init("run_eval", args)
    cfg = settings.load("eval.yaml")
    if "detection" not in cfg:
        raise ConfigError("eval.yaml needs a detection: section")
    results = settings.path("results")

    if args.rescore:
        run_dir = results / args.rescore
        if not run_dir.is_dir():
            raise GccvtError(f"no such run: {run_dir}")
        run_info = json.loads((run_dir / "run.json").read_text()) if (run_dir / "run.json").is_file() else {}
    else:
        mcfg = settings.load("models.yaml")
        model = args.model or mcfg.get("default")
        spec = (mcfg.get("models") or {}).get(model)
        if not spec:
            raise ConfigError(f"model {model!r} not in config/models.yaml (have: {sorted(mcfg.get('models') or {})})")

        rows = []
        for r in Manifest(settings.path("manifest")).rows.values():
            if (args.clip and r["clip_id"] not in args.clip) or (args.task and r["task"] != args.task):
                continue
            if args.country and r["country"] != settings.normalize_country(args.country):
                continue
            if args.condition and r["condition"] != args.condition.lower():
                continue
            rows.append(r)
        gts, unlabeled = {}, []
        for r in rows:
            try:
                gt = load_gt(settings, r, cfg)
            except GccvtError as e:
                log.error("%s: %s", r["clip_id"], e)
                gt = {}
            if gt:
                gts[r["clip_id"]] = gt
            else:
                unlabeled.append(r["clip_id"])
        rows = [r for r in rows if r["clip_id"] in gts]
        if unlabeled:
            log.info("%d clip(s) without ground truth skipped (label them into data/annotations/<clip_id>/gt/)",
                     len(unlabeled))
        if not rows:
            log.warning("No labelled clips to evaluate.")
            return 0
        needed = {role for r in rows for role in TASK_ROLES[r["task"]]}
        missing = {TASK_ROLES[r["task"]][0] for r in rows} - {k for k, v in spec.items() if v}
        if missing:
            log.warning("model %s has no %s - those detection metrics will be zero", model, sorted(missing))
        log.info("Evaluating model %s on %d clip(s), %d labelled frames", model, len(rows),
                 sum(min(len(g), args.limit_frames or len(g)) for g in gts.values()))
        if args.dry_run:
            for r in rows:
                print(f"{r['clip_id']:45} {len(gts[r['clip_id']]):6d} frames")
            return 0

        run_id = args.run_id or f"{model}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        run_dir = results / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        run_info = {"run_id": run_id, "model": model, "model_spec": spec,
                    "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "clips": [r["clip_id"] for r in rows], "unlabeled_clips": unlabeled,
                    "limit_frames": args.limit_frames}
        (run_dir / "run.json").write_text(json.dumps(run_info, indent=2, ensure_ascii=False, default=str))
        models = load_models(settings, spec, needed, log)
        try:
            infer(settings, rows, gts, models, cfg, run_dir, args.limit_frames, log)
        finally:
            for adapter in {id(a): a for a in models.values()}.values():
                adapter.close()

    frames = load_predictions(run_dir)
    result, ocr = compute_metrics(frames, cfg, run_dir)
    result = {"run_id": run_dir.name, "model": run_info.get("model"), "scored": datetime.now(timezone.utc)
              .isoformat(timespec="seconds"), "eval_config": cfg,
              "unlabeled_clips": run_info.get("unlabeled_clips", []), **result}
    (run_dir / "metrics.json").write_text(json.dumps(clean(result), indent=2, ensure_ascii=False))
    failures = find_failures(settings, frames, ocr, cfg, run_dir, log)
    (run_dir / "failures.json").write_text(json.dumps(clean(failures), indent=2, ensure_ascii=False))
    for line in summary_lines(clean(result)):
        log.info("%s", line)
    log.info("Results: %s", settings.rel(run_dir))
    return 0
