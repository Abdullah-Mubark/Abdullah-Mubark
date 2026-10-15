"""Detection, plate OCR and face recognition metrics (pure numpy)."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from ..text_norm import cer, digits_only, levenshtein


# ---------------------------------------------------------------- detection
def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between boxes a (N,4) and b (M,4) in xyxy."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def match(preds: list[dict], gts: list[dict], iou_thr: float) -> tuple[list[int], list[int]]:
    """Greedy COCO-style matching by descending score.

    Returns (pred_to_gt, gt_to_pred) index lists, -1 where unmatched.
    """
    order = sorted(range(len(preds)), key=lambda i: -preds[i]["score"])
    ious = iou_matrix(np.array([p["xyxy"] for p in preds], float).reshape(-1, 4),
                      np.array([g["xyxy"] for g in gts], float).reshape(-1, 4))
    p2g, g2p = [-1] * len(preds), [-1] * len(gts)
    for pi in order:
        best, best_iou = -1, iou_thr
        for gi in range(len(gts)):
            if g2p[gi] == -1 and ious[pi, gi] >= best_iou:
                best, best_iou = gi, ious[pi, gi]
        if best >= 0:
            p2g[pi], g2p[best] = best, pi
    return p2g, g2p


def average_precision(scores: list[float], tps: list[bool], n_gt: int) -> float:
    """COCO 101-point interpolated AP."""
    if n_gt == 0:
        return float("nan")
    if not scores:
        return 0.0
    order = np.argsort(-np.asarray(scores), kind="stable")
    tp = np.asarray(tps, float)[order]
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    recall = ctp / n_gt
    precision = ctp / np.maximum(ctp + cfp, 1e-9)
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    points = np.linspace(0, 1, 101)
    idx = np.searchsorted(recall, points, side="left")
    return float(np.mean([precision[i] if i < len(precision) else 0.0 for i in idx]))


def detection_metrics(frames: Iterable[dict], label: str, iou_thr: float, score_thr: float,
                      map_ious: list[float]) -> dict[str, Any]:
    """frames: [{"preds": [...], "gt": [...], "iou"?: float}] with boxes {xyxy, score?, label}.

    A frame's own "iou" (e.g. looser for ChokePoint eye-derived boxes) overrides iou_thr.
    """
    frames = list(frames)
    per_iou: dict[float, tuple[list[float], list[bool]]] = {t: ([], []) for t in set(map_ious) | {iou_thr}}
    n_gt = tp = fp = fn = 0
    for f in frames:
        preds = [p for p in f["preds"] if p["label"] == label]
        gts = [g for g in f["gt"] if g["label"] == label]
        n_gt += len(gts)
        for t, (sc, tps) in per_iou.items():
            p2g, _ = match(preds, gts, f.get("iou", iou_thr) if t == iou_thr else t)
            sc.extend(p["score"] for p in preds)
            tps.extend(m >= 0 for m in p2g)
        kept = [p for p in preds if p["score"] >= score_thr]
        p2g, g2p = match(kept, gts, f.get("iou", iou_thr))
        tp += sum(m >= 0 for m in p2g)
        fp += sum(m < 0 for m in p2g)
        fn += sum(m < 0 for m in g2p)
    aps = {t: average_precision(*per_iou[t], n_gt) for t in per_iou}
    return {
        "frames": len(frames), "gt": n_gt, "tp": tp, "fp": fp, "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "recall": tp / (tp + fn) if tp + fn else float("nan"),
        "ap": aps[iou_thr],                         # AP at the task IoU (per-frame override allowed)
        "map@[.5:.95]": float(np.nanmean([aps[t] for t in map_ious])) if n_gt else float("nan"),
    }


# ---------------------------------------------------------------- plate OCR
def ocr_metrics(pairs: list[tuple[str, str]]) -> dict[str, Any]:
    """pairs: (normalized reference, normalized hypothesis); missing read = ''."""
    if not pairs:
        return {"plates": 0}
    exact = sum(r == h for r, h in pairs)
    dig = [(digits_only(r), digits_only(h)) for r, h in pairs]
    dig = [(r, h) for r, h in dig if r]
    edits = sum(levenshtein(r, h) for r, h in pairs)
    chars = sum(len(r) for r, _ in pairs)
    return {
        "plates": len(pairs),
        "exact_match": exact / len(pairs),
        "cer": edits / chars if chars else float("nan"),           # micro: total edits / total chars
        "mean_plate_cer": float(np.mean([cer(r, h) for r, h in pairs])),
        "digits_exact_match": sum(r == h for r, h in dig) / len(dig) if dig else float("nan"),
    }


# ---------------------------------------------------------------- face recognition
def _l2n(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def identification(gallery: dict[str, np.ndarray], probes: list[tuple[str, np.ndarray]]) -> dict[str, Any]:
    """Rank-1 accuracy of probes (true_id, embedding) against gallery templates {id: embedding}."""
    known = [(pid, e) for pid, e in probes if pid in gallery]
    if not gallery or not known:
        return {"probes": 0, "probes_without_gallery": len(probes) - len(known)}
    ids = sorted(gallery)
    g = _l2n(np.stack([gallery[i] for i in ids]))
    p = _l2n(np.stack([e for _, e in known]))
    sims = p @ g.T
    top = sims.argmax(1)
    correct = sum(ids[t] == pid for t, (pid, _) in zip(top, known))
    genuine = [sims[k, ids.index(pid)] for k, (pid, _) in enumerate(known)]
    impostor = [sims[k, j] for k, (pid, _) in enumerate(known) for j, gid in enumerate(ids) if gid != pid]
    return {"probes": len(known), "probes_without_gallery": len(probes) - len(known),
            "rank1": correct / len(known), **verification(genuine, impostor)}


def verification(genuine: list[float], impostor: list[float]) -> dict[str, float]:
    if not genuine or not impostor:
        return {}
    gen, imp = np.asarray(genuine), np.asarray(impostor)
    thr = np.unique(np.concatenate([gen, imp]))
    far = np.array([(imp >= t).mean() for t in thr])
    frr = np.array([(gen < t).mean() for t in thr])
    i = int(np.argmin(np.abs(far - frr)))
    ok = far <= 0.01
    return {"eer": float((far[i] + frr[i]) / 2),
            "tar@far=1%": float(1 - frr[ok].min()) if ok.any() else 0.0}
