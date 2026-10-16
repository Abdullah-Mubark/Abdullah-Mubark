"""Drop near-duplicate frames within each clip using perceptual hashing.

Each frame is compared with the last few kept frames; if the Hamming distance between hashes is
within the threshold it is a duplicate. By default duplicates are moved to {clip}/_duplicates/
so the step is reversible - changing the settings and re-running restores and re-checks them.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from ..cli import base_parser, init
from ..errors import ConfigError, GccvtError
from .common import (DUP_DIR, add_filter_args, frames_dir, read_index, read_state, select_clips,
                     write_index, write_state)

STATE = ".dedupe.json"
METHODS = ("phash", "dhash")
ACTIONS = ("move", "delete", "mark")


def _dct_matrix(n: int) -> np.ndarray:
    k = np.arange(n)
    m = np.cos(np.pi * (2 * k[None, :] + 1) * k[:, None] / (2 * n)) * np.sqrt(2 / n)
    m[0] /= np.sqrt(2)
    return m


def phash(img: Image.Image, hash_size: int = 8) -> int:
    """DCT perceptual hash (same idea as imagehash.phash)."""
    n = hash_size * 4
    px = np.asarray(img.convert("L").resize((n, n), Image.Resampling.LANCZOS), dtype=np.float64)
    d = _dct_matrix(n)
    low = (d @ px @ d.T)[:hash_size, :hash_size]
    bits = (low > np.median(low)).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def dhash(img: Image.Image, hash_size: int = 8) -> int:
    px = np.asarray(img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS), dtype=np.int16)
    bits = (px[:, 1:] > px[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def image_hash(path: Path, method: str, hash_size: int) -> int:
    with Image.open(path) as img:
        return phash(img, hash_size) if method == "phash" else dhash(img, hash_size)


def restore(folder: Path) -> int:
    dup = folder / DUP_DIR
    if not dup.is_dir():
        return 0
    n = 0
    for f in dup.iterdir():
        f.replace(folder / f.name)
        n += 1
    dup.rmdir()
    return n


def dedupe_clip(folder: Path, cfg: dict, force: bool, log) -> tuple[int, int] | None:
    """Returns (kept, duplicates), or None if already up to date."""
    method, action = cfg["method"], cfg["action"]
    if method not in METHODS:
        raise ConfigError(f"dedupe.method must be one of {METHODS}, got {method!r}")
    if action not in ACTIONS:
        raise ConfigError(f"dedupe.action must be one of {ACTIONS}, got {action!r}")
    want = {k: cfg[k] for k in ("method", "hash_size", "threshold", "window", "action")}
    state = read_state(folder / STATE)
    if not force and state == want:
        return None
    rows = read_index(folder)
    if not rows:
        raise GccvtError(f"{folder}: no index.csv - run extract_frames first")
    if state and state.get("action") == "delete":
        raise GccvtError(f"{folder.name}: duplicates were deleted earlier; re-run extract_frames --force first")
    restore(folder)

    hash_size, threshold, window = int(cfg["hash_size"]), int(cfg["threshold"]), max(1, int(cfg["window"]))
    recent: list[tuple[str, int]] = []
    kept = dups = 0
    for r in rows:
        path = folder / r["frame"]
        if not path.is_file():
            r["status"], r["dup_of"] = "missing", ""
            continue
        h = image_hash(path, method, hash_size)
        r["hash"] = f"{h:0{hash_size * hash_size // 4}x}"
        match = next((name for name, kh in reversed(recent) if hamming(h, kh) <= threshold), None)
        if match:
            r["status"], r["dup_of"] = "duplicate", match
            dups += 1
            if action == "move":
                (folder / DUP_DIR).mkdir(exist_ok=True)
                path.replace(folder / DUP_DIR / path.name)
            elif action == "delete":
                path.unlink()
        else:
            r["status"], r["dup_of"] = "kept", ""
            kept += 1
            recent = (recent + [(r["frame"], h)])[-window:]
    write_index(folder, rows)
    write_state(folder / STATE, want)
    return kept, dups


def effective_config(cfg: dict, row: dict) -> dict:
    over = cfg.get("overrides") or {}
    task_o = (over.get("task") or {}).get(row["task"]) or {}
    src_o = (over.get("source") or {}).get(row["source"]) or {}
    return {**cfg, **task_o, **src_o}


def main(argv: list[str] | None = None) -> int:
    p = base_parser(__doc__.splitlines()[0])
    add_filter_args(p)
    p.add_argument("--threshold", type=int, help="override dedupe.threshold")
    p.add_argument("--undo", action="store_true", help="move duplicates back and clear dedupe results")
    args = p.parse_args(argv)
    settings, log = init("dedupe", args)
    cfg = {"method": "phash", "hash_size": 8, "threshold": 6, "window": 3, "action": "move",
           **(settings.load("prep.yaml").get("dedupe") or {})}
    if args.threshold is not None:
        cfg["threshold"] = args.threshold

    rows = [r for r in select_clips(settings, args) if frames_dir(settings, r["clip_id"]).is_dir()]
    folders = [frames_dir(settings, r["clip_id"]) for r in rows]
    if not folders:
        log.info("No extracted frames found - run extract_frames first.")
        return 0
    if args.dry_run:
        for f in folders:
            print(f.name)
        return 0

    if args.undo:
        for f in folders:
            n = restore(f)
            rows = read_index(f)
            write_index(f, [{**r, "status": "kept" if r["status"] == "duplicate" else r["status"], "dup_of": ""}
                            for r in rows])
            (f / STATE).unlink(missing_ok=True)
            if n:
                log.info("%s: restored %d frames", f.name, n)
        return 0

    total_k = total_d = failed = 0
    for r, f in zip(rows, folders):
        clip_cfg = effective_config(cfg, r)
        if not clip_cfg.get("enabled", True):
            log.debug("Dedupe disabled for %s (overrides)", f.name)
            continue
        try:
            res = dedupe_clip(f, clip_cfg, args.force, log)
        except GccvtError as e:
            log.error("%s", e)
            failed += 1
            continue
        if res is None:
            log.debug("Up to date: %s", f.name)
            continue
        k, d = res
        total_k, total_d = total_k + k, total_d + d
        log.info("%s: kept %d, duplicates %d", f.name, k, d)
    log.info("Dedupe: %d frames kept, %d duplicates (%s, threshold %s)", total_k, total_d,
             cfg["method"], cfg["threshold"])
    return 1 if failed else 0
