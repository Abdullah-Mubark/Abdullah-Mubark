"""Compare two evaluation runs: what differed (code, packages, models, config, data) and how the
headline metrics moved."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from ..cli import base_parser, init
from ..errors import GccvtError

SKIP = {"run_id", "created"}  # always differ


def flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else str(k)))
        return out
    return {prefix: obj}


def load_run(run_dir: Path) -> tuple[dict, dict]:
    if not run_dir.is_dir():
        raise GccvtError(f"no such run: {run_dir}")
    run = json.loads((run_dir / "run.json").read_text()) if (run_dir / "run.json").is_file() else {}
    metrics = json.loads((run_dir / "metrics.json").read_text()) if (run_dir / "metrics.json").is_file() else {}
    return run, metrics


def setup_diff(run_a: dict, run_b: dict) -> list[tuple[str, Any, Any]]:
    keys = ("model", "model_spec", "clips", "limit_frames", "provenance")
    a = flatten({k: run_a.get(k) for k in keys})
    b = flatten({k: run_b.get(k) for k in keys})
    return [(k, a.get(k), b.get(k)) for k in sorted(set(a) | set(b))
            if k not in SKIP and a.get(k) != b.get(k)]


def headline(m: dict) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for task, d in (m.get("detection") or {}).items():
        for k in ("precision", "recall", "ap", "map@[.5:.95]"):
            out[f"{task} detection {k}"] = d["overall"].get(k)
    for mode, d in (m.get("plate_ocr") or {}).items():
        for k in ("exact_match", "cer", "digits_exact_match"):
            out[f"plate OCR {mode} {k}"] = d["overall"].get(k)
    for lvl in ("frame_level", "track_level"):
        o = (m.get("face_recognition") or {}).get(lvl) or {}
        for k in ("rank1", "eer"):
            if k in o:
                out[f"face recognition {lvl} {k}"] = o[k]
    return out


def metric_diff(m_a: dict, m_b: dict) -> list[tuple[str, Any, Any, float | None]]:
    a, b = headline(m_a), headline(m_b)
    rows = []
    for k in sorted(set(a) | set(b)):
        va, vb = a.get(k), b.get(k)
        delta = vb - va if isinstance(va, (int, float)) and isinstance(vb, (int, float)) else None
        rows.append((k, va, vb, delta))
    return rows


def _short(v: Any, n: int = 60) -> str:
    s = json.dumps(v, ensure_ascii=False, default=str) if not isinstance(v, str) else v
    return s if len(s) <= n else s[: n - 1] + "…"


def _pct(v: Any) -> str:
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{100 * v:6.2f}%"


def main(argv: list[str] | None = None) -> int:
    p = base_parser(__doc__.splitlines()[0])
    p.add_argument("run_a", help="run id (folder under data/results)")
    p.add_argument("run_b", help="run id to compare against run_a")
    args = p.parse_args(argv)
    settings, log = init("compare_runs", args)
    results = settings.path("results")
    run_a, m_a = load_run(results / args.run_a)
    run_b, m_b = load_run(results / args.run_b)

    diffs = setup_diff(run_a, run_b)
    print(f"Setup: {args.run_a}  vs  {args.run_b}")
    if not diffs:
        print("  identical code, packages, model files, config and data")
    for k, va, vb in diffs:
        print(f"  {k}\n      {_short(va)}\n   -> {_short(vb)}")
    print("\nMetrics:")
    for k, va, vb, d in metric_diff(m_a, m_b):
        ds = "" if d is None else f"  ({'+' if d >= 0 else ''}{100 * d:.2f} pts)"
        print(f"  {k:42} {_pct(va)} -> {_pct(vb)}{ds}")
    return 0
