"""Build an HTML report (tables, charts, annotated clips, worst failures) for an evaluation run.

Writes data/results/{run_id}/report.html plus clips/*.mp4. Open report.html from that folder.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from html import escape
from pathlib import Path
from typing import Any

from ..cli import base_parser, init
from ..config import Settings
from ..errors import GccvtError
from ..eval import metrics as M
from ..eval.runner import load_predictions, plate_pairs
from .charts import hbar
from .clips import render_run_clips

COND_VARS = {"day": "--series-1", "night": "--series-2"}
TASK_NAMES = {"plate": "Plate", "face": "Face"}


# ---------------------------------------------------------------- formatting
def pct(v: Any, digits: int = 1) -> str:
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{100 * v:.{digits}f}%"


def num(v: Any) -> str:
    return "n/a" if v is None else f"{v:,}" if isinstance(v, int) else str(v)


def table(headers: list[str], rows: list[list[str]], num_cols: set[int] | None = None, cls: str = "") -> str:
    num_cols = num_cols if num_cols is not None else set(range(1, len(headers)))
    head = "".join(f'<th class="{"num" if i in num_cols else ""}">{escape(h)}</th>' for i, h in enumerate(headers))
    body = "".join("<tr>" + "".join(f'<td class="{"num" if i in num_cols else ""}">{c}</td>'
                                    for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f'<div class="table-wrap {cls}"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def tile(label: str, value: str, note: str = "") -> str:
    return (f'<div class="tile"><div class="tile-label">{escape(label)}</div>'
            f'<div class="tile-value">{escape(value)}</div><div class="tile-note">{escape(note)}</div></div>')


# ---------------------------------------------------------------- cross breakdowns for charts
def cross(frames: list[dict], cfg: dict) -> dict:
    """Detection and OCR metrics per (country, condition), computed from predictions."""
    det = cfg["detection"]
    lo, hi, step = det.get("map_iou", [0.5, 0.95, 0.05])
    map_ious = [round(lo + k * step, 3) for k in range(int(round((hi - lo) / step)) + 1)]
    out: dict[str, Any] = {"detection": {}, "ocr": {}}
    for task in ("plate", "face"):
        groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for f in frames:
            if f["task"] == task:
                groups[(f["country"], f["condition"])].append(f)
        out["detection"][task] = {k: M.detection_metrics(v, task, float(det["iou"][task]),
                                                         float(det.get("score_threshold", 0.5)), map_ious)
                                  for k, v in groups.items()}
    e2e, _ = plate_pairs(frames, cfg)
    groups = defaultdict(list)
    for r in e2e:
        groups[(r["country"], r["condition"])].append((r["ref"], r["hyp"]))
    out["ocr"] = {k: M.ocr_metrics(v) for k, v in groups.items()}
    return out


def condition_series(data: dict[tuple[str, str], dict], key: str) -> tuple[list[str], list[tuple[str, str, dict]]]:
    countries = sorted({c for c, _ in data})
    conds = sorted({d for _, d in data})
    series = []
    for i, cond in enumerate(conds):
        var = COND_VARS.get(cond, f"--series-{min(i + 1, 3)}")
        series.append((cond if cond != "-" else "all", var,
                       {c: data[(c, cond)].get(key) for c in countries if (c, cond) in data}))
    return countries, series


# ---------------------------------------------------------------- sections
def header(m: dict, run: dict) -> str:
    clips = len(run.get("clips", []))
    unl = len(m.get("unlabeled_clips") or [])
    meta = [f"run <b>{escape(m['run_id'])}</b>", f"scored {escape(str(m.get('scored', ''))[:16].replace('T', ' '))}",
            f"{m.get('frames', 0):,} labelled frames", f"{clips} clips"]
    if unl:
        meta.append(f"{unl} clips without ground truth (not scored)")
    if run.get("limit_frames"):
        meta.append(f"limited to {run['limit_frames']} frames per clip")
    return (f'<header><p class="eyebrow">gcc-vision-testbench · evaluation</p>'
            f'<h1>{escape(str(m.get("model") or "model"))}</h1><p class="meta">{" · ".join(meta)}</p></header>')


def tiles(m: dict, thr: float) -> str:
    t = []
    for task in ("plate", "face"):
        d = m.get("detection", {}).get(task)
        if d:
            o = d["overall"]
            t.append(tile(f"{TASK_NAMES[task]} detection AP", pct(o["ap"]), f"{o['gt']:,} objects"))
            t.append(tile(f"{TASK_NAMES[task]} recall", pct(o["recall"]), f"precision {pct(o['precision'])} at score ≥ {thr:g}"))
    e2e = m.get("plate_ocr", {}).get("end_to_end", {}).get("overall")
    if e2e:
        t.append(tile("Plate exact match", pct(e2e["exact_match"]), f"end-to-end · {e2e['plates']:,} plates"))
        t.append(tile("Plate CER", pct(e2e["cer"]), "character error rate · lower is better"))
    rec = (m.get("face_recognition") or {}).get("track_level") or {}
    if rec.get("probes"):
        t.append(tile("Face rank-1", pct(rec["rank1"]), f"per track · {rec['probes']} tracks"))
    return f'<section class="tiles">{"".join(t)}</section>' if t else ""


def charts_section(m: dict, cx: dict, thr: float) -> str:
    out = []
    for task in ("plate", "face"):
        data = cx["detection"].get(task) or {}
        if not data:
            continue
        name = TASK_NAMES[task]
        if any(c != "-" for c, _ in data):
            cats, series = condition_series(data, "ap")
            out.append(hbar(f"{name} detection AP by country", "Day vs night · AP at the task IoU", cats, series))
            cats, series = condition_series(data, "recall")
            out.append(hbar(f"{name} recall by country", f"Day vs night · detections with score ≥ {thr:g}", cats, series))
        else:  # e.g. ChokePoint: no country - chart per clip instead
            by_clip = m["detection"][task]["by"].get("clip_id", {})
            cats = sorted(by_clip)
            out.append(hbar(f"{name} detection AP by clip", "AP at the task IoU", cats,
                            [("AP", "--series-1", {c: by_clip[c]["ap"] for c in cats})]))
    if cx["ocr"]:
        cats, series = condition_series(cx["ocr"], "exact_match")
        out.append(hbar("Plate exact match by country", "End-to-end · day vs night", cats, series))
        cats, series = condition_series(cx["ocr"], "cer")
        top = max([v for _, _, d in series for v in d.values() if v is not None] + [0.0])
        vmax = next((4 * st for st in (0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.5) if 4 * st >= top), top)
        out.append(hbar("Plate character error rate by country", "End-to-end · lower is better", cats, series,
                        vmax=vmax))
    if not out:
        return ""
    return f'<section><h2>By country and condition</h2><div class="grid-2">{"".join(out)}</div></section>'


def breakdown_tables(m: dict) -> str:
    parts = []
    for task, d in (m.get("detection") or {}).items():
        for field, groups in d["by"].items():
            if field == "clip_id" or len(groups) < 1:
                continue
            rows = [[escape(k), num(v["frames"]), num(v["gt"]), pct(v["precision"]), pct(v["recall"]),
                     pct(v["ap"]), pct(v["map@[.5:.95]"])] for k, v in groups.items()]
            parts.append(f"<h3>{TASK_NAMES[task]} detection by {escape(field)}</h3>" + table(
                [field, "frames", "objects", "precision", "recall", "AP", "mAP .5:.95"], rows))
    for mode, d in (m.get("plate_ocr") or {}).items():
        label = mode.replace("_", "-")
        rows = [["all", num(d["overall"]["plates"]), pct(d["overall"]["exact_match"]), pct(d["overall"]["cer"]),
                 pct(d["overall"]["digits_exact_match"])]]
        for field, groups in d["by"].items():
            if field == "clip_id":
                continue
            rows += [[f"{escape(field)}: {escape(k)}", num(v["plates"]), pct(v["exact_match"]), pct(v["cer"]),
                      pct(v["digits_exact_match"])] for k, v in groups.items()]
        parts.append(f"<h3>Plate reading ({label})</h3>" + table(
            ["group", "plates", "exact match", "CER", "digits exact"], rows))
    rec = m.get("face_recognition") or {}
    if rec:
        rows = [[lvl.replace("_", " "), num(o.get("probes")), pct(o.get("rank1")), pct(o.get("eer")),
                 pct(o.get("tar@far=1%")), num(o.get("probes_without_gallery"))]
                for lvl, o in rec.items() if isinstance(o, dict)]
        parts.append(f"<h3>Face recognition (ChokePoint protocol)</h3>"
                     f"<p class=\"sub\">{rec.get('gallery_identities', 0)} gallery identities · "
                     f"{rec.get('embed_failures', 0)} embedding failures</p>" + table(
                         ["level", "probes", "rank-1", "EER", "TAR @ FAR 1%", "no gallery"], rows))
    return f'<section><h2>Breakdowns</h2>{"".join(parts)}</section>' if parts else ""


def video(entry: dict) -> str:
    return (f'<video controls muted playsinline preload="none" src="{escape(entry["video"])}" '
            f'poster="{escape(entry["poster"])}"></video>')


def clips_section(m: dict, frames: list[dict], clip_files: dict[str, dict]) -> str:
    info: dict[str, dict] = {}
    for f in frames:
        info.setdefault(f["clip_id"], {"country": f["country"], "condition": f["condition"], "task": f["task"],
                                       "source": f["source"], "frames": 0})
        info[f["clip_id"]]["frames"] += 1
    rows, cards = [], []
    for cid in sorted(info, key=lambda c: (_clip_ap(m, info[c]["task"], c) is None, _clip_ap(m, info[c]["task"], c) or 0)):
        i = info[cid]
        det = ((m.get("detection") or {}).get(i["task"]) or {}).get("by", {}).get("clip_id", {}).get(cid, {})
        ocr = (((m.get("plate_ocr") or {}).get("end_to_end") or {}).get("by", {}).get("clip_id", {}) or {}).get(cid, {})
        link = f'<a href="{escape(clip_files[cid]["video"])}">{escape(cid)}</a>' if cid in clip_files else escape(cid)
        rows.append([link, escape(i["country"]), escape(i["condition"]), num(i["frames"]), num(det.get("gt")),
                     pct(det.get("precision")), pct(det.get("recall")), pct(det.get("ap")),
                     pct(ocr.get("exact_match")) if ocr else "–", pct(ocr.get("cer")) if ocr else "–"])
        if cid in clip_files:
            cards.append(
                f'<figure class="card video-card">{video(clip_files[cid])}<figcaption><b>{escape(cid)}</b>'
                f'<span class="sub">{escape(i["country"])} · {escape(i["condition"])} · AP {pct(det.get("ap"))}'
                f' · recall {pct(det.get("recall"))}</span></figcaption></figure>')
    t = table(["clip", "country", "condition", "frames", "objects", "precision", "recall", "AP", "plate exact",
               "plate CER"], rows, num_cols=set(range(3, 10)))
    vids = (f'<p class="sub">Each labelled frame of the clip with ground truth and detections drawn on it '
            f'(worst clips first). Green = found, blue = detection, red = missed, orange = false positive.</p>'
            f'<div class="grid-3">{"".join(cards)}</div>') if cards else ""
    return f'<section><h2>Clips</h2>{vids}<h3>Per-clip results</h3>{t}</section>'


def _clip_ap(m, task, cid):
    v = (((m.get("detection") or {}).get(task) or {}).get("by", {}).get("clip_id", {}).get(cid) or {}).get("ap")
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else v


def failures_section(f: dict, failure_clips: dict[str, dict]) -> str:
    titles = {"missed_detections": ("Missed detections", "largest objects first"),
              "false_positives": ("False positives", "highest score first"),
              "plate_misreads": ("Plate misreads", "highest character error first")}
    parts = []
    for kind, (title, order) in titles.items():
        items = f.get(kind) or []
        total = (f.get("counts") or {}).get(kind, len(items))
        if not items:
            continue
        cards = []
        for i, it in enumerate(items):
            clip = failure_clips.get(f"{kind}:{i}")
            media = (video(clip) if clip else
                     f'<img src="{escape(it["crop"])}" alt="Crop around the failure">' if it.get("crop") else "")
            if kind == "plate_misreads":
                detail = (f'<span class="plate" dir="auto">GT {escape(it["raw_ref"])}</span>'
                          f'<span class="plate" dir="auto">read {escape(it["raw_hyp"]) or "—"}</span>'
                          f'<span class="sub">CER {pct(it["cer"], 0)}{"" if it.get("detected", True) else " · plate not detected"}</span>')
            elif kind == "false_positives":
                detail = f'<span class="sub">score {it["score"]:.2f} · {escape(it["label"])}</span>'
            else:
                detail = f'<span class="sub">{escape(it["label"])} · {it.get("area", 0):,} px²</span>'
            cards.append(f'<figure class="card fail-card">{media}<figcaption>{detail}<span class="sub">'
                         f'{escape(it["clip_id"])} · {escape(Path(it["frame"]).name)}</span></figcaption></figure>')
        parts.append(f'<h3>{title} <span class="count">{len(items)} of {total:,} · {order}</span></h3>'
                     f'<div class="grid-4">{"".join(cards)}</div>')
    return (f'<section><h2>Worst failures</h2><p class="sub">Short clips around each failure (the failure frame '
            f'is outlined in yellow) with a still crop where no clip is available.</p>{"".join(parts)}</section>'
            if parts else "")


def config_section(m: dict, run: dict) -> str:
    snap = {"model": run.get("model_spec"), "eval": m.get("eval_config")}
    return (f'<section><details><summary>Model and evaluation settings</summary><pre>'
            f'{escape(json.dumps(snap, indent=2, ensure_ascii=False, default=str))}</pre></details></section>')


# ---------------------------------------------------------------- page
CSS = """
:root{color-scheme:light;--surface:#f6f6f4;--surface-1:#fcfcfb;--line:#e4e3df;--grid:#ecebe7;
--text-primary:#0b0b0b;--text-secondary:#52514e;--text-muted:#7a7975;--series-1:#2a78d6;--series-2:#eb6834;
--series-3:#1baf7a;--accent:#2a78d6}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--surface:#111110;
--surface-1:#1a1a19;--line:#2e2e2c;--grid:#262624;--text-primary:#f0efec;--text-secondary:#c3c2b7;
--text-muted:#8f8e88;--series-1:#3987e5;--series-2:#d95926;--series-3:#199e70;--accent:#3987e5}}
:root[data-theme="dark"]{color-scheme:dark;--surface:#111110;--surface-1:#1a1a19;--line:#2e2e2c;--grid:#262624;
--text-primary:#f0efec;--text-secondary:#c3c2b7;--text-muted:#8f8e88;--series-1:#3987e5;--series-2:#d95926;
--series-3:#199e70;--accent:#3987e5}
*{box-sizing:border-box}
body{margin:0;background:var(--surface);color:var(--text-primary);
font:14px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,"Noto Sans Arabic","Noto Sans",sans-serif}
main{max-width:1240px;margin:0 auto;padding:32px 16px 64px}
header{margin-bottom:24px}.eyebrow{margin:0;color:var(--text-muted);font-size:12px;letter-spacing:.04em;text-transform:uppercase}
h1{margin:4px 0 6px;font-size:28px;font-weight:600}h2{font-size:19px;font-weight:600;margin:40px 0 12px}
h3{font-size:15px;font-weight:600;margin:24px 0 8px}.meta,.sub{color:var(--text-secondary);margin:0}
.sub{font-size:12.5px}.count{font-weight:400;color:var(--text-muted);font-size:12.5px;margin-left:6px}
.tiles{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:12px}
.tile{background:var(--surface-1);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.tile-label{color:var(--text-secondary);font-size:12.5px}.tile-value{font-size:28px;font-weight:600;margin:2px 0}
.tile-note{color:var(--text-muted);font-size:12px}
.card{background:var(--surface-1);border:1px solid var(--line);border-radius:10px;margin:0;padding:14px;min-width:0}
.grid-2,.grid-3,.grid-4{display:grid;gap:12px}.grid-2{grid-template-columns:repeat(auto-fill,minmax(min(100%,520px),1fr))}
.grid-3{grid-template-columns:repeat(auto-fill,minmax(min(100%,360px),1fr))}
.grid-4{grid-template-columns:repeat(auto-fill,minmax(min(100%,240px),1fr))}
.chart-card h3{margin:0}.chart-scroll{overflow-x:auto}
.chart{width:100%;min-width:540px;height:auto;display:block;margin-top:8px}
.chart .grid{stroke:var(--grid);stroke-width:1}.chart .axis{stroke:var(--line);stroke-width:1}
.chart text{font-size:11.5px;fill:var(--text-secondary)}.chart .cat{fill:var(--text-primary)}
.chart .tick{fill:var(--text-muted);font-size:10.5px}.chart .hit{fill:transparent}
.chart .mark:focus{outline:none}.chart .mark:focus path,.chart .mark:hover path{filter:brightness(1.12)}
.legend{display:flex;gap:16px;margin-top:8px;font-size:12.5px;color:var(--text-secondary)}
.legend i{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:6px;vertical-align:-1px}
details{margin-top:8px}summary{cursor:pointer;color:var(--text-secondary);font-size:12.5px}
.table-wrap{overflow-x:auto;background:var(--surface-1);border:1px solid var(--line);border-radius:10px}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:7px 12px;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap}
th{font-weight:600;color:var(--text-secondary);font-size:12px}tbody tr:last-child td{border-bottom:0}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}a{color:var(--accent)}
video,.fail-card img{width:100%;border-radius:6px;background:#000;display:block}
video{aspect-ratio:16/9;object-fit:contain}
figcaption{display:flex;flex-direction:column;gap:2px;margin-top:8px;font-size:13px;min-width:0}
figcaption b{overflow-wrap:anywhere}.plate{font-size:14px;font-weight:600;unicode-bidi:plaintext}
pre{background:var(--surface-1);border:1px solid var(--line);border-radius:10px;padding:12px;overflow:auto;font-size:12px}
#tip{position:fixed;pointer-events:none;background:var(--text-primary);color:var(--surface-1);padding:5px 8px;
border-radius:6px;font-size:12px;opacity:0;transition:opacity .08s;z-index:9}
"""

JS = """
const tip=document.getElementById('tip');
function show(e,el){tip.textContent=el.dataset.tip;tip.style.opacity=1;
const r=el.getBoundingClientRect();const x=e&&e.clientX?e.clientX:r.left+r.width/2;const y=e&&e.clientY?e.clientY:r.top;
tip.style.left=Math.min(x+12,innerWidth-tip.offsetWidth-8)+'px';tip.style.top=Math.max(8,y-36)+'px'}
document.querySelectorAll('.mark').forEach(el=>{el.addEventListener('mousemove',e=>show(e,el));
el.addEventListener('focus',()=>show(null,el));['mouseleave','blur'].forEach(t=>el.addEventListener(t,()=>tip.style.opacity=0))});
"""


def build_html(m: dict, run: dict, frames: list[dict], failures: dict, clip_out: dict, cfg: dict) -> str:
    thr = float(cfg["detection"].get("score_threshold", 0.5))
    cx = cross(frames, cfg)
    body = "".join([
        header(m, run), tiles(m, thr), charts_section(m, cx, thr),
        clips_section(m, frames, clip_out.get("clips", {})),
        failures_section(failures, clip_out.get("failure_clips", {})),
        breakdown_tables(m), config_section(m, run),
    ])
    title = f"Eval {escape(str(m.get('model') or ''))}"
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>'
            f'<style>{CSS}</style></head><body><main>{body}</main><div id="tip" role="tooltip"></div>'
            f'<script>{JS}</script></body></html>')


def latest_run(results: Path) -> Path:
    runs = sorted((p for p in results.glob("*/metrics.json")), key=lambda p: p.stat().st_mtime)
    if not runs:
        raise GccvtError(f"no scored runs in {results} - run 'make eval' first")
    return runs[-1].parent


def main(argv: list[str] | None = None) -> int:
    p = base_parser(__doc__.splitlines()[0])
    p.add_argument("--run", metavar="RUN_ID", help="run to report on (default: most recently scored)")
    p.add_argument("--no-clips", action="store_true", help="skip rendering annotated video clips")
    args = p.parse_args(argv)
    settings, log = init("build_report", args)
    results = settings.path("results")
    run_dir = results / args.run if args.run else latest_run(results)
    if not (run_dir / "metrics.json").is_file():
        raise GccvtError(f"{run_dir} has no metrics.json - run 'make eval' (or --rescore) first")
    m = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8")) if (run_dir / "run.json").is_file() else {}
    failures = json.loads((run_dir / "failures.json").read_text(encoding="utf-8")) \
        if (run_dir / "failures.json").is_file() else {}
    cfg = dict(m.get("eval_config") or settings.load("eval.yaml"))
    cfg["report"] = settings.load("eval.yaml").get("report") or cfg.get("report") or {}
    frames = load_predictions(run_dir)
    if args.dry_run:
        log.info("Would build %s from %d frames", run_dir / "report.html", len(frames))
        return 0

    clip_out: dict = {}
    if not args.no_clips:
        clip_out = render_run_clips(settings, frames, failures, cfg, run_dir)
    (run_dir / "report.html").write_text(build_html(m, run, frames, failures, clip_out, cfg), encoding="utf-8")
    log.info("Report: %s (%d clips, %d failure clips)", settings.rel(run_dir / "report.html"),
             len(clip_out.get("clips", {})), len(clip_out.get("failure_clips", {})))
    return 0
