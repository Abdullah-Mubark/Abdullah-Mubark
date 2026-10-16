"""Inline-SVG horizontal bar charts (grouped), styled by CSS tokens defined in the report page."""

from __future__ import annotations

import math
from html import escape

BAR, GAP, GROUP_GAP = 14, 2, 14
LABEL_W, VALUE_W, PLOT_W = 150, 56, 420


def _num(v) -> float | None:
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)


def _bar_path(x0: float, y: float, w: float, h: float, r: float = 4) -> str:
    """Bar from baseline x0, square at the baseline, 4px rounded data end."""
    r = min(r, w, h / 2)
    if w <= 0:
        return ""
    return (f"M{x0:.1f},{y:.1f} H{x0 + w - r:.1f} Q{x0 + w:.1f},{y:.1f} {x0 + w:.1f},{y + r:.1f} "
            f"V{y + h - r:.1f} Q{x0 + w:.1f},{y + h:.1f} {x0 + w - r:.1f},{y + h:.1f} H{x0:.1f} Z")


def hbar(title: str, subtitle: str, categories: list[str], series: list[tuple[str, str, dict]],
         fmt=lambda v: f"{100 * v:.1f}%", vmax: float | None = None, tick_fmt=lambda v: f"{100 * v:.0f}%") -> str:
    """series: [(name, css_var, {category: value})]. Values None/NaN are shown as 'n/a'."""
    vals = [_num(s[2].get(c)) for s in series for c in categories]
    vmax = vmax or max([v for v in vals if v is not None] + [1.0])
    group_h = len(series) * BAR + (len(series) - 1) * GAP
    top = 8
    height = top + len(categories) * (group_h + GROUP_GAP) + 22
    width = LABEL_W + PLOT_W + VALUE_W
    x0 = LABEL_W
    parts = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}" '
             f'preserveAspectRatio="xMinYMin meet">']
    for t in (0, 0.25, 0.5, 0.75, 1.0):
        x = x0 + PLOT_W * t
        parts.append(f'<line class="grid" x1="{x:.1f}" x2="{x:.1f}" y1="{top - 4}" y2="{height - 18}"/>')
        parts.append(f'<text class="tick" x="{x:.1f}" y="{height - 4}" text-anchor="middle">'
                     f'{escape(tick_fmt(vmax * t))}</text>')
    for ci, cat in enumerate(categories):
        gy = top + ci * (group_h + GROUP_GAP)
        parts.append(f'<text class="cat" x="{LABEL_W - 10}" y="{gy + group_h / 2 + 4:.1f}" text-anchor="end">'
                     f'{escape(_short(cat))}<title>{escape(cat)}</title></text>')
        for si, (name, var, data) in enumerate(series):
            y = gy + si * (BAR + GAP)
            v = _num(data.get(cat))
            tip = escape(f"{cat} · {name}: {fmt(v) if v is not None else 'n/a'}")
            if v is None:
                parts.append(f'<text class="val" x="{x0 + 4}" y="{y + BAR - 3}">n/a</text>')
                continue
            w = PLOT_W * min(v, vmax) / vmax if vmax else 0
            parts.append(f'<g class="mark" tabindex="0" data-tip="{tip}">'
                         f'<rect class="hit" x="{x0}" y="{y - 1}" width="{PLOT_W + VALUE_W}" height="{BAR + 2}"/>'
                         f'<path d="{_bar_path(x0, y, max(w, 1.5), BAR)}" fill="var({var})"/></g>')
            parts.append(f'<text class="val" x="{x0 + w + 6:.1f}" y="{y + BAR - 3}">{escape(fmt(v))}</text>')
    parts.append(f'<line class="axis" x1="{x0}" x2="{x0}" y1="{top - 4}" y2="{height - 18}"/>')
    parts.append("</svg>")
    legend = ""
    if len(series) > 1:
        legend = '<div class="legend">' + "".join(
            f'<span><i style="background:var({var})"></i>{escape(name)}</span>' for name, var, _ in series) + "</div>"
    table = _table(categories, series, fmt)
    return (f'<figure class="card chart-card"><figcaption><h3>{escape(title)}</h3>'
            f'<p class="sub">{escape(subtitle)}</p></figcaption>{legend}<div class="chart-scroll">{"".join(parts)}</div>'
            f'<details><summary>Table</summary>{table}</details></figure>')


def _table(categories, series, fmt) -> str:
    head = "".join(f"<th>{escape(n)}</th>" for n, _, _ in series)
    rows = "".join(
        f"<tr><td>{escape(c)}</td>" + "".join(
            f'<td class="num">{escape(fmt(v)) if (v := _num(d.get(c))) is not None else "n/a"}</td>'
            for _, _, d in series) + "</tr>" for c in categories)
    return f'<table><thead><tr><th></th>{head}</tr></thead><tbody>{rows}</tbody></table>'


def _short(s: str, n: int = 22) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"
