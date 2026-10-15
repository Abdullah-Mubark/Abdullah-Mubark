"""Print a summary of manifest.csv (counts and hours by task / country / condition)."""

from __future__ import annotations

from collections import defaultdict

from ..cli import base_parser, init
from ..manifest import Manifest


def main(argv: list[str] | None = None) -> int:
    args = base_parser(__doc__.splitlines()[0]).parse_args(argv)
    settings, log = init("manifest_summary", args)
    manifest = Manifest(settings.path("manifest"))
    if not manifest.rows:
        log.info("Manifest is empty: %s", settings.path("manifest"))
        return 0
    groups: dict[tuple, list[float]] = defaultdict(lambda: [0, 0.0, 0])
    for r in manifest.rows.values():
        g = groups[(r["task"], r["source"], r["country"] or "-", r["condition"] or "-")]
        g[0] += 1
        g[1] += float(r["duration_s"] or 0)
        g[2] += int(r["frame_count"] or 0)
    print(f"{'task':6} {'source':11} {'country':8} {'cond':6} {'clips':>6} {'minutes':>9} {'frames/imgs':>12}")
    for k in sorted(groups):
        n, secs, frames = groups[k]
        print(f"{k[0]:6} {k[1]:11} {k[2]:8} {k[3]:6} {n:6d} {secs / 60:9.1f} {frames:12d}")
    print(f"\n{len(manifest.rows)} rows in {settings.rel(settings.path('manifest'))}")
    return 0
