"""Self-contained HTML acceptance report (no dependencies beyond the standard library).

``html_report(results)`` turns CMAT results for one or more candidates into a single page:
a verdict summary, the accuracy-against-safety chart and one condition checklist per
candidate, with light and dark themes and a phone-width layout.
"""

from __future__ import annotations

import html
import math
from collections.abc import Mapping
from datetime import date
from pathlib import Path

from hybrid_guard.cmat import CMATResult

CITATION = ("A. I. Asl, B. M. El-Den. Aggregate error margins cannot certify physics-informed wind-power estimators: "
            "a coverage-matched acceptance test with sealed multi-farm SCADA evidence. Manuscript submitted, 2026.")

CSS = """
/* Layout: one column read like a test certificate; summary, chart, then one checklist row per candidate. */
:root {
  --paper: #f6f8f8; --sheet: #ffffff; --ink: #142028; --muted: #59686f; --line: #d5dee0;
  --accent: #1c6773; --accept: #1d7f55; --reject: #b8392e; --abstain: #96640f;
  --accept-soft: #e3f2ea; --reject-soft: #f8e5e2; --abstain-soft: #f6ecd8; --region: rgba(29, 127, 85, 0.08);
  --font-display: "IBM Plex Sans Condensed", "Arial Narrow", "Segoe UI", system-ui, sans-serif;
  --font-body: "IBM Plex Sans", "Segoe UI", system-ui, -apple-system, sans-serif;
  --font-data: "IBM Plex Mono", "Cascadia Mono", Consolas, ui-monospace, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --paper: #0e1619; --sheet: #142024; --ink: #e2eaec; --muted: #94a5ab; --line: #27363b;
  --accent: #66b6c2; --accept: #55c48f; --reject: #ee7a6c; --abstain: #e1aa52;
  --accept-soft: #173328; --reject-soft: #3a1f1c; --abstain-soft: #372b17; --region: rgba(85, 196, 143, 0.10);
  color-scheme: dark; } }
:root[data-theme="dark"] {
  --paper: #0e1619; --sheet: #142024; --ink: #e2eaec; --muted: #94a5ab; --line: #27363b;
  --accent: #66b6c2; --accept: #55c48f; --reject: #ee7a6c; --abstain: #e1aa52;
  --accept-soft: #173328; --reject-soft: #3a1f1c; --abstain-soft: #372b17; --region: rgba(85, 196, 143, 0.10);
  color-scheme: dark; }
* { box-sizing: border-box; }
body { background: var(--paper); color: var(--ink); font: 15px/1.55 var(--font-body);
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
.page { max-width: 1000px; margin: 0 auto; padding-block: 32px 48px; display: grid; gap: 28px; }
header { display: grid; gap: 14px; border-bottom: 2px solid var(--ink); padding-bottom: 18px; }
.eyebrow { font: 600 12px/1 var(--font-data); letter-spacing: 0.12em; text-transform: uppercase; color: var(--accent); }
h1 { font: 600 clamp(26px, 4vw, 36px)/1.1 var(--font-display); margin: 0; text-wrap: balance; letter-spacing: -0.01em; }
h2 { font: 600 19px/1.2 var(--font-display); margin: 0; }
.settings { display: flex; flex-wrap: wrap; gap: 6px 22px; margin: 0; font-size: 13px; color: var(--muted); }
.settings div { display: flex; gap: 6px; }
.settings dt { margin: 0; }
.settings dd { margin: 0; font-family: var(--font-data); color: var(--ink); font-variant-numeric: tabular-nums; }
.lede { font-size: 17px; max-width: 68ch; margin: 0; }
.lede strong { font-weight: 600; }
.tally { display: flex; flex-wrap: wrap; gap: 10px; }
.tally span { font: 600 13px/1 var(--font-data); padding: 8px 12px; border-radius: 4px; border: 1px solid var(--line); background: var(--sheet); }
.tally b { font-size: 17px; margin-right: 6px; }
.v-ACCEPT { color: var(--accept); } .v-REJECT { color: var(--reject); } .v-ABSTAIN { color: var(--abstain); }
section { display: grid; gap: 12px; min-width: 0; }
.chart-wrap { background: var(--sheet); border: 1px solid var(--line); border-radius: 6px; padding: 14px; overflow-x: auto; }
.chart-wrap svg { display: block; width: 100%; min-width: 560px; height: auto; }
.axis text, .tick { font: 11px var(--font-data); fill: var(--muted); }
.axis-title { font: 12px var(--font-body); fill: var(--muted); }
.grid { stroke: var(--line); stroke-width: 1; }
.pt-label { font: 11.5px var(--font-body); fill: var(--ink); paint-order: stroke; stroke: var(--sheet); stroke-width: 4px; stroke-linejoin: round; }
.note { font-size: 13px; color: var(--muted); margin: 0; }
.legend { display: flex; flex-wrap: wrap; gap: 14px; font-size: 13px; color: var(--muted); }
.legend i { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; vertical-align: -1px; }
.rows { display: grid; gap: 10px; }
.cand { background: var(--sheet); border: 1px solid var(--line); border-radius: 6px; padding: 14px 16px;
  display: grid; grid-template-columns: minmax(180px, 1.3fr) repeat(5, minmax(0, 1fr)); gap: 12px; align-items: start; }
.cand-head { display: grid; gap: 6px; min-width: 0; }
.cand-name { font: 600 15px/1.25 var(--font-display); overflow-wrap: anywhere; }
.cand-sub { font-size: 12px; color: var(--muted); }
.chip { justify-self: start; font: 700 11px/1 var(--font-data); letter-spacing: 0.08em; padding: 5px 8px; border-radius: 3px; }
.chip.ACCEPT { background: var(--accept-soft); color: var(--accept); }
.chip.REJECT { background: var(--reject-soft); color: var(--reject); }
.chip.ABSTAIN { background: var(--abstain-soft); color: var(--abstain); }
.cond { display: grid; gap: 3px; min-width: 0; font-size: 12.5px; }
.cond-name { display: flex; align-items: center; gap: 6px; font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); }
.cond-val { font-family: var(--font-data); font-variant-numeric: tabular-nums; font-size: 13px; }
.cond-why { color: var(--muted); font-size: 12px; }
.mark { width: 14px; height: 14px; flex: none; }
.mark.pass { color: var(--accept); } .mark.fail { color: var(--reject); } .mark.na { color: var(--muted); }
details { font-size: 14px; }
summary { cursor: pointer; font-weight: 600; }
summary:focus-visible, a:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.howto { display: grid; gap: 6px; margin: 10px 0 0; padding-left: 18px; max-width: 75ch; }
footer { border-top: 1px solid var(--line); padding-top: 14px; font-size: 12.5px; color: var(--muted); display: grid; gap: 6px; }
footer cite { font-style: normal; color: var(--ink); }
@media (max-width: 760px) {
  .cand { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .cand-head { grid-column: 1 / -1; }
}
"""

ICONS = {
    "pass": '<svg class="mark pass" viewBox="0 0 16 16" aria-hidden="true"><path d="M3 8.5l3.2 3L13 4.5" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>',
    "fail": '<svg class="mark fail" viewBox="0 0 16 16" aria-hidden="true"><path d="M4 4l8 8M12 4l-8 8" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"/></svg>',
    "na": '<svg class="mark na" viewBox="0 0 16 16" aria-hidden="true"><path d="M4 8h8" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>',
}


def _e(value) -> str:
    return html.escape(str(value), quote=True)


def _signed(value: float, digits: int = 4) -> str:
    return f"{value:+.{digits}f}".replace("-", "−")


def _conditions(r: CMATResult) -> list[tuple[str, str, str, str]]:
    s = r.settings
    out = [("Aggregate", "pass" if r.nrmse_upper < s["margin"] else "fail", f"{_signed(r.nrmse_upper)}",
            f"NRMSE change upper bound, must be below {s['margin']}")]
    support = r.changed_rows >= s["min_rows"]
    out.append(("Support", "pass" if support else "fail", f"{r.changed_rows:,} rows", f"changed by more than {s['materiality']}, need {s['min_rows']}"))
    if not support:
        out += [(name, "na", "not tested", "too few changed rows") for name in ("Net benefit", "Safety")]
        out.append(("Skill", "na", "not tested", "too few changed rows" if r.selective else "full hybrid, no selection"))
        return out
    out.append(("Net benefit", "pass" if r.net_benefit_lower > 0 else "fail", f"{_signed(r.net_benefit_lower, 3)}",
                f"benefit {r.benefit:.3f} minus harm {r.harm:.3f}, lower bound above 0"))
    out.append(("Safety", "pass" if r.harm_upper <= s["harm_bound"] else "fail", f"{r.harm_upper:.3f}",
                f"material harm upper bound, at most {s['harm_bound']}"))
    if r.selective:
        out.append(("Skill", "pass" if r.skill_gap_lower > 0 else "fail", f"{_signed(r.skill_gap_lower, 3)}",
                    f"beats random selection (base rate {r.base_rate:.3f})"))
    else:
        out.append(("Skill", "na", "n/a", "full hybrid, no selection"))
    return out


def _ticks(lo: float, hi: float, count: int = 5) -> list[float]:
    span = hi - lo
    raw = span / max(count, 1)
    magnitude = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1
    step = min((m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw), default=magnitude * 10)
    start = math.ceil(lo / step) * step
    ticks, value = [], start
    while value <= hi + 1e-12:
        ticks.append(round(value, 10))
        value += step
    return ticks


def _chart(results: Mapping[str, CMATResult]) -> str:
    plotted = {k: r for k, r in results.items() if r.harm is not None}
    if not plotted:
        return '<p class="note">No candidate changed enough rows to plot.</p>'
    first = next(iter(results.values())).settings
    margin, bound = first["margin"], first["harm_bound"]
    xs = [v for r in plotted.values() for v in (r.nrmse_lower, r.nrmse_upper)] + [0.0, margin]
    x_lo, x_hi = min(xs), max(xs)
    pad = (x_hi - x_lo) * 0.08 or 0.001
    x_lo, x_hi = x_lo - pad, x_hi + pad * 2.5
    y_hi = min(1.0, max(0.5, max((r.harm_upper or r.harm) for r in plotted.values()) * 1.15))
    width, height, left, right, top, bottom = 760, 420, 64, 24, 18, 52
    pw, ph = width - left - right, height - top - bottom

    def sx(v): return left + (v - x_lo) / (x_hi - x_lo) * pw
    def sy(v): return top + (1 - v / y_hi) * ph

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="NRMSE change against material harm for each candidate">']
    region_right = min(sx(margin), left + pw)
    parts.append(f'<rect x="{left}" y="{sy(bound):.1f}" width="{max(region_right - left, 0):.1f}" height="{sy(0) - sy(bound):.1f}" fill="var(--region)"/>')
    for t in _ticks(0, y_hi):
        parts.append(f'<line class="grid" x1="{left}" x2="{left + pw}" y1="{sy(t):.1f}" y2="{sy(t):.1f}"/>')
        parts.append(f'<text class="tick" x="{left - 8}" y="{sy(t) + 4:.1f}" text-anchor="end">{t:.2f}</text>')
    for t in _ticks(x_lo, x_hi):
        parts.append(f'<text class="tick" x="{sx(t):.1f}" y="{top + ph + 18}" text-anchor="middle">{_signed(t, 3) if t else "0"}</text>')
    parts.append(f'<line x1="{sx(0):.1f}" x2="{sx(0):.1f}" y1="{top}" y2="{top + ph}" stroke="var(--muted)" stroke-width="1"/>')
    parts.append(f'<line x1="{sx(margin):.1f}" x2="{sx(margin):.1f}" y1="{top}" y2="{top + ph}" stroke="var(--muted)" stroke-width="1" stroke-dasharray="2 4"/>')
    parts.append(f'<line x1="{left}" x2="{left + pw}" y1="{sy(bound):.1f}" y2="{sy(bound):.1f}" stroke="var(--reject)" stroke-width="1.2" stroke-dasharray="6 4"/>')
    parts.append(f'<text class="tick" x="{left + pw}" y="{sy(bound) - 6:.1f}" text-anchor="end" style="fill: var(--reject)">harm bound {bound}</text>')
    parts.append(f'<text class="tick" x="{sx(margin) + 4:.1f}" y="{top + 12}">margin +{margin}</text>')
    parts.append(f'<text class="tick" x="{left + 6}" y="{sy(bound) + 16:.1f}" style="fill: var(--accept)">accept region</text>')
    parts.append(f'<text class="axis-title" x="{left + pw / 2:.1f}" y="{height - 10}" text-anchor="middle">NRMSE change against the incumbent (left of 0 is more accurate)</text>')
    parts.append(f'<text class="axis-title" transform="translate(16 {top + ph / 2:.1f}) rotate(-90)" text-anchor="middle">material harm on changed rows</text>')
    points = [(sx(r.nrmse_difference), sy(r.harm)) for r in plotted.values()]
    placed: list[tuple[float, float, float, float]] = []
    for name, r in sorted(plotted.items(), key=lambda kv: -kv[1].harm):
        color = f"var(--{r.decision.lower()})"
        cx, cy = sx(r.nrmse_difference), sy(r.harm)
        upper = sy(r.harm_upper if r.harm_upper is not None else r.harm)
        tip = (f"{name}: {r.decision}. NRMSE change {_signed(r.nrmse_difference)} "
               f"[{_signed(r.nrmse_lower)}, {_signed(r.nrmse_upper)}]; harm {r.harm:.3f}"
               + (f" (upper {r.harm_upper:.3f})" if r.harm_upper is not None else "") + f"; coverage {r.coverage:.3f}")
        parts.append(f'<g><title>{_e(tip)}</title>'
                     f'<line x1="{sx(r.nrmse_lower):.1f}" x2="{sx(r.nrmse_upper):.1f}" y1="{cy:.1f}" y2="{cy:.1f}" stroke="{color}" stroke-width="2" stroke-linecap="round"/>'
                     f'<line x1="{cx:.1f}" x2="{cx:.1f}" y1="{cy:.1f}" y2="{upper:.1f}" stroke="{color}" stroke-width="2" stroke-linecap="round"/>'
                     f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="12" fill="transparent"/>'
                     f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5.5" fill="{color}" stroke="var(--sheet)" stroke-width="2"/></g>')
        w = 6.4 * len(name)
        # labels go on the side with more room: left of points in the right half, right otherwise
        lx = cx - 12 - w if cx > left + pw / 2 else cx + 12
        lx = min(max(lx, left + 2), left + pw - w)
        ly = cy + 4

        def clashes(x, y):
            box_hits = any(abs(y - py) < 14 and x < px + pw2 and px < x + w for px, py, pw2, _h in placed)
            point_hits = any(abs((y - 4) - qy) < 9 and x - 4 < qx < x + w + 4 for qx, qy in points)
            return box_hits or point_hits

        for step in range(24):
            offset = (step // 2 + 1) * 14 * (1 if step % 2 == 0 else -1) if step else 0
            if not clashes(lx, cy + 4 + offset):
                ly = cy + 4 + offset
                break
        placed.append((lx, ly, w, 14))
        if abs(ly - (cy + 4)) > 1:
            anchor_x = lx + w if lx < cx else lx
            parts.append(f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{anchor_x:.1f}" y2="{ly - 4:.1f}" stroke="var(--muted)" stroke-width="0.8"/>')
        parts.append(f'<text class="pt-label" x="{lx:.1f}" y="{ly:.1f}">{_e(name)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def html_report(results: Mapping[str, CMATResult], *, title: str = "Acceptance report", dataset: str | None = None,
                notes: Mapping[str, str] | None = None, standalone: bool = True) -> str:
    """Render CMAT results for one or more candidates as a self-contained HTML page."""
    if not results:
        raise ValueError("no results to report")
    notes = notes or {}
    first = next(iter(results.values()))
    s = first.settings
    counts = {v: sum(r.decision == v for r in results.values()) for v in ("ACCEPT", "REJECT", "ABSTAIN")}
    agg_pass = [k for k, r in results.items() if r.nrmse_upper < s["margin"]]
    agg_accept = [k for k in agg_pass if results[k].decision == "ACCEPT"]
    lede = (f"<strong>{counts['ACCEPT']} of {len(results)}</strong> candidates were accepted. "
            f"{len(agg_pass)} passed the aggregate error margin, and {len(agg_accept)} of those "
            f"{'was' if len(agg_accept) == 1 else 'were'} also safe row by row.")
    settings = [("rows", f"{first.rows:,}"), ("materiality", s["materiality"]), ("harm bound", s["harm_bound"]),
                ("margin", s["margin"]), ("min. changed rows", s["min_rows"]), ("bootstrap", f"{s['replicates']:,} over {s['blocks']:,} blocks")]
    rows = []
    for name, r in results.items():
        cells = "".join(f'<div class="cond"><div class="cond-name">{ICONS[state]}{_e(cond)}</div>'
                        f'<div class="cond-val">{_e(value)}</div><div class="cond-why">{_e(why)}</div></div>'
                        for cond, state, value, why in _conditions(r))
        sub = _e(notes.get(name, "")) + (" &middot; " if notes.get(name) else "") + f"coverage {r.coverage:.3f}"
        rows.append(f'<article class="cand"><div class="cand-head"><div class="cand-name">{_e(name)}</div>'
                    f'<span class="chip {r.decision}">{r.decision}</span><div class="cand-sub">{sub}</div></div>{cells}</article>')
    abstained = [k for k, r in results.items() if r.harm is None]
    body = f"""
<div class="page">
<header>
  <div class="eyebrow">hybrid-guard &middot; coverage-matched acceptance test</div>
  <h1>{_e(title)}</h1>
  {f'<p class="note">{_e(dataset)}</p>' if dataset else ''}
  <dl class="settings">{''.join(f'<div><dt>{_e(k)}</dt><dd>{_e(v)}</dd></div>' for k, v in settings)}</dl>
</header>
<section>
  <p class="lede">{lede}</p>
  <div class="tally">{''.join(f'<span class="v-{v}"><b>{counts[v]}</b>{v.lower()}</span>' for v in ('ACCEPT', 'REJECT', 'ABSTAIN'))}</div>
</section>
<section>
  <h2>Accuracy against row-level safety</h2>
  <div class="legend"><span><i style="background: var(--accept)"></i>accepted</span><span><i style="background: var(--reject)"></i>rejected</span><span><i style="background: var(--abstain)"></i>abstained</span></div>
  <div class="chart-wrap">{_chart(results)}</div>
  <p class="note">Horizontal whiskers: 95% interval of the NRMSE change. Vertical whiskers: upper bound of material harm. Hover a point for its values.{' Not plotted (too few changed rows): ' + _e(', '.join(abstained)) + '.' if abstained else ''}</p>
</section>
<section>
  <h2>Conditions by candidate</h2>
  <div class="rows">{''.join(rows)}</div>
</section>
<details>
  <summary>How to read this report</summary>
  <ul class="howto">
    <li>A candidate replaces the incumbent model on the selected rows. Only changes larger than the materiality level count.</li>
    <li>A changed row is harmed when its absolute error grows by more than the materiality level, and improved when it shrinks by more.</li>
    <li>Accept needs every condition: the aggregate margin, enough changed rows, more improvements than harms, few harms with confidence, and for a selective policy, fewer harms than random selection at the same coverage.</li>
    <li>Passing the aggregate margin alone says little: at low coverage even random selection passes it.</li>
  </ul>
</details>
<footer>
  <div>Generated by hybrid-guard on {date.today().isoformat()}. If you use this test, please cite: <cite>{_e(CITATION)}</cite></div>
</footer>
</div>"""
    head = (f'<title>{_e(title)}</title>\n<meta name="claude:viewport-fit-sides" content="cover">\n'
            '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
            '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&amp;family=IBM+Plex+Sans+Condensed:wght@600&amp;family=IBM+Plex+Sans:wght@400;600&amp;display=swap">\n'
            f"<style>{CSS}</style>")
    if not standalone:
        return head + body
    return ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            f"{head}\n</head>\n<body>{body}\n</body>\n</html>\n")


def save_report(path, results: Mapping[str, CMATResult], **kwargs) -> Path:
    path = Path(path)
    path.write_text(html_report(results, **kwargs), encoding="utf-8")
    return path
