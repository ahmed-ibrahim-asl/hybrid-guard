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
/* Layout: one column read like a bench test certificate; summary, chart, then one checklist row per candidate.
   Identity: eng-asl.com tokens. Dark first, ink ramp, one gold accent at low area, instrument status colours,
   near-square 4px corners, hairlines instead of shadows, Archivo for text, Space Mono for labels and data. */
:root {
  --page: #0B0D11; --surface: #12161C; --raised: #1C2129; --ink: #E6E8EB; --muted: #929CAB;
  --hair: #262D37; --hair-strong: #39424F; --accent: #D9A441; --accent-text: #D9A441;
  --ok: #5FA37A; --alert: #C4553D; --idle: #78828F;
  --ok-a12: rgba(95, 163, 122, 0.12); --alert-a12: rgba(196, 85, 61, 0.12); --idle-a12: rgba(120, 130, 143, 0.14);
  --region: rgba(95, 163, 122, 0.09); --grid-line: rgba(230, 232, 235, 0.045);
  --inset-top: inset 0 1px 0 rgba(230, 232, 235, 0.055);
  --font-sans: "Archivo", "Helvetica Neue", Arial, sans-serif;
  --font-mono: "Space Mono", "SFMono-Regular", Menlo, Consolas, monospace;
  color-scheme: dark;
}
@media (prefers-color-scheme: light) { :root:not([data-theme="dark"]) {
  --page: #F4F5F7; --surface: #FFFFFF; --raised: #FFFFFF; --ink: #0B0D11; --muted: #3E4652;
  --hair: #D5D9DE; --hair-strong: #B9BFC8; --accent: #B8842C; --accent-text: #8F6520;
  --ok: #326E4E; --alert: #A8402B; --idle: #727B89;
  --ok-a12: rgba(63, 133, 96, 0.13); --alert-a12: rgba(168, 64, 43, 0.13); --idle-a12: rgba(114, 123, 137, 0.14);
  --region: rgba(63, 133, 96, 0.08); --grid-line: rgba(11, 13, 17, 0.05);
  --inset-top: inset 0 1px 0 rgba(11, 13, 17, 0.05);
  color-scheme: light; } }
:root[data-theme="light"] {
  --page: #F4F5F7; --surface: #FFFFFF; --raised: #FFFFFF; --ink: #0B0D11; --muted: #3E4652;
  --hair: #D5D9DE; --hair-strong: #B9BFC8; --accent: #B8842C; --accent-text: #8F6520;
  --ok: #326E4E; --alert: #A8402B; --idle: #727B89;
  --ok-a12: rgba(63, 133, 96, 0.13); --alert-a12: rgba(168, 64, 43, 0.13); --idle-a12: rgba(114, 123, 137, 0.14);
  --region: rgba(63, 133, 96, 0.08); --grid-line: rgba(11, 13, 17, 0.05);
  --inset-top: inset 0 1px 0 rgba(11, 13, 17, 0.05);
  color-scheme: light; }
* { box-sizing: border-box; }
body { background: var(--page); color: var(--ink); font: 400 15px/1.65 var(--font-sans);
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
.page { max-width: 1040px; margin: 0 auto; padding-block: 36px 56px; display: grid; gap: 32px; }
header { display: grid; gap: 16px; border-bottom: 1px solid var(--hair-strong); padding-bottom: 22px; }
.brandline { display: flex; align-items: center; gap: 12px; }
.mark { width: 30px; height: 33px; color: var(--accent); flex: none; }
.eyebrow { font: 400 12px/1.2 var(--font-mono); letter-spacing: 0.12em; text-transform: uppercase; color: var(--accent-text); }
h1 { font: 600 clamp(28px, 4.4vw, 44px)/1.08 var(--font-sans); letter-spacing: -0.025em; margin: 0; text-wrap: balance; }
h2 { font: 600 clamp(19px, 2.2vw, 24px)/1.14 var(--font-sans); letter-spacing: -0.015em; margin: 0; }
.dataset { margin: 0; color: var(--muted); font-size: 14px; }
.settings { display: flex; flex-wrap: wrap; gap: 8px 26px; margin: 0; }
.settings div { display: grid; gap: 2px; }
.settings dt { font: 400 11px/1.2 var(--font-mono); letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted); }
.settings dd { margin: 0; font: 700 14px/1.3 var(--font-mono); font-variant-numeric: tabular-nums; }
.lede { font-size: clamp(17px, 2vw, 20px); line-height: 1.5; max-width: 62ch; margin: 0; }
.lede strong { font-weight: 600; color: var(--accent-text); }
.tally { display: flex; flex-wrap: wrap; gap: 10px; }
.tally span { display: inline-flex; align-items: baseline; gap: 8px; font: 400 12px/1 var(--font-mono); letter-spacing: 0.12em;
  text-transform: uppercase; padding: 10px 14px; border-radius: 4px; border: 1px solid var(--hair); background: var(--surface); box-shadow: var(--inset-top); }
.tally b { font: 700 20px/1 var(--font-mono); letter-spacing: 0; }
.v-ACCEPT b { color: var(--ok); } .v-REJECT b { color: var(--alert); } .v-ABSTAIN b { color: var(--idle); }
section { display: grid; gap: 14px; min-width: 0; }
.section-head { display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between; gap: 8px 20px; }
.chart-wrap { background: var(--surface); border: 1px solid var(--hair); border-radius: 4px; padding: 14px; overflow-x: auto; box-shadow: var(--inset-top); }
.chart-wrap svg { display: block; width: 100%; min-width: 560px; height: auto; }
.tick { font: 400 11px var(--font-mono); fill: var(--muted); }
.axis-title { font: 400 12px var(--font-sans); fill: var(--muted); }
.grid { stroke: var(--hair); stroke-width: 1; }
.pt-label { font: 500 12px var(--font-sans); fill: var(--ink); paint-order: stroke; stroke: var(--surface); stroke-width: 4px; stroke-linejoin: round; }
.note { font-size: 13px; color: var(--muted); margin: 0; max-width: 90ch; }
.legend { display: flex; flex-wrap: wrap; gap: 16px; font: 400 11px/1.2 var(--font-mono); letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted); }
.legend i { display: inline-block; width: 9px; height: 9px; border-radius: 999px; margin-right: 7px; vertical-align: 0; }
.rows { display: grid; gap: 10px; }
.cand { background: var(--surface); border: 1px solid var(--hair); border-radius: 4px; padding: 16px 18px; box-shadow: var(--inset-top);
  display: grid; grid-template-columns: minmax(190px, 1.25fr) repeat(5, minmax(0, 1fr)); gap: 14px; align-items: start; }
.cand.ACCEPT { border-color: color-mix(in srgb, var(--ok) 45%, var(--hair)); }
.cand-head { display: grid; gap: 7px; min-width: 0; }
.cand-name { font: 600 16px/1.25 var(--font-sans); letter-spacing: -0.01em; overflow-wrap: anywhere; }
.cand-sub { font-size: 12.5px; color: var(--muted); line-height: 1.45; }
.chip { justify-self: start; font: 700 11px/1 var(--font-mono); letter-spacing: 0.12em; padding: 6px 9px; border-radius: 2px; }
.chip.ACCEPT { background: var(--ok-a12); color: var(--ok); }
.chip.REJECT { background: var(--alert-a12); color: var(--alert); }
.chip.ABSTAIN { background: var(--idle-a12); color: var(--idle); }
.cond { display: grid; gap: 4px; min-width: 0; font-size: 12.5px; padding-left: 12px; border-left: 1px solid var(--hair); }
.cond-name { display: flex; align-items: center; gap: 6px; font: 400 10.5px/1.2 var(--font-mono); letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted); }
.cond-val { font: 700 14px/1.3 var(--font-mono); font-variant-numeric: tabular-nums; }
.cond-why { color: var(--muted); font-size: 12px; line-height: 1.45; }
.tick-mark { width: 13px; height: 13px; flex: none; }
.tick-mark.pass { color: var(--ok); } .tick-mark.fail { color: var(--alert); } .tick-mark.na { color: var(--idle); }
details { font-size: 14px; border-top: 1px solid var(--hair); padding-top: 14px; }
summary { cursor: pointer; font: 400 12px/1.2 var(--font-mono); letter-spacing: 0.12em; text-transform: uppercase; color: var(--accent-text); }
summary:focus-visible, a:focus-visible { outline: 2px solid var(--accent); outline-offset: 3px; border-radius: 2px; }
.howto { display: grid; gap: 8px; margin: 12px 0 0; padding-left: 18px; max-width: 75ch; }
footer { border-top: 1px solid var(--hair); padding-top: 16px; font-size: 12.5px; color: var(--muted); display: grid; gap: 6px; }
footer cite { font-style: normal; color: var(--ink); }
@media (max-width: 780px) {
  .cand { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .cand-head { grid-column: 1 / -1; }
}
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
"""

MARK = ('<svg class="mark" viewBox="0 0 96 104" fill="none" aria-hidden="true"><path d="M48 2 92 27v50L48 102 4 77V27Z" '
        'stroke="currentColor" stroke-width="3"/><g stroke="currentColor" stroke-width="3"><path d="M48 26l13 7.5v15L48 56l-13-7.5v-15Z"/>'
        '<path d="M31 55l13 7.5v15L31 85l-13-7.5v-15Z"/><path d="M65 55l13 7.5v15L65 85l-13-7.5v-15Z"/></g></svg>')

ICONS = {
    "pass": '<svg class="tick-mark pass" viewBox="0 0 16 16" aria-hidden="true"><path d="M3 8.5l3.2 3L13 4.5" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>',
    "fail": '<svg class="tick-mark fail" viewBox="0 0 16 16" aria-hidden="true"><path d="M4 4l8 8M12 4l-8 8" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"/></svg>',
    "na": '<svg class="tick-mark na" viewBox="0 0 16 16" aria-hidden="true"><path d="M4 8h8" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>',
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
    parts.append('<defs><pattern id="bp" width="24" height="24" patternUnits="userSpaceOnUse">'
                 '<path d="M24 0H0V24" fill="none" stroke="var(--grid-line)" stroke-width="1"/></pattern></defs>')
    parts.append(f'<rect x="{left}" y="{top}" width="{pw}" height="{ph}" fill="url(#bp)"/>')
    region_right = min(sx(margin), left + pw)
    parts.append(f'<rect x="{left}" y="{sy(bound):.1f}" width="{max(region_right - left, 0):.1f}" height="{sy(0) - sy(bound):.1f}" fill="var(--region)"/>')
    for t in _ticks(0, y_hi):
        parts.append(f'<line class="grid" x1="{left}" x2="{left + pw}" y1="{sy(t):.1f}" y2="{sy(t):.1f}"/>')
        parts.append(f'<text class="tick" x="{left - 8}" y="{sy(t) + 4:.1f}" text-anchor="end">{t:.2f}</text>')
    for t in _ticks(x_lo, x_hi):
        parts.append(f'<text class="tick" x="{sx(t):.1f}" y="{top + ph + 18}" text-anchor="middle">{_signed(t, 3) if t else "0"}</text>')
    parts.append(f'<line x1="{sx(0):.1f}" x2="{sx(0):.1f}" y1="{top}" y2="{top + ph}" stroke="var(--muted)" stroke-width="1"/>')
    parts.append(f'<line x1="{sx(margin):.1f}" x2="{sx(margin):.1f}" y1="{top}" y2="{top + ph}" stroke="var(--muted)" stroke-width="1" stroke-dasharray="2 4"/>')
    parts.append(f'<line x1="{left}" x2="{left + pw}" y1="{sy(bound):.1f}" y2="{sy(bound):.1f}" stroke="var(--alert)" stroke-width="1.2" stroke-dasharray="6 4"/>')
    parts.append(f'<text class="tick" x="{left + pw}" y="{sy(bound) - 6:.1f}" text-anchor="end" style="fill: var(--alert)">harm bound {bound}</text>')
    parts.append(f'<text class="tick" x="{sx(margin) + 4:.1f}" y="{top + 12}">margin +{margin}</text>')
    parts.append(f'<text class="tick" x="{left + 6}" y="{sy(bound) + 16:.1f}" style="fill: var(--ok)">accept region</text>')
    parts.append(f'<text class="axis-title" x="{left + pw / 2:.1f}" y="{height - 10}" text-anchor="middle">NRMSE change against the incumbent (left of 0 is more accurate)</text>')
    parts.append(f'<text class="axis-title" transform="translate(16 {top + ph / 2:.1f}) rotate(-90)" text-anchor="middle">material harm on changed rows</text>')
    points = [(sx(r.nrmse_difference), sy(r.harm)) for r in plotted.values()]
    placed: list[tuple[float, float, float, float]] = []
    for name, r in sorted(plotted.items(), key=lambda kv: -kv[1].harm):
        color = {"ACCEPT": "var(--ok)", "REJECT": "var(--alert)", "ABSTAIN": "var(--idle)"}[r.decision]
        cx, cy = sx(r.nrmse_difference), sy(r.harm)
        upper = sy(r.harm_upper if r.harm_upper is not None else r.harm)
        tip = (f"{name}: {r.decision}. NRMSE change {_signed(r.nrmse_difference)} "
               f"[{_signed(r.nrmse_lower)}, {_signed(r.nrmse_upper)}]; harm {r.harm:.3f}"
               + (f" (upper {r.harm_upper:.3f})" if r.harm_upper is not None else "") + f"; coverage {r.coverage:.3f}")
        parts.append(f'<g><title>{_e(tip)}</title>'
                     f'<line x1="{sx(r.nrmse_lower):.1f}" x2="{sx(r.nrmse_upper):.1f}" y1="{cy:.1f}" y2="{cy:.1f}" stroke="{color}" stroke-width="2" stroke-linecap="round"/>'
                     f'<line x1="{cx:.1f}" x2="{cx:.1f}" y1="{cy:.1f}" y2="{upper:.1f}" stroke="{color}" stroke-width="2" stroke-linecap="round"/>'
                     f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="12" fill="transparent"/>'
                     f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5.5" fill="{color}" stroke="var(--surface)" stroke-width="2"/></g>')
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
  <div class="brandline">{MARK}<div class="eyebrow">hybrid-guard &middot; coverage-matched acceptance test</div></div>
  <h1>{_e(title)}</h1>
  {f'<p class="dataset">{_e(dataset)}</p>' if dataset else ''}
  <dl class="settings">{''.join(f'<div><dt>{_e(k)}</dt><dd>{_e(v)}</dd></div>' for k, v in settings)}</dl>
</header>
<section>
  <p class="lede">{lede}</p>
  <div class="tally">{''.join(f'<span class="v-{v}"><b>{counts[v]}</b>{v.lower()}</span>' for v in ('ACCEPT', 'REJECT', 'ABSTAIN'))}</div>
</section>
<section>
  <div class="section-head"><h2>Accuracy against row-level safety</h2>
  <div class="legend"><span><i style="background: var(--ok)"></i>accepted</span><span><i style="background: var(--alert)"></i>rejected</span><span><i style="background: var(--idle)"></i>abstained</span></div></div>
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
            '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;600;700&amp;family=Space+Mono:wght@400;700&amp;display=swap">\n'
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
