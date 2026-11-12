"""Built-in charts. Each function takes documented inputs and returns a ``Chart``.

A ``Chart`` can be embedded in the HTML report, saved as an SVG for a paper (``chart.save("a.svg")``)
or as a small HTML page (``chart.save("a.html")``). Charts are plain SVG with hover tooltips
(<title> elements), drawn in the eng-asl.com style with light and dark themes.

    accuracy_safety(results)          A  NRMSE change against material harm per candidate
    conditions(results)               B  pass or fail of every CMAT condition per candidate
    changes(results)                  D  share of changed rows improved, neither, harmed
    coverage_harm(curves)             E  harm as a selector covers more rows
    harm_map(table)                   F  harm by condition, from analysis.harm_by_condition
    monitor_chart(series)             G  harm per period, from analysis.monitor
    leaderboard_table(results)        H  ranked table (HTML)
    capacity(curve)                   I  irreducible error and fixable share against delta (Paper A)
    gap(distribution)                 J  distribution of the prior's error (Paper A)
"""

from __future__ import annotations

import html
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from hybrid_guard.analysis import change_breakdown, condition_table, leaderboard
from hybrid_guard.cmat import CMATResult

TOKENS_DARK = ("--page:#0B0D11;--surface:#12161C;--raised:#1C2129;--ink:#E6E8EB;--muted:#929CAB;--hair:#262D37;"
               "--hair-strong:#39424F;--accent:#D9A441;--accent-text:#D9A441;--ok:#5FA37A;--alert:#C4553D;--idle:#78828F;"
               "--ok-a12:rgba(95,163,122,.14);--alert-a12:rgba(196,85,61,.14);--idle-a12:rgba(120,130,143,.16);"
               "--series-1:#B8842C;--series-2:#1F9AB8;--region:rgba(95,163,122,.09);--grid-line:rgba(230,232,235,.045)")
TOKENS_LIGHT = ("--page:#F4F5F7;--surface:#FFFFFF;--raised:#FFFFFF;--ink:#0B0D11;--muted:#3E4652;--hair:#D5D9DE;"
                "--hair-strong:#B9BFC8;--accent:#B8842C;--accent-text:#8F6520;--ok:#326E4E;--alert:#A8402B;--idle:#727B89;"
                "--ok-a12:rgba(63,133,96,.14);--alert-a12:rgba(168,64,43,.14);--idle-a12:rgba(114,123,137,.15);"
                "--series-1:#B8842C;--series-2:#1F9AB8;--region:rgba(63,133,96,.08);--grid-line:rgba(11,13,17,.05)")
FONTS = "--font-sans:'Archivo','Helvetica Neue',Arial,sans-serif;--font-mono:'Space Mono','SFMono-Regular',Menlo,Consolas,monospace"
CHART_CSS = (".hg-tick{font:400 10.5px var(--font-mono);fill:var(--muted)}"
             ".hg-axis{font:400 11.5px var(--font-sans);fill:var(--muted)}"
             ".hg-lbl{font:500 11.5px var(--font-sans);fill:var(--ink);paint-order:stroke;stroke:var(--surface);stroke-width:4px;stroke-linejoin:round}"
             ".hg-cell{font:700 11px var(--font-mono);fill:var(--ink)}")
VERDICT = {"ACCEPT": "var(--ok)", "REJECT": "var(--alert)", "ABSTAIN": "var(--idle)"}
SERIES = ("var(--series-1)", "var(--series-2)", "var(--accent)", "var(--idle)")


def _e(value) -> str:
    return html.escape(str(value), quote=True)


def _signed(value: float, digits: int = 4) -> str:
    return f"{value:+.{digits}f}".replace("-", "−")


def _fmt(value, digits: int = 3) -> str:
    return "n/a" if value is None or (isinstance(value, float) and not math.isfinite(value)) else f"{value:.{digits}f}"


def _ticks(lo: float, hi: float, count: int = 5) -> list[float]:
    raw = (hi - lo) / max(count, 1)
    if raw <= 0:
        return [lo]
    magnitude = 10 ** math.floor(math.log10(raw))
    step = next((m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw), magnitude * 10)
    out, value = [], math.ceil(lo / step) * step
    while value <= hi + 1e-12:
        out.append(round(value, 10))
        value += step
    return out


def _lin(d0, d1, r0, r1):
    span = (d1 - d0) or 1.0
    return lambda v: r0 + (v - d0) / span * (r1 - r0)


def _marker(x: float, y: float, decision: str, r: float = 5.5) -> str:
    c = VERDICT.get(decision, "var(--idle)")
    if decision == "REJECT":
        return (f'<rect x="{x - r * .85:.1f}" y="{y - r * .85:.1f}" width="{r * 1.7:.1f}" height="{r * 1.7:.1f}" '
                f'transform="rotate(45 {x:.1f} {y:.1f})" style="fill:{c};stroke:var(--surface);stroke-width:2"/>')
    if decision == "ABSTAIN":
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" style="fill:var(--surface);stroke:{c};stroke-width:2"/>'
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" style="fill:{c};stroke:var(--surface);stroke-width:2"/>'


def _axes(sx, sy, xt, yt, box, xfmt=lambda v: f"{v:g}", yfmt=lambda v: f"{v:.2f}") -> list[str]:
    left, top, right, bottom = box
    out = []
    for v in yt:
        out.append(f'<line x1="{left}" x2="{right}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" style="stroke:var(--hair)"/>')
        out.append(f'<text class="hg-tick" x="{left - 8}" y="{sy(v) + 4:.1f}" text-anchor="end">{_e(yfmt(v))}</text>')
    for v in xt:
        out.append(f'<text class="hg-tick" x="{sx(v):.1f}" y="{bottom + 16}" text-anchor="middle">{_e(xfmt(v))}</text>')
    return out


def _line(x1, y1, x2, y2, style: str) -> str:
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" style="{style}"/>'


def _text(x, y, text, cls="hg-tick", anchor="start", style="") -> str:
    extra = f' style="{style}"' if style else ""
    return f'<text class="{cls}" x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}"{extra}>{_e(text)}</text>'


def _bound_line(box, sy, bound, label_right=True) -> list[str]:
    left, _, right, _ = box
    return [_line(left, sy(bound), right, sy(bound), "stroke:var(--alert);stroke-dasharray:6 4"),
            _text(right if label_right else left + 4, sy(bound) - 6, f"harm bound {bound:g}", anchor="end" if label_right else "start",
                  style="fill:var(--alert)")]


@dataclass
class Chart:
    """A rendered chart. ``body`` is SVG markup (``kind="svg"``) or an HTML fragment (``kind="html"``)."""

    key: str
    title: str
    caption: str
    body: str
    width: int = 0
    height: int = 0
    kind: str = "svg"

    def svg(self, *, standalone: bool = True, theme: str = "auto") -> str:
        """SVG markup. ``standalone`` adds the colour tokens; ``theme`` is "auto", "dark" or "light"."""
        if self.kind != "svg":
            raise ValueError("this chart is an HTML table; use .html()")
        style = ""
        if standalone:
            if theme == "light":
                tokens = f".hg{{{TOKENS_LIGHT};{FONTS}}}"
            elif theme == "dark":
                tokens = f".hg{{{TOKENS_DARK};{FONTS}}}"
            else:
                tokens = f".hg{{{TOKENS_DARK};{FONTS}}}@media (prefers-color-scheme: light){{.hg{{{TOKENS_LIGHT}}}}}"
            style = f"<style>{tokens}{CHART_CSS}.hg{{background:var(--surface)}}</style>"
        ns = ' xmlns="http://www.w3.org/2000/svg"' if standalone else ""
        return (f'<svg{ns} class="hg" viewBox="0 0 {self.width} {self.height}" role="img" aria-label="{_e(self.title)}">'
                f"{style}{self.body}</svg>")

    def html(self, *, theme: str = "auto") -> str:
        """A complete HTML page holding this chart (for a quick look or sharing)."""
        content = self.body if self.kind == "html" else self.svg(standalone=False)
        light = f"@media (prefers-color-scheme: light){{:root{{{TOKENS_LIGHT}}}}}" if theme == "auto" else ""
        root = TOKENS_LIGHT if theme == "light" else TOKENS_DARK
        from hybrid_guard.report import TABLE_CSS

        return ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
                f"<title>{_e(self.title)}</title><style>:root{{{root};{FONTS}}}{light}"
                "body{margin:0;padding:24px;background:var(--page);color:var(--ink);font:15px/1.5 var(--font-sans)}"
                f"svg{{width:100%;height:auto;max-width:1000px;display:block}}{CHART_CSS}{TABLE_CSS}</style></head><body>"
                f"<h1 style=\"font:600 22px var(--font-sans)\">{_e(self.title)}</h1><p style=\"color:var(--muted)\">{_e(self.caption)}</p>"
                f"{content}</body></html>")

    def save(self, path, **kwargs) -> Path:
        """Save as ``.svg`` (for papers) or ``.html``."""
        path = Path(path)
        text = self.svg(**kwargs) if path.suffix.lower() == ".svg" else self.html(**kwargs)
        path.write_text(text, encoding="utf-8")
        return path

    def _repr_html_(self) -> str:  # Jupyter
        return self.body if self.kind == "html" else self.svg()


def _cand_tip(name: str, r: CMATResult) -> str:
    harm = f"; harm {_fmt(r.harm)}" + (f" (upper {_fmt(r.harm_upper)})" if r.harm_upper is not None else "")
    return (f"{name}: {r.decision}. {r.settings.get('metric', 'NRMSE')} change {_signed(r.nrmse_difference)} [{_signed(r.nrmse_lower)}, "
            f"{_signed(r.nrmse_upper)}]{harm if r.harm is not None else ''}; coverage {r.coverage:.3f}")


def accuracy_safety(results: Mapping[str, CMATResult], *, title: str = "Accuracy against row-level safety") -> Chart:
    """A: one point per candidate. x = NRMSE change with 95% interval, y = material harm with upper bound."""
    plotted = {k: r for k, r in results.items() if r.harm is not None}
    first = next(iter(results.values())).settings
    margin, bound = first["margin"], first["harm_bound"]
    caption = f"Horizontal whiskers: 95% interval of the {first.get('metric', 'NRMSE')} change. Vertical whiskers: upper bound of material harm."
    if not plotted:
        return Chart("accuracy_safety", title, caption, '<text class="hg-axis" x="20" y="40">No candidate changed enough rows.</text>', 600, 80)
    xs = [v for r in plotted.values() for v in (r.nrmse_lower, r.nrmse_upper)] + [0.0, margin]
    lo, hi = min(xs), max(xs)
    pad = (hi - lo) * 0.08 or 0.001
    lo, hi = lo - pad, hi + pad * 2.5
    y_hi = min(1.0, max(0.5, max((r.harm_upper or r.harm) for r in plotted.values()) * 1.15))
    W, H, box = 760, 420, (64, 18, 736, 368)
    sx, sy = _lin(lo, hi, box[0], box[2]), _lin(0, y_hi, box[3], box[1])
    p = [f'<rect x="{box[0]}" y="{sy(bound):.1f}" width="{max(min(sx(margin), box[2]) - box[0], 0):.1f}" height="{sy(0) - sy(bound):.1f}" style="fill:var(--region)"/>']
    p += _axes(sx, sy, _ticks(lo, hi, 5), _ticks(0, y_hi), box, lambda v: _signed(v, 3) if v else "0")
    p += [_line(sx(0), box[1], sx(0), box[3], "stroke:var(--muted)"),
          _line(sx(margin), box[1], sx(margin), box[3], "stroke:var(--muted);stroke-dasharray:2 4"),
          _text(sx(margin) + 4, box[1] + 12, f"margin +{margin:g}")]
    p += _bound_line(box, sy, bound)
    p += [_text(box[0] + 6, sy(bound) + 16, "accept region", style="fill:var(--ok)"),
          _text((box[0] + box[2]) / 2, H - 8, f"{first.get('metric', 'NRMSE')} change against the incumbent (left of 0 is more accurate)", "hg-axis", "middle"),
          f'<text class="hg-axis" transform="translate(16 {(box[1] + box[3]) / 2:.1f}) rotate(-90)" text-anchor="middle">material harm on changed rows</text>']
    points = [(sx(r.nrmse_difference), sy(r.harm)) for r in plotted.values()]
    placed: list[tuple[float, float, float]] = []
    for name, r in sorted(plotted.items(), key=lambda kv: -kv[1].harm):
        c = VERDICT[r.decision]
        x, y = sx(r.nrmse_difference), sy(r.harm)
        up = sy(r.harm_upper if r.harm_upper is not None else r.harm)
        p.append(f"<g><title>{_e(_cand_tip(name, r))}</title>" + _line(sx(r.nrmse_lower), y, sx(r.nrmse_upper), y, f"stroke:{c};stroke-width:2")
                 + _line(x, y, x, up, f"stroke:{c};stroke-width:2") + f'<circle cx="{x:.1f}" cy="{y:.1f}" r="13" style="fill:transparent"/>'
                 + _marker(x, y, r.decision) + "</g>")
        w = 6.4 * len(name)
        lx = x - 12 - w if x > (box[0] + box[2]) / 2 else x + 12
        lx = min(max(lx, box[0] + 2), box[2] - w)
        ly = y + 4
        for k in range(24):
            off = 0 if k == 0 else ((k - 1) // 2 + 1) * 14 * (1 if k % 2 else -1)
            cy = y + 4 + off
            hit = any(abs(cy - py) < 14 and lx < px + pw and px < lx + w for px, py, pw in placed) or \
                any(abs(cy - 4 - qy) < 9 and lx - 4 < qx < lx + w + 4 for qx, qy in points)
            if not hit:
                ly = cy
                break
        placed.append((lx, ly, w))
        if abs(ly - (y + 4)) > 1:
            p.append(_line(x, y, lx + w if lx < x else lx, ly - 4, "stroke:var(--muted);stroke-width:.8"))
        p.append(_text(lx, ly, name, "hg-lbl"))
    return Chart("accuracy_safety", title, caption, "".join(p), W, H)


def conditions(results: Mapping[str, CMATResult], *, title: str = "Condition matrix") -> Chart:
    """B: rows are candidates, columns the five CMAT conditions; each cell shows pass or fail and the value."""
    table = condition_table(results)
    cols = ["aggregate", "support", "net_benefit", "safety", "skill"]
    names = list(results)
    W, row_h, top, left = 760, 34, 30, 200
    cw = (W - left - 8) / len(cols)
    H = top + len(names) * row_h + 8
    p = [_text(left + i * cw + cw / 2, 18, c.replace("_", " ").upper(), anchor="middle") for i, c in enumerate(cols)]
    for r_i, name in enumerate(names):
        y = top + r_i * row_h
        p.append(_text(0, y + row_h / 2 + 4, name, "hg-lbl"))
        for i, cond in enumerate(cols):
            row = table.loc[(table["candidate"] == name) & (table["condition"] == cond)].iloc[0]
            state = row["state"]
            fill = {"pass": "var(--ok-a12)", "fail": "var(--alert-a12)"}.get(state, "var(--idle-a12)")
            ink = {"pass": "var(--ok)", "fail": "var(--alert)"}.get(state, "var(--idle)")
            mark = {"pass": "✓", "fail": "✗"}.get(state, "–")
            value = row["value"]
            if state == "n/a" or not np.isfinite(value):
                shown = "n/a"
            elif cond == "support":
                shown = f"{int(value):,}"
            elif cond in ("aggregate",):
                shown = _signed(value, 4)
            elif cond in ("net_benefit", "skill"):
                shown = _signed(value, 3)
            else:
                shown = f"{value:.3f}"
            x = left + i * cw + 3
            p.append(f"<g><title>{_e(f'{name}: {cond.replace(chr(95), chr(32))} {state}. {row.requirement}')}</title>"
                     f'<rect x="{x:.1f}" y="{y + 3}" width="{cw - 6:.1f}" height="{row_h - 6}" rx="2" style="fill:{fill}"/>'
                     + _text(x + 8, y + row_h / 2 + 4, mark, "hg-cell", style=f"fill:{ink}")
                     + _text(x + cw - 14, y + row_h / 2 + 4, shown, "hg-cell", "end") + "</g>")
    return Chart("conditions", title, "Each cell: pass or fail, and the measured value. Hover for the requirement.", "".join(p), W, H)


def changes(results: Mapping[str, CMATResult], *, title: str = "What happened to the changed rows") -> Chart:
    """D: for each candidate, the share of materially changed rows that improved, did neither or were harmed."""
    table = change_breakdown(results)
    W, row_h, top, left, right = 760, 32, 8, 200, 750
    H = top + len(table) * row_h + 34
    sx = _lin(0, 1, left, right)
    p = []
    for i, row in table.iterrows():
        y = top + i * row_h
        p.append(_text(0, y + row_h / 2 + 4, row.candidate, "hg-lbl"))
        if not np.isfinite(row.harmed):
            p.append(_text(left, y + row_h / 2 + 4, "too few changed rows", style="fill:var(--idle)"))
            continue
        x0, seg = 0.0, []
        for v, col in ((row.improved, "var(--ok)"), (row.neither, "var(--hair-strong)"), (row.harmed, "var(--alert)")):
            x, w = sx(x0), sx(x0 + v) - sx(x0)
            seg.append(f'<rect x="{x + 1:.1f}" y="{y + 6}" width="{max(w - 2, 0):.1f}" height="{row_h - 12}" rx="2" style="fill:{col}"/>')
            if w > 34:
                seg.append(_text(x + w / 2, y + row_h / 2 + 4, f"{round(v * 100)}%", "hg-cell", "middle", "fill:var(--page)"))
            x0 += v
        tip = (f"{row.candidate}: of {row.changed_rows:,} changed rows, improved {round(row.improved * 100)}%, "
               f"neither {round(row.neither * 100)}%, harmed {round(row.harmed * 100)}%")
        p.append(f"<g><title>{_e(tip)}</title>{''.join(seg)}</g>")
    ly = H - 12
    for k, (lab, col) in enumerate((("improved", "var(--ok)"), ("neither", "var(--hair-strong)"), ("harmed", "var(--alert)"))):
        x = left + k * 110
        p.append(f'<rect x="{x}" y="{ly - 9}" width="10" height="10" rx="2" style="fill:{col}"/>' + _text(x + 16, ly, lab.upper()))
    return Chart("changes", title, "Share of materially changed rows that got better, worse or neither.", "".join(p), W, H)


def coverage_harm(curves: Mapping[str, pd.DataFrame], *, chosen: Mapping[str, float] | None = None, harm_bound: float = 0.10,
                  title: str = "Coverage against harm") -> Chart:
    """E: ``curves`` maps a name to ``coverage_harm_curve(...)`` output; ``chosen`` maps a name to a chosen coverage."""
    chosen = chosen or {}
    W, H, box = 760, 360, (56, 14, 740, 312)
    y_hi = 0.6
    for c in curves.values():
        y_hi = max(y_hi, float(np.nanmax(c["harm"])) * 1.05 if len(c) else y_hi)
    y_hi = min(1.0, y_hi)
    sx, sy = _lin(0, 1, box[0], box[2]), _lin(0, y_hi, box[3], box[1])
    p = _axes(sx, sy, _ticks(0, 1, 5), _ticks(0, y_hi), box, lambda v: f"{v:.1f}")
    p += [_line(box[0], sy(harm_bound), box[2], sy(harm_bound), "stroke:var(--alert);stroke-dasharray:6 4"),
          _text(box[2], sy(harm_bound) + 14, f"harm bound {harm_bound:g}", anchor="end", style="fill:var(--alert)"),
          _text((box[0] + box[2]) / 2, H - 6, "coverage (share of rows the selector acts on)", "hg-axis", "middle")]
    for i, (name, c) in enumerate(curves.items()):
        col = SERIES[i % len(SERIES)]
        pts = c[["coverage", "harm"]].dropna().to_numpy()
        pts = pts[pts[:, 0] > 0.002]
        if not len(pts):
            continue
        p.append('<path d="' + "".join(f"{'L' if k else 'M'}{sx(a):.1f} {sy(min(b, y_hi)):.1f}" for k, (a, b) in enumerate(pts))
                 + f'" style="fill:none;stroke:{col};stroke-width:2"/>')
        base = c.attrs.get("material_base_rate")
        if base is not None:
            p.append(_line(box[0], sy(base), box[2], sy(base), f"stroke:{col};stroke-width:1;stroke-dasharray:1 4"))
        last = pts[-1]
        p.append(_text(sx(last[0]) - 4, sy(last[1]) + (18 if i % 2 else -10), name, "hg-lbl", "end", f"fill:{col}"))
        if name in chosen:
            k = int(np.argmin(np.abs(pts[:, 0] - chosen[name])))
            x, y = sx(pts[k, 0]), sy(pts[k, 1])
            p.append(f"<g><title>{_e(f'{name}: chosen coverage {chosen[name]:.3f}, harm {pts[k, 1]:.3f}' + (f'; random selection {base:.3f}' if base is not None else ''))}</title>"
                     f'<circle cx="{x:.1f}" cy="{y:.1f}" r="14" style="fill:transparent"/>'
                     f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" style="fill:{col};stroke:var(--surface);stroke-width:2"/></g>')
    return Chart("coverage_harm", title, "Lines: material harm as coverage grows. Dotted: random selection. Dot: the chosen threshold.",
                 "".join(p), W, H)


def harm_map(table: pd.DataFrame, *, scale_max: float = 0.5, title: str = "Where it fails",
             row_order: Sequence | None = None, column_order: Sequence | None = None) -> Chart:
    """F: heat map from ``analysis.harm_by_condition``. Darker = larger share of changed rows harmed."""
    rows = list(row_order) if row_order is not None else list(dict.fromkeys(table["row"]))
    cols = list(column_order) if column_order is not None else list(dict.fromkeys(table["column"]))
    W, left, top = 760, 140, 26
    cw = (W - left - 8) / max(len(cols), 1)
    ch = 64
    H = top + len(rows) * ch + 34
    p = [_text(left + i * cw + cw / 2, 16, str(c), anchor="middle") for i, c in enumerate(cols)]
    for r_i, rname in enumerate(rows):
        p.append(_text(0, top + r_i * ch + ch / 2 + 4, str(rname), "hg-lbl"))
        for i, cname in enumerate(cols):
            cell = table.loc[(table["row"] == rname) & (table["column"] == cname)]
            harm = float(cell["harm"].iloc[0]) if len(cell) else float("nan")
            n = int(cell["changed_rows"].iloc[0]) if len(cell) else 0
            a = 0.0 if not np.isfinite(harm) else min(1.0, harm / scale_max)
            reliable = bool(cell["reliable"].iloc[0]) if len(cell) else False
            x, y = left + i * cw + 2, top + r_i * ch + 2
            label = "n/a" if not np.isfinite(harm) else f"{harm:.2f}" + ("" if reliable else "*")
            p.append(f"<g><title>{_e(f'{rname}, {cname}: harm {_fmt(harm)} of {n:,} changed rows' + ('' if reliable else ' (few rows)'))}</title>"
                     f'<rect x="{x:.1f}" y="{y}" width="{cw - 4:.1f}" height="{ch - 4}" rx="2" style="fill:var(--alert);fill-opacity:{0.12 + 0.78 * a:.3f}"/>'
                     + _text(x + (cw - 4) / 2, y + (ch - 4) / 2 + 4, label, "hg-cell", "middle", "fill:var(--page)" if a > 0.55 else "") + "</g>")
    p.append(_text(left, H - 10, f"darker = larger share of changed rows harmed (scale 0 to {scale_max:g}); * fewer than the minimum rows", "hg-axis"))
    return Chart("harm_map", title, "Material harm on changed rows, split by condition.", "".join(p), W, H)


def monitor_chart(series: Mapping[str, pd.DataFrame], *, harm_bound: float = 0.10, title: str = "Period by period") -> Chart:
    """G: ``series`` maps a name to ``analysis.monitor(...)`` output. Ringed points are periods above the bound."""
    periods = list(dict.fromkeys(p for s in series.values() for p in s["period"]))
    y_hi = max([0.3] + [float(np.nanmax(s["harm"].astype(float))) * 1.15 for s in series.values() if s["harm"].notna().any()])
    y_hi = min(1.0, y_hi)
    W, H, box = 1000, 300, (52, 14, 820, 256)
    idx = {p: i for i, p in enumerate(periods)}
    sx = _lin(0, max(len(periods) - 1, 1), box[0] + 20, box[2] - 20)
    sy = _lin(0, y_hi, box[3], box[1])
    p = _axes(sx, sy, list(range(len(periods))), _ticks(0, y_hi, 4), box, lambda v: str(periods[int(v)]))
    p += [_line(box[0], sy(harm_bound), box[2], sy(harm_bound), "stroke:var(--alert);stroke-dasharray:6 4"),
          _text(box[0] + 4, sy(harm_bound) - 6, f"harm bound {harm_bound:g}", style="fill:var(--alert)")]
    ends = []
    for i, (name, s) in enumerate(series.items()):
        col = SERIES[i % len(SERIES)]
        pts = s.loc[s["harm"].notna()]
        if pts.empty:
            continue
        p.append('<path d="' + "".join(f"{'L' if k else 'M'}{sx(idx[r.period]):.1f} {sy(r.harm):.1f}" for k, r in enumerate(pts.itertuples()))
                 + f'" style="fill:none;stroke:{col};stroke-width:2"/>')
        for r in pts.itertuples():
            x, y = sx(idx[r.period]), sy(r.harm)
            ring = f'<circle cx="{x:.1f}" cy="{y:.1f}" r="9" style="fill:none;stroke:var(--alert);stroke-width:1.5"/>' if r.harm > harm_bound else ""
            upper = "" if r.harm_upper is None or not np.isfinite(r.harm_upper) else f" (upper {r.harm_upper:.3f})"
            p.append(f"<g><title>{_e(f'{name}, {r.period}: harm {r.harm:.3f}{upper} on {r.changed_rows:,} changed rows')}</title>{ring}"
                     f'<circle cx="{x:.1f}" cy="{y:.1f}" r="14" style="fill:transparent"/>'
                     f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" style="fill:{col};stroke:var(--surface);stroke-width:2"/></g>')
        last = pts.iloc[-1]
        ly = sy(last.harm) + 4
        while any(abs(ly - e) < 14 for e in ends):
            ly += 14
        ends.append(ly)
        p.append(_text(sx(idx[last.period]) + 12, ly, name, "hg-lbl", style=f"fill:{col}"))
    return Chart("monitor", title, "Material harm per period. Ringed points are periods above the safety bound.", "".join(p), W, H)


def leaderboard_table(results: Mapping[str, CMATResult], *, title: str = "Leaderboard") -> Chart:
    """H: ranked HTML table, accepted first, then by harm, with inline bars and the harm bound tick."""
    table = leaderboard(results)
    bound = next(iter(results.values())).settings["harm_bound"]
    max_d = max(float(table["nrmse_difference"].abs().max()), 1e-9)
    rows = []
    for r in table.itertuples():
        w = abs(r.nrmse_difference) / max_d * 50
        left = 50 if r.nrmse_difference >= 0 else 50 - w
        dcol = "var(--alert)" if r.nrmse_difference >= 0 else "var(--ok)"
        harm = "n/a" if r.harm is None or not np.isfinite(r.harm) else f"{r.harm:.3f}"
        hw = 0 if harm == "n/a" else min(r.harm, 1.0) * 100
        rows.append(f'<tr><td class="num">{r.rank}</td><td><b>{_e(r.candidate)}</b></td><td><span class="chip {r.decision}">{r.decision}</span></td>'
                    f'<td class="num">{_signed(r.nrmse_difference)}</td><td><div class="bar"><span style="left:{left:.1f}%;width:{w:.1f}%;background:{dcol}"></span>'
                    f'<span style="left:50%;width:1px;background:var(--muted)"></span></div></td>'
                    f'<td class="num">{harm}</td><td><div class="bar"><span style="left:0;width:{hw:.1f}%;background:{VERDICT[r.decision]}"></span>'
                    f'<i class="bound" style="left:{bound * 100:.1f}%"></i></div></td><td class="num">{r.coverage:.3f}</td></tr>')
    metric = next(iter(results.values())).settings.get('metric', 'NRMSE')
    body = ('<div class="hg-table"><table><thead><tr><th>#</th><th>Candidate</th><th>Verdict</th>' + f'<th>{_e(metric)} change</th><th></th>'
            '<th>Harm</th><th></th><th>Coverage</th></tr></thead><tbody>' + "".join(rows) + "</tbody></table></div>")
    return Chart("leaderboard", title, "Accepted first, then by material harm. Red tick: the harm bound.", body, kind="html")


def capacity(curve: pd.DataFrame, *, delta_mark: float | None = 0.10, title: str = "Capacity curves") -> Chart:
    """I (Paper A): two panels from ``capacity_curve(...)``: irreducible error and fixable share against delta."""
    W, H = 760, 300
    d = curve["delta"].to_numpy(float)
    d_hi = float(d.max()) if len(d) else 1.0
    panels = (("IRREDUCIBLE ERROR S", "irreducible_error", float(curve["irreducible_error"].max()) * 1.1 or 1.0, lambda v: f"{v:.3f}"),
              ("FIXABLE SHARE C", "capacity_coverage", 1.0, lambda v: f"{v:.1f}"))
    p = []
    for k, (label, col, y_hi, yf) in enumerate(panels):
        box = (60 + k * 370, 28, 40 + k * 370 + 320, 250)
        sx, sy = _lin(0, d_hi, box[0], box[2]), _lin(0, y_hi, box[3], box[1])
        p.append(_text(box[0], 14, label))
        p += _axes(sx, sy, _ticks(0, d_hi, 4), _ticks(0, y_hi, 4), box, lambda v: f"{v:.1f}", yf)
        if delta_mark is not None:
            p.append(_line(sx(delta_mark), box[1], sx(delta_mark), box[3], "stroke:var(--muted);stroke-dasharray:2 4"))
        values = curve[col].to_numpy(float)
        p.append('<path d="' + "".join(f"{'L' if i else 'M'}{sx(a):.1f} {sy(b):.1f}" for i, (a, b) in enumerate(zip(d, values)))
                 + '" style="fill:none;stroke:var(--accent);stroke-width:2"/>')
        for a, s_, c_ in zip(d, curve["irreducible_error"], curve["capacity_coverage"]):
            p.append(f"<g><title>{_e(f'delta {a:.2f}: irreducible error {s_:.4f}, fixable share {c_ * 100:.1f}%')}</title>"
                     f'<circle cx="{sx(a):.1f}" cy="{sy(s_ if k == 0 else c_):.1f}" r="6" style="fill:transparent"/></g>')
        p.append(_text((box[0] + box[2]) / 2, H - 6, "correction limit δ", "hg-axis", "middle"))
    return Chart("capacity", title, "Before training: the error floor of a bounded hybrid and the share of rows it can fix.", "".join(p), W, H)


def gap(distribution: pd.DataFrame, *, deltas: Sequence[float] = (0.10, 0.20, 0.40), max_gap: float = 0.8,
        title: str = "How far the physics is off") -> Chart:
    """J (Paper A): histogram from ``analysis.gap_distribution``; bars left of the first delta are fixable."""
    W, H, box = 760, 300, (60, 14, 740, 256)
    dist = distribution.loc[distribution["gap_start"] < max_gap]
    y_hi = float(dist["share"].max()) * 1.1 if len(dist) else 1.0
    sx, sy = _lin(0, max_gap, box[0], box[2]), _lin(0, y_hi, box[3], box[1])
    p = _axes(sx, sy, _ticks(0, max_gap, 4), _ticks(0, y_hi, 4), box, lambda v: f"{v:.1f}", lambda v: f"{v * 100:.0f}%")
    first = deltas[0] if deltas else 0.0
    for r in dist.itertuples():
        x0, x1 = sx(r.gap_start), sx(r.gap_end)
        fixable = r.gap_end <= first + 1e-9
        p.append(f"<g><title>{_e(f'gap {r.gap_start:.2f} to {r.gap_end:.2f}: {r.share * 100:.1f}% of rows' + (f', fixable with delta = {first:g}' if fixable else ''))}</title>"
                 f'<rect x="{x0 + .5:.1f}" y="{sy(r.share):.1f}" width="{max(x1 - x0 - 1, 1):.1f}" height="{sy(0) - sy(r.share):.1f}" '
                 f'style="fill:{"var(--ok)" if fixable else "var(--idle)"}"/></g>')
    for k, dlt in enumerate(deltas):
        p += [_line(sx(dlt), box[1], sx(dlt), box[3], "stroke:var(--accent);stroke-dasharray:3 3"),
              _text(sx(dlt) + 4, box[1] + 12, (f"δ = {dlt:g}" if k == 0 else f"{dlt:g}"), style="fill:var(--accent-text)")]
    p.append(_text((box[0] + box[2]) / 2, H - 6, "gap between physics prior and measured value |y − p|", "hg-axis", "middle"))
    return Chart("gap", title, "Before training: rows left of δ can be fixed by a bounded hybrid; rows to the right cannot.", "".join(p), W, H)
