"""Build the documentation site (docs/index.html) for GitHub Pages.

Usage, from the repository root:  python docs/build_site.py
Charts are generated from examples/data/hill_of_towie_2025_sample.csv.gz with the package itself.
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import hybrid_guard as hg  # noqa: E402
from hybrid_guard import charts  # noqa: E402
from hybrid_guard.report import CSS, MARK, TABLE_CSS  # noqa: E402

REPO = "https://github.com/ahmed-ibrahim-asl/hybrid-guard"
COLAB = "https://colab.research.google.com/github/ahmed-ibrahim-asl/hybrid-guard/blob/main/examples/"


def code(text: str) -> str:
    return f'<pre class="code"><code>{html.escape(text.strip())}</code></pre>'


def figure(chart, caption: str) -> str:
    inner = chart.body if chart.kind == "html" else f'<div class="chart-wrap">{chart.svg(standalone=False)}</div>'
    return f'<figure class="fig">{inner}<figcaption>{html.escape(caption)}</figcaption></figure>'


def main() -> int:
    df = pd.read_csv(ROOT / "examples" / "data" / "hill_of_towie_2025_sample.csv.gz", parse_dates=["timestamp"])
    meta = json.loads((ROOT / "examples" / "data" / "hill_of_towie_2025_sample.json").read_text(encoding="utf-8"))
    y, t = df["power"].to_numpy(), df["timestamp"]
    transfer, wake = df["data_model_transfer"].to_numpy(), df["physics_wake_mean"].to_numpy()
    selected = df["harm_score"].to_numpy() <= meta["harm_threshold"]
    results = {
        "Retrained local model": hg.cmat(y, df["data_model_old"], df["data_model_local"], timestamps=t),
        "Physics-features hybrid": hg.cmat(y, df["data_model_local"], df["hybrid_physics_features"], timestamps=t),
        "Harm classifier": hg.cmat(y, transfer, wake, selected, timestamps=t),
        "Random selection": hg.random_matched(y, transfer, wake, float(selected.mean()), timestamps=t),
    }
    prior = df["physics_no_wake"].to_numpy()
    table = hg.harm_by_condition(y, df["data_model_local"], df["hybrid_physics_features"],
                                 rows=np.where(df["waked"] == 1, "waked", "free stream"),
                                 columns=hg.bands(df["wind_speed"], [3, 5, 8, 11, 14, 25], " m/s"))
    figs = {
        "accuracy": figure(charts.accuracy_safety(results), "A. Accuracy against row-level safety."),
        "conditions": figure(charts.conditions(results), "B. Every condition for every candidate."),
        "changes": figure(charts.changes(results), "D. What happened to the rows each candidate changed."),
        "coverage": figure(charts.coverage_harm({"Harm classifier": hg.coverage_harm_curve(y, transfer, wake, df["harm_score"])},
                                                chosen={"Harm classifier": float(selected.mean())}), "E. Harm as a selector covers more rows."),
        "map": figure(charts.harm_map(table, row_order=["free stream", "waked"],
                                      column_order=["3-5 m/s", "5-8 m/s", "8-11 m/s", "11-14 m/s", "14-25 m/s"]), "F. Where a hybrid fails."),
        "monitor": figure(charts.monitor_chart({"Harm classifier": hg.monitor(y, transfer, wake, selected, timestamps=t)}),
                          "G. Month-by-month monitoring after deployment."),
        "leader": figure(charts.leaderboard_table(results), "H. Leaderboard."),
        "capacity": figure(charts.capacity(hg.capacity_curve(y, prior, np.round(np.arange(0, 0.82, 0.02), 2))), "I. Capacity curves before training."),
        "gap": figure(charts.gap(hg.gap_distribution(y, prior)), "J. How far the physics prior is off."),
    }
    api = [
        ("cmat(y, incumbent, candidate, selected=None, timestamps=...)", "Coverage-matched acceptance test for regression. Returns a result with decision, reasons and statistics."),
        ("cmat_classification(y, incumbent, candidate, selected=None, groups=...)", "The same test for labels: cascades, fallbacks and deferral to a human."),
        ("random_matched(y, incumbent, candidate, coverage)", "The comparator every selector must beat: random selection at the same coverage."),
        ("coverage_dilution(y, incumbent, candidate, selected)", "The identity behind the problem: MSE change = coverage x mean excess on selected rows."),
        ("capacity_diagnostics(y, prior, delta) / capacity_curve(y, prior)", "Before training: irreducible error and fixable share of a bounded hybrid."),
        ("coverage_harm_curve(...) / choose_threshold(...)", "Harm along a selector's score; a threshold with a safety margin below the bound."),
        ("harm_by_condition(..., rows=, columns=) / bands(values, edges)", "Harm split by conditions such as wind speed or wake."),
        ("monitor(..., timestamps=, period='M')", "CMAT per month (or week, quarter) to catch drift after deployment."),
        ("condition_table / change_breakdown / leaderboard / gap_distribution", "The tables behind every chart, as pandas DataFrames."),
        ("charts.*", "Nine charts; each returns a Chart with .svg(), .html() and .save('fig.svg')."),
        ("html_report(results, charts=[...], extra=[...], layout=...)", "A self-contained report in light and dark themes."),
    ]
    api_rows = "".join(f"<tr><td><code>{html.escape(a)}</code></td><td>{html.escape(b)}</td></tr>" for a, b in api)
    site_css = """
.hero{display:grid;gap:18px;padding-block:8px 6px}
.hero .lede{font-size:clamp(18px,2.2vw,22px);max-width:60ch}
.pills{display:flex;flex-wrap:wrap;gap:10px}
.pill{display:inline-flex;align-items:center;gap:8px;padding:9px 14px;border:1px solid var(--hair);border-radius:4px;background:var(--surface);
  color:var(--ink);text-decoration:none;font:400 12px/1 var(--font-mono);letter-spacing:.12em;text-transform:uppercase;box-shadow:var(--inset-top)}
.pill.primary{border-color:var(--accent);color:var(--accent-text)}
.pill:hover{background:var(--raised)}
.two{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}
.panel{background:var(--surface);border:1px solid var(--hair);border-radius:4px;padding:18px;display:grid;gap:10px;box-shadow:var(--inset-top);min-width:0}
.panel h3{font:600 18px/1.25 var(--font-sans);margin:0}
.code{background:var(--page);border:1px solid var(--hair);border-radius:4px;padding:14px;overflow-x:auto;font:400 13px/1.55 var(--font-mono);margin:0}
.figs{display:grid;grid-template-columns:minmax(0,1fr);gap:22px}
.fig{margin:0;display:grid;gap:8px;min-width:0}
.fig.wide{grid-column:1/-1}
figcaption{font-size:13px;color:var(--muted)}
.api{width:100%;border-collapse:collapse;font-size:14px;min-width:620px}
.api td{border-bottom:1px solid var(--hair);padding:10px 8px;vertical-align:top}
.api td:first-child{width:46%}
.api code{font:400 12.5px var(--font-mono);color:var(--accent-text)}
.scroll{overflow-x:auto}
a{color:var(--accent-text)}
@media (max-width:860px){.two,.figs{grid-template-columns:minmax(0,1fr)}}
"""
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>hybrid-guard</title>
<meta name="description" content="Check physics-ML hybrid models before training and before deployment.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;600;700&amp;family=Space+Mono:wght@400;700&amp;display=swap">
<style>{CSS}{charts.CHART_CSS}{TABLE_CSS}{site_css}</style></head>
<body><div class="page">
<header>
  <div class="brandline">{MARK}<div class="eyebrow">hybrid-guard &middot; v{hg.__version__} &middot; MIT</div></div>
  <div class="hero">
    <h1>Good on average is not the same as safe.</h1>
    <p class="lede">hybrid-guard checks models that mix physics with machine learning, or switch between models, at two moments: before you train them, and before you trust them.</p>
    <div class="pills"><a class="pill primary" href="{REPO}">Code on GitHub</a><a class="pill" href="{COLAB}01_wind_quickstart.ipynb">Wind tutorial in Colab</a>
    <a class="pill" href="{COLAB}02_beyond_wind_expert_handoff.ipynb">Expert hand-off tutorial</a><a class="pill" href="#cite">How to cite</a></div>
  </div>
</header>

<section class="two">
  <div class="panel"><div class="eyebrow">Before training</div><h3>Can a bounded hybrid fix the physics?</h3>
    <p>If the physics prior is off by more than the correction your hybrid allows, no training can fix those rows. The capacity diagnostics measure this from training data alone.</p></div>
  <div class="panel"><div class="eyebrow">Before deployment</div><h3>Is the new model safe row by row?</h3>
    <p>A model that changes a few percent of predictions can pass any small error margin while harming most of the rows it touches. The coverage-matched acceptance test (CMAT) checks the rows that changed, and compares with random selection.</p></div>
</section>

<section><div class="section-head"><h2>Install and run</h2></div>
{code('pip install hybrid-guard')}
{code('''from hybrid_guard import cmat, html_report

result = cmat(y, incumbent=data_model, candidate=hybrid, timestamps=times)
print(result.decision)        # ACCEPT, REJECT or ABSTAIN
print(result.summary())       # every condition in plain words

html_report({"hybrid": result}, title="Acceptance report")''')}
<p class="note">Command line: <code>hybrid-guard check predictions.csv --target y --incumbent base --candidate hybrid --time timestamp --html report.html</code></p>
</section>

<section><div class="section-head"><h2>What it shows</h2></div>
<p class="note">Real data: three Siemens turbines at Hill of Towie, sealed year 2025 (Hill of Towie open dataset, RES on behalf of TRIG, CC BY 4.0). Hover any mark for its values.</p>
<div class="figs">
<div class="fig wide">{figs['accuracy']}</div>{figs['conditions']}{figs['changes']}{figs['coverage']}{figs['map']}
<div class="fig wide">{figs['monitor']}</div><div class="fig wide">{figs['leader']}</div>{figs['capacity']}{figs['gap']}
</div></section>

<section><div class="section-head"><h2>Reference</h2></div>
<div class="scroll"><table class="api"><tbody>{api_rows}</tbody></table></div></section>

<section id="cite"><div class="section-head"><h2>How to cite</h2></div>
<div class="two">
<div class="panel"><div class="eyebrow">Capacity diagnostics</div><p>A. I. Asl, B. M. El-Den. Diagnosing physics-data mismatch in hybrid learning through correction capacity: a training-only framework for wind power forecasting. Manuscript under review, 2026.</p></div>
<div class="panel"><div class="eyebrow">CMAT and the coverage-dilution identity</div><p>A. I. Asl, B. M. El-Den. Aggregate error margins cannot certify physics-informed wind-power estimators: a coverage-matched acceptance test with sealed multi-farm SCADA evidence. Manuscript submitted, 2026.</p></div>
</div></section>

<footer><div>hybrid-guard is MIT licensed. Built by Ahmed Ibrahim Asl and B. M. El-Den. <a href="{REPO}">Source</a> &middot; <a href="https://eng-asl.com">eng-asl.com</a></div></footer>
</div></body></html>
"""
    out = ROOT / "docs" / "index.html"
    out.write_text(page, encoding="utf-8")
    (ROOT / "docs" / ".nojekyll").write_text("", encoding="utf-8")
    print(out, len(page))
    return 0


if __name__ == "__main__":
    sys.exit(main())
