import numpy as np
import pandas as pd
import pytest

from hybrid_guard import (bands, capacity_curve, change_breakdown, charts, cmat, condition_table, coverage_harm_curve,
                          gap_distribution, harm_by_condition, html_report, leaderboard, monitor)


@pytest.fixture(scope="module")
def setup():
    rng = np.random.default_rng(0)
    n = 6 * 24 * 120
    times = pd.date_range("2025-01-01", periods=n, freq="10min")
    y = rng.uniform(0, 1, n)
    f = y + rng.normal(0, 0.06, n)
    noise = rng.normal(0, 0.15, n)
    h = y + noise
    score = np.abs(noise) + rng.normal(0, 0.03, n)
    ws = rng.uniform(3, 20, n)
    results = {"blend": cmat(y, f, 0.6 * f + 0.4 * h, timestamps=times, replicates=200),
               "router": cmat(y, f, h, score < 0.03, timestamps=times, replicates=200),
               "unchanged": cmat(y, f, f.copy(), timestamps=times, replicates=200)}
    return dict(times=times, y=y, f=f, h=h, score=score, ws=ws, results=results)


def test_tables_have_documented_columns(setup):
    r = setup["results"]
    assert set(condition_table(r).columns) == {"candidate", "decision", "condition", "state", "value", "requirement"}
    assert len(condition_table(r)) == 15
    br = change_breakdown(r)
    ok = br.dropna()
    assert np.allclose(ok[["improved", "neither", "harmed"]].sum(axis=1), 1.0)
    lb = leaderboard(r)
    assert list(lb["rank"]) == [1, 2, 3] and lb.iloc[-1]["decision"] in ("REJECT", "ABSTAIN")


def test_harm_by_condition_and_bands(setup):
    s = setup
    labels = bands(s["ws"], [3, 8, 14, 25], " m/s")
    assert set(labels) <= {"3-8 m/s", "8-14 m/s", "14-25 m/s", ""}
    table = harm_by_condition(s["y"], s["f"], s["h"], rows=labels, columns=np.where(s["ws"] > 10, "high", "low"))
    assert {"row", "column", "changed_rows", "harm", "benefit", "reliable"} <= set(table.columns)
    assert (table["harm"].dropna().between(0, 1)).all()


def test_monitor_returns_one_row_per_month(setup):
    s = setup
    m = monitor(s["y"], s["f"], s["h"], s["score"] < 0.03, timestamps=s["times"], replicates=100)
    assert len(m) == 4 and {"period", "harm", "harm_upper", "decision", "above_bound"} <= set(m.columns)


def test_every_chart_renders_svg_or_html(setup, tmp_path):
    s = setup
    r = setup["results"]
    curve = coverage_harm_curve(s["y"], s["f"], s["h"], s["score"], points=20)
    built = [
        charts.accuracy_safety(r), charts.conditions(r), charts.changes(r), charts.leaderboard_table(r),
        charts.coverage_harm({"router": curve}, chosen={"router": 0.1}),
        charts.harm_map(harm_by_condition(s["y"], s["f"], s["h"], rows=bands(s["ws"], [3, 8, 14, 25]))),
        charts.monitor_chart({"router": monitor(s["y"], s["f"], s["h"], s["score"] < 0.03, timestamps=s["times"], replicates=100)}),
        charts.capacity(capacity_curve(s["y"], s["f"], np.arange(0, 0.42, 0.02))),
        charts.gap(gap_distribution(s["y"], s["f"])),
    ]
    for chart in built:
        if chart.kind == "svg":
            text = chart.svg()
            assert text.startswith("<svg") and "</svg>" in text and "<title>" in text or chart.key == "accuracy_safety"
            path = chart.save(tmp_path / f"{chart.key}.svg", theme="light")
            assert path.read_text(encoding="utf-8").startswith('<svg xmlns="http://www.w3.org/2000/svg"')
        page = chart.save(tmp_path / f"{chart.key}.html")
        assert "<!doctype html>" in page.read_text(encoding="utf-8")


def test_report_chart_selection_and_layouts(setup):
    s = setup
    r = setup["results"]
    extra = [charts.gap(gap_distribution(s["y"], s["f"]))]
    page = html_report(r, charts=["conditions", "changes"], extra=extra, layout="leaderboard")
    assert page.index('id="chart-leaderboard"') < page.index('id="chart-conditions"') < page.index('id="chart-gap"')
    assert 'id="chart-accuracy_safety"' not in page
    with pytest.raises(ValueError):
        html_report(r, charts=["nonexistent"])
