import numpy as np
import pandas as pd

from hybrid_guard import choose_threshold, cmat, coverage_harm_curve, html_report
from hybrid_guard.cli import main as cli_main


def _setup(n=6 * 24 * 60, seed=0):
    rng = np.random.default_rng(seed)
    times = pd.date_range("2025-01-01", periods=n, freq="10min")
    y = rng.uniform(0, 1, n)
    f = y + rng.normal(0, 0.06, n)
    noise = rng.normal(0, 0.15, n)
    h = y + noise
    score = np.abs(noise) + rng.normal(0, 0.02, n)  # informative score: low where the candidate is good
    return times, y, f, h, score


def test_coverage_harm_curve_rises_with_coverage_for_an_informative_score():
    _, y, f, h, score = _setup()
    curve = coverage_harm_curve(y, f, h, score, points=20)
    assert curve["coverage"].is_monotonic_increasing
    harm = curve["harm"].dropna().to_numpy()
    assert harm[-1] > harm[len(harm) // 4]
    assert 0 < curve.attrs["material_base_rate"] < 1


def test_choose_threshold_respects_the_safety_margin():
    times, y, f, _, _ = _setup()
    rng = np.random.default_rng(5)
    good = rng.random(len(y)) < 0.3
    h = np.where(good, y, y + rng.normal(0, 0.3, len(y)))  # exact on good rows, poor elsewhere
    score = np.where(good, 0.0, 1.0) + rng.normal(0, 0.1, len(y))
    chosen = choose_threshold(y, f, h, score, timestamps=times, candidates=12, replicates=300)
    assert chosen["threshold"] is not None
    assert chosen["result"].harm_upper <= 0.07 + 1e-12


def test_html_report_contains_every_candidate_and_both_themes(tmp_path):
    times, y, f, h, score = _setup(n=6 * 24 * 30)
    results = {"blend": cmat(y, f, 0.5 * (f + h), timestamps=times, replicates=200),
               "router": cmat(y, f, h, score < 0.02, timestamps=times, replicates=200),
               "none": cmat(y, f, f.copy(), timestamps=times, replicates=200)}
    page = html_report(results, title="Test report")
    for name in results:
        assert name in page
    assert "prefers-color-scheme: light" in page and '[data-theme="light"]' in page and "<svg" in page
    assert html_report(results, standalone=False).startswith("<title>")


def test_cli_writes_report(tmp_path):
    times, y, f, h, score = _setup(n=6 * 24 * 20)
    csv = tmp_path / "preds.csv"
    pd.DataFrame({"time": times, "y": y, "base": f, "hyb": 0.7 * f + 0.3 * h, "sel": score < 0.03}).to_csv(csv, index=False)
    out = tmp_path / "report.html"
    code = cli_main(["check", str(csv), "--target", "y", "--incumbent", "base", "--candidate", "blend=hyb",
                     "--candidate", "router=hyb", "--selected", "router=sel", "--time", "time", "--replicates", "200",
                     "--html", str(out)])
    assert code == 0 and out.exists() and "blend" in out.read_text(encoding="utf-8")
