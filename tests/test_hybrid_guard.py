import numpy as np
import pandas as pd
import pytest

from hybrid_guard import (ABSTAIN, ACCEPT, REJECT, admissible_mean_excess, capacity_curve, capacity_diagnostics,
                          cmat, coverage_dilution, irreducible_error, random_matched, random_router_passes,
                          smallest_sufficient_delta)


def _data(n=24 * 6 * 90, seed=0):
    rng = np.random.default_rng(seed)
    times = pd.date_range("2025-01-01", periods=n, freq="10min")
    y = rng.uniform(0.0, 1.0, n)
    incumbent = y + rng.normal(0.0, 0.08, n)
    return times, y, incumbent, rng


def test_irreducible_error_is_the_projection_bound():
    y = np.array([0.0, 0.5, 1.0])
    p = np.array([0.3, 0.5, 0.6])
    assert np.allclose(irreducible_error(y, p, 0.1), [0.04, 0.0, 0.09])
    d = capacity_diagnostics(y, p, 0.1)
    assert d["capacity_coverage"] == pytest.approx(1 / 3)


def test_capacity_curve_is_monotone_and_finds_sufficient_delta():
    rng = np.random.default_rng(1)
    y = rng.uniform(0, 1, 5000)
    p = y + rng.normal(0, 0.2, 5000)
    curve = capacity_curve(y, p)
    assert np.all(np.diff(curve["irreducible_error"]) <= 1e-12)
    assert np.all(np.diff(curve["capacity_coverage"]) >= -1e-12)
    delta = smallest_sufficient_delta(y, p, 1e-4)
    assert delta is not None and 0.3 < delta < 1.0


def test_coverage_dilution_identity_holds_exactly():
    _, y, f, rng = _data()
    h = y + rng.normal(0.0, 0.2, len(y))
    selected = rng.random(len(y)) < 0.03
    d = coverage_dilution(y, f, h, selected)
    assert abs(d["identity_residual"]) < 1e-12
    assert admissible_mean_excess(0.08, 0.03) > admissible_mean_excess(0.08, 0.3)


def test_random_router_passes_margin_at_low_coverage():
    _, y, f, rng = _data()
    h = y + rng.normal(0.0, 0.10, len(y))
    assert random_router_passes(y, f, h, coverage=0.01)
    assert not random_router_passes(y, f, h, coverage=0.9)


def test_identical_candidate_abstains_and_clear_improvement_is_accepted():
    times, y, f, _ = _data()
    assert cmat(y, f, f.copy(), timestamps=times, replicates=300).decision == ABSTAIN
    better = cmat(y, f, y + 0.2 * (f - y), timestamps=times, replicates=300)
    assert better.decision == ACCEPT and better.harm < 0.05
    worse = cmat(y, f, y + 3.0 * (f - y), timestamps=times, replicates=300)
    assert worse.decision == REJECT and any("safety failed" in r for r in worse.reasons)


def test_random_selection_is_rejected_for_lack_of_skill():
    times, y, f, rng = _data(seed=3)
    h = y + rng.normal(0.0, 0.08, len(y))
    result = random_matched(y, f, h, 0.05, timestamps=times, replicates=500)
    assert result.decision == REJECT and any("skill failed" in r for r in result.reasons)


def test_summary_and_groups_and_iid_warning():
    times, y, f, _ = _data(n=2000)
    result = cmat(y, f, y + 0.5 * (f - y), groups=np.arange(2000) // 144, replicates=200)
    assert "CMAT decision" in result.summary()
    with pytest.warns(UserWarning):
        cmat(y, f, y + 0.5 * (f - y), replicates=100)
