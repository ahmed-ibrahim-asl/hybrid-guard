import numpy as np
import pandas as pd

from hybrid_guard import ABSTAIN, ACCEPT, REJECT, charts, html_report
from hybrid_guard.classification import cmat_classification


def _cases(n=20_000, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    difficulty = rng.uniform(0, 1, n)
    small = np.where(rng.random(n) < 0.95 - 0.5 * difficulty, y, 1 - y)   # cheap model, worse on hard cases
    expert = np.where(rng.random(n) < 0.97, y, 1 - y)                      # expert, good everywhere
    groups = np.arange(n) // 200
    return y, difficulty, small, expert, groups


def test_deferring_hard_cases_to_an_expert_is_accepted():
    y, difficulty, small, expert, groups = _cases()
    result = cmat_classification(y, small, expert, difficulty > 0.6, groups=groups, replicates=500)
    assert result.decision == ACCEPT and result.nrmse_difference < 0
    assert result.settings["metric"] == "error rate"


def test_deferring_to_a_bad_reviewer_is_rejected_and_random_deferral_lacks_skill():
    y, difficulty, small, _, groups = _cases()
    rng = np.random.default_rng(3)
    bad = np.where(rng.random(len(y)) < 0.6, y, 1 - y)
    assert cmat_classification(y, small, bad, difficulty > 0.6, groups=groups, replicates=500).decision == REJECT
    random_sel = rng.random(len(y)) < 0.4
    noisy = np.where(rng.random(len(y)) < 0.80, y, 1 - y)
    result = cmat_classification(y, small, noisy, random_sel, groups=groups, replicates=500)
    assert result.decision == REJECT


def test_identical_labels_abstain_and_report_renders():
    y, _, small, expert, groups = _cases(n=5000)
    assert cmat_classification(y, small, small.copy(), groups=groups, replicates=200).decision == ABSTAIN
    results = {"defer hard cases": cmat_classification(y, small, expert, np.arange(len(y)) % 3 == 0, groups=groups, replicates=200)}
    page = html_report(results, charts=["accuracy_safety", "conditions", "changes", "leaderboard"])
    assert "error rate change" in page
    assert charts.accuracy_safety(results).svg().startswith("<svg")
