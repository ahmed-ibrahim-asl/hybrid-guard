"""Choosing a selection threshold: the coverage-harm curve and a safety-margin threshold rule.

Scores are "lower is safer": a row is selected when its score is at or below the threshold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hybrid_guard.cmat import ACCEPT, cmat


def coverage_harm_curve(y, incumbent, candidate, score, *, materiality: float = 0.01, points: int = 50) -> pd.DataFrame:
    """Point estimates of coverage, material harm, benefit and NRMSE change along the score.

    Cheap (no bootstrap); use it to see the trade-off, then confirm a threshold with ``cmat``.
    """
    y = np.asarray(y, float)
    f = np.asarray(incumbent, float)
    h = np.asarray(candidate, float)
    s = np.asarray(score, float)
    ok = np.isfinite(s)
    order = np.argsort(np.where(ok, s, np.inf), kind="mergesort")
    gain = np.abs(f - y) - np.abs(h - y)
    changed = (np.abs(h - f) > materiality)[order]
    harm = (changed & (gain[order] < -materiality)).cumsum()
    benefit = (changed & (gain[order] > materiality)).cumsum()
    n_changed = changed.cumsum()
    excess = ((h - y) ** 2 - (f - y) ** 2)[order].cumsum()
    base_sq = float(np.mean((f - y) ** 2))
    n, finite = len(y), int(ok.sum())
    rows = []
    for k in np.unique(np.linspace(1, finite, num=min(points, finite)).astype(int)):
        i = k - 1
        c = n_changed[i]
        rows.append({"threshold": float(s[order][i]), "selected": int(k), "coverage": k / n, "changed": int(c),
                     "harm": float(harm[i] / c) if c else np.nan, "benefit": float(benefit[i] / c) if c else np.nan,
                     "nrmse_difference": float(np.sqrt(base_sq + excess[i] / n) - np.sqrt(base_sq))})
    out = pd.DataFrame(rows)
    out.attrs["material_base_rate"] = float(np.mean((np.abs(h - f) > materiality) & (-gain > materiality)))
    return out


def choose_threshold(y, incumbent, candidate, score, *, harm_bound: float = 0.10, safety_margin: float = 0.03,
                     candidates: int = 25, replicates: int = 1000, **cmat_kwargs) -> dict:
    """Largest-coverage threshold that CMAT accepts with harm upper bound at most ``harm_bound - safety_margin``.

    Thresholds tuned right at the bound tend to drift above it on new data; the margin keeps a
    buffer. Run this on calibration data and confirm the chosen threshold on later data.
    """
    s = np.asarray(score, float)
    finite = s[np.isfinite(s)]
    if not len(finite):
        return {"threshold": None, "result": None, "reason": "no finite scores"}
    target = harm_bound - safety_margin
    best = None
    for threshold in np.unique(np.quantile(finite, np.linspace(0.01, 1.0, candidates))):
        selected = np.isfinite(s) & (s <= threshold)
        result = cmat(y, incumbent, candidate, selected, selective=True, harm_bound=harm_bound, replicates=replicates, **cmat_kwargs)
        if result.decision == ACCEPT and result.harm_upper is not None and result.harm_upper <= target:
            if best is None or result.coverage > best[1].coverage:
                best = (float(threshold), result)
    if best is None:
        return {"threshold": None, "result": None, "reason": f"no threshold accepted with harm upper bound at most {target:.3f}"}
    return {"threshold": best[0], "result": best[1], "reason": f"largest accepted coverage with harm upper bound at most {target:.3f}"}
