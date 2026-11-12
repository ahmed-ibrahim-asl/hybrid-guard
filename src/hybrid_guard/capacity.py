"""Correction-capacity diagnostics for bounded hybrid models (before training).

A bounded additive hybrid predicts ``clip(p + r, lo, hi)`` with ``|r| <= delta``. For any
admissible correction the squared error on a row is at least

    B_delta(y, p) = max(|y - p| - delta, 0) ** 2,

because the reachable predictions form an interval of half-width ``delta`` around the prior
``p``. Two training-only summaries follow:

    S_delta = mean(B_delta)          irreducible error floor of the architecture
    C_delta = mean(|y - p| <= delta) share of rows whose prior gap is fully correctable

Both use training targets and the physical prior only, so they can be computed before any
hybrid is trained and without touching validation or test labels. A high ``S_delta`` proves
that the stated correction limit cannot fit the prior; a low value says correction is
feasible, not that training will succeed.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def _arrays(y, prior) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float)
    prior = np.asarray(prior, dtype=float)
    if y.shape != prior.shape:
        raise ValueError("target and prior must have the same shape")
    keep = np.isfinite(y) & np.isfinite(prior)
    if not keep.any():
        raise ValueError("no finite target-prior pairs")
    return y[keep], prior[keep]


def irreducible_error(y, prior, delta: float) -> np.ndarray:
    """Row-wise lower bound ``max(|y - p| - delta, 0)**2`` on the squared error."""
    if delta < 0:
        raise ValueError("delta must be non-negative")
    y, prior = _arrays(y, prior)
    return np.maximum(np.abs(y - prior) - float(delta), 0.0) ** 2


def capacity_diagnostics(y, prior, delta: float) -> dict[str, float]:
    """Return ``S_delta`` (irreducible error), ``C_delta`` (coverage) and the prior's own MSE."""
    y, prior = _arrays(y, prior)
    gap = np.abs(y - prior)
    return {
        "delta": float(delta),
        "irreducible_error": float(np.mean(np.maximum(gap - delta, 0.0) ** 2)),
        "capacity_coverage": float(np.mean(gap <= delta)),
        "prior_mse": float(np.mean(gap**2)),
        "rows": int(len(y)),
    }


def capacity_curve(y, prior, deltas: Sequence[float] | None = None) -> pd.DataFrame:
    """``S_delta`` and ``C_delta`` over a grid of correction limits (default 0 to 1 in 0.01 steps)."""
    grid = np.round(np.arange(0.0, 1.0001, 0.01), 10) if deltas is None else np.asarray(deltas, dtype=float)
    return pd.DataFrame([capacity_diagnostics(y, prior, d) for d in grid])


def smallest_sufficient_delta(y, prior, max_irreducible_error: float, deltas: Sequence[float] | None = None) -> float | None:
    """Smallest correction limit whose irreducible error is at most ``max_irreducible_error``."""
    curve = capacity_curve(y, prior, deltas)
    ok = curve.loc[curve["irreducible_error"] <= max_irreducible_error, "delta"]
    return None if ok.empty else float(ok.min())
