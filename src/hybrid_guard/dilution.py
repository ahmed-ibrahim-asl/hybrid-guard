"""The coverage-dilution identity: why an aggregate margin cannot certify selective substitution.

If a candidate prediction replaces an incumbent only on selected rows (coverage ``C``), the
change in mean squared error is exactly

    Delta = MSE_routed - MSE_incumbent = C * mean excess loss on selected rows.

An NRMSE noninferiority margin ``m`` therefore passes whenever the mean excess on selected
rows is below ``(2 m NRMSE_incumbent + m**2) / C``, a bound that grows as coverage shrinks and
says nothing about how many selected rows were harmed. A random selector at the same coverage
has expected excess equal to the all-row mean, so it passes too when coverage is small.
"""

from __future__ import annotations

import numpy as np


def coverage_dilution(y, incumbent, candidate, selected) -> dict[str, float]:
    """Coverage, mean excess on selected rows and the MSE change, with the identity checked."""
    y = np.asarray(y, float)
    f = np.asarray(incumbent, float)
    h = np.asarray(candidate, float)
    sel = np.asarray(selected, bool)
    excess = (h - y) ** 2 - (f - y) ** 2
    routed = np.where(sel, h, f)
    delta = float(np.mean((routed - y) ** 2) - np.mean((f - y) ** 2))
    coverage = float(sel.mean())
    mean_excess = float(excess[sel].mean()) if sel.any() else 0.0
    return {"coverage": coverage, "mean_excess_selected": mean_excess, "delta_mse": delta,
            "identity_residual": delta - coverage * mean_excess,
            "nrmse_incumbent": float(np.sqrt(np.mean((f - y) ** 2))),
            "nrmse_routed": float(np.sqrt(np.mean((routed - y) ** 2)))}


def admissible_mean_excess(nrmse_incumbent: float, coverage: float, margin: float = 0.002) -> float:
    """Largest mean excess on selected rows that still passes an NRMSE margin at this coverage."""
    if coverage <= 0:
        return float("inf")
    return (2.0 * margin * nrmse_incumbent + margin**2) / coverage


def random_router_passes(y, incumbent, candidate, coverage: float, margin: float = 0.002) -> bool:
    """Whether random substitution at ``coverage`` passes the margin in expectation."""
    y = np.asarray(y, float)
    f = np.asarray(incumbent, float)
    h = np.asarray(candidate, float)
    mean_excess_all = float(np.mean((h - y) ** 2 - (f - y) ** 2))
    return mean_excess_all < admissible_mean_excess(float(np.sqrt(np.mean((f - y) ** 2))), coverage, margin)
