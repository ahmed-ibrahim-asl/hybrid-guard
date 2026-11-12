"""Matplotlib figures (optional dependency: ``pip install hybrid-guard[plot]``)."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from hybrid_guard.cmat import CMATResult

VERDICT_COLORS = {"ACCEPT": "#1f8a5b", "REJECT": "#c8453b", "ABSTAIN": "#b7832f"}


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover
        raise ImportError("plots need matplotlib: pip install hybrid-guard[plot]") from error
    return plt


def plot_accuracy_safety(results: Mapping[str, CMATResult], ax=None, *, annotate: bool = True):
    """NRMSE change (x, with 95% interval) against material harm (y, with upper bound) per candidate."""
    plt = _plt()
    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 4.2))
    first = next(iter(results.values()))
    margin, bound = first.settings["margin"], first.settings["harm_bound"]
    ax.axhspan(0, bound, xmin=0, xmax=1, color="#1f8a5b", alpha=0.06, zorder=0)
    ax.axhline(bound, color="#c8453b", linestyle="--", linewidth=1)
    ax.axvline(0, color="#888888", linewidth=0.8)
    ax.axvline(margin, color="#888888", linewidth=0.8, linestyle=":")
    for name, r in results.items():
        if r.harm is None:
            continue
        color = VERDICT_COLORS[r.decision]
        ax.errorbar(r.nrmse_difference, r.harm, xerr=[[r.nrmse_difference - r.nrmse_lower], [r.nrmse_upper - r.nrmse_difference]],
                    yerr=[[0], [max((r.harm_upper or r.harm) - r.harm, 0)]], fmt="o", color=color, ecolor=color,
                    elinewidth=1, capsize=2, markersize=6, markeredgecolor="white", zorder=3)
        if annotate:
            ax.annotate(name, (r.nrmse_difference, r.harm), textcoords="offset points", xytext=(5, 4), fontsize=7)
    ax.set_xlabel("NRMSE change against the incumbent (negative = more accurate)")
    ax.set_ylabel("material harm on changed rows")
    ax.set_ylim(bottom=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return ax


def plot_coverage_harm(curve, ax=None, *, harm_bound: float = 0.10):
    """Material harm against coverage from ``coverage_harm_curve``, with the base rate and bound."""
    plt = _plt()
    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(curve["coverage"], curve["harm"], color="#21606e", linewidth=2)
    base = curve.attrs.get("material_base_rate")
    if base is not None:
        ax.axhline(base, color="#888888", linestyle=":", linewidth=1)
        ax.annotate("random selection", (curve["coverage"].max(), base), xytext=(-4, 4), textcoords="offset points",
                    ha="right", fontsize=7, color="#666666")
    ax.axhline(harm_bound, color="#c8453b", linestyle="--", linewidth=1)
    ax.set_xlabel("coverage (share of rows selected)")
    ax.set_ylabel("material harm on changed rows")
    ax.set_ylim(0, max(1.0, float(np.nanmax(curve["harm"])) if len(curve) else 1.0))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return ax


def plot_capacity(curve, axes=None):
    """Irreducible error (top) and capacity coverage (bottom) against the correction limit."""
    plt = _plt()
    if axes is None:
        _, axes = plt.subplots(2, 1, figsize=(6.4, 5.0), sharex=True)
    top, bottom = axes
    top.plot(curve["delta"], curve["irreducible_error"], color="#21606e", linewidth=2)
    top.set_ylabel("irreducible error S")
    bottom.plot(curve["delta"], curve["capacity_coverage"], color="#21606e", linewidth=2)
    bottom.set_ylabel("capacity coverage C")
    bottom.set_ylim(0, 1)
    bottom.set_xlabel("correction limit delta")
    for ax in axes:
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    return axes
