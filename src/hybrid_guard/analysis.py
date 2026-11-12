"""Tables behind the charts. Every function returns a pandas DataFrame with documented columns,
so the numbers can be used without the built-in charts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from hybrid_guard.cmat import CMATResult, cmat

CONDITIONS = ("aggregate", "support", "net_benefit", "safety", "skill")


def condition_table(results: Mapping[str, CMATResult]) -> pd.DataFrame:
    """One row per candidate and condition.

    Columns: candidate, decision, condition, state ("pass", "fail" or "n/a"), value, requirement.
    """
    rows = []
    for name, r in results.items():
        s = r.settings
        support = r.changed_rows >= s["min_rows"]
        checks = {
            "aggregate": (r.nrmse_upper < s["margin"], r.nrmse_upper, f"NRMSE change upper bound below {s['margin']}"),
            "support": (support, float(r.changed_rows), f"at least {s['min_rows']} rows changed by more than {s['materiality']}"),
            "net_benefit": (r.net_benefit_lower > 0 if support else None, r.net_benefit_lower, "lower bound of benefit minus harm above 0"),
            "safety": (r.harm_upper <= s["harm_bound"] if support else None, r.harm_upper, f"harm upper bound at most {s['harm_bound']}"),
            "skill": ((r.skill_gap_lower > 0) if (support and r.selective) else None, r.skill_gap_lower,
                      "harm below random selection at the same coverage" if r.selective else "not applicable to a full hybrid"),
        }
        for condition, (passed, value, requirement) in checks.items():
            rows.append({"candidate": name, "decision": r.decision, "condition": condition,
                         "state": "n/a" if passed is None else ("pass" if passed else "fail"),
                         "value": np.nan if value is None else float(value), "requirement": requirement})
    return pd.DataFrame(rows)


def change_breakdown(results: Mapping[str, CMATResult]) -> pd.DataFrame:
    """Share of materially changed rows that improved, were harmed or neither.

    Columns: candidate, decision, changed_rows, improved, neither, harmed (shares summing to 1).
    """
    rows = []
    for name, r in results.items():
        improved = r.benefit if r.benefit is not None else np.nan
        harmed = r.harm if r.harm is not None else np.nan
        rows.append({"candidate": name, "decision": r.decision, "changed_rows": r.changed_rows, "improved": improved,
                     "neither": max(0.0, 1.0 - improved - harmed) if r.harm is not None else np.nan, "harmed": harmed})
    return pd.DataFrame(rows)


def leaderboard(results: Mapping[str, CMATResult]) -> pd.DataFrame:
    """Candidates ranked: accepted first, then abstained, then rejected; within each, by material harm.

    Columns: rank, candidate, decision, nrmse_difference, nrmse_lower, nrmse_upper, harm, harm_upper,
    benefit, coverage, changed_rows.
    """
    order = {"ACCEPT": 0, "ABSTAIN": 1, "REJECT": 2}
    rows = [{"candidate": k, "decision": r.decision, "nrmse_difference": r.nrmse_difference, "nrmse_lower": r.nrmse_lower,
             "nrmse_upper": r.nrmse_upper, "harm": r.harm, "harm_upper": r.harm_upper, "benefit": r.benefit,
             "coverage": r.coverage, "changed_rows": r.changed_rows} for k, r in results.items()]
    table = pd.DataFrame(rows)
    table["_o"] = table["decision"].map(order)
    table = table.sort_values(["_o", "harm"], na_position="last", kind="mergesort").drop(columns="_o").reset_index(drop=True)
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    return table


def bands(values, edges: Sequence[float], unit: str = "") -> np.ndarray:
    """Label each value with its band, e.g. bands(wind_speed, [3, 5, 8, 11, 14, 25], " m/s") -> "5-8 m/s"."""
    values = np.asarray(values, float)
    labels = np.array([f"{a:g}-{b:g}{unit}" for a, b in zip(edges[:-1], edges[1:])] + [""], dtype=object)
    index = np.searchsorted(edges, values, side="right") - 1
    index = np.where((index < 0) | (index >= len(edges) - 1), len(edges) - 1, index)
    return labels[index]


def harm_by_condition(y, incumbent, candidate, selected=None, *, rows, columns=None, materiality: float = 0.01,
                      min_rows: int = 30) -> pd.DataFrame:
    """Material harm and benefit on changed rows, split by one or two grouping variables.

    ``rows`` and ``columns`` are arrays of labels (see ``bands``). Columns of the result: row, column,
    changed_rows, harm, benefit, reliable (at least ``min_rows`` changed rows).
    """
    y = np.asarray(y, float)
    f = np.asarray(incumbent, float)
    h = np.asarray(candidate, float)
    sel = np.ones(len(y), bool) if selected is None else np.asarray(selected, bool)
    changed = sel & (np.abs(h - f) > materiality)
    gain = np.abs(f - y) - np.abs(h - y)
    frame = pd.DataFrame({"row": np.asarray(rows, dtype=object), "column": "all" if columns is None else np.asarray(columns, dtype=object),
                          "changed": changed, "harm": changed & (gain < -materiality), "benefit": changed & (gain > materiality)})
    frame = frame.loc[frame["row"].astype(str).ne("") & frame["column"].astype(str).ne("")]
    grouped = frame.groupby(["row", "column"], sort=False)[["changed", "harm", "benefit"]].sum().reset_index()
    with np.errstate(invalid="ignore", divide="ignore"):
        grouped["harm"] = np.where(grouped["changed"] > 0, grouped["harm"] / grouped["changed"], np.nan)
        grouped["benefit"] = np.where(grouped["changed"] > 0, grouped["benefit"] / grouped["changed"], np.nan)
    grouped = grouped.rename(columns={"changed": "changed_rows"})
    grouped["changed_rows"] = grouped["changed_rows"].astype(int)
    grouped["reliable"] = grouped["changed_rows"] >= min_rows
    return grouped


def monitor(y, incumbent, candidate, selected=None, *, timestamps, period: str = "M", selective: bool | None = None,
            replicates: int = 1000, **cmat_kwargs) -> pd.DataFrame:
    """Run CMAT per calendar period ("M" month, "W" week, "Q" quarter) to watch a deployed model drift.

    Columns: period, rows, changed_rows, harm, harm_upper, benefit, nrmse_difference, decision, above_bound.
    """
    times = pd.to_datetime(pd.Series(timestamps)).reset_index(drop=True)
    periods = times.dt.to_period(period)
    y = np.asarray(y, float)
    f = np.asarray(incumbent, float)
    h = np.asarray(candidate, float)
    sel = None if selected is None else np.asarray(selected, bool)
    if selective is None:
        selective = sel is not None and not bool(sel.all())
    rows = []
    for period in sorted(periods.unique()):
        m = (periods == period).to_numpy()
        r = cmat(y[m], f[m], h[m], None if sel is None else sel[m], timestamps=times[m].to_numpy(), selective=selective,
                 replicates=replicates, **cmat_kwargs)
        bound = r.settings["harm_bound"]
        rows.append({"period": str(period), "rows": r.rows, "changed_rows": r.changed_rows, "harm": r.harm, "harm_upper": r.harm_upper,
                     "benefit": r.benefit, "nrmse_difference": r.nrmse_difference, "decision": r.decision,
                     "above_bound": bool(r.harm is not None and r.harm > bound)})
    return pd.DataFrame(rows)


def gap_distribution(y, prior, *, width: float = 0.02, max_gap: float = 1.0) -> pd.DataFrame:
    """Distribution of the prior's absolute error |y - p| (Paper A).

    Columns: gap_start, gap_end, rows, share. Rows beyond ``max_gap`` fall in the last bin.
    """
    gap = np.abs(np.asarray(y, float) - np.asarray(prior, float))
    gap = gap[np.isfinite(gap)]
    edges = np.round(np.arange(0.0, max_gap + width / 2, width), 10)
    counts, _ = np.histogram(np.minimum(gap, max_gap - 1e-12), bins=edges)
    return pd.DataFrame({"gap_start": edges[:-1], "gap_end": edges[1:], "rows": counts, "share": counts / max(len(gap), 1)})
