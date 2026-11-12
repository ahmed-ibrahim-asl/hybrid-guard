"""Coverage-matched acceptance test (CMAT) for hybrid models, routers and fallback models.

A candidate prediction ``h`` replaces an incumbent ``f`` on some rows (all rows for a full
hybrid, selected rows for a router). Only material changes count: the change set is
``S = {selected rows with |h - f| > m}``. On ``S`` a row is materially harmed when the
candidate's absolute error exceeds the incumbent's by more than ``m``, and materially improved
in the reverse case. CMAT accepts only if every condition holds:

1. aggregate   upper bound of NRMSE(routed) - NRMSE(incumbent) below ``margin``;
2. support     at least ``min_rows`` materially changed rows (otherwise ABSTAIN);
3. net benefit lower bound of (benefit share - harm share) on S above zero;
4. safety      upper bound of the harm share on S at most ``harm_bound``;
5. skill       for selective policies, lower bound of (harm base rate of the candidate over
               all rows - harm share on S) above zero, i.e. better than random selection at
               the same coverage.

Bounds come from a paired bootstrap over whole blocks of rows (for time series, whole days
by default), recomputing every ratio in each replicate. Because acceptance needs every
condition, the probability of accepting an unsafe or unskilled selection is asymptotically
bounded by the level of a single condition (intersection-union principle).
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

ACCEPT, REJECT, ABSTAIN = "ACCEPT", "REJECT", "ABSTAIN"


@dataclass(frozen=True)
class CMATResult:
    decision: str
    reasons: list[str]
    rows: int
    selected_rows: int
    changed_rows: int
    coverage: float
    change_coverage: float
    nrmse_incumbent: float
    nrmse_candidate: float
    nrmse_difference: float
    nrmse_lower: float
    nrmse_upper: float
    harm: float | None
    harm_upper: float | None
    benefit: float | None
    net_benefit_lower: float | None
    base_rate: float
    skill_gap_lower: float | None
    selective: bool
    settings: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        lines = [f"CMAT decision: {self.decision}",
                 f"  rows {self.rows:,}, selected {self.selected_rows:,} (coverage {self.coverage:.3f}), materially changed {self.changed_rows:,}",
                 f"  NRMSE {self.nrmse_incumbent:.4f} -> {self.nrmse_candidate:.4f}, difference {self.nrmse_difference:+.4f} "
                 f"[{self.nrmse_lower:+.4f}, {self.nrmse_upper:+.4f}]"]
        if self.harm is not None:
            lines.append(f"  material harm {self.harm:.3f} (upper {self.harm_upper if self.harm_upper is None else round(self.harm_upper, 3)}), "
                         f"benefit {self.benefit:.3f}, base rate {self.base_rate:.3f}")
        lines += [f"  - {reason}" for reason in self.reasons]
        return "\n".join(lines)


def _labels(n: int, timestamps, groups, block: str | None) -> np.ndarray:
    if groups is not None:
        labels, _ = pd.factorize(pd.Series(groups), sort=True)
        return labels
    if timestamps is not None:
        floored = pd.to_datetime(pd.Series(timestamps)).dt.floor(block or "24h")
        labels, _ = pd.factorize(floored, sort=True)
        return labels
    warnings.warn("no timestamps or groups given: rows are resampled independently, which understates "
                  "uncertainty for dependent data", stacklevel=3)
    return np.arange(n)


def block_sums(y, incumbent, candidate, selected=None, *, candidate_all=None, materiality: float = 0.01,
               timestamps=None, groups=None, block: str | None = "24h") -> np.ndarray:
    """Per-block sums used by the bootstrap (seven columns)."""
    y = np.asarray(y, float)
    f = np.asarray(incumbent, float)
    h = np.asarray(candidate, float)
    if not (y.shape == f.shape == h.shape):
        raise ValueError("y, incumbent and candidate must have the same shape")
    h_all = h if candidate_all is None else np.asarray(candidate_all, float)
    sel = np.ones(len(y), bool) if selected is None else np.asarray(selected, bool)
    routed = np.where(sel, h, f)
    gain = np.abs(f - y) - np.abs(routed - y)
    changed = sel & (np.abs(h - f) > materiality)
    harm = changed & (gain < -materiality)
    benefit = changed & (gain > materiality)
    base = (np.abs(h_all - f) > materiality) & ((np.abs(h_all - y) - np.abs(f - y)) > materiality)
    labels = _labels(len(y), timestamps, groups, block)
    columns = [np.ones(len(y)), (routed - y) ** 2, (f - y) ** 2, changed, harm, benefit, base, sel]
    return np.column_stack([np.bincount(labels, weights=np.asarray(c, float), minlength=labels.max() + 1) for c in columns])


def _stats(t: np.ndarray):
    n, routed, base_sq, changed, harm, benefit, base = (t[..., i] for i in range(7))
    nrmse = np.sqrt(routed / n) - np.sqrt(base_sq / n)
    with np.errstate(invalid="ignore", divide="ignore"):
        h = np.where(changed > 0, harm / changed, np.nan)
        g = np.where(changed > 0, benefit / changed, np.nan)
    return nrmse, h, g, base / n


def cmat(y, incumbent, candidate, selected=None, *, timestamps=None, groups=None, block: str | None = "24h",
         candidate_all=None, selective: bool | None = None, materiality: float = 0.01, min_rows: int = 30,
         margin: float = 0.002, harm_bound: float = 0.10, alpha: float = 0.05, replicates: int = 10_000,
         seed: int = 42, chunk: int = 500) -> CMATResult:
    """Run CMAT. ``selected`` defaults to all rows (a full hybrid).

    ``selective`` defaults to True when ``selected`` is given and not all True. For a router,
    pass the candidate prediction on every row (``candidate``); only selected rows are used.
    """
    sel = np.ones(len(np.asarray(y)), bool) if selected is None else np.asarray(selected, bool)
    if selective is None:
        selective = selected is not None and not bool(sel.all())
    sums = block_sums(y, incumbent, candidate, sel, candidate_all=candidate_all, materiality=materiality,
                      timestamps=timestamps, groups=groups, block=block)
    total = sums.sum(axis=0)
    nrmse, harm, benefit, base = (float(v) for v in _stats(total[:7]))
    rng = np.random.default_rng(seed)
    count = len(sums)
    b_n, b_h, b_net, b_gap = [], [], [], []
    for start in range(0, replicates, chunk):
        size = min(chunk, replicates - start)
        draws = sums[rng.integers(0, count, size=(size, count))].sum(axis=1)[..., :7]
        d_n, d_h, d_g, d_b = _stats(draws)
        b_n.append(d_n)
        b_h.append(d_h)
        b_net.append(d_g - d_h)
        b_gap.append(d_b - d_h)
    b_n, b_h, b_net, b_gap = (np.concatenate(v) for v in (b_n, b_h, b_net, b_gap))
    lo_q, hi_q = alpha / 2.0, 1.0 - alpha / 2.0
    nrmse_lower, nrmse_upper = (float(v) for v in np.quantile(b_n, [lo_q, hi_q]))
    changed_rows = int(total[3])
    settings = {"materiality": materiality, "min_rows": min_rows, "margin": margin, "harm_bound": harm_bound,
                "alpha": alpha, "replicates": replicates, "seed": seed, "blocks": int(count)}
    common = dict(rows=int(total[0]), selected_rows=int(total[7]), changed_rows=changed_rows,
                  coverage=float(total[7] / total[0]), change_coverage=float(total[3] / total[0]), nrmse_incumbent=float(np.sqrt(total[2] / total[0])),
                  nrmse_candidate=float(np.sqrt(total[1] / total[0])), nrmse_difference=nrmse,
                  nrmse_lower=nrmse_lower, nrmse_upper=nrmse_upper, base_rate=base, selective=bool(selective),
                  settings=settings)
    aggregate_pass = nrmse_upper < margin
    reasons = [f"aggregate margin {'passed' if aggregate_pass else 'failed'}: NRMSE upper bound {nrmse_upper:+.4f} "
               f"{'<' if aggregate_pass else '>='} {margin}"]
    if changed_rows < min_rows:
        reasons.append(f"only {changed_rows} rows changed by more than {materiality}; at least {min_rows} are needed")
        return CMATResult(decision=ABSTAIN, reasons=reasons, harm=None if changed_rows == 0 else harm, harm_upper=None,
                          benefit=None if changed_rows == 0 else benefit, net_benefit_lower=None, skill_gap_lower=None, **common)
    finite = np.isfinite(b_h)
    harm_upper = float(np.quantile(b_h[finite], hi_q))
    net_lower = float(np.quantile(b_net[finite], lo_q))
    gap_lower = float(np.quantile(b_gap[finite], lo_q)) if selective else None
    net_pass, safety_pass = net_lower > 0.0, harm_upper <= harm_bound
    reasons.append(f"net benefit {'passed' if net_pass else 'failed'}: lower bound of benefit minus harm {net_lower:+.3f}")
    reasons.append(f"safety {'passed' if safety_pass else 'failed'}: harm upper bound {harm_upper:.3f} "
                   f"{'<=' if safety_pass else '>'} {harm_bound}")
    skill_pass = True
    if selective:
        skill_pass = gap_lower > 0.0
        reasons.append(f"skill {'passed' if skill_pass else 'failed'}: lower bound of (base rate - harm) {gap_lower:+.3f}; "
                       f"{'better than' if skill_pass else 'not shown to beat'} random selection at the same coverage")
    accept = aggregate_pass and net_pass and safety_pass and skill_pass
    return CMATResult(decision=ACCEPT if accept else REJECT, reasons=reasons, harm=harm, harm_upper=harm_upper, benefit=benefit,
                      net_benefit_lower=net_lower, skill_gap_lower=gap_lower, **common)


def random_matched(y, incumbent, candidate, coverage: float, *, seed: int = 42, **kwargs) -> CMATResult:
    """CMAT for random substitution at a given coverage, the comparator every selector must beat."""
    n = len(np.asarray(y))
    selected = np.zeros(n, bool)
    count = int(round(coverage * n))
    if count:
        selected[np.random.default_rng(seed).choice(n, size=count, replace=False)] = True
    return cmat(y, incumbent, candidate, selected, selective=True, **kwargs)
