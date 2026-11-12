"""CMAT for classifiers: model cascades, fallback models and deferral to a human expert.

A candidate classifier (or a human) replaces an incumbent classifier on selected cases. A case is
changed when the two labels differ. On changed cases, harm means the candidate is wrong where the
incumbent was right, and benefit the reverse. The aggregate condition bounds the change in error
rate. Everything else (support, net benefit, safety, skill against random selection at the same
coverage, bootstrap over blocks) is identical to the regression test.

    from hybrid_guard.classification import cmat_classification
    r = cmat_classification(y_true, small_model_labels, large_model_labels, selected=routed_to_large)
"""

from __future__ import annotations

import numpy as np

from hybrid_guard.cmat import ABSTAIN, ACCEPT, REJECT, CMATResult, _labels


def block_sums_classification(y, incumbent, candidate, selected=None, *, candidate_all=None, timestamps=None, groups=None,
                              block: str | None = "24h") -> np.ndarray:
    y = np.asarray(y)
    f = np.asarray(incumbent)
    h = np.asarray(candidate)
    if not (y.shape == f.shape == h.shape):
        raise ValueError("y, incumbent and candidate must have the same shape")
    h_all = h if candidate_all is None else np.asarray(candidate_all)
    sel = np.ones(len(y), bool) if selected is None else np.asarray(selected, bool)
    routed = np.where(sel, h, f)
    inc_wrong, routed_wrong = f != y, routed != y
    changed = sel & (h != f)
    harm = changed & (h != y) & (f == y)
    benefit = changed & (h == y) & (f != y)
    base = (h_all != f) & (h_all != y) & (f == y)
    labels = _labels(len(y), timestamps, groups, block)
    columns = [np.ones(len(y)), routed_wrong, inc_wrong, changed, harm, benefit, base, sel, h_all != f]
    return np.column_stack([np.bincount(labels, weights=np.asarray(c, float), minlength=labels.max() + 1) for c in columns])


def _stats(t: np.ndarray, base_definition: str = "changed"):
    n, routed, inc, changed, harm, benefit, base = (t[..., i] for i in range(7))
    diff = routed / n - inc / n
    with np.errstate(invalid="ignore", divide="ignore"):
        h = np.where(changed > 0, harm / changed, np.nan)
        g = np.where(changed > 0, benefit / changed, np.nan)
    if base_definition == "changed":
        with np.errstate(invalid="ignore", divide="ignore"):
            return diff, h, g, np.where(t[..., 8] > 0, base / t[..., 8], np.nan)
    return diff, h, g, base / n


def cmat_classification(y, incumbent, candidate, selected=None, *, timestamps=None, groups=None, block: str | None = "24h",
                        candidate_all=None, selective: bool | None = None, min_rows: int = 30, margin: float = 0.01,
                        harm_bound: float = 0.10, alpha: float = 0.05, replicates: int = 10_000, seed: int = 42,
                        base_definition: str = "changed",
                        chunk: int = 500) -> CMATResult:
    """CMAT on predicted labels. ``margin`` bounds the increase in error rate (default one point)."""
    n_rows = len(np.asarray(y))
    sel = np.ones(n_rows, bool) if selected is None else np.asarray(selected, bool)
    if selective is None:
        selective = selected is not None and not bool(sel.all())
    sums = block_sums_classification(y, incumbent, candidate, sel, candidate_all=candidate_all, timestamps=timestamps,
                                     groups=groups, block=block)
    total = sums.sum(axis=0)
    diff, harm, benefit, base = (float(v) for v in _stats(total, base_definition))
    rng = np.random.default_rng(seed)
    count = len(sums)
    b_d, b_h, b_net, b_gap = [], [], [], []
    for start in range(0, replicates, chunk):
        size = min(chunk, replicates - start)
        draws = sums[rng.integers(0, count, size=(size, count))].sum(axis=1)
        d_d, d_h, d_g, d_b = _stats(draws, base_definition)
        b_d.append(d_d)
        b_h.append(d_h)
        b_net.append(d_g - d_h)
        b_gap.append(d_b - d_h)
    b_d, b_h, b_net, b_gap = (np.concatenate(v) for v in (b_d, b_h, b_net, b_gap))
    lo_q, hi_q = alpha / 2.0, 1.0 - alpha / 2.0
    d_lo, d_hi = (float(v) for v in np.quantile(b_d, [lo_q, hi_q]))
    changed_rows = int(total[3])
    settings = {"metric": "error rate", "base_definition": base_definition, "materiality": "label change", "min_rows": min_rows, "margin": margin,
                "harm_bound": harm_bound, "alpha": alpha, "replicates": replicates, "seed": seed, "blocks": int(count)}
    common = dict(rows=int(total[0]), selected_rows=int(total[7]), changed_rows=changed_rows, coverage=float(total[7] / total[0]),
                  change_coverage=float(total[3] / total[0]), nrmse_incumbent=float(total[2] / total[0]),
                  nrmse_candidate=float(total[1] / total[0]), nrmse_difference=diff, nrmse_lower=d_lo, nrmse_upper=d_hi,
                  base_rate=base, selective=bool(selective), settings=settings)
    aggregate_pass = d_hi < margin
    reasons = [f"aggregate margin {'passed' if aggregate_pass else 'failed'}: error-rate change upper bound {d_hi:+.4f} "
               f"{'<' if aggregate_pass else '>='} {margin}"]
    if changed_rows < min_rows:
        reasons.append(f"only {changed_rows} cases changed label; at least {min_rows} are needed")
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
        reasons.append(f"skill {'passed' if skill_pass else 'failed'}: lower bound of (base rate - harm) {gap_lower:+.3f}")
    accept = aggregate_pass and net_pass and safety_pass and skill_pass
    return CMATResult(decision=ACCEPT if accept else REJECT, reasons=reasons, harm=harm, harm_upper=harm_upper, benefit=benefit,
                      net_benefit_lower=net_lower, skill_gap_lower=gap_lower, **common)
