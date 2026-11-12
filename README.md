# hybrid-guard

Two checks for any model that combines a physics prior with machine learning, or that switches between models.

| When | Question | Tool |
|---|---|---|
| Before training | Can a bounded hybrid correct its physics prior at all? | `capacity_diagnostics`, `capacity_curve` |
| Before acceptance | Is the hybrid or router safe row by row, and better than random selection, or does it only look good on average? | `cmat` |

## Why an aggregate error margin is not enough

If a candidate replaces an incumbent model on a share `C` of rows, the change in mean squared error is exactly `C` times the mean excess loss on those rows. At low coverage a selector with no skill passes any small error margin, and a hybrid that is more accurate on average can still make a third of the rows it changes materially worse. `coverage_dilution` shows this decomposition for your own predictions.

## Install

```
pip install hybrid-guard
```

## Before training: correction capacity

```python
from hybrid_guard import capacity_diagnostics, capacity_curve

d = capacity_diagnostics(y_train, prior_train, delta=0.10)
d["irreducible_error"]   # lower bound on the training MSE of a hybrid p + r with |r| <= 0.10
d["capacity_coverage"]   # share of rows whose prior gap is fully correctable
curve = capacity_curve(y_train, prior_train)   # the same over delta = 0, 0.01, ..., 1
```

## Before acceptance: the coverage-matched acceptance test

```python
from hybrid_guard import cmat

# full hybrid against the data-only model
result = cmat(y_test, incumbent=data_only_pred, candidate=hybrid_pred, timestamps=times)
print(result.summary())

# a router that substitutes physics on selected rows
result = cmat(y_test, incumbent=data_only_pred, candidate=physics_pred, selected=router_mask, timestamps=times)
result.decision          # "ACCEPT", "REJECT" or "ABSTAIN"
result.reasons           # which conditions passed or failed, in plain words
```

CMAT counts only material changes (by default more than 0.01 in the target's units, for example one percent of rated power) and accepts only when all of the following hold: the aggregate margin, at least 30 changed rows, more material benefits than harms, an upper confidence bound on material harm of at most 0.10, and, for selective policies, harm below that of random selection at the same coverage. Bounds come from a bootstrap over whole blocks of rows (whole days for time series). Every threshold is an argument.

## How to cite

If you use the capacity diagnostics, cite:

> A. I. Asl, B. M. El-Den. Diagnosing physics-data mismatch in hybrid learning through correction capacity: a training-only framework for wind power forecasting. Manuscript under review, 2026.

If you use CMAT or the coverage-dilution identity, cite:

> A. I. Asl, B. M. El-Den. Aggregate error margins cannot certify physics-informed wind-power estimators: a coverage-matched acceptance test with sealed multi-farm SCADA evidence. Manuscript submitted, 2026.

See `CITATION.cff` for machine-readable entries. References will be updated when the papers are published.

## License

MIT.
