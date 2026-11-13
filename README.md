# hybrid-guard

[![tests](https://github.com/ahmed-ibrahim-asl/hybrid-guard/actions/workflows/tests.yml/badge.svg)](https://github.com/ahmed-ibrahim-asl/hybrid-guard/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/hybrid-guard.svg)](https://pypi.org/project/hybrid-guard/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Wind tutorial](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ahmed-ibrahim-asl/hybrid-guard/blob/main/examples/01_wind_quickstart.ipynb)

**Good on average is not the same as safe.** Two checks for any model that combines a physics prior with machine learning, or that switches between models: before you train it, and before you trust it.

Documentation and live charts: https://eng-asl.com/hybrid-guard/ (mirror: https://ahmed-ibrahim-asl.github.io/hybrid-guard/)

| When | Question | Tool |
|---|---|---|
| Before training | Can a bounded hybrid correct its physics prior at all? | `capacity_diagnostics`, `capacity_curve` |
| Before deployment | Is the hybrid or router safe row by row, and better than random selection? | `cmat`, `cmat_classification` |
| After deployment | Is it still safe this month? | `monitor` |

## Why an aggregate error margin is not enough

If a candidate replaces an incumbent model on a share `C` of rows, the change in mean squared error is exactly `C` times the mean excess loss on those rows. At low coverage a selector with no skill passes any small error margin, and a hybrid that is more accurate on average can still make a third of the rows it changes materially worse. `coverage_dilution` shows this decomposition for your own predictions.

## Install

```
pip install hybrid-guard            # core: numpy and pandas only
pip install "hybrid-guard[plot]"    # matplotlib figures as well
```

## Before training: correction capacity

```python
from hybrid_guard import capacity_diagnostics, capacity_curve, charts

d = capacity_diagnostics(y_train, prior_train, delta=0.10)
d["irreducible_error"]   # lower bound on the training MSE of a hybrid p + r with |r| <= 0.10
d["capacity_coverage"]   # share of rows whose prior gap is fully correctable
charts.capacity(capacity_curve(y_train, prior_train)).save("capacity.svg")
```

## Before deployment: the coverage-matched acceptance test

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

CMAT counts only material changes (by default more than 0.01 in the target's units, for example one percent of rated power) and accepts only when all of the following hold: the aggregate margin, at least 30 changed rows, more material benefits than harms, an upper confidence bound on material harm of at most 0.10, and, for selective policies, harm below that of random selection at the same coverage. Bounds come from a bootstrap over whole blocks of rows (whole days for time series). Because every condition must hold, the chance of accepting an unsafe or unskilled selection is bounded by the level of a single condition. Every threshold is an argument.

For labels (model cascades, fallbacks, deferral to a human):

```python
from hybrid_guard import cmat_classification
r = cmat_classification(y_true, cheap_model_labels, expert_labels, selected=handed_over, groups=batch_id)
```

## After deployment: monitoring

```python
from hybrid_guard import monitor, charts
watch = monitor(y, incumbent, candidate, selected, timestamps=times, period="M")
charts.monitor_chart({"router": watch}).save("monitor.svg")
```

## Charts

Every chart is a function in `hybrid_guard.charts` that returns a `Chart`. Put it in a report, save it as an SVG for a paper (`chart.save("fig.svg", theme="light")`) or as a small HTML page (`chart.save("fig.html")`). In Jupyter it displays inline. The table behind each chart comes from `hybrid_guard.analysis`, so you can also plot the numbers yourself.

| Chart | Function | Input | Table behind it |
|---|---|---|---|
| Accuracy against safety | `charts.accuracy_safety(results)` | dict of `cmat` results | `leaderboard(results)` |
| Condition matrix | `charts.conditions(results)` | dict of `cmat` results | `condition_table(results)` |
| What happened to changed rows | `charts.changes(results)` | dict of `cmat` results | `change_breakdown(results)` |
| Leaderboard | `charts.leaderboard_table(results)` | dict of `cmat` results | `leaderboard(results)` |
| Coverage against harm | `charts.coverage_harm(curves, chosen=...)` | dict of `coverage_harm_curve(...)` | `coverage_harm_curve(...)` |
| Where it fails | `charts.harm_map(table)` | `harm_by_condition(..., rows=, columns=)` | same |
| Period by period | `charts.monitor_chart(series)` | dict of `monitor(..., timestamps=)` | `monitor(...)` |
| Capacity curves (before training) | `charts.capacity(curve)` | `capacity_curve(y, prior)` | same |
| How far the physics is off (before training) | `charts.gap(distribution)` | `gap_distribution(y, prior)` | same |

## Reports

```python
from hybrid_guard import html_report, charts, monitor

page = html_report(results,
                   charts=["accuracy_safety", "conditions", "changes", "leaderboard"],
                   extra=[charts.monitor_chart({"router": monitor(y, base, physics, mask, timestamps=t)})],
                   layout="certificate")   # or "leaderboard" to put the ranked table first
```

From the command line:

```
hybrid-guard check predictions.csv --target y --incumbent base --candidate hybrid --time timestamp --html report.html
```

## Tutorials

- [Wind-farm data](examples/01_wind_quickstart.ipynb): before training, before deployment, thresholds, where a hybrid fails, monitoring. Three Siemens turbines at Hill of Towie (CC BY 4.0).
- [Handing cases to an expert](examples/02_beyond_wind_expert_handoff.ipynb): classification, a simulated reviewer and real handwritten digits.

## Reproducing the paper

The paper's CMAT v2 used a stricter random-selection base rate (harm count over all rows). Pass `base_definition="rows"` to reproduce its numbers exactly; on the sealed data the two definitions lead to the same decision in all 16 policies checked.

## How to cite

If you use the capacity diagnostics, cite:

> A. I. Asl, B. M. El-Den. Diagnosing physics-data mismatch in hybrid learning through correction capacity: a training-only framework for wind power forecasting. Manuscript under review, 2026.

If you use CMAT or the coverage-dilution identity, cite:

> A. I. Asl, B. M. El-Den. Aggregate error margins cannot certify physics-informed wind-power estimators: a coverage-matched acceptance test with sealed multi-farm SCADA evidence. Manuscript submitted, 2026.

See `CITATION.cff` for machine-readable entries. References will be updated when the papers are published.

## License

MIT.
