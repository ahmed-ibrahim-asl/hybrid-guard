# Changelog

## 0.4.0

- `cmat_classification` for labels: model cascades, fallback models and deferral to a human expert.
- Skill condition: the random-selection base rate is now the candidate's harm share among the rows it would change (`base_definition="changed"`, the expected harm share of random selection). `base_definition="rows"` reproduces the original paper, whose comparison is stricter.
- Report, charts and summaries name the metric (NRMSE or error rate).
- Tutorial notebooks (wind data, expert hand-off) and a documentation site.

## 0.3.0

- Chart library: nine charts with SVG and HTML output (`hybrid_guard.charts`).
- Tables behind the charts (`hybrid_guard.analysis`), `monitor` for period-by-period checks and `harm_by_condition`.
- Report: choose charts and layout.

## 0.2.0

- HTML acceptance report, coverage-harm curve, safety-margin threshold rule, matplotlib figures, command-line tool.

## 0.1.0

- Correction-capacity diagnostics, coverage-matched acceptance test, coverage-dilution identity.
