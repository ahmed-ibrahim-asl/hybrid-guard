---
title: 'hybrid-guard: acceptance testing for physics-informed and model-switching predictors'
tags:
  - Python
  - physics-informed machine learning
  - model evaluation
  - selective prediction
  - acceptance testing
  - wind energy
authors:
  - name: Ahmed Ibrahim Asl
    orcid: 0009-0001-4557-3641
    affiliation: "1, 2"
  - name: B. M. El-Den
    corresponding: true
    affiliation: 1
affiliations:
  - name: Electronics and Communications Engineering Department, Faculty of Engineering, Delta University for Science and Technology, Gamasa, Egypt
    index: 1
  - name: Mechatronics Department, Faculty of Engineering, Mansoura University, Mansoura, Egypt
    index: 2
date: 12 November 2026
bibliography: paper.bib
---

# Summary

Many predictors combine a physical model with machine learning, or switch between several models: a router substitutes a physics prediction on selected rows, a cheap model hands hard cases to an expensive model or a human, a hybrid corrects a physics prior with a learned residual. Such systems are usually accepted on an aggregate error: the new system is kept if its average error is no worse than before. `hybrid-guard` is a Python package that checks these systems at the two moments that matter. Before training, it measures whether a bounded hybrid can correct its physics prior at all, using training data only [@asl2026capacity]. Before deployment, it runs a coverage-matched acceptance test (CMAT) that asks whether the rows a candidate changes are improved, whether few of them are harmed, and whether the selection beats random selection at the same coverage [@asl2026cmat]. After deployment, it repeats the test period by period to detect drift. Results come with plain-language reasons, nine charts that can be saved as SVG, and a self-contained HTML report.

# Statement of need

When a candidate replaces an incumbent model on a share $C$ of rows, the change in mean squared error is exactly $C$ times the mean excess loss on those rows. An aggregate margin therefore dilutes whatever happens on the changed rows by coverage: a selector with no skill passes it at low coverage, and a hybrid that is better on average can still make a large share of the rows it changes materially worse. In a preregistered study on public data from five wind farms, a physics router passed a noninferiority margin on every sealed test period while a fifth to a half of its substitutions were harmful, and hybrids that improved accuracy on every period still worsened 21 to 41 percent of the rows they changed [@asl2026cmat]. The same structure appears in hidden stratification and in average treatment effects that conceal harmed subgroups [@oakdenrayner2020; @kent2018].

Selective prediction reports risk as a function of coverage [@chow1970; @herbei2006], and distribution-free methods calibrate thresholds with risk guarantees [@bates2021; @angelopoulos2023]. Work on learning to defer and on routing between language models trains the selector itself [@madras2018predict; @mozannar2020consistent; @ong2024routellm], and physics-informed machine learning studies how to combine priors with data [@karpatne2017; @willard2022]. None of these provides a ready acceptance test that compares a selection with random selection at the same coverage, counts only material changes and bounds harm on the changed rows. `hybrid-guard` fills that gap with a small dependency footprint (NumPy and pandas) and works on any array of predictions, whatever model produced them.

# Functionality

- `capacity_diagnostics` and `capacity_curve` compute, from training data, the irreducible error $\max(|y-p|-\delta,0)^2$ of a hybrid whose correction is bounded by $\delta$, and the share of rows whose prior gap is correctable.
- `cmat` (regression) and `cmat_classification` (labels) apply five conditions: an aggregate margin, a minimum number of materially changed rows, more material benefits than harms, an upper confidence bound on material harm, and, for selective policies, harm below the random-selection base rate. Intervals come from a bootstrap over whole blocks of dependent rows. Because acceptance requires every condition, it is an intersection-union test [@berger1982] whose asymptotic error is bounded by that of one condition under block-bootstrap consistency [@kunsch1989].
- `coverage_harm_curve` and `choose_threshold` trace harm along a selector's score and choose a threshold with a margin below the bound; `harm_by_condition` splits harm by operating conditions; `monitor` repeats the test per period.
- `hybrid_guard.charts` draws nine charts (accuracy against safety, a condition matrix, the fate of changed rows, coverage against harm, a harm map, monitoring, a leaderboard and two capacity charts); each returns an object that renders in a notebook, in the HTML report or as an SVG file.
- A command-line interface checks a CSV or Parquet file without code.

The package reproduces the published statistics of its companion study exactly on the sealed Hill of Towie data [@hilloftowie2026], and its tutorials cover wind-farm data and expert hand-off in classification.

# AI usage disclosure

An AI coding assistant was used to write and test parts of the code, documentation and this paper. The authors reviewed the output and take responsibility for the software and the text.

# References
