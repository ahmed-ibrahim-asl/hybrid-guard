"""hybrid-guard: check physics-ML hybrids before training and before acceptance.

Before training:   capacity diagnostics show whether a bounded hybrid can correct its prior.
Before acceptance: the coverage-matched acceptance test (CMAT) asks whether a hybrid, router
                   or fallback model is safe row by row and better than random selection,
                   not only within an aggregate error margin.

Charts live in ``hybrid_guard.charts``; the tables behind them in ``hybrid_guard.analysis``.
"""

from hybrid_guard import charts
from hybrid_guard.analysis import (bands, change_breakdown, condition_table, gap_distribution, harm_by_condition, leaderboard,
                                   monitor)
from hybrid_guard.classification import cmat_classification
from hybrid_guard.capacity import capacity_curve, capacity_diagnostics, irreducible_error, smallest_sufficient_delta
from hybrid_guard.cmat import ABSTAIN, ACCEPT, REJECT, CMATResult, block_sums, cmat, random_matched
from hybrid_guard.dilution import admissible_mean_excess, coverage_dilution, random_router_passes
from hybrid_guard.report import html_report, save_report
from hybrid_guard.thresholds import choose_threshold, coverage_harm_curve

__version__ = "0.4.0"
__all__ = ["ABSTAIN", "ACCEPT", "REJECT", "CMATResult", "admissible_mean_excess", "bands", "block_sums", "capacity_curve",
           "capacity_diagnostics", "change_breakdown", "charts", "choose_threshold", "cmat", "cmat_classification", "condition_table",
           "coverage_dilution", "coverage_harm_curve", "gap_distribution", "harm_by_condition", "html_report",
           "irreducible_error", "leaderboard", "monitor", "random_matched", "random_router_passes", "save_report",
           "smallest_sufficient_delta"]
