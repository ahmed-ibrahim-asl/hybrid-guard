"""hybrid-guard: check physics-ML hybrids before training and before acceptance.

Before training:   capacity diagnostics show whether a bounded hybrid can correct its prior.
Before acceptance: the coverage-matched acceptance test (CMAT) asks whether a hybrid, router
                   or fallback model is safe row by row and better than random selection,
                   not only within an aggregate error margin.
"""

from hybrid_guard.capacity import capacity_curve, capacity_diagnostics, irreducible_error, smallest_sufficient_delta
from hybrid_guard.cmat import ABSTAIN, ACCEPT, REJECT, CMATResult, block_sums, cmat, random_matched
from hybrid_guard.dilution import admissible_mean_excess, coverage_dilution, random_router_passes

__version__ = "0.1.0"
__all__ = ["ABSTAIN", "ACCEPT", "REJECT", "CMATResult", "admissible_mean_excess", "block_sums", "capacity_curve",
           "capacity_diagnostics", "cmat", "coverage_dilution", "irreducible_error", "random_matched",
           "random_router_passes", "smallest_sufficient_delta"]
