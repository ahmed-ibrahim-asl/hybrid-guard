"""Synthetic walk-through: a hybrid that is more accurate on average but not safe row by row."""

import numpy as np
import pandas as pd

from hybrid_guard import capacity_diagnostics, cmat, coverage_dilution, random_matched

rng = np.random.default_rng(0)
n = 6 * 24 * 180  # half a year of ten-minute rows
times = pd.date_range("2025-01-01", periods=n, freq="10min")
y = np.clip(rng.beta(2, 3, n), 0, 1)                      # measured power, per unit
prior = np.clip(y + rng.normal(0.05, 0.15, n), 0, 1)     # physics prior with bias and noise
data_model = y + rng.normal(0, 0.06, n)                  # data-only model
hybrid = 0.8 * data_model + 0.2 * prior                  # a blended hybrid

print("Before training: can a bounded residual with |r| <= 0.10 fix the prior?")
print(capacity_diagnostics(y, prior, delta=0.10))

print("\nBefore acceptance: full hybrid against the data-only model")
print(cmat(y, data_model, hybrid, timestamps=times).summary())

print("\nA physics router that substitutes the prior on 3% of rows at random")
selected = rng.random(n) < 0.03
print(coverage_dilution(y, data_model, prior, selected))
print(random_matched(y, data_model, prior, 0.03, timestamps=times).summary())
