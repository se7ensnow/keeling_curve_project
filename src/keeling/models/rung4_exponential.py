"""Rung 4: trend c0 + a*exp(r*t) with r as a truly non-linear parameter."""

import numpy as np
from scipy.optimize import minimize_scalar

from .base import Model
from .rung3_harmonic import harmonic_features


class ExponentialSeasonalModel(Model):
    name = "exponential_seasonal"

    def __init__(self, harmonics=4, rate_bounds=(0.005, 0.1)):
        self.harmonics = harmonics
        self.rate_bounds = rate_bounds
        self.rate_ = None
        self.coef_ = None

    def _features(self, t, rate):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        # expm1 makes c0 the trend value at t = 0 and stops c0 and a from cancelling
        features = [np.ones(len(t)), np.expm1(rate * t)]
        return np.column_stack(features + harmonic_features(t, self.harmonics))

    @property
    def n_params(self):
        return 2 + 2 * self.harmonics + 1  # c0, a, the harmonics, and the rate r

    def _sse(self, t, y, rate):
        X = self._features(t, rate)
        coef = np.linalg.lstsq(X, y, rcond=None)[0]
        return np.sum((y - X @ coef) ** 2)

    def sse_profile(self, t, y, rates):
        """Sum of squared errors for each rate, with the linear coefficients re-fitted every time."""
        t = np.asarray(t, dtype=float)
        y = np.asarray(y, dtype=float)
        return np.array([self._sse(t, y, r) for r in rates])

    def fit(self, t, y):
        t = np.asarray(t, dtype=float)
        y = np.asarray(y, dtype=float)
        # Least squares for the linear coefficients inside a 1-D search over r
        result = minimize_scalar(
            lambda r: self._sse(t, y, r), bounds=self.rate_bounds, method="bounded"
        )
        self.rate_ = result.x
        self.coef_ = np.linalg.lstsq(self._features(t, self.rate_), y, rcond=None)[0]
        return self

    def predict(self, t):
        return self._features(t, self.rate_) @ self.coef_

    def trend(self, t):
        t = np.asarray(t, dtype=float)
        return self.coef_[0] + self.coef_[1] * np.expm1(self.rate_ * t)

    def growth_rate(self, t):
        t = np.asarray(t, dtype=float)
        return self.coef_[1] * self.rate_ * np.exp(self.rate_ * t)

    def params(self):
        return {
            "harmonics": self.harmonics,
            "rate_per_year": float(self.rate_),
            "time_unit": "t = years since 1958-03",
            "coef": [float(c) for c in self.coef_],
        }
