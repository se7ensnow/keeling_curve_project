"""Rung 3: polynomial trend + annual/semi-annual sines and cosines (linear in parameters)."""

import numpy as np

from .base import Model


def harmonic_features(t, harmonics):
    t = np.atleast_1d(np.asarray(t, dtype=float))
    features = []
    for k in range(1, harmonics + 1):
        features.append(np.sin(2 * np.pi * k * t))
        features.append(np.cos(2 * np.pi * k * t))
    return features


class PolynomialSeasonalModel(Model):
    name = "polynomial_seasonal"

    def __init__(self, degree=2, harmonics=4):
        self.degree = degree
        self.harmonics = harmonics
        self.coef_ = None

    @property
    def n_params(self):
        return self.degree + 1 + 2 * self.harmonics

    def _features(self, t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        x = t / 10  # decades, keeps the higher powers well conditioned
        features = [x**d for d in range(self.degree + 1)]
        return np.column_stack(features + harmonic_features(t, self.harmonics))

    def fit(self, t, y):
        self.coef_ = np.linalg.lstsq(self._features(t), np.asarray(y), rcond=None)[0]
        return self

    def predict(self, t):
        return self._features(t) @ self.coef_

    def trend(self, t):
        x = np.asarray(t, dtype=float) / 10
        return sum(self.coef_[d] * x**d for d in range(self.degree + 1))

    def growth_rate(self, t):
        """d(trend)/dt in ppm/yr."""
        x = np.asarray(t, dtype=float) / 10
        return sum(d * self.coef_[d] * x ** (d - 1) for d in range(1, self.degree + 1)) / 10

    def params(self):
        return {
            "degree": self.degree,
            "harmonics": self.harmonics,
            "time_unit": "t = years since 1958-03; trend uses x = t/10",
            "coef": [float(c) for c in self.coef_],
        }
