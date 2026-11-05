"""Rung 1: one straight line fitted to the whole record by least squares."""

import numpy as np

from .base import Model


class LinearModel(Model):
    name = "linear"

    n_params = 2

    def __init__(self):
        self.intercept_ = None
        self.slope_ = None

    def fit(self, t, y):
        t = np.asarray(t, dtype=float)
        X = np.column_stack([np.ones(len(t)), t])
        self.intercept_, self.slope_ = np.linalg.lstsq(X, np.asarray(y), rcond=None)[0]
        return self

    def predict(self, t):
        return self.intercept_ + self.slope_ * np.asarray(t, dtype=float)

    def trend(self, t):
        return self.predict(t)

    def growth_rate(self, t):
        return np.full_like(np.asarray(t, dtype=float), self.slope_)

    def params(self):
        return {
            "intercept_ppm": float(self.intercept_),
            "slope_ppm_per_year": float(self.slope_),
        }
