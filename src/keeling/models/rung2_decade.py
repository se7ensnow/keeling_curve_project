"""Rung 2: a straight line per decade; slope = growth rate in ppm/yr."""

import numpy as np
import pandas as pd

from .base import Model
from .rung1_linear import LinearModel


def decade_of(t):
    """Calendar decade (1960, 1970, ...) of each time t, in years since March 1958."""
    months_since_jan_1958 = np.rint(np.asarray(t, dtype=float) * 12).astype(int) + 2
    return (1958 + months_since_jan_1958 // 12) // 10 * 10


class DecadeLinearModel(Model):
    """One LinearModel per calendar decade.

    Decades with fewer than `min_months` months are skipped: the record starts in March 1958, so
    the 1950s hold 22 months, less than two seasonal cycles, and their slope is mostly seasonal.
    The lines are separate fits, so the prediction jumps at the start of each decade.
    """

    name = "decade_linear"

    def __init__(self, min_months: int = 60):
        self.min_months = min_months
        self.lines_ = {}   # decade -> LinearModel
        self.months_ = {}  # decade -> number of months fitted

    @property
    def n_params(self):
        return 2 * len(self.lines_)

    def fit(self, t, y):
        t = np.asarray(t, dtype=float)
        y = np.asarray(y, dtype=float)
        decades = decade_of(t)
        self.lines_, self.months_ = {}, {}
        for decade in np.unique(decades):
            mask = decades == decade
            if mask.sum() < self.min_months:
                continue
            self.lines_[int(decade)] = LinearModel().fit(t[mask], y[mask])
            self.months_[int(decade)] = int(mask.sum())
        return self

    def _line_for(self, decades):
        """The fitted line each month uses: its own decade's, or the nearest fitted decade's."""
        known = np.array(sorted(self.lines_))
        return np.array([int(known[np.argmin(np.abs(known - d))]) for d in decades])

    def predict(self, t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        which = self._line_for(decade_of(t))
        out = np.empty(len(t))
        for decade in np.unique(which):
            mask = which == decade
            out[mask] = self.lines_[int(decade)].predict(t[mask])
        return out

    def trend(self, t):
        return self.predict(t)

    def growth_rate(self, t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        which = self._line_for(decade_of(t))
        return np.array([self.lines_[int(d)].slope_ for d in which])

    def table(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "decade": decade,
            "months": self.months_[decade],
            "slope_ppm_per_year": line.slope_,
            "intercept_ppm": line.intercept_,
        } for decade, line in self.lines_.items()])

    def params(self):
        return {
            "min_months": self.min_months,
            "time_unit": "t = years since 1958-03",
            "decades": {
                str(decade): {
                    "months": self.months_[decade],
                    "slope_ppm_per_year": float(line.slope_),
                    "intercept_ppm": float(line.intercept_),
                } for decade, line in self.lines_.items()
            },
        }


def fit_by_decade(df: pd.DataFrame, min_months: int = 60) -> pd.DataFrame:
    """The per-decade slopes and intercepts of a DataFrame with `t` and `average` columns."""
    return DecadeLinearModel(min_months).fit(df["t"], df["average"]).table()
