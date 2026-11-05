"""Rung 2: a straight line per decade; slope = growth rate in ppm/yr."""

import pandas as pd

from .rung1_linear import LinearModel


def fit_by_decade(df: pd.DataFrame, min_months: int = 60) -> pd.DataFrame:
    """One LinearModel per calendar decade; returns slope, intercept and month count.

    Decades with fewer than `min_months` months are skipped: the record starts in March 1958, so
    the 1950s hold 22 months, less than two seasonal cycles, and their slope is mostly seasonal.
    """
    rows = []
    for decade, group in df.groupby((df["year"] // 10) * 10):
        if len(group) < min_months:
            continue
        model = LinearModel().fit(group["t"], group["average"])
        rows.append({
            "decade": int(decade),
            "months": len(group),
            "slope_ppm_per_year": model.slope_,
            "intercept_ppm": model.intercept_,
        })
    return pd.DataFrame(rows)
