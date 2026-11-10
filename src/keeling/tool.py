"""The tool the project brief asks for.

It takes the monthly record and returns a decomposition into trend and seasonal cycle, the current
growth rate, and a forecast with an error bar. The models are fitted on Mar 1958 - Dec 2015 only.
"""

import numpy as np
import pandas as pd

from . import evaluate
from .split import train_test_split

ERROR_BAR_NOTE = (
    "90th percentile of the absolute error of annual-mean forecasts made 20 years ahead "
    "(models refitted with data to 1985 ... 1995, scored 2005 ... 2015)"
)


def decompose(model, df):
    """Split the measurements into trend + seasonal cycle + what the model leaves over."""
    observed = df["average"].to_numpy(dtype=float)
    trend = np.asarray(model.trend(df["t"]), dtype=float)
    fitted = np.asarray(model.predict(df["t"]), dtype=float)
    return pd.DataFrame({
        "date": df["date"].to_numpy(),
        "observed": observed,
        "trend": trend,
        "seasonal": fitted - trend,
        "residual": observed - fitted,
    })


def analyse(co2, model_name=evaluate.EXPONENTIAL, target_year=2035):
    """Fit on the training years, then report the decomposition, growth rate and forecast.

    Returns (summary dict, decomposition DataFrame, forecast table for all three models).
    """
    train, _ = train_test_split(co2)
    models, selection = evaluate.fit_models(train)
    model = models[model_name]

    best = selection.iloc[0]
    factories = evaluate.model_factories(int(best["degree"]), int(best["harmonics"]))
    backtest = evaluate.backtest_forecasts(train, factories)
    forecasts = evaluate.forecast_year(models, backtest, year=target_year)

    decomposition = decompose(model, co2)
    last = co2.iloc[-1]

    # peak-to-trough of the seasonal cycle over the latest full calendar year of data
    full_years = co2.groupby("year")["month"].count()
    latest_full = int(full_years[full_years == 12].index.max())
    season = decomposition.loc[decomposition["date"].dt.year == latest_full, "seasonal"]

    year_ago = co2[co2["date"] == last["date"] - pd.DateOffset(years=1)]
    chosen = forecasts.set_index("Model").loc[model_name]

    summary = {
        "model": model_name,
        "fitted_on": f"{train['date'].min():%Y-%m} to {train['date'].max():%Y-%m}",
        "latest_month": f"{last['date']:%Y-%m}",
        "latest_measured_ppm": float(last["average"]),
        "model_trend_ppm": float(decomposition["trend"].iloc[-1]),
        "model_seasonal_ppm": float(decomposition["seasonal"].iloc[-1]),
        "seasonal_peak_to_trough_ppm": float(season.max() - season.min()),
        "seasonal_cycle_year": latest_full,
        "growth_rate_now_ppm_per_year": float(np.ravel(model.growth_rate(last["t"]))[0]),
        "measured_change_last_12_months_ppm": (
            float(last["average"] - year_ago["average"].iloc[0]) if len(year_ago) else None),
        "forecast": {
            "year": target_year,
            "mean_ppm": float(chosen[f"{target_year} mean (ppm)"]),
            "error_bar_ppm": float(chosen["Error bar (ppm)"]),
            "low_ppm": float(chosen["Low (ppm)"]),
            "high_ppm": float(chosen["High (ppm)"]),
            "error_bar_is": ERROR_BAR_NOTE,
        },
    }
    return summary, decomposition, forecasts
