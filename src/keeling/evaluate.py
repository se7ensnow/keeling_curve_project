"""Held-out errors (ppm), rung 3 model selection, comparison table, and rung 5 limit tests."""

import numpy as np
import pandas as pd

from .data import time_index
from .models import ExponentialSeasonalModel, LinearModel, PolynomialSeasonalModel

# Rung 3 is chosen on 2011-2015, which is still inside the training period.
# The held-out years (2016 onward) play no part in any choice.
VALIDATION_START = pd.Timestamp("2011-01-01")

LINEAR = "Linear"
POLY = "Polynomial + seasonal"
EXPONENTIAL = "Exponential + seasonal"


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def select_rung3(train, max_degree=8, max_harmonics=12):
    """Grid over polynomial degree and harmonics: fit before 2011, score on 2011-2015."""
    fit = train[train["date"] < VALIDATION_START]
    val = train[train["date"] >= VALIDATION_START]

    rows = []
    for degree in range(1, max_degree + 1):
        for harmonics in range(0, max_harmonics + 1):
            model = PolynomialSeasonalModel(degree, harmonics).fit(fit["t"], fit["average"])
            rows.append({
                "degree": degree,
                "harmonics": harmonics,
                "parameters": model.n_params,
                "train_rmse": rmse(fit["average"], model.predict(fit["t"])),
                "val_rmse": rmse(val["average"], model.predict(val["t"])),
            })
    return pd.DataFrame(rows).sort_values("val_rmse", ignore_index=True)


def model_factories(degree, harmonics):
    """Fresh, unfitted models; the backtests need a new one for every cutoff."""
    return {
        LINEAR: LinearModel,
        POLY: lambda: PolynomialSeasonalModel(degree, harmonics),
        EXPONENTIAL: lambda: ExponentialSeasonalModel(harmonics),
    }


def fit_models(train):
    """Choose degree and harmonics on validation data, then fit all three on the full training set."""
    selection = select_rung3(train)
    degree = int(selection.loc[0, "degree"])
    harmonics = int(selection.loc[0, "harmonics"])

    models = {name: make() for name, make in model_factories(degree, harmonics).items()}
    for model in models.values():
        model.fit(train["t"], train["average"])
    return models, selection


def comparison_table(models, train, test):
    """One row per model: parameters, training error, held-out error and spread."""
    rows = []
    for name, model in models.items():
        pred = model.predict(test["t"])
        rows.append({
            "Model": name,
            "Parameters": model.n_params,
            "Train RMSE (ppm)": rmse(train["average"], model.predict(train["t"])),
            "Test MAE (ppm)": mae(test["average"], pred),
            "Test RMSE (ppm)": rmse(test["average"], pred),
            "Residual SD (ppm)": float(np.std(test["average"] - pred, ddof=1)),
        })
    return pd.DataFrame(rows).sort_values("Test RMSE (ppm)", ignore_index=True)


def yearly_rmse(models, test):
    rows = []
    for year, group in test.groupby("year"):
        for name, model in models.items():
            rows.append({"Year": int(year), "Model": name,
                         "RMSE (ppm)": rmse(group["average"], model.predict(group["t"]))})
    return pd.DataFrame(rows)


def annual_mean_error(models, test):
    """Measured minus model, averaged over each held-out year (ppm). The seasonal wiggle cancels."""
    rows = []
    for year, group in test.groupby("year"):
        for name, model in models.items():
            rows.append({"Year": int(year), "Model": name, "Months": len(group),
                         "Mean error (ppm)": float(np.mean(group["average"] - model.predict(group["t"])))})
    return pd.DataFrame(rows)


def annual_horizon(annual, threshold=1.0):
    """First held-out year whose mean error passes `threshold` ppm, and the largest annual error."""
    rows = []
    for name, group in annual.groupby("Model", sort=False):
        over = group[group["Mean error (ppm)"].abs() > threshold]
        rows.append({"Model": name,
                     "First year over": int(over["Year"].iloc[0]) if len(over) else np.nan,
                     "Max abs annual error (ppm)": float(group["Mean error (ppm)"].abs().max())})
    table = pd.DataFrame(rows)
    table["First year over"] = table["First year over"].astype("Int64")
    return table


def maunakea_table(models, test):
    """Held-out error for the Maunakea months against the other held-out months."""
    rows = []
    for name, model in models.items():
        for flag, group in test.groupby("is_maunakea"):
            rows.append({"Model": name, "Maunakea": bool(flag), "Months": len(group),
                         "RMSE (ppm)": rmse(group["average"], model.predict(group["t"]))})
    return pd.DataFrame(rows)


def error_horizon(dates, y_true, y_pred, threshold=1.0, window=12):
    """How many months after the first held-out month the error passes `threshold` ppm.

    Two readings: the first single month with |error| > threshold, and the first month where
    the rolling RMSE over `window` months passes it. The rolling one is not fooled by a single
    noisy month.
    """
    dates = pd.DatetimeIndex(dates)
    err = pd.Series(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float), index=dates)

    def months_after_start(ts):
        return (ts.year - dates[0].year) * 12 + ts.month - dates[0].month

    single = err[err.abs() > threshold]
    rolling = np.sqrt((err ** 2).rolling(window).mean())
    rolling = rolling[rolling > threshold]

    return {
        "first_month_over": single.index[0] if len(single) else pd.NaT,
        "months_single": months_after_start(single.index[0]) if len(single) else np.nan,
        "first_rolling_over": rolling.index[0] if len(rolling) else pd.NaT,
        "months_rolling": months_after_start(rolling.index[0]) if len(rolling) else np.nan,
    }


def horizon_table(models, test, threshold=1.0):
    rows = []
    for name, model in models.items():
        info = error_horizon(test["date"], test["average"], model.predict(test["t"]), threshold)
        rows.append({"Model": name, **info})
    return pd.DataFrame(rows)


def backtest_forecasts(train, factories, cutoffs=range(1985, 1996), horizon=20):
    """Fit up to each cutoff year, predict the annual mean `horizon` years later, record the error.

    The real forecast also looks about 20 years past its training data (2015 to 2035),
    so these errors are the honest size of that forecast's error.
    """
    rows = []
    for cutoff in cutoffs:
        history = train[train["year"] <= cutoff]
        target = train[train["year"] == cutoff + horizon]
        for name, make in factories.items():
            model = make().fit(history["t"], history["average"])
            error = float(target["average"].mean() - model.predict(target["t"]).mean())
            rows.append({"Model": name, "Train until": cutoff, "Target year": cutoff + horizon,
                         "Annual error (ppm)": error, "Absolute error (ppm)": abs(error)})
    return pd.DataFrame(rows)


def forecast_year(models, backtest, year=2035, quantile=0.9):
    """Annual mean forecast for `year` with the backtest error (90th percentile) as the error bar."""
    months = pd.date_range(f"{year}-01-01", f"{year}-12-01", freq="MS")
    t = time_index(months.year, months.month).to_numpy(dtype=float)
    spread = backtest.groupby("Model")["Absolute error (ppm)"].quantile(quantile)

    rows = []
    for name, model in models.items():
        mean = float(np.mean(model.predict(t)))
        rows.append({"Model": name, f"{year} mean (ppm)": mean, "Error bar (ppm)": float(spread[name]),
                     "Low (ppm)": mean - float(spread[name]), "High (ppm)": mean + float(spread[name])})
    return pd.DataFrame(rows)
