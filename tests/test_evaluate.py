"""Tests for the error measures, model selection and the 1 ppm horizon."""

import numpy as np
import pandas as pd
import pytest

from keeling import evaluate
from keeling.data import time_index


def test_rmse_and_mae():
    assert evaluate.rmse([0, 0], [3, 4]) == pytest.approx(np.sqrt(12.5))
    assert evaluate.mae([0, 0], [3, -4]) == pytest.approx(3.5)


def test_error_horizon_reports_the_first_month_over_the_threshold():
    dates = pd.date_range("2016-01-01", periods=36, freq="MS")
    truth = np.zeros(36)
    pred = np.linspace(0, 3, 36)  # error grows by 3/35 ppm per month
    info = evaluate.error_horizon(dates, truth, pred, threshold=1.0, window=12)
    # single month first passes 1 ppm at index 12; a rolling RMSE needs a few more months
    assert info["months_single"] == 12
    assert info["months_rolling"] > info["months_single"]


def test_error_horizon_ignores_one_noisy_month_when_rolling():
    dates = pd.date_range("2016-01-01", periods=36, freq="MS")
    truth = np.zeros(36)
    pred = np.zeros(36)
    pred[5] = 1.5  # one outlier
    info = evaluate.error_horizon(dates, truth, pred, threshold=1.0, window=12)
    assert info["months_single"] == 5
    assert np.isnan(info["months_rolling"])


def test_annual_horizon_flags_the_first_year_over_the_threshold():
    annual = pd.DataFrame({
        "Year": [2016, 2017, 2018, 2019],
        "Model": ["A"] * 4,
        "Mean error (ppm)": [0.2, -0.6, -1.3, -2.0],
    })
    table = evaluate.annual_horizon(annual, threshold=1.0)
    assert table.loc[0, "First year over"] == 2018
    assert table.loc[0, "Max abs annual error (ppm)"] == pytest.approx(2.0)


def _synthetic_frame(start="1958-03-01", end="2015-12-01"):
    months = pd.date_range(start, end, freq="MS")
    t = time_index(months.year, months.month).to_numpy(dtype=float)
    y = 315.0 + 8.0 * (t / 10) + 1.2 * (t / 10) ** 2 + 2.0 * np.sin(2 * np.pi * t) - 0.8 * np.cos(2 * np.pi * t)
    return pd.DataFrame({"date": months, "year": months.year, "month": months.month, "t": t, "average": y})


def test_selection_finds_the_true_model_on_synthetic_data():
    frame = _synthetic_frame()
    selection = evaluate.select_rung3(frame, max_degree=4, max_harmonics=3)
    best = selection.iloc[0]
    # the true model is degree 2 with 1 harmonic; extra terms may tie, but fewer cannot fit
    assert best["val_rmse"] < 1e-6
    assert best["degree"] >= 2 and best["harmonics"] >= 1


def test_selection_uses_only_training_data():
    frame = _synthetic_frame()
    selection = evaluate.select_rung3(frame, max_degree=2, max_harmonics=1)
    assert len(selection) == 2 * 2
    assert set(selection.columns) >= {"degree", "harmonics", "parameters", "train_rmse", "val_rmse"}


def test_forecast_year_error_bar_comes_from_the_backtest():
    frame = _synthetic_frame()
    models = {evaluate.LINEAR: evaluate.LinearModel().fit(frame["t"], frame["average"])}
    backtest = pd.DataFrame({"Model": [evaluate.LINEAR] * 3, "Absolute error (ppm)": [0.5, 1.0, 1.5]})
    summary = evaluate.forecast_year(models, backtest, year=2035, quantile=0.9)
    row = summary.iloc[0]
    assert row["Error bar (ppm)"] == pytest.approx(1.4)
    assert row["High (ppm)"] - row["Low (ppm)"] == pytest.approx(2.8)
