"""Tests for the decomposition / growth rate / forecast tool."""

import numpy as np
import pandas as pd
import pytest

from keeling import evaluate, tool
from keeling.data import load_co2_data, time_index
from keeling.models import PolynomialSeasonalModel


def test_decomposition_adds_back_up_to_the_measurements():
    months = pd.date_range("1958-03-01", "2015-12-01", freq="MS")
    t = time_index(months.year, months.month).to_numpy(dtype=float)
    rng = np.random.default_rng(1)
    y = 315.0 + 8.0 * (t / 10) + 2.0 * np.sin(2 * np.pi * t) + rng.normal(0, 0.3, len(t))
    df = pd.DataFrame({"date": months, "t": t, "average": y})

    model = PolynomialSeasonalModel(2, 2).fit(df["t"], df["average"])
    out = tool.decompose(model, df)

    assert (out["trend"] + out["seasonal"] + out["residual"]).to_numpy() == pytest.approx(y)
    assert out["seasonal"].abs().max() < 3  # the yearly cycle, not the trend
    assert out["trend"].is_monotonic_increasing


@pytest.fixture(scope="module")
def analysis():
    return tool.analyse(load_co2_data(), evaluate.EXPONENTIAL, target_year=2035)


def test_summary_has_growth_rate_and_a_forecast_with_an_error_bar(analysis):
    summary, decomposition, forecasts = analysis
    f = summary["forecast"]
    assert summary["fitted_on"] == "1958-03 to 2015-12"
    assert 1.5 < summary["growth_rate_now_ppm_per_year"] < 4.0
    assert f["year"] == 2035
    assert f["error_bar_ppm"] > 0
    assert f["low_ppm"] == pytest.approx(f["mean_ppm"] - f["error_bar_ppm"])
    assert f["high_ppm"] == pytest.approx(f["mean_ppm"] + f["error_bar_ppm"])
    assert 430 < f["mean_ppm"] < 480
    assert 3 < summary["seasonal_peak_to_trough_ppm"] < 10


def test_decomposition_covers_every_month_of_the_record(analysis):
    _, decomposition, forecasts = analysis
    assert len(decomposition) == len(load_co2_data())
    assert set(forecasts["Model"]) == {evaluate.LINEAR, evaluate.POLY, evaluate.EXPONENTIAL}


def test_measured_change_over_the_last_12_months_uses_the_data(analysis):
    summary, *_ = analysis
    co2 = load_co2_data()
    expected = co2["average"].iloc[-1] - co2["average"].iloc[-13]
    assert summary["measured_change_last_12_months_ppm"] == pytest.approx(expected)
