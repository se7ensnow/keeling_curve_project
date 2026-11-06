"""Tests for the comparison tables. They need neither the model nor a download."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

FOLDER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FOLDER))
sys.path.insert(0, str(FOLDER.parent / "src"))

import compare  # noqa: E402


def make_frame(months=24):
    dates = pd.date_range("2016-01-01", periods=months, freq="MS")
    return pd.DataFrame({"date": dates, "year": dates.year, "average": np.linspace(400.0, 405.0, months)})


def test_held_out_table_puts_the_more_accurate_model_first():
    frame = make_frame()
    truth = frame["average"].to_numpy()
    table = compare.held_out_table(frame, {"close": truth + 0.1, "far": truth + 5.0})
    assert list(table["Model"]) == ["close", "far"]
    assert table.loc[0, "Test RMSE (ppm)"] == pytest.approx(0.1)
    assert table.loc[0, "Test MAE (ppm)"] == pytest.approx(0.1)
    assert table.loc[1, "Test RMSE (ppm)"] == pytest.approx(5.0)


def test_timesfm_is_reported_with_no_parameters_fitted_here():
    frame = make_frame()
    truth = frame["average"].to_numpy()
    table = compare.held_out_table(frame, {compare.NAME: truth + 0.5})
    assert table.loc[0, "Parameters fitted here"] == 0


def test_annual_table_averages_the_signed_error_of_each_year():
    frame = make_frame(24)
    truth = frame["average"].to_numpy()
    pred = truth.copy()
    pred[frame["year"].to_numpy() == 2016] -= 1.0  # model too low in 2016: measured minus model = +1
    pred[frame["year"].to_numpy() == 2017] += 2.0  # model too high in 2017: measured minus model = -2
    table = compare.annual_table(frame, {"m": pred}).set_index("Year")
    assert table.loc[2016, "m"] == pytest.approx(1.0)
    assert table.loc[2017, "m"] == pytest.approx(-2.0)


def test_interval_coverage_counts_the_months_inside_the_range():
    frame = make_frame(10)
    truth = frame["average"].to_numpy()
    forecast = pd.DataFrame({"date": frame["date"], "q10": truth - 1.0, "q90": truth + 1.0})
    forecast.loc[:3, ["q10", "q90"]] += 10.0  # four months where the range misses the measurement
    out = compare.interval_coverage(frame, forecast).iloc[0]
    assert out["Held-out months"] == 10
    assert out["Share inside"] == pytest.approx(0.6)
    assert out["Mean width (ppm)"] == pytest.approx(2.0)


def test_predictions_refuse_a_forecast_that_does_not_reach_the_held_out_months():
    frame = make_frame(12)
    short = pd.DataFrame({"date": frame["date"].iloc[:6], "point": np.zeros(6)})
    with pytest.raises(SystemExit):
        compare.all_predictions({}, frame.assign(t=0.0), short)
