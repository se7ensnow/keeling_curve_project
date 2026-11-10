"""Tests for the numbers behind the presentation page."""

import json

import numpy as np
import pytest

from keeling import evaluate, presentation
from keeling.data import load_co2_data
from keeling.models import LinearModel, PolynomialSeasonalModel
from keeling.split import train_test_split


@pytest.fixture(scope="module")
def co2():
    return load_co2_data()


@pytest.fixture(scope="module")
def data(co2):
    return presentation.build(co2)


def test_everything_is_plain_json(data):
    assert json.loads(json.dumps(data)) == data


def test_periods_follow_the_fixed_split(data):
    meta = data["meta"]
    assert meta["train"] == ["1958-03", "2015-12"]
    assert meta["fit"] == ["1958-03", "2010-12"]
    assert meta["validation"] == ["2011-01", "2015-12"]
    assert meta["held_out"][0] == "2016-01"
    assert meta["fit_months"] + meta["validation_months"] == meta["train_months"]
    assert meta["train_months"] + meta["held_out_months"] == meta["record_months"]
    assert meta["maunakea"][0] == "2022-12" and meta["maunakea"][-1] == "2023-07"


def test_polynomial_of_degree_one_is_the_rung_1_line(data):
    line, poly = data["linear"], data["polynomial"]["1"]
    assert poly["coef_per_year"][0] == pytest.approx(line["intercept"])
    assert poly["coef_per_year"][1] == pytest.approx(line["slope"])
    assert poly["train_rmse"] == pytest.approx(line["train_rmse"])


def test_per_year_coefficients_describe_the_same_curve(data):
    t = np.linspace(0, 57, 50)
    for fit in data["polynomial"].values():
        in_decades = sum(c * (t / 10) ** k for k, c in enumerate(fit["coef"]))
        in_years = sum(c * t**k for k, c in enumerate(fit["coef_per_year"]))
        assert in_years == pytest.approx(in_decades)


def test_training_error_falls_with_degree_and_with_harmonics(data):
    by_degree = [data["polynomial"][str(d)]["train_rmse"] for d in presentation.POLY_DEGREES]
    by_count = [data["harmonic"]["fits"][str(h)]["train_rmse"] for h in presentation.HARMONIC_COUNTS]
    assert by_degree == sorted(by_degree, reverse=True)
    assert by_count == sorted(by_count, reverse=True)


def test_harmonic_fits_are_the_package_model(co2, data):
    train, _ = train_test_split(co2)
    for count in presentation.HARMONIC_COUNTS:
        model = PolynomialSeasonalModel(2, count).fit(train["t"], train["average"])
        fit = data["harmonic"]["fits"][str(count)]
        assert fit["coef"] == pytest.approx(model.coef_)
        assert len(fit["amplitudes"]) == count
    # the annual wave dominates: each further harmonic is smaller than the first
    amplitudes = data["harmonic"]["fits"]["4"]["amplitudes"]
    assert amplitudes[0] == max(amplitudes) and 2 < amplitudes[0] < 4


def test_heatmap_holds_the_validation_errors_and_its_best_cell(data):
    grid, best = data["selection"], data["selection"]["best"]
    assert (best["degree"], best["harmonics"]) == (2, 4)
    row, column = grid["degrees"].index(best["degree"]), grid["harmonics"].index(best["harmonics"])
    assert grid["val_rmse"][row][column] == pytest.approx(best["val_rmse"])
    # the cell on the slide is the lowest of the whole search, not only of the part shown
    assert best["val_rmse"] == pytest.approx(min(min(r) for r in grid["val_rmse"]))
    assert grid["parameters"][row][column] == 2 + 1 + 2 * 4
    assert grid["searched"] == {"degrees": [1, 8], "harmonics": [0, 12]}


def test_selected_model_is_the_validated_choice(data):
    selected, four = data["selected"], data["harmonic"]["fits"]["4"]
    assert (selected["degree"], selected["harmonics"], selected["parameters"]) == (2, 4, 11)
    assert selected["coef"] == pytest.approx(four["coef"])
    assert selected["train_rmse"] == pytest.approx(four["train_rmse"])


def test_exponential_rate_minimises_the_plotted_objective(data):
    expo = data["exponential"]
    rates, errors = np.array(expo["scan"]["rate"]), np.array(expo["scan"]["rmse"])
    assert expo["bounds"] == [0.005, 0.1]
    assert rates[0] == pytest.approx(0.005) and rates[-1] == pytest.approx(0.1)
    assert expo["bounds"][0] < expo["rate"] < expo["bounds"][1]
    # the optimiser's r* sits at the bottom of the sampled curve, within one grid step
    assert abs(rates[errors.argmin()] - expo["rate"]) <= rates[1] - rates[0]
    assert expo["train_rmse"] <= errors.min() + 1e-6


def test_exponential_slide_form_is_the_fitted_trend(data):
    """c0 + a*exp(r t) on the slides equals the class's c0 + a*(exp(r t) - 1)."""
    expo = data["exponential"]
    t = np.linspace(0, 57, 20)
    as_fitted = expo["coef"][0] + expo["coef"][1] * np.expm1(expo["rate"] * t)
    on_slide = expo["c0"] + expo["a"] * np.exp(expo["rate"] * t)
    assert on_slide == pytest.approx(as_fitted)


def test_metrics_table_matches_the_evaluation(co2, data):
    train, test = train_test_split(co2)
    models, _ = evaluate.fit_models(train)
    rows = {row["model"]: row for row in data["metrics"]}
    assert list(rows) == [evaluate.LINEAR, evaluate.POLY, evaluate.EXPONENTIAL]
    for name, model in models.items():
        assert rows[name]["test_rmse"] == pytest.approx(evaluate.rmse(test["average"], model.predict(test["t"])))
        assert rows[name]["train_mae"] == pytest.approx(evaluate.mae(train["average"], model.predict(train["t"])))
        assert rows[name]["train_mae"] <= rows[name]["train_rmse"]  # always true of MAE and RMSE
        assert rows[name]["test_mae"] <= rows[name]["test_rmse"]


def test_decade_lines_cover_their_own_decades(co2, data):
    train, _ = train_test_split(co2)
    assert [d["decade"] for d in data["decades"]] == [1960, 1970, 1980, 1990, 2000, 2010]
    sixties = data["decades"][0]
    inside = train[train["year"].between(1960, 1969)]
    line = LinearModel().fit(inside["t"], inside["average"])
    assert sixties["slope"] == pytest.approx(line.slope_)
    assert sixties["t_start"] == pytest.approx(inside["t"].min())
    assert sixties["t_end"] == pytest.approx(inside["t"].max())
    assert data["decades"][-1]["months"] == 72  # 2010-2015 only
