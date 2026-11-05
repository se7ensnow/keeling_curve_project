"""Model tests, including recovery of known parameters from synthetic data."""

import json

import numpy as np
import pytest

from keeling.models import (
    ExponentialSeasonalModel,
    LinearModel,
    PolynomialSeasonalModel,
    fit_by_decade,
)

T = np.arange(0, 58, 1 / 12)  # monthly times, years since 1958-03


def seasonal(t, b1=2.0, c1=-0.8):
    return b1 * np.sin(2 * np.pi * t) + c1 * np.cos(2 * np.pi * t)


def test_linear_recovers_known_line():
    y = 310.0 + 1.5 * T
    model = LinearModel().fit(T, y)
    assert model.intercept_ == pytest.approx(310.0)
    assert model.slope_ == pytest.approx(1.5)
    assert model.predict([10.0]) == pytest.approx([325.0])


def test_polynomial_seasonal_recovers_known_coefficients():
    x = T / 10
    y = 315.0 + 8.0 * x + 1.2 * x**2 + seasonal(T)
    model = PolynomialSeasonalModel(degree=2, harmonics=1).fit(T, y)
    # order: x^0, x^1, x^2, sin, cos
    assert model.coef_ == pytest.approx([315.0, 8.0, 1.2, 2.0, -0.8], abs=1e-6)
    assert model.predict(T) == pytest.approx(y, abs=1e-6)


def test_polynomial_is_linear_in_its_parameters():
    """Doubling the data doubles the coefficients: the defining property of a linear model."""
    y = 315.0 + 8.0 * (T / 10) + seasonal(T)
    one = PolynomialSeasonalModel(2, 2).fit(T, y)
    two = PolynomialSeasonalModel(2, 2).fit(T, 2 * y)
    assert two.coef_ == pytest.approx(2 * one.coef_, rel=1e-8, abs=1e-8)


def test_exponential_recovers_known_rate_without_noise():
    rate = 0.02
    y = 313.0 + 55.0 * np.expm1(rate * T) + seasonal(T)
    model = ExponentialSeasonalModel(harmonics=1).fit(T, y)
    assert model.rate_ == pytest.approx(rate, abs=1e-4)
    assert model.coef_[0] == pytest.approx(313.0, abs=0.01)
    assert model.coef_[1] == pytest.approx(55.0, rel=0.01)


def test_exponential_recovers_known_rate_with_noise():
    rng = np.random.default_rng(0)
    rate = 0.016
    y = 313.0 + 59.0 * np.expm1(rate * T) + seasonal(T) + rng.normal(0, 0.3, len(T))
    model = ExponentialSeasonalModel(harmonics=1).fit(T, y)
    assert model.rate_ == pytest.approx(rate, abs=1e-3)


def test_exponential_rate_does_not_scale_with_the_data():
    """Doubling the data doubles c0 and a (linear) but leaves r unchanged (non-linear)."""
    y = 313.0 + 59.0 * np.expm1(0.016 * T) + seasonal(T)
    one = ExponentialSeasonalModel(harmonics=1).fit(T, y)
    two = ExponentialSeasonalModel(harmonics=1).fit(T, 2 * y)
    assert two.coef_ == pytest.approx(2 * one.coef_, rel=1e-4, abs=1e-4)
    assert two.rate_ == pytest.approx(one.rate_, abs=1e-5)
    assert one.n_params == len(one.coef_) + 1


@pytest.mark.parametrize("make", [
    lambda: LinearModel(),
    lambda: PolynomialSeasonalModel(3, 2),
    lambda: ExponentialSeasonalModel(2),
])
def test_growth_rate_is_the_derivative_of_the_trend(make):
    y = 313.0 + 59.0 * np.expm1(0.016 * T) + seasonal(T)
    model = make().fit(T, y)
    t = np.array([5.0, 20.0, 40.0])
    h = 1e-4
    numeric = (model.trend(t + h) - model.trend(t - h)) / (2 * h)
    assert np.asarray(model.growth_rate(t)) == pytest.approx(numeric, rel=1e-4)


def test_fit_by_decade_finds_each_slope():
    t = np.arange(0, 60, 1 / 12)
    year = 1958 + (t + 2 / 12)  # Mar 1958 start
    import pandas as pd

    slope = np.where(year < 1980, 1.0, 2.0)
    y = 310 + np.cumsum(slope / 12)
    df = pd.DataFrame({"year": np.floor(year).astype(int), "t": t, "average": y})
    table = fit_by_decade(df)
    assert table.loc[table["decade"] == 1960, "slope_ppm_per_year"].iloc[0] == pytest.approx(1.0, abs=1e-6)
    assert table.loc[table["decade"] == 1990, "slope_ppm_per_year"].iloc[0] == pytest.approx(2.0, abs=1e-6)


def test_parameters_are_saved_as_json(tmp_path):
    y = 310.0 + 1.5 * T
    model = LinearModel().fit(T, y)
    path = tmp_path / "params" / "linear.json"
    model.save_json(path)
    saved = json.loads(path.read_text())
    assert saved["model"] == "linear"
    assert saved["slope_ppm_per_year"] == pytest.approx(1.5)
