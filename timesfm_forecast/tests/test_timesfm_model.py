"""Tests for the TimesFM wrapper, using a stand-in model so nothing is downloaded."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

FOLDER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FOLDER))

import timesfm_model  # noqa: E402


class StandInModel:
    """Records what it was asked and answers with a fixed ramp, in the shape TimesFM uses."""

    def __init__(self, quantile_shape="flat"):
        self.quantile_shape = quantile_shape
        self.calls = []

    def predict_batch(self, contexts, horizon, return_quantiles, use_symmetric_averaging):
        self.calls.append({"contexts": contexts, "horizon": horizon,
                           "return_quantiles": return_quantiles, "symmetric": use_symmetric_averaging})
        point = np.arange(horizon, dtype=float)
        quantiles = np.stack([point + k for k in range(9)], axis=-1)  # (horizon, 9)
        if self.quantile_shape == "batched":
            quantiles = quantiles[None]  # (1, horizon, 9)
        yield SimpleNamespace(ts_id=None, forecast=point, quantiles=quantiles)


def test_forecast_returns_point_and_nine_quantiles():
    model = StandInModel()
    point, quantiles = timesfm_model.forecast(model, np.linspace(315, 400, 100), horizon=24)
    assert point.shape == (24,)
    assert quantiles.shape == (24, 9)
    assert list(timesfm_model.QUANTILE_NAMES) == ["q10", "q20", "q30", "q40", "q50", "q60", "q70", "q80", "q90"]


def test_forecast_asks_for_quantiles_and_passes_one_float32_series():
    model = StandInModel()
    timesfm_model.forecast(model, list(range(50)), horizon=12)
    call = model.calls[0]
    assert call["horizon"] == 12 and call["return_quantiles"] is True
    assert len(call["contexts"]) == 1 and call["contexts"][0].dtype == np.float32


def test_a_batched_quantile_array_is_flattened_to_horizon_by_nine():
    point, quantiles = timesfm_model.forecast(StandInModel("batched"), np.arange(64.0), horizon=10)
    assert quantiles.shape == (10, 9)


def test_an_unexpected_output_shape_fails_loudly():
    class Wrong(StandInModel):
        def predict_batch(self, contexts, horizon, return_quantiles, use_symmetric_averaging):
            yield SimpleNamespace(forecast=np.zeros(horizon + 3), quantiles=np.zeros((horizon, 9)))

    with pytest.raises(RuntimeError, match="expected"):
        timesfm_model.forecast(Wrong(), np.arange(64.0), horizon=10)


@pytest.mark.parametrize("bad, message", [
    (np.arange(10.0), "at least"),
    (np.array([1.0] * 40 + [np.nan]), "NaN"),
    (np.ones((2, 50)), "1-D"),
])
def test_clean_context_refuses_series_timesfm_should_not_get(bad, message):
    with pytest.raises(ValueError, match=message):
        timesfm_model.clean_context(bad)


def test_horizon_must_be_positive():
    with pytest.raises(ValueError):
        timesfm_model.forecast(StandInModel(), np.arange(64.0), horizon=0)


def test_versions_names_the_packages_behind_a_run():
    found = timesfm_model.versions()
    assert set(found) == {"timesfm", "torch", "numpy"}
    assert found["numpy"] is not None
