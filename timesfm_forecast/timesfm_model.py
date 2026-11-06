"""A thin wrapper around TimesFM 3.0, so nothing else in this folder touches its API directly.

TimesFM is a pretrained time-series foundation model (https://github.com/google-research/timesfm).
Here it is used zero-shot: it is given a series and a horizon and nothing is trained or tuned.
The 3.0 weights are under Google's non-commercial licence, which is fine for coursework.
"""

from importlib import metadata

import numpy as np

DEFAULT_CHECKPOINT = "google/timesfm-3.0-pytorch"  # one 1.32 GB file, cached by Hugging Face
QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
QUANTILE_NAMES = [f"q{round(q * 100)}" for q in QUANTILES]
MIN_CONTEXT = 32  # one input patch


def clean_context(values):
    """The series as a 1-D float32 array, refusing anything TimesFM should not be handed."""
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 1:
        raise ValueError(f"expected one series (1-D), got shape {array.shape}")
    if len(array) < MIN_CONTEXT:
        raise ValueError(f"need at least {MIN_CONTEXT} points of context, got {len(array)}")
    if not np.isfinite(array).all():
        raise ValueError("the context contains NaN or infinite values")
    return array


def load_model(checkpoint=DEFAULT_CHECKPOINT, device="cpu"):
    """Load the checkpoint, downloading it on first use (about 1.3 GB)."""
    # imported here so the helpers in this file work, and are testable, without PyTorch
    from timesfm3 import ModelConfig, TimesFM3Evaluator

    return TimesFM3Evaluator(ModelConfig(checkpoint_path=checkpoint, per_core_batch_size=1, device=device))


def forecast(model, context, horizon):
    """Forecast one series: returns the point forecast (horizon,) and the quantiles (horizon, 9)."""
    context = clean_context(context)
    if horizon < 1:
        raise ValueError("horizon must be at least 1")

    outputs = model.predict_batch([context], horizon=horizon, return_quantiles=True, use_symmetric_averaging=True)
    output = next(iter(outputs))

    point = np.asarray(output.forecast, dtype=float).reshape(-1)
    quantiles = np.asarray(output.quantiles, dtype=float)
    quantiles = quantiles.reshape(-1, quantiles.shape[-1])  # (1, horizon, 9) or (horizon, 9)

    expected = (horizon, len(QUANTILES))
    if point.shape != (horizon,) or quantiles.shape != expected:
        raise RuntimeError(
            f"TimesFM returned forecast {point.shape} and quantiles {quantiles.shape}; expected ({horizon},) and {expected}")
    return point, quantiles


def versions():
    """The package versions behind a run, so a result can be traced back to them."""
    found = {}
    for name in ("timesfm", "torch", "numpy"):
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            found[name] = None
    return found
