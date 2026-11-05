"""One fit/predict class per model rung."""

from .rung1_linear import LinearModel
from .rung2_decade import fit_by_decade
from .rung3_harmonic import PolynomialSeasonalModel
from .rung4_exponential import ExponentialSeasonalModel

__all__ = [
    "LinearModel",
    "fit_by_decade",
    "PolynomialSeasonalModel",
    "ExponentialSeasonalModel",
]
