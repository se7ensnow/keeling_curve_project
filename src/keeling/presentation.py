"""The numbers behind the presentation page (frontend/presentation.html).

Every value is computed from the NOAA record with the same model classes as the results, so the
page shows fitted coefficients and evaluated errors only: nothing on it is illustrative.
"""

import numpy as np

from . import evaluate
from .models import DecadeLinearModel, PolynomialModel, PolynomialSeasonalModel
from .models.rung2_decade import decade_of
from .split import train_test_split

POLY_DEGREES = (1, 2, 3, 4)       # the degree buttons on the polynomial slide
HARMONIC_COUNTS = (1, 2, 3, 4)    # the harmonic buttons, and the columns of the validation heatmap
HARMONIC_TREND_DEGREE = 2         # the harmonic slide keeps the same quadratic trend throughout
RATE_SCAN = np.linspace(0.005, 0.1, 191)  # where the objective is plotted; the optimiser is not a grid


def _ym(timestamp):
    return f"{timestamp:%Y-%m}"


def _floats(values):
    return [float(v) for v in values]


def _span(frame):
    return [_ym(frame["date"].iloc[0]), _ym(frame["date"].iloc[-1])]


def _amplitudes(wave_coefficients):
    """Amplitude of each harmonic from its (sin, cos) pair: a sin(u) + b cos(u) = A sin(u + phase)."""
    pairs = np.asarray(wave_coefficients, dtype=float).reshape(-1, 2)
    return _floats(np.hypot(pairs[:, 0], pairs[:, 1]))


def build(co2):
    """Everything the presentation page needs, as plain JSON-serialisable values."""
    train, test = train_test_split(co2)
    fit = train[train["date"] < evaluate.VALIDATION_START]
    validation = train[train["date"] >= evaluate.VALIDATION_START]
    t = train["t"].to_numpy(dtype=float)
    y = train["average"].to_numpy(dtype=float)

    models, selection = evaluate.fit_models(train)
    linear = models[evaluate.LINEAR]
    selected = models[evaluate.POLY]
    exponential = models[evaluate.EXPONENTIAL]

    meta = {
        "source": "NOAA GML monthly mean CO2, Mauna Loa (co2_mm_mlo.txt)",
        "time_origin": _ym(co2["date"].iloc[0]),  # t = 0; t counts years from here
        "record": _span(co2), "record_months": len(co2),
        "train": _span(train), "train_months": len(train),
        "fit": _span(fit), "fit_months": len(fit),
        "validation": _span(validation), "validation_months": len(validation),
        "held_out": _span(test), "held_out_months": len(test),
        "scripps_until": _ym(co2.loc[co2["is_scripps"], "date"].max()),
        "interpolated": [_ym(d) for d in co2.loc[co2["is_interpolated"], "date"]],
        "maunakea": [_ym(d) for d in co2.loc[co2["is_maunakea"], "date"]],
    }

    # Rung 2: one straight line per decade
    by_decade = DecadeLinearModel().fit(t, y)
    decades = []
    for decade, line in by_decade.lines_.items():
        inside = t[decade_of(t) == decade]
        decades.append({
            "decade": decade, "months": by_decade.months_[decade],
            "slope": float(line.slope_), "intercept": float(line.intercept_),
            "t_start": float(inside.min()), "t_end": float(inside.max()),
        })

    # Rung 3, trend only: PolynomialModel(degree=d) for each degree button
    polynomial = {}
    for degree in POLY_DEGREES:
        model = PolynomialModel(degree=degree).fit(train["t"], train["average"])
        polynomial[str(degree)] = {
            "coef": _floats(model.coef_),  # in x = t/10 (decades), exactly as the class fits it
            "coef_per_year": _floats(c / 10**k for k, c in enumerate(model.coef_)),  # the same curve in t
            "train_rmse": evaluate.rmse(y, model.predict(t)),
        }

    # Rung 3 with seasons: the same quadratic trend plus 1, 2, 3 or 4 harmonics
    harmonic = {}
    for count in HARMONIC_COUNTS:
        model = PolynomialSeasonalModel(HARMONIC_TREND_DEGREE, count).fit(train["t"], train["average"])
        harmonic[str(count)] = {
            "coef": _floats(model.coef_),
            "amplitudes": _amplitudes(model.coef_[HARMONIC_TREND_DEGREE + 1:]),
            "train_rmse": evaluate.rmse(y, model.predict(t)),
        }

    # Model selection: fitted before 2011, scored on 2011-2015; the heatmap shows 1-4 harmonics
    shown = selection[selection["harmonics"].isin(HARMONIC_COUNTS)]
    degrees = sorted(int(d) for d in shown["degree"].unique())

    def grid(column, cast=float):
        pivot = shown.pivot(index="degree", columns="harmonics", values=column)
        return [[cast(pivot.loc[d, h]) for h in HARMONIC_COUNTS] for d in degrees]

    best = selection.iloc[0]
    selection_out = {
        "degrees": degrees, "harmonics": list(HARMONIC_COUNTS),
        "val_rmse": grid("val_rmse"), "fit_rmse": grid("train_rmse"), "parameters": grid("parameters", int),
        "best": {"degree": int(best["degree"]), "harmonics": int(best["harmonics"]),
                 "val_rmse": float(best["val_rmse"])},
        "searched": {"degrees": [int(selection["degree"].min()), int(selection["degree"].max())],
                     "harmonics": [int(selection["harmonics"].min()), int(selection["harmonics"].max())]},
    }

    selected_out = {
        "degree": selected.degree, "harmonics": selected.harmonics, "parameters": selected.n_params,
        "coef": _floats(selected.coef_),
        "trend_per_year": _floats(c / 10**k for k, c in enumerate(selected.coef_[:selected.degree + 1])),
        "amplitudes": _amplitudes(selected.coef_[selected.degree + 1:]),
        "train_rmse": evaluate.rmse(y, selected.predict(t)),
    }

    # Rung 4: r is searched; for every r the other coefficients come from least squares
    scan_sse = exponential.sse_profile(t, y, RATE_SCAN)
    exponential_out = {
        "rate": float(exponential.rate_), "bounds": _floats(exponential.rate_bounds),
        "harmonics": exponential.harmonics, "parameters": exponential.n_params,
        "coef": _floats(exponential.coef_),  # c0 + a*(exp(r t) - 1) + waves, as the class fits it
        # the same trend written c0 + a*exp(r t), the form used on the slides
        "c0": float(exponential.coef_[0] - exponential.coef_[1]), "a": float(exponential.coef_[1]),
        "amplitudes": _amplitudes(exponential.coef_[2:]),
        "train_rmse": evaluate.rmse(y, exponential.predict(t)),
        "scan": {"rate": [round(float(r), 6) for r in RATE_SCAN],
                 "rmse": [round(float(np.sqrt(s / len(y))), 5) for s in scan_sse]},
    }

    metrics = []
    for name, model in models.items():
        fitted, predicted = model.predict(train["t"]), model.predict(test["t"])
        metrics.append({
            "key": model.name, "model": name, "parameters": model.n_params,
            "train_mae": evaluate.mae(train["average"], fitted),
            "train_rmse": evaluate.rmse(train["average"], fitted),
            "test_mae": evaluate.mae(test["average"], predicted),
            "test_rmse": evaluate.rmse(test["average"], predicted),
        })

    return {
        "meta": meta,
        "linear": {"intercept": float(linear.intercept_), "slope": float(linear.slope_),
                   "train_rmse": evaluate.rmse(y, linear.predict(t))},
        "decades": decades,
        "polynomial": polynomial,
        "harmonic": {"trend_degree": HARMONIC_TREND_DEGREE, "fits": harmonic},
        "selection": selection_out,
        "selected": selected_out,
        "exponential": exponential_out,
        "metrics": metrics,
    }
