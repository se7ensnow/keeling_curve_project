"""Fit the three models on 1958-2015 and export parameters and predictions.

Writes results/params/<model>.json and frontend/data.js (read by frontend/index.html).

    python scripts/export_frontend.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # so the script also runs without `pip install -e .`

from keeling.data import load_co2_data, time_index  # noqa: E402
from keeling.evaluate import (  # noqa: E402
    backtest_forecasts,
    failure_case,
    fit_models,
    forecast_year,
    model_factories,
    rate_by_cutoff,
    rmse,
)
from keeling.tool import ERROR_BAR_NOTE  # noqa: E402
from keeling.split import HELD_OUT_START, train_test_split  # noqa: E402

FORECAST_YEARS = 50
FAILURE_CUTOFF = 1985   # the failure case: models that only saw 1958-1985 ...
FAILURE_HORIZON = 20    # ... judged on the annual mean 20 years later (2005)


def r2(values):
    return [round(float(v), 2) for v in values]


def main():
    co2 = load_co2_data()
    train, test = train_test_split(co2)

    last = co2["date"].max()
    months = pd.date_range(co2["date"].min(), last + pd.DateOffset(years=FORECAST_YEARS), freq="MS")
    t = time_index(months.year, months.month).to_numpy(dtype=float)

    observed = co2.set_index("date")["average"].reindex(months)

    fitted, selection = fit_models(train)  # degree and harmonics are chosen on 2011-2015 validation data
    best = selection.iloc[0]
    backtest = backtest_forecasts(train, model_factories(int(best["degree"]), int(best["harmonics"])))
    summary_2035 = forecast_year(fitted, backtest, year=2035).set_index("Model")

    models = {}
    for label, model in fitted.items():
        key = model.name
        model.save_json(ROOT / "results" / "params" / f"{key}.json")

        models[key] = {
            "label": label,
            "params": model.params(),
            "train_rmse": rmse(train["average"], model.predict(train["t"])),
            "test_rmse": rmse(test["average"], model.predict(test["t"])),
            "pred": r2(model.predict(t)),
            "trend": r2(model.trend(t)),
            "growth": [round(float(v), 3) for v in model.growth_rate(t)],
            "forecast_2035": {
                # 3 decimals so the page's 1-decimal display rounds like the report (424.548 -> 424.5)
                "mean": round(float(summary_2035.loc[label, "2035 mean (ppm)"]), 3),
                "error_bar": round(float(summary_2035.loc[label, "Error bar (ppm)"]), 3),
                "low": round(float(summary_2035.loc[label, "Low (ppm)"]), 3),
                "high": round(float(summary_2035.loc[label, "High (ppm)"]), 3),
            },
        }
        print(f"{label:24s} train RMSE {models[key]['train_rmse']:.3f}  "
              f"held-out RMSE {models[key]['test_rmse']:.3f} ppm")

    # The honest failure case for the closing section: every model fitted on data to 1985 only,
    # its monthly predictions up to the latest measurement, and how the exponential's rate r
    # depends on the history it sees (same numbers as `run_results.py failure`).
    factories = model_factories(int(best["degree"]), int(best["harmonics"]))
    early, early_table = failure_case(train, factories, cutoff=FAILURE_CUTOFF, horizon=FAILURE_HORIZON)
    early_table = early_table.set_index("Model")
    shown = months <= last
    rates = rate_by_cutoff(train, int(best["harmonics"]))
    failure = {
        "cutoff": FAILURE_CUTOFF,
        "target_year": FAILURE_CUTOFF + FAILURE_HORIZON,
        "measured_mean": round(float(early_table["Measured mean (ppm)"].iloc[0]), 2),
        "models": {
            model.name: {
                "label": label,
                "target_mean": round(float(early_table.loc[label, "Model mean (ppm)"]), 2),
                "error": round(float(early_table.loc[label, "Model minus measured (ppm)"]), 2),
                "pred": r2(model.predict(t[shown])),
            }
            for label, model in early.items()
        },
        "rate_by_cutoff": [{"until": int(row["Train until"]), "rate": round(float(row["Rate r (1/yr)"]), 4)}
                           for _, row in rates.iterrows()],
    }

    data = {
        "failure": failure,
        "meta": {
            "source": "NOAA GML monthly mean CO2, Mauna Loa (co2_mm_mlo.txt)",
            "first_month": months[0].strftime("%Y-%m"),
            "last_observed": last.strftime("%Y-%m"),
            "held_out_start": HELD_OUT_START.strftime("%Y-%m"),
            "forecast_years": FORECAST_YEARS,
            "error_bar_note": ERROR_BAR_NOTE,
        },
        "months": [m.strftime("%Y-%m") for m in months],
        "observed": [None if pd.isna(v) else round(float(v), 2) for v in observed],
        "maunakea": co2.loc[co2["is_maunakea"], "date"].dt.strftime("%Y-%m").tolist(),
        "models": models,
    }

    out = ROOT / "frontend" / "data.js"
    out.write_text("window.CO2_DATA = " + json.dumps(data, separators=(",", ":")) + ";\n")
    print(f"Wrote {out.relative_to(ROOT)} ({len(months)} months)")


if __name__ == "__main__":
    main()
