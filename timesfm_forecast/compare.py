"""Compare TimesFM with the project's three models, on the same held-out months and the same 2035 target.

    timesfm_forecast/.venv/Scripts/python timesfm_forecast/compare.py
    timesfm_forecast/.venv/Scripts/python timesfm_forecast/compare.py --backtest

Run run_forecast.py first. With --backtest TimesFM is also run on the 20-years-ahead backtests the
other models use for their 2035 error bar (11 forecasts, so it takes a while).

Writes tables and a figure to results/ next to this script.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))  # the project's models and split, used read-only
sys.path.insert(0, str(HERE))

import timesfm_model  # noqa: E402
from keeling import evaluate  # noqa: E402
from keeling.data import load_co2_data  # noqa: E402
from keeling.split import train_test_split  # noqa: E402

RESULTS = HERE / "results"
NAME = "TimesFM 3.0 (zero-shot)"
COLORS = {
    "Measured": "#0b0b0b",
    evaluate.LINEAR: "#2a78d6",
    evaluate.POLY: "#eb6834",
    evaluate.EXPONENTIAL: "#1baf7a",
    NAME: "#6250d6",
}


def save_table(df, name):
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / name, index=False, float_format="%.4f")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print()


def all_predictions(models, test, forecast):
    """Predictions for the held-out months from every model, as one frame."""
    frame = test[["date", "year", "average"]].merge(forecast[["date", "point"]], on="date", how="left")
    if frame["point"].isna().any():
        sys.exit("The TimesFM forecast does not cover every held-out month. Run run_forecast.py with --horizon 240.")
    out = {NAME: frame["point"].to_numpy()}
    for name, model in models.items():
        out[name] = model.predict(test["t"])
    return frame, out


def held_out_table(frame, predictions):
    rows = []
    for name, pred in predictions.items():
        error = frame["average"].to_numpy() - pred
        rows.append({
            "Model": name,
            "Parameters fitted here": 0 if name == NAME else {evaluate.LINEAR: 2}.get(name, 11),
            "Test MAE (ppm)": evaluate.mae(frame["average"], pred),
            "Test RMSE (ppm)": evaluate.rmse(frame["average"], pred),
            "Residual SD (ppm)": float(np.std(error, ddof=1)),
        })
    return pd.DataFrame(rows).sort_values("Test RMSE (ppm)", ignore_index=True)


def annual_table(frame, predictions):
    """Measured minus model, averaged over each held-out year."""
    rows = []
    for year, group in frame.groupby("year"):
        for name, pred in predictions.items():
            err = group["average"].to_numpy() - pd.Series(pred, index=frame.index).loc[group.index].to_numpy()
            rows.append({"Year": int(year), "Model": name, "Mean error (ppm)": float(err.mean())})
    return pd.DataFrame(rows).pivot(index="Year", columns="Model", values="Mean error (ppm)").reset_index()


def timesfm_backtest(model, train, cutoffs=range(1985, 1996), years=20):
    """Same backtests as evaluate.backtest_forecasts: data to each cutoff, annual mean `years` later."""
    rows = []
    for cutoff in cutoffs:
        history = train[train["year"] <= cutoff]
        target = train[train["year"] == cutoff + years]
        point, _ = timesfm_model.forecast(model, history["average"].to_numpy(dtype=float), years * 12)
        error = float(target["average"].mean() - point[-12:].mean())
        rows.append({"Model": NAME, "Train until": cutoff, "Target year": cutoff + years,
                     "Annual error (ppm)": error, "Absolute error (ppm)": abs(error)})
    return pd.DataFrame(rows)


def interval_coverage(frame, forecast, low="q10", high="q90"):
    """How often the measured months fall inside TimesFM's 10-90% range (80% would be well calibrated)."""
    merged = frame[["date", "average"]].merge(forecast[["date", low, high]], on="date")
    inside = (merged["average"] >= merged[low]) & (merged["average"] <= merged[high])
    return pd.DataFrame([{
        "Range": f"{low}-{high}",
        "Held-out months": len(merged),
        "Share inside": float(inside.mean()),
        "Share if well calibrated": 0.8,
        "Mean width (ppm)": float((merged[high] - merged[low]).mean()),
    }])


def plot(test, frame, predictions, forecast, co2):
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(11, 9))

    top.plot(frame["date"], frame["average"], color=COLORS["Measured"], lw=2, label="Measured")
    for name, pred in predictions.items():
        top.plot(frame["date"], pred, color=COLORS[name], lw=1.6, label=name)
    top.set(title="Held-out months (Jan 2016 onward): nothing was fitted on these", ylabel="CO2 (ppm)")
    top.legend(frameon=False)

    recent = co2[co2["year"] >= 1990]
    bottom.plot(recent["date"], recent["average"], color=COLORS["Measured"], lw=1.3, label="Measured")
    future = pd.date_range("2016-01-01", forecast["date"].max(), freq="MS")
    t = (future.year - 1958) + (future.month - 3) / 12
    for name, model in MODELS_FOR_PLOT.items():
        bottom.plot(future, model.predict(np.asarray(t)), color=COLORS[name], lw=1.3, alpha=0.9, label=name)
    bottom.plot(forecast["date"], forecast["point"], color=COLORS[NAME], lw=1.8, label=NAME)
    if {"q10", "q90"} <= set(forecast.columns):
        bottom.fill_between(forecast["date"], forecast["q10"], forecast["q90"], color=COLORS[NAME], alpha=0.18,
                            label="TimesFM 10-90% range (single months)")
    bottom.axvline(pd.Timestamp("2016-01-01"), color="gray", ls="--", alpha=0.6)
    bottom.set(title="Forecast to 2035, all from data up to Dec 2015", ylabel="CO2 (ppm)", xlabel="Year")
    bottom.legend(frameon=False, loc="upper left")
    for ax in (top, bottom):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.25)

    RESULTS.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(RESULTS / "timesfm_comparison.png", dpi=120)
    plt.close(fig)


MODELS_FOR_PLOT = {}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backtest", action="store_true", help="also run TimesFM on the 20-year backtests")
    parser.add_argument("--checkpoint", default=timesfm_model.DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    csv = RESULTS / "timesfm_forecast.csv"
    if not csv.exists():
        sys.exit("results/timesfm_forecast.csv is missing. Run run_forecast.py first.")
    forecast = pd.read_csv(csv, parse_dates=["date"])

    co2 = load_co2_data()
    train, test = train_test_split(co2)
    models, selection = evaluate.fit_models(train)
    MODELS_FOR_PLOT.update(models)

    frame, predictions = all_predictions(models, test, forecast)
    save_table(held_out_table(frame, predictions), "timesfm_heldout.csv")
    save_table(annual_table(frame, predictions), "timesfm_annual_mean_error.csv")
    save_table(interval_coverage(frame, forecast), "timesfm_interval_coverage.csv")

    # 2035: the project's models with their backtest error bars, and TimesFM
    best = selection.iloc[0]
    backtest = evaluate.backtest_forecasts(train, evaluate.model_factories(int(best["degree"]), int(best["harmonics"])))
    summary = evaluate.forecast_year(models, backtest, year=2035)
    mean_2035 = float(forecast.loc[forecast["date"].dt.year == 2035, "point"].mean())
    row = {"Model": NAME, "2035 mean (ppm)": mean_2035}
    saved = RESULTS / "timesfm_backtest.csv"
    tf_back = None
    if args.backtest:
        model = timesfm_model.load_model(args.checkpoint, args.device)
        tf_back = timesfm_backtest(model, train)
        save_table(tf_back, "timesfm_backtest.csv")
    elif saved.exists():
        tf_back = pd.read_csv(saved)  # from an earlier --backtest run, so the error bar is not lost
    if tf_back is not None:
        bar = float(tf_back["Absolute error (ppm)"].quantile(0.9))
        row.update({"Error bar (ppm)": bar, "Low (ppm)": mean_2035 - bar, "High (ppm)": mean_2035 + bar})
    summary = pd.concat([summary, pd.DataFrame([row])], ignore_index=True)
    save_table(summary, "timesfm_forecast_2035.csv")

    plot(test, frame, predictions, forecast, co2)
    print("Wrote tables and timesfm_comparison.png to timesfm_forecast/results/")


if __name__ == "__main__":
    main()
