"""Plotting helpers; figures are written to results/figures/."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

FIGURES = Path(__file__).resolve().parents[2] / "results" / "figures"

COLORS = {
    "Observed": "#0b0b0b",
    "Linear": "#2a78d6",
    "Polynomial + seasonal": "#eb6834",
    "Exponential + seasonal": "#1baf7a",
}

plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "legend.frameon": False,
})


def _save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def baseline(train, model):
    """Rung 1: the straight line, and its residuals against time."""
    pred = model.predict(train["t"])
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    top.scatter(train["date"], train["average"], s=6, alpha=0.5, color=COLORS["Observed"], label="Measured")
    top.plot(train["date"], pred, color=COLORS["Linear"], lw=2,
             label=f"Line: {model.slope_:.2f} ppm/yr")
    top.set(ylabel="CO2 (ppm)", title="Rung 1: one straight line, 1958-2015")
    top.legend()
    bottom.scatter(train["date"], train["average"] - pred, s=6, alpha=0.6, color=COLORS["Linear"])
    bottom.axhline(0, color="gray", ls="--")
    bottom.set(xlabel="Year", ylabel="Residual (ppm)", title="Residuals: a curve plus a yearly wave")
    return _save(fig, "rung1_linear_baseline.png")


def decades(train, table):
    """Rung 2: growth rate (slope) per decade."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    labels = [f"{d}s\n({m} mo)" for d, m in zip(table["decade"], table["months"])]
    ax.bar(labels, table["slope_ppm_per_year"], color=COLORS["Linear"])
    ax.set(ylabel="Slope (ppm/yr)", title="Rung 2: growth rate per decade")
    return _save(fig, "rung2_decade_slopes.png")


def selection_heatmap(selection):
    """Rung 3: validation RMSE over polynomial degree and number of harmonics."""
    pivot = selection.pivot(index="degree", columns="harmonics", values="val_rmse")
    fig, ax = plt.subplots(figsize=(9, 5))
    image = ax.imshow(pivot.to_numpy(), cmap="YlGnBu_r", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns)
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.grid(False)
    best = selection.iloc[0]
    ax.set(xlabel="Harmonics", ylabel="Polynomial degree",
           title=f"Rung 3: validation RMSE (ppm), best = degree {int(best['degree'])}, "
                 f"{int(best['harmonics'])} harmonics")
    fig.colorbar(image, ax=ax, label="Validation RMSE (ppm)")
    return _save(fig, "rung3_selection_heatmap.png")


def rate_scan(model, train, rates):
    """Rung 4: the sum of squared errors as a function of the non-linear parameter r."""
    sse = model.sse_profile(train["t"], train["average"], rates)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(rates, sse, color=COLORS["Exponential + seasonal"], lw=2)
    ax.axvline(model.rate_, color="crimson", ls="--", label=f"Best r = {model.rate_:.4f} per year")
    ax.set(xlabel="r (1/yr)", ylabel="Sum of squared errors",
           title="Rung 4: the rate r searched; the other coefficients solved by least squares each time")
    ax.legend()
    return _save(fig, "rung4_rate_scan.png")


def trends(train, models):
    """The fitted trends (seasonal cycle removed) of the two curved models."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for name in ("Polynomial + seasonal", "Exponential + seasonal"):
        ax.plot(train["date"], models[name].trend(train["t"]), color=COLORS[name], lw=2, label=name)
    ax.set(xlabel="Year", ylabel="Trend (ppm)", title="Trends of the two curved models")
    ax.legend()
    return _save(fig, "rung4_trends.png")


def heldout(test, models):
    """Held-out months: measured values against all models, plus their errors."""
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    top.plot(test["date"], test["average"], color=COLORS["Observed"], lw=2, label="Measured")
    for name, model in models.items():
        pred = model.predict(test["t"])
        top.plot(test["date"], pred, color=COLORS[name], lw=1.6, label=name)
        bottom.plot(test["date"], test["average"] - pred, color=COLORS[name], lw=1.6, label=name)
    bottom.axhline(0, color="gray", ls="--")
    top.set(ylabel="CO2 (ppm)", title="Held-out months (Jan 2016 onward)")
    bottom.set(xlabel="Year", ylabel="Measured - model (ppm)")
    top.legend()
    return _save(fig, "rung5_heldout.png")


def forecast(co2, train, models, summary, year=2035):
    """Rung 5: all models out to `year`, with the error bars on the annual mean."""
    future = pd.date_range("2016-01-01", f"{year}-12-01", freq="MS")
    t = (future.year - 1958) + (future.month - 3) / 12
    fig, ax = plt.subplots(figsize=(11, 5.5))
    recent = co2[co2["year"] >= 1990]
    ax.plot(recent["date"], recent["average"], color=COLORS["Observed"], lw=1.4, label="Measured")
    for name, model in models.items():
        ax.plot(future, model.predict(np.asarray(t)), color=COLORS[name], lw=1.8, label=name)
    mid = pd.Timestamp(f"{year}-07-01")
    for i, row in summary.iterrows():
        ax.errorbar(mid + pd.Timedelta(days=40 * (i - 1)), row[f"{year} mean (ppm)"], yerr=row["Error bar (ppm)"],
                    fmt="o", color=COLORS[row["Model"]], capsize=4)
    ax.axvline(pd.Timestamp("2016-01-01"), color="gray", ls="--", alpha=0.6)
    ax.set(xlabel="Year", ylabel="CO2 (ppm)", title=f"Forecast to {year}: fitted on 1958-2015 only")
    ax.legend()
    return _save(fig, "rung5_forecast.png")


def horizon(yearly):
    """Rung 5: held-out RMSE for each year, against the 1 ppm line."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for name, group in yearly.groupby("Model"):
        ax.plot(group["Year"], group["RMSE (ppm)"], marker="o", color=COLORS[name], label=name)
    ax.axhline(1, color="crimson", ls="--", label="1 ppm")
    ax.set(xlabel="Year", ylabel="RMSE (ppm)", title="Held-out RMSE by year")
    ax.legend()
    return _save(fig, "rung5_rmse_by_year.png")
