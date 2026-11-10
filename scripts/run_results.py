"""One command per result.

    python scripts/run_results.py rung1      straight line, residuals
    python scripts/run_results.py rung2      one line per decade
    python scripts/run_results.py rung3      choose degree and harmonics on validation data
    python scripts/run_results.py rung4      exponential trend, search over r
    python scripts/run_results.py compare    held-out table, errors by year, Maunakea months, 1 ppm horizon
    python scripts/run_results.py failure    the failure case: models fitted on 1958-1985 only, and r against history
    python scripts/run_results.py forecast   backtests and the 2035 forecast with an error bar
    python scripts/run_results.py all        everything above

Figures go to results/figures/, tables to results/tables/, parameters to results/params/.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # so the script also runs without `pip install -e .`

from keeling import evaluate, plots  # noqa: E402
from keeling.data import load_co2_data  # noqa: E402
from keeling.models import DecadeLinearModel, LinearModel  # noqa: E402
from keeling.split import train_test_split  # noqa: E402

TABLES = ROOT / "results" / "tables"
PARAMS = ROOT / "results" / "params"


def save_table(df, name):
    TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES / name, index=False, float_format="%.4f")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print()


def rung1(co2, train, test, models, selection):
    model = LinearModel().fit(train["t"], train["average"])
    model.save_json(PARAMS / "linear.json")
    plots.baseline(train, model)
    resid = train["average"] - model.predict(train["t"])
    print(f"Rung 1: slope {model.slope_:.3f} ppm/yr, intercept {model.intercept_:.2f} ppm, "
          f"train RMSE {evaluate.rmse(train['average'], model.predict(train['t'])):.2f} ppm")
    print(f"Residuals run from {resid.min():.1f} to {resid.max():.1f} ppm\n")


def rung2(co2, train, test, models, selection):
    model = DecadeLinearModel().fit(train["t"], train["average"])
    model.save_json(PARAMS / "decade_linear.json")
    table = model.table()
    save_table(table, "rung2_decades.csv")
    plots.decades(train, table)


def rung3(co2, train, test, models, selection):
    save_table(selection.head(15), "rung3_selection_top15.csv")
    selection.to_csv(TABLES / "rung3_selection_all.csv", index=False, float_format="%.4f")
    plots.selection_heatmap(selection)
    models[evaluate.POLY].save_json(PARAMS / "polynomial_seasonal.json")
    best = selection.iloc[0]
    print(f"Rung 3: degree {int(best['degree'])}, {int(best['harmonics'])} harmonics, "
          f"{int(best['parameters'])} parameters, validation RMSE {best['val_rmse']:.3f} ppm\n")


def rung4(co2, train, test, models, selection):
    model = models[evaluate.EXPONENTIAL]
    model.save_json(PARAMS / "exponential_seasonal.json")
    plots.rate_scan(model, train, np.linspace(0.001, 0.12, 200))
    plots.trends(train, models)
    print(f"Rung 4: r = {model.rate_:.5f} per year, {model.n_params} parameters\n")


def compare(co2, train, test, models, selection):
    save_table(evaluate.comparison_table(models, train, test), "rung5_comparison.csv")
    save_table(evaluate.maunakea_table(models, test), "rung5_maunakea.csv")
    save_table(evaluate.horizon_table(models, test), "rung5_horizon_monthly.csv")
    annual = evaluate.annual_mean_error(models, test)
    save_table(annual.pivot(index="Year", columns="Model", values="Mean error (ppm)").reset_index(),
               "rung5_annual_mean_error.csv")
    save_table(evaluate.annual_horizon(annual), "rung5_horizon_annual.csv")
    yearly = evaluate.yearly_rmse(models, test)
    save_table(yearly.pivot(index="Year", columns="Model", values="RMSE (ppm)").reset_index(),
               "rung5_rmse_by_year.csv")
    plots.heldout(test, models)
    plots.horizon(yearly)


def failure(co2, train, test, models, selection):
    """The honest failure case: the exponential, fitted on 1958-1985, overshoots 2005 by about 21 ppm."""
    best = selection.iloc[0]
    harmonics = int(best["harmonics"])
    factories = evaluate.model_factories(int(best["degree"]), harmonics)
    fitted, table = evaluate.failure_case(train, factories, cutoff=1985)
    rates = evaluate.rate_by_cutoff(train, harmonics)
    save_table(table, "rung5_failure_case.csv")
    save_table(rates, "rung5_rate_by_cutoff.csv")
    plots.failure_case(train, fitted, table, rates, cutoff=1985, target_year=2005)


def forecast(co2, train, test, models, selection):
    best = selection.iloc[0]
    factories = evaluate.model_factories(int(best["degree"]), int(best["harmonics"]))
    backtest = evaluate.backtest_forecasts(train, factories)
    save_table(backtest, "rung5_backtest.csv")
    summary = evaluate.forecast_year(models, backtest, year=2035)
    save_table(summary, "rung5_forecast_2035.csv")
    plots.forecast(co2, train, models, summary, year=2035)


COMMANDS = {"rung1": rung1, "rung2": rung2, "rung3": rung3, "rung4": rung4,
            "compare": compare, "failure": failure, "forecast": forecast}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("result", choices=[*COMMANDS, "all"])
    args = parser.parse_args()

    co2 = load_co2_data()
    train, test = train_test_split(co2)
    models, selection = evaluate.fit_models(train)

    for name in COMMANDS if args.result == "all" else [args.result]:
        print(f"== {name} ==")
        COMMANDS[name](co2, train, test, models, selection)


if __name__ == "__main__":
    main()
