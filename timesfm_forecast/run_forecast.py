"""Zero-shot TimesFM forecast of the Keeling curve.

    timesfm_forecast/.venv/Scripts/python timesfm_forecast/run_forecast.py
    timesfm_forecast/.venv/Scripts/python timesfm_forecast/run_forecast.py --horizon 240 --device cpu

The model only sees March 1958 - December 2015, the same training years as the project's own
models, and nothing is fitted: it is a pretrained model used as it is ("zero-shot"). Its forecast
starts in January 2016, so the held-out months and 2035 are both genuine forecasts.

Writes results/timesfm_forecast.csv and results/timesfm_run.json next to this script.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))  # the project's loader and split, used read-only
sys.path.insert(0, str(HERE))                 # this folder's TimesFM wrapper

import timesfm_model  # noqa: E402
from keeling.data import load_co2_data  # noqa: E402
from keeling.split import train_test_split  # noqa: E402

RESULTS = HERE / "results"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=timesfm_model.DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--horizon", type=int, default=240, help="months to forecast after Dec 2015 (240 = to Dec 2035)")
    args = parser.parse_args()

    co2 = load_co2_data()
    train, _ = train_test_split(co2)
    context = train["average"].to_numpy(dtype=float)

    started = time.time()
    model = timesfm_model.load_model(args.checkpoint, args.device)
    loaded = time.time()
    point, quantiles = timesfm_model.forecast(model, context, args.horizon)
    finished = time.time()

    dates = pd.date_range(train["date"].iloc[-1] + pd.DateOffset(months=1), periods=args.horizon, freq="MS")
    frame = pd.DataFrame({"date": dates, "point": point})
    for k, name in enumerate(timesfm_model.QUANTILE_NAMES):
        frame[name] = quantiles[:, k]

    RESULTS.mkdir(exist_ok=True)
    frame.to_csv(RESULTS / "timesfm_forecast.csv", index=False, float_format="%.4f")
    info = {
        "checkpoint": args.checkpoint,
        "device": args.device,
        "context": f"{train['date'].iloc[0]:%Y-%m} to {train['date'].iloc[-1]:%Y-%m}",
        "context_months": int(len(context)),
        "horizon_months": args.horizon,
        "forecast": f"{dates[0]:%Y-%m} to {dates[-1]:%Y-%m}",
        "fitted_on_the_data": False,
        "seconds_to_load": round(loaded - started, 1),
        "seconds_to_forecast": round(finished - loaded, 1),
        "versions": timesfm_model.versions(),
    }
    (RESULTS / "timesfm_run.json").write_text(json.dumps(info, indent=2))

    print(f"TimesFM ({args.checkpoint}) on {args.device}")
    print(f"  context  {info['context']} ({info['context_months']} months)")
    print(f"  forecast {info['forecast']} ({args.horizon} months) in {info['seconds_to_forecast']} s")
    print(f"  Dec 2035 point forecast: {frame['point'].iloc[-1]:.2f} ppm" if args.horizon >= 240 else "")
    print("Wrote timesfm_forecast/results/timesfm_forecast.csv and timesfm_run.json")


if __name__ == "__main__":
    main()
