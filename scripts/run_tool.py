"""The tool from the project brief: monthly record in; decomposition, growth rate and forecast out.

    python scripts/run_tool.py
    python scripts/run_tool.py --model polynomial_seasonal --target-year 2040
    python scripts/run_tool.py --data path/to/co2_mm_mlo.txt

Writes results/tables/tool_decomposition.csv (observed, trend, seasonal, residual for every month)
and results/tool_summary.json.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # so the script also runs without `pip install -e .`

from keeling import evaluate, tool  # noqa: E402
from keeling.data import RAW_PATH, load_co2_data  # noqa: E402

LABELS = {
    "linear": evaluate.LINEAR,
    "polynomial_seasonal": evaluate.POLY,
    "exponential_seasonal": evaluate.EXPONENTIAL,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=RAW_PATH, help="NOAA monthly file (default: data/raw)")
    parser.add_argument("--model", choices=LABELS, default="exponential_seasonal")
    parser.add_argument("--target-year", type=int, default=2035)
    args = parser.parse_args()

    co2 = load_co2_data(args.data)
    summary, decomposition, forecasts = tool.analyse(co2, LABELS[args.model], args.target_year)

    tables = ROOT / "results" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    decomposition.to_csv(tables / "tool_decomposition.csv", index=False, float_format="%.4f")
    (ROOT / "results" / "tool_summary.json").write_text(json.dumps(summary, indent=2))

    f = summary["forecast"]
    change = summary["measured_change_last_12_months_ppm"]
    print(f"Model: {LABELS[args.model]}, fitted on {summary['fitted_on']}")
    print(f"Latest month {summary['latest_month']}: measured {summary['latest_measured_ppm']:.2f} ppm")
    print(f"  trend {summary['model_trend_ppm']:.2f} ppm + seasonal {summary['model_seasonal_ppm']:+.2f} ppm")
    print(f"  seasonal cycle {summary['seasonal_peak_to_trough_ppm']:.1f} ppm peak to trough ({summary['seasonal_cycle_year']})")
    print(f"Growth rate now: {summary['growth_rate_now_ppm_per_year']:.2f} ppm/yr (model trend)")
    if change is not None:
        print(f"  measured change over the last 12 months: {change:+.2f} ppm")
    print(f"Forecast {f['year']} annual mean: {f['mean_ppm']:.1f} +/- {f['error_bar_ppm']:.1f} ppm "
          f"({f['low_ppm']:.1f} to {f['high_ppm']:.1f})")
    print(f"  error bar: {f['error_bar_is']}")
    print("All models:")
    print(forecasts.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    print(f"\nWrote results/tables/tool_decomposition.csv and results/tool_summary.json")


if __name__ == "__main__":
    main()
