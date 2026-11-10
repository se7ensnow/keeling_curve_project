"""Compute the numbers shown on the presentation page.

Writes frontend/presentation-data.js (read by frontend/presentation.html, next to data.js).

    python scripts/export_presentation.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # so the script also runs without `pip install -e .`

from keeling import presentation  # noqa: E402
from keeling.data import load_co2_data  # noqa: E402


def main():
    data = presentation.build(load_co2_data())

    out = ROOT / "frontend" / "presentation-data.js"
    out.write_text("window.CO2_PRESENTATION = " + json.dumps(data, separators=(",", ":")) + ";\n")

    best, expo = data["selection"]["best"], data["exponential"]
    print(f"Linear: {data['linear']['intercept']:.2f} ppm + {data['linear']['slope']:.3f} ppm/yr")
    for degree, fit in data["polynomial"].items():
        print(f"Polynomial degree {degree}: train RMSE {fit['train_rmse']:.3f} ppm")
    for count, fit in data["harmonic"]["fits"].items():
        print(f"Degree 2 + {count} harmonic(s): train RMSE {fit['train_rmse']:.3f} ppm")
    print(f"Selected on validation: degree {best['degree']}, {best['harmonics']} harmonics "
          f"({best['val_rmse']:.3f} ppm)")
    print(f"Exponential: r* = {expo['rate']:.5f} per year, train RMSE {expo['train_rmse']:.3f} ppm")
    print(f"Wrote {out.relative_to(ROOT)} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
