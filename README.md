# keeling_curve_project

DSM-101/102 project 4: *Linear or not?* using the Mauna Loa CO2 record (the Keeling curve).
We go from a linear baseline up to a truly non-linear trend and make a forecast for 2035 with an error bar.

## Layout

```
data/raw/            NOAA file, exactly as downloaded (read-only)
data/processed/      cleaned data written by the code
data/README.md       source, download date, notes
src/keeling/         Python package
  data.py            loader / cleaning
  split.py           fixed held-out split (train <= 2015-12)
  models/            one fit/predict class per rung (1-4)
  evaluate.py        rung 3 selection, held-out errors, comparison table, rung 5 limits, backtests
  plots.py           figures
  tool.py            the brief's tool: decomposition, growth rate, forecast with error bar
scripts/             one script per result
tests/               pytest tests (incl. synthetic data)
results/figures/     plots
results/params/      fitted parameters as JSON
frontend/            interactive explorer (index.html + generated data.js)
notebooks/           exploration only
report/              4-5 page report (PDF)
decisions.md         decision log
```

## Setup

```bash
pip install -e ".[dev]"
pytest
```

## Results: one command each

All commands run from the repo root. Figures go to `results/figures/`, tables to `results/tables/`, parameters (JSON) to `results/params/`.

| Result | Command |
|---|---|
| Rung 1: one straight line, residual plot | `python scripts/run_results.py rung1` |
| Rung 2: a line per decade, growth rate in ppm/yr | `python scripts/run_results.py rung2` |
| Rung 3: polynomial + seasonal, degree and harmonics chosen on 2011-2015 | `python scripts/run_results.py rung3` |
| Rung 4: exponential trend, rate r searched non-linearly | `python scripts/run_results.py rung4` |
| Rung 5: held-out table, errors by year, Maunakea months, 1 ppm horizon | `python scripts/run_results.py compare` |
| Rung 5: 20-year backtests and the 2035 forecast with an error bar | `python scripts/run_results.py forecast` |
| Rung 5: the failure case, models fitted on 1958-1985 only, and the exponential's r against history | `python scripts/run_results.py failure` |
| All of the above | `python scripts/run_results.py all` |
| The tool: trend/seasonal decomposition, growth rate now, forecast with error bar (`--model`, `--target-year`, `--data`) | `python scripts/run_tool.py` |
| Tests | `pytest` |
| Fit linear, polynomial + seasonal and exponential + seasonal on 1958-2015; write `results/params/*.json` and `frontend/data.js` | `python scripts/export_frontend.py` |
| Interactive explorer: pick a month and a model, see measured vs predicted CO2 and forecasts up to 50 years ahead | `python -m http.server 8000 --directory frontend`, then open http://localhost:8000 |

`frontend/index.html` also opens directly from disk (double-click), because the data is loaded as a script rather than fetched.
