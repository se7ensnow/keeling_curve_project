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
  evaluate.py        held-out errors, comparison table, rung 5 limits
  plots.py           figures
scripts/             one script per result
tests/               pytest tests (incl. synthetic data)
results/figures/     plots
results/params/      fitted parameters as JSON
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

_TODO_
