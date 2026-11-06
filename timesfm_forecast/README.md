# TimesFM on the Keeling curve

An extra, kept apart from the rest of the project: how does a pretrained time-series foundation model,
[TimesFM 3.0](https://github.com/google-research/timesfm) from Google Research, do on the same task as the
project's three models?

Nothing in this folder is used by the main project, and nothing outside it was changed. The scripts only
*read* the project's loader and split (`../src/keeling`) so that every model is scored on exactly the same
months.

## What "zero-shot" means here

TimesFM is given the 694 months from March 1958 to December 2015 and asked for the next 240 months. Nothing is
fitted, tuned or trained on this data. It is the same cut as the other models: 2016 onward is held out, and its
forecast for 2016 to Sep 2026 is scored against what was measured.

## Files

| File | What it does |
|---|---|
| `timesfm_model.py` | The only place that touches TimesFM's API: load the checkpoint, forecast one series, check the output shapes |
| `run_forecast.py` | Forecast Jan 2016 to Dec 2035 from the 1958-2015 data; writes `results/timesfm_forecast.csv` and `timesfm_run.json` |
| `compare.py` | Score TimesFM against the project's three models; held-out table, error by year, 2035 forecast, figure |
| `tests/` | 14 tests, using a stand-in model so they need no download |
| `requirements.txt` | What this folder needs (PyTorch is large, so it has its own virtual environment) |
| `results/` | Tables and `timesfm_comparison.png` |

## Setup

```bash
python -m venv timesfm_forecast/.venv
```

```bash
PYYAML_FORCE_LIBYAML=0 timesfm_forecast/.venv/Scripts/python -m pip install -r timesfm_forecast/requirements.txt
```

`PYYAML_FORCE_LIBYAML=0` is only needed where PyYAML has no prebuilt wheel for your Python (it was the case for
Python 3.15 on Windows) and there is no C compiler: it builds PyYAML without its optional C extension.

The first run downloads the checkpoint, `google/timesfm-3.0-pytorch` (one file, 1.32 GB), into the Hugging Face
cache (`~/.cache/huggingface`), outside the repository. Loading it takes about two minutes on a CPU; the forecast
itself takes under a second.

## Run

```bash
timesfm_forecast/.venv/Scripts/python timesfm_forecast/run_forecast.py
```

```bash
timesfm_forecast/.venv/Scripts/python timesfm_forecast/compare.py --backtest
```

```bash
timesfm_forecast/.venv/Scripts/python -m pytest timesfm_forecast/tests
```

`--backtest` also runs TimesFM on the 20-years-ahead backtests the other models use for their 2035 error bar.
Once `results/timesfm_backtest.csv` exists, plain `compare.py` reuses it.

## Results

Held-out months, Jan 2016 to Sep 2026 (129 months), nothing fitted on them:

| Model | Parameters fitted here | Test RMSE (ppm) |
|---|---|---|
| Exponential + seasonal | 11 | 0.76 |
| Polynomial + seasonal | 11 | 2.20 |
| **TimesFM 3.0 (zero-shot)** | **0** (330M pretrained) | **3.22** |
| Linear | 2 | 13.74 |

- **It captures the seasonal cycle but not the acceleration.** TimesFM's yearly wiggle matches the measurements
  closely, but its trend is almost a straight line (about 1.96 ppm/yr over 2016-2035) while the real growth
  rate kept rising. Its mean error grows from 0.9 ppm in 2016 to about 5.4 ppm in 2025 and 2026.
- **2035 annual mean:** 440.7 ppm, with an error bar of 14.2 ppm from the same 20-year backtests as the other
  models. For comparison: linear 424.5, polynomial 449.7, exponential 456.1 ppm.
- **The backtests all err the same way:** for 20-year-ahead annual means, TimesFM was 8.9 to 14.6 ppm too low
  in every one of the 11 backtests.
- **Its 10-90% range is overconfident.** The measured month fell inside it only 29.5% of the time, where a
  calibrated range would give 80%.

## Caveats worth stating in the report

- **Possible leakage.** TimesFM was pretrained on a large public corpus. It may well have seen Mauna Loa CO2
  data, including the months after 2015. This cannot be ruled out, so its held-out score is, if anything,
  flattering to it; it still loses to the 11-parameter exponential model.
- **Default detrending.** TimesFM removes a straight line from its input by default (`use_linear_detrending`),
  which suits a series that is nearly straight but is part of why it does not bend upward.
- **Not a like-for-like comparison.** The project's models are fitted to this series with a handful of
  parameters; TimesFM has 330 million parameters fitted to other data. "Zero parameters fitted here" does not
  mean "zero parameters".
- **Licence.** The 3.0 weights are under Google's non-commercial licence, which is fine for coursework.
- **One horizon.** All numbers come from a single 240-month forecast, produced in one pass.
