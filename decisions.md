# Decision log

Every choice that changes a result: decision, alternatives, evidence.
Numbers come from `python scripts/run_results.py all` (tables in `results/tables/`).

## 2026-11-02 - Fixed split and the column we model
- **Decision:** Fit on Mar 1958 - Dec 2015 (694 months), hold out Jan 2016 onward (129 months), and model the `average` column, not the deseasonalised one.
- **Alternatives:** A different cut year; modelling `deseasonalized` (it removes the cycle we are asked to model).
- **Evidence:** The split and the column are fixed by the project brief. `tests/test_data.py` checks that train and test do not overlap.

## 2026-11-02 - Scripps months are kept; "interpolated" means no measured days
- **Decision:** Keep all 194 Scripps months (Mar 1958 - Apr 1974). Flag a NOAA-era month as interpolated only when `ndays < 0`, which is Dec 1975 alone. Apr 1984 is kept as a measured month.
- **Alternatives:** Drop every row with a negative st.dev (this would delete the first 16 years of the record, and also Apr 1984); flag by `sdev < 0` as the notebook first did.
- **Evidence:** The file header says Scripps rows have no ndays/st.dev/unc, so they are all negative. Apr 1984 has `ndays = 2`, `sdev = -9.99`: measured, but too few days for a st.dev. Dec 1975 has `ndays = -1`. See `data/README.md`.

## 2026-11-02 - Maunakea months stay in, flagged, and are scored separately
- **Decision:** Keep Dec 2022 - Jul 2023 (8 months, Maunakea station) in the data, flag them, and report the held-out error for them separately.
- **Alternatives:** Drop them; fill them by interpolating the Mauna Loa months around the gap.
- **Evidence:** All 8 fall in the held-out period, so they cannot bias any fit. Held-out RMSE on those 8 months vs the other 121 (`rung5_maunakea.csv`): exponential 0.42 vs 0.78 ppm, polynomial 1.82 vs 2.23, linear 15.7 vs 13.6. The two curved models are not worse on them, so there is no case for dropping them.

## 2026-11-02 - Rung 3 size chosen on 2011-2015, not on training error
- **Decision:** Fit on 1958-2010, score on 2011-2015, pick the lowest validation RMSE from degree 1-8 and 0-12 harmonics: degree 2 with 4 harmonics (11 parameters, validation RMSE 0.504 ppm). Then refit on all of 1958-2015.
- **Alternatives:** Pick the lowest training error; pick fewer harmonics (2 harmonics is 0.511 ppm, 7 parameters).
- **Evidence:** Training error keeps falling with degree while validation error rises: degree 3 with 4 harmonics has train RMSE 0.725 (better than 0.756) but validation RMSE 1.318 (much worse than 0.504). Beyond 2 harmonics the gain is 0.007 ppm, within noise, so 4 is the formal minimum, not a strong preference (`rung3_selection_top15.csv`).

## 2026-11-02 - Rung 4 uses the same number of harmonics as rung 3
- **Decision:** `c0 + a(e^{rt} - 1)` plus 4 harmonics, so rungs 3 and 4 have 11 parameters each and differ only in the shape of the trend.
- **Alternatives:** Choose the harmonics separately for the exponential model.
- **Evidence:** Same parameter count makes the held-out comparison about trend shape alone. Held-out RMSE 0.762 (exponential) vs 2.202 ppm (polynomial).

## 2026-11-02 - Exponential trend written with expm1, and r searched in a bounded 1-D search
- **Decision:** Write the trend as `c0 + a(e^{rt} - 1)`; search only r in (0.005, 0.1) with `minimize_scalar`, solving c0, a and the harmonic coefficients by least squares inside every step.
- **Alternatives:** `c0 + a e^{rt}`; `scipy.optimize.curve_fit` on all parameters at once.
- **Evidence:** Same family of curves, but here c0 is the trend in Mar 1958 (313.5 ppm) rather than a number that nearly cancels `a` (254.6 ppm and 58.95 ppm in the other form). A scan of 200 rates over 0.001-0.12 (`rung4_rate_scan.png`) has one minimum at 0.0159, inside the bounds. The fitted r is 0.01590 per year.

## 2026-11-02 - Decades with under 60 months are skipped in rung 2
- **Decision:** Fit a line per calendar decade, but skip a decade with fewer than 60 months. That drops the "1950s".
- **Alternatives:** Keep it; use ten-year windows starting in Mar 1958.
- **Evidence:** The 1950s hold 22 months (under two yearly cycles) and gave a slope of -0.39 ppm/yr, which is the seasonal cycle, not CO2 falling. The 2010s keep 72 months (2010-2015). Slopes: 0.77, 1.19, 1.56, 1.53, 1.99, 2.10 ppm/yr for the 1960s to the 2010s (`rung2_decades.csv`).

## 2026-11-02 - The "1 ppm horizon" is read on annual mean error
- **Decision:** Answer rung 5 question 5 with the annual mean error. The monthly numbers are reported too, but not used for the answer.
- **Alternatives:** First single month with an error over 1 ppm; a 12-month rolling RMSE.
- **Evidence:** Monthly error is dominated by the yearly wiggle and year-to-year noise: all three models pass 1 ppm in the first 0-3 months, and the first full 12-month RMSE window is already above 1 ppm for all three (`rung5_horizon_monthly.csv`). On annual means the exponential never passes 1 ppm (largest 0.85 ppm), while the polynomial and linear models pass it in 2016 (`rung5_horizon_annual.csv`). 2026 holds only 9 months.

## 2026-11-02 - The 2035 error bar comes from 20-year-ahead backtests
- **Decision:** Error bar = 90th percentile of the absolute error of the annual-mean forecast, 20 years after training stops, for cutoffs 1985-1995 (11 per model).
- **Alternatives:** The least-squares prediction interval; the held-out RMSE.
- **Evidence:** A least-squares interval assumes independent errors, but monthly CO2 residuals are strongly correlated. The real forecast also looks about 20 years past its data (2015 to 2035). Caveat: the 11 backtests overlap heavily, and the exponential's errors shrink as it sees more data (21.5 ppm for a 1985 cutoff, 6.8 ppm for 1995), so its bar of 21.5 ppm is conservative (`rung5_backtest.csv`, `rung5_forecast_2035.csv`).

## 2026-11-05 - Front end is a static page fed by the Python models
- **Decision:** `scripts/export_frontend.py` fits the models and writes `frontend/data.js`; `frontend/index.html` only reads it.
- **Alternatives:** A Streamlit app (needs a server and Python at demo time); refitting the models in JavaScript (a second copy of the models that could drift from the report).
- **Evidence:** A static page can be hosted for free and opens from disk. The numbers in the page come from the same classes the tests check.

## 2026-11-05 - The failure case is the exponential model fitted on 1958-1985
- **Decision:** Show the exponential + seasonal model, fitted on data to 1985 only, as the honest failure: its 2005 annual mean is 21.5 ppm too high. Also show how its fitted r changes with the history it sees.
- **Alternatives:** The linear model's 2035 forecast (424.5 ppm, below the 426.07 ppm already measured in Sep 2026, so it is a poor example because it fails obviously); the polynomial model (9.7 ppm too high).
- **Evidence:** The best held-out model is the one that can fail worst. r is 0.047 with data to 1975, 0.037 to 1985, 0.020 to 1995 and 0.016 to 2015 (`rung5_rate_by_cutoff.csv`, `rung5_failure_case.png`). Over 27 years the record is nearly straight, so r is poorly pinned down and the exponential overshoots.

## 2026-11-05 - The growth rate "now" is reported two ways
- **Decision:** `run_tool.py` reports the model's trend growth at the latest month (2.79 ppm/yr) and the measured change over the last 12 months (+1.70 ppm), side by side.
- **Alternatives:** Report only the model's trend growth; report only the measured 12-month change.
- **Evidence:** They differ by 1.1 ppm/yr. The 12-month change matches the brief's own figure (426.07 - 424.37 = 1.70 ppm) and carries year-to-year variability; the model trend is smooth. One number alone would hide that.

## 2026-11-10 - Presentation page shows the fitted numbers in the form used on the slides
- **Decision:** `frontend/presentation.html` gets every number from `scripts/export_presentation.py`. Polynomial coefficients are shown per year (the slide formula is a sum of beta_k t^k), although `PolynomialModel` fits in decades (x = t/10). The exponential trend is shown as c0 + a exp(r t), although the class fits c0 + a (exp(r t) - 1).
- **Alternatives:** Show the coefficients exactly as stored by the classes; type the numbers into the page by hand.
- **Evidence:** Both rewrites describe the same curves, and `tests/test_presentation.py` checks that. Per year, degree 1 reproduces the rung 1 line (307.51 ppm + 1.514 ppm/yr), so the two slides agree. In the browser, the curves drawn from the exported coefficients match the Python predictions to the 0.01 ppm rounding of `data.js` for all 1423 months.

## 2026-11-10 - Residual slide zooms into 1983-1987
- **Decision:** The "zoom into several years" button on the residual slide shows January 1983 to December 1987, with the vertical axis unchanged.
- **Alternatives:** 2010-2015, as in the notebook; any other five years.
- **Evidence:** Of all five-year stretches of the training period, this is where the yearly mean of the straight line's residual changes least: it stays within 0.29 ppm (slope 0.01 ppm per year). So the slow curvature is flat here and only the yearly wave is left. In 2010-2014 the yearly mean climbs by 0.72 ppm per year, and the wave rides on that slope.
