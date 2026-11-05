# Data

## Source

NOAA Global Monitoring Laboratory, *Trends in Atmospheric Carbon Dioxide*, monthly mean CO2 at Mauna Loa.

- URL: https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.txt
- File in `raw/`: `co2_mm_mlo.txt` (header says: File Creation Fri Oct 9 13:08:07 2026; last month Sep 2026)
- Downloaded on: 2026-11-02
- SHA-256: `0ce4e7749217b4af2cca8e0bdf263856ad59b0a2946d4e98c7b22ccfbff21cad`
- Citation: the file header asks for "relevant citation text" but gives none. The NOAA trends page asks for:
  Dr. Xin Lan, NOAA/GML (gml.noaa.gov/ccgg/trends/) and Dr. Ralph Keeling, Scripps Institution of Oceanography (scrippsco2.ucsd.edu/). _Check against the website before the report._

## Rules

- `raw/` is read-only: files are kept exactly as downloaded, and code never writes there.
- Cleaned / derived data goes to `processed/`.

## Notes

- Units: ppm (mole fraction in dry air).
- Models use the monthly average column, not the deseasonalised one (that is only used as a check).
- Format: whitespace-separated, lines starting with `#` are comments. Columns: year, month, decimal date, monthly average, deseasonalised, #days, st.dev of days, unc. of monthly mean.
- Missing months: there are no missing values; missing months were already interpolated by NOAA and are marked by a negative st.dev / uncertainty (`-9.99` / `-0.99`).
  - Mar 1958 - Apr 1974 (Scripps data): no #days / st.dev / unc. available, so all are negative. These are not necessarily interpolated.
  - NOAA era (May 1974 on): only Dec 1975 is interpolated (`ndays = -1`, no measured days).
  - Apr 1984 also has st.dev `-9.99`, but it has 2 measured days, so it is a measured month with too few days for a st.dev. It is kept as measured.
- Values are corrected to the centre of the month using the average seasonal cycle.
- Mauna Loa eruption gap: Dec 2022 - Jul 2023 values come from Maunakea. Treatment: see `decisions.md`.
