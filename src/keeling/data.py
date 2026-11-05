"""Load and clean the NOAA monthly Mauna Loa CO2 record (data/raw/ is read-only)."""

from pathlib import Path

import pandas as pd

RAW_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "co2_mm_mlo.txt"

COLUMNS = [
    "year", "month", "decimal_date", "average", "deseasonalized",
    "ndays", "sdev", "unc",
]


def time_index(year, month):
    """Years since March 1958, the start of the record."""
    return (year - 1958) + (month - 3) / 12


def load_co2_data(path: str | Path = RAW_PATH) -> pd.DataFrame:
    """Read the whitespace-separated NOAA text file and add flags and the time index t."""
    df = pd.read_csv(path, comment="#", sep=r"\s+", header=None, names=COLUMNS)

    df["date"] = pd.to_datetime(df[["year", "month"]].assign(day=1))
    df["t"] = time_index(df["year"], df["month"])

    # Scripps months have no ndays/sdev/unc, so a negative value there does not mean interpolated.
    # In the NOAA era a month with no measured days has ndays = -1. Apr 1984 (2 days) also has
    # sdev = -9.99, but it is measured, so sdev < 0 alone would flag it wrongly.
    df["is_scripps"] = df["date"] <= "1974-04-01"
    df["is_interpolated"] = ~df["is_scripps"] & (df["ndays"] < 0)
    df["is_maunakea"] = df["date"].between("2022-12-01", "2023-07-01")

    return df
