"""Tests for the data loader, the time index and the fixed split."""

import pandas as pd
import pytest

from keeling.data import RAW_PATH, load_co2_data, time_index
from keeling.split import train_test_split


@pytest.fixture(scope="module")
def co2():
    return load_co2_data()


def test_loader_reads_first_and_last_rows(co2):
    first = co2.iloc[0]
    assert (first["year"], first["month"]) == (1958, 3)
    assert first["average"] == pytest.approx(315.71)
    assert co2["date"].is_monotonic_increasing
    assert co2["date"].iloc[-1] >= pd.Timestamp("2026-09-01")


def test_months_are_consecutive_and_unique(co2):
    gaps = co2["date"].diff().dropna().dt.days
    assert co2["date"].is_unique
    assert gaps.between(28, 31).all()


def test_no_comment_lines_or_missing_values_leak_into_the_data(co2):
    assert len(co2.columns) >= 8
    assert co2[["average", "deseasonalized", "t"]].notna().all().all()
    assert (co2["average"] > 300).all()


def test_time_index_starts_at_zero_in_march_1958():
    assert time_index(1958, 3) == 0
    assert time_index(1959, 3) == pytest.approx(1.0)
    assert time_index(1958, 9) == pytest.approx(0.5)


def test_flags_mark_the_documented_months(co2):
    assert co2["is_scripps"].sum() == 194  # Mar 1958 - Apr 1974
    assert co2["is_maunakea"].sum() == 8  # Dec 2022 - Jul 2023
    maunakea = co2.loc[co2["is_maunakea"], "date"]
    assert maunakea.min() == pd.Timestamp("2022-12-01")
    assert maunakea.max() == pd.Timestamp("2023-07-01")
    # Scripps months carry no st.dev, so they must never count as interpolated
    assert not (co2["is_scripps"] & co2["is_interpolated"]).any()
    # Only Dec 1975 had no measured days; Apr 1984 has 2 days and a -9.99 st.dev but is measured
    assert co2.loc[co2["is_interpolated"], "date"].tolist() == [pd.Timestamp("1975-12-01")]


def test_split_is_fixed_and_does_not_overlap(co2):
    train, test = train_test_split(co2)
    assert train["date"].max() == pd.Timestamp("2015-12-01")
    assert test["date"].min() == pd.Timestamp("2016-01-01")
    assert len(train) + len(test) == len(co2)
    assert set(train["date"]).isdisjoint(test["date"])


def test_loading_never_modifies_the_raw_file():
    before = RAW_PATH.read_bytes()
    load_co2_data()
    assert RAW_PATH.read_bytes() == before
