"""Fixed train/held-out split: train Mar 1958 - Dec 2015, hold out Jan 2016 onward."""

import pandas as pd

HELD_OUT_START = pd.Timestamp("2016-01-01")


def train_test_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df["date"] < HELD_OUT_START].copy()
    test = df[df["date"] >= HELD_OUT_START].copy()
    return train, test
