"""Fixed UTC windows; indices are positional and usable with pandas iloc."""

import numpy as np
import pandas as pd

TIME = "scheduled_departure_utc"
OBSERVED = "actual_departure_utc"
START = pd.Timestamp("2025-01-01", tz="UTC")
DEVELOPMENT_END = pd.Timestamp("2025-10-31", tz="UTC")
TEST_START = pd.Timestamp("2025-11-02", tz="UTC")
TEST_END = pd.Timestamp("2026-01-02", tz="UTC")
VALIDATION_WINDOWS = tuple(
    (pd.Timestamp(f"2025-{month:02d}-01", tz="UTC"),
     min(pd.Timestamp(f"2025-{month + 1:02d}-01", tz="UTC"), DEVELOPMENT_END))
    for month in range(7, 11)
)
GAP = pd.Timedelta(days=1)


def utc_column(frame, name):
    values = frame[name]
    if not isinstance(values.dtype, pd.DatetimeTZDtype) or values.isna().any():
        raise ValueError(f"{name} must contain nonmissing timezone-aware timestamps")
    return values.dt.tz_convert("UTC")


def get_folds(frame):
    """Return expanding train and monthly validation indices from development data."""
    scheduled = utc_column(frame, TIME)
    observed = utc_column(frame, OBSERVED)
    if not scheduled.between(START, DEVELOPMENT_END, inclusive="left").all():
        raise ValueError("get_folds accepts only the frozen development period")

    eligible = observed > scheduled - pd.Timedelta(hours=1)
    folds = []
    for start, end in VALIDATION_WINDOWS:
        cutoff = start - GAP
        train = np.flatnonzero(eligible & (scheduled < cutoff) & (observed < cutoff))
        validate = np.flatnonzero(eligible & scheduled.between(start, end, inclusive="left"))
        if not len(train) or not len(validate):
            raise ValueError(f"Empty train or validation set for {start:%Y-%m}")
        folds.append((train, validate))
    return folds


def get_test_split(frame):
    """Return final development and holdout indices. Call only at final evaluation."""
    scheduled = utc_column(frame, TIME)
    observed = utc_column(frame, OBSERVED)
    development = scheduled.between(START, DEVELOPMENT_END, inclusive="left")
    test = scheduled.between(TEST_START, TEST_END, inclusive="left")
    eligible = observed > scheduled - pd.Timedelta(hours=1)
    train = np.flatnonzero(eligible & development & (observed < DEVELOPMENT_END))
    holdout = np.flatnonzero(eligible & test)
    if not len(train) or not len(holdout):
        raise ValueError("Final evaluation requires both development and holdout data")
    return train, holdout
