"""Feature engineering pipeline for Flight Departure Delay Prediction (Phase A).

All features are strictly computed relative to a pre-departure checkpoint cutoff:
    cutoff_utc = scheduled_departure_utc - checkpoint_delta

Guarantees:
1. Positional index invariance: output X has identical index and row order to input df.
2. Anti-leakage: no target outcomes or events timestamped > cutoff_utc are used.
3. Explicit missingness: 4-state indicator flags for inbound aircraft status.
"""

from typing import Dict
import numpy as np
import pandas as pd

from src.splits import OBSERVED, TIME

CHECKPOINTS: Dict[str, pd.Timedelta] = {
    "t6h": pd.Timedelta(hours=6),
    "t1h": pd.Timedelta(hours=1),
}


def build_features(df: pd.DataFrame, checkpoint_delta: pd.Timedelta) -> pd.DataFrame:
    """Build feature matrix X for a given prediction checkpoint delta.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned flight records containing scheduled_departure_utc, actual_departure_utc,
        LATE, Tail_Number, tail_number_missing, Origin, Dest, etc.
    checkpoint_delta : pd.Timedelta
        Time before scheduled departure at which prediction is performed (e.g. 6h or 1h).

    Returns
    -------
    pd.DataFrame
        Feature matrix X with identical index and row order as df.
    """
    if TIME not in df.columns:
        raise ValueError(f"Input DataFrame must contain {TIME}")
    if not isinstance(checkpoint_delta, pd.Timedelta):
        raise TypeError("checkpoint_delta must be a pd.Timedelta")

    # Cutoff timestamp per flight
    cutoff = df[TIME] - checkpoint_delta

    # Initialize feature matrix preserving identical index and ordering
    X = pd.DataFrame(index=df.index)

    # =========================================================================
    # Group A: Calendar, Schedule & Route (Static at booking, zero leak risk)
    # =========================================================================
    sched_dt = df[TIME].dt
    hour_float = sched_dt.hour + sched_dt.minute / 60.0
    X["hour_sin"] = np.sin(2 * np.pi * hour_float / 24.0).astype("float32")
    X["hour_cos"] = np.cos(2 * np.pi * hour_float / 24.0).astype("float32")

    dow = sched_dt.dayofweek
    X["dow_sin"] = np.sin(2 * np.pi * dow / 7.0).astype("float32")
    X["dow_cos"] = np.cos(2 * np.pi * dow / 7.0).astype("float32")
    X["month"] = sched_dt.month.astype("uint8")

    if "CRSElapsedTime" in df.columns:
        X["crs_elapsed_min"] = pd.to_numeric(df["CRSElapsedTime"], errors="coerce").fillna(0.0).astype("float32")
    if "Distance" in df.columns:
        X["distance"] = pd.to_numeric(df["Distance"], errors="coerce").fillna(0.0).astype("float32")

    for col_name, feature_name in [("Reporting_Airline", "carrier"), ("Origin", "origin"), ("Dest", "dest")]:
        if col_name in df.columns:
            X[feature_name] = df[col_name].astype("category")

    # =========================================================================
    # Group B: Inbound Aircraft Chain (Tail_Number tracking)
    # =========================================================================
    X = _add_inbound_chain_features(df, cutoff, X)

    # =========================================================================
    # Group C: Origin Airport Congestion & Delays by T (Strictly Backward-Looking)
    # =========================================================================
    X = _add_airport_congestion_features(df, cutoff, X)

    # Strict invariant assertion to safeguard evaluation harness iloc indexing
    if not X.index.equals(df.index):
        raise RuntimeError("Feature matrix index does not match input DataFrame index!")

    return X


def _add_inbound_chain_features(df: pd.DataFrame, cutoff: pd.Series, X: pd.DataFrame) -> pd.DataFrame:
    """Compute previous leg status and arrival delay strictly bounded by cutoff."""
    tail_col = "Tail_Number"
    missing_tail_col = "tail_number_missing"
    sched_arr_col = "scheduled_arrival_utc"
    act_arr_col = "actual_arrival_utc"

    has_tail_data = tail_col in df.columns and missing_tail_col in df.columns
    if not has_tail_data:
        # Fallback if tail tracking columns are absent
        X["inbound_missing_no_tail"] = np.uint8(1)
        X["inbound_first_leg_of_day"] = np.uint8(0)
        X["inbound_not_landed_by_T"] = np.uint8(0)
        X["inbound_arrival_delay_min"] = np.float32(0.0)
        X["turnaround_buffer_min"] = np.float32(np.nan)
        X["inbound_prior_late"] = np.float32(0.0)
        return X

    # Create temporary sorted view for sequence tracking within tail
    tail_series = df[tail_col].astype("string").str.strip().replace("", pd.NA)
    is_tail_missing = df[missing_tail_col].astype(bool) | tail_series.isna()

    chain = pd.DataFrame({
        "tail": tail_series,
        "sched_dep": df[TIME],
        "sched_arr": df[sched_arr_col] if sched_arr_col in df.columns else df[TIME],
        "act_arr": df[act_arr_col] if act_arr_col in df.columns else pd.Series(pd.NaT, index=df.index),
        "late": df["LATE"] if "LATE" in df.columns else pd.Series(0, index=df.index),
    }, index=df.index)

    # Filter only rows with valid tail numbers to build chains
    valid_tail_mask = ~is_tail_missing
    valid_chain = chain.loc[valid_tail_mask].sort_values(["tail", "sched_dep"])

    # Shift by 1 within each tail group to inspect previous flight leg
    grouped = valid_chain.groupby("tail", observed=True)
    prev_sched_dep = grouped["sched_dep"].shift(1)
    prev_sched_arr = grouped["sched_arr"].shift(1)
    prev_act_arr = grouped["act_arr"].shift(1)
    prev_late = grouped["late"].shift(1)

    # Reindex back to original DataFrame index
    aligned_prev_sched_dep = prev_sched_dep.reindex(df.index)
    aligned_prev_sched_arr = prev_sched_arr.reindex(df.index)
    aligned_prev_act_arr = prev_act_arr.reindex(df.index)
    aligned_prev_late = prev_late.reindex(df.index)

    # 4-State Mutually Exclusive Indicator Logic
    # 1. No tail available
    state_no_tail = is_tail_missing

    # 2. First leg of calendar day for this tail (no preceding flight)
    state_first_leg = (~state_no_tail) & aligned_prev_sched_dep.isna()

    # 3. Inbound flight not landed by checkpoint cutoff:
    #    Either actual arrival timestamp > cutoff, OR actual arrival is NaT (e.g. diverted)
    state_not_landed = (
        (~state_no_tail)
        & (~state_first_leg)
        & (aligned_prev_act_arr.isna() | (aligned_prev_act_arr > cutoff))
    )

    # 4. Inbound successfully landed and observable before cutoff
    state_landed_known = (~state_no_tail) & (~state_first_leg) & (~state_not_landed)

    X["inbound_missing_no_tail"] = state_no_tail.astype("uint8")
    X["inbound_first_leg_of_day"] = state_first_leg.astype("uint8")
    X["inbound_not_landed_by_T"] = state_not_landed.astype("uint8")

    # Inbound arrival delay in minutes: (actual_arrival - scheduled_arrival)
    raw_arrival_delay = (
        (aligned_prev_act_arr - aligned_prev_sched_arr).dt.total_seconds() / 60.0
    ).fillna(0.0)
    X["inbound_arrival_delay_min"] = (
        raw_arrival_delay.where(state_landed_known, other=0.0).astype("float32")
    )

    # Turnaround buffer in minutes: (scheduled_departure of current - actual_arrival of prior)
    raw_turnaround = (
        (df[TIME] - aligned_prev_act_arr).dt.total_seconds() / 60.0
    )
    X["turnaround_buffer_min"] = (
        raw_turnaround.where(state_landed_known, other=np.nan).astype("float32")
    )

    # Prior leg late flag
    X["inbound_prior_late"] = (
        aligned_prev_late.where(state_landed_known, other=0.0).fillna(0.0).astype("float32")
    )

    return X


def _add_airport_congestion_features(df: pd.DataFrame, cutoff: pd.Series, X: pd.DataFrame) -> pd.DataFrame:
    """Compute origin airport departures and delay rate in [cutoff - 3h, cutoff].

    Uses efficient searchsorted logic per origin airport on actual departures.
    Strictly backward looking: only actual departures <= cutoff are counted.
    """
    dep_count_series = pd.Series(0, index=df.index, dtype="int32")
    late_rate_series = pd.Series(np.nan, index=df.index, dtype="float32")

    has_origin_and_observed = "Origin" in df.columns and OBSERVED in df.columns and "LATE" in df.columns
    if not has_origin_and_observed:
        X["origin_departures_3h"] = dep_count_series
        X["origin_late_rate_3h"] = late_rate_series
        return X

    # Prepare observed departure events
    valid_observed = df[[OBSERVED, "Origin", "LATE"]].dropna(subset=[OBSERVED, "Origin"])
    three_hours = pd.Timedelta(hours=3)

    # Group by Origin and perform binary search on sorted actual departure timestamps
    grouped_observed = valid_observed.groupby("Origin")
    for origin, group in grouped_observed:
        match_idx = df.index[df["Origin"] == origin]
        if len(match_idx) == 0:
            continue

        origin_cutoffs = cutoff.loc[match_idx]
        sorted_group = group.sort_values(OBSERVED)
        t_events = sorted_group[OBSERVED].values
        late_events = sorted_group["LATE"].values
        late_cumsum = np.concatenate([[0], np.cumsum(late_events)])

        # Search window [cutoff - 3h, cutoff]
        window_starts = (origin_cutoffs - three_hours).values
        window_ends = origin_cutoffs.values

        idx_left = np.searchsorted(t_events, window_starts, side="left")
        idx_right = np.searchsorted(t_events, window_ends, side="right")

        counts = idx_right - idx_left
        late_sums = late_cumsum[idx_right] - late_cumsum[idx_left]

        rates = np.full(len(counts), np.nan, dtype="float32")
        valid_counts_mask = counts > 0
        rates[valid_counts_mask] = late_sums[valid_counts_mask] / counts[valid_counts_mask]

        dep_count_series.loc[match_idx] = counts.astype("int32")
        late_rate_series.loc[match_idx] = rates

    X["origin_departures_3h"] = dep_count_series
    X["origin_late_rate_3h"] = late_rate_series
    return X
