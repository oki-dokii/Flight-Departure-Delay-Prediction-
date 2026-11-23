"""Feature engineering pipeline for Flight Departure Delay Prediction (Phase A).

All features are strictly computed relative to a pre-departure checkpoint cutoff:
    cutoff_utc = scheduled_departure_utc - checkpoint_delta

Guarantees:
1. Positional index invariance: output X has identical index and row order to input df.
2. Anti-leakage: no target outcomes or events timestamped > cutoff_utc are used.
3. Target self-exclusion: airport congestion excludes target flight even if departed early.
4. Audited predictor subset: features derived from retrospective BTS Tail_Number are omitted
   from the primary audited predictor set unless live dispatch assignment evidence is provided.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from src.splits import OBSERVED, TIME

CHECKPOINTS: Dict[str, pd.Timedelta] = {
    "t360m": pd.Timedelta(minutes=360),
    "t60m": pd.Timedelta(minutes=60),
    "t6h": pd.Timedelta(hours=6),
    "t1h": pd.Timedelta(hours=1),
}

AUDITED_PREDICTORS: List[Dict[str, str]] = [
    {"name": "hour_sin", "role": "numeric"},
    {"name": "hour_cos", "role": "numeric"},
    {"name": "dow_sin", "role": "numeric"},
    {"name": "dow_cos", "role": "numeric"},
    {"name": "month", "role": "numeric"},
    {"name": "crs_elapsed_min", "role": "numeric"},
    {"name": "distance", "role": "numeric"},
    {"name": "carrier", "role": "categorical"},
    {"name": "origin", "role": "categorical"},
    {"name": "dest", "role": "categorical"},
    {"name": "origin_departures_3h", "role": "numeric"},
    {"name": "origin_late_rate_3h", "role": "numeric"},
]

EXPERIMENTAL_TAIL_PREDICTORS: List[Dict[str, str]] = [
    {"name": "inbound_missing_no_tail", "role": "indicator"},
    {"name": "inbound_first_leg_of_day", "role": "indicator"},
    {"name": "inbound_not_landed_by_T", "role": "indicator"},
    {"name": "inbound_arrival_delay_min", "role": "numeric"},
    {"name": "turnaround_buffer_min", "role": "numeric"},
    {"name": "inbound_prior_late", "role": "indicator"},
]


def build_features(
    df: pd.DataFrame,
    checkpoint_delta: pd.Timedelta,
    history_df: Optional[pd.DataFrame] = None,
    include_tail: bool = False,
) -> pd.DataFrame:
    """Build feature matrix X for a given prediction checkpoint delta.

    Parameters
    ----------
    df : pd.DataFrame
        Target flight records containing scheduled_departure_utc, actual_departure_utc,
        LATE, Origin, Dest, etc.
    checkpoint_delta : pd.Timedelta
        Time before scheduled departure at which prediction is performed (e.g. 360m or 60m).
    history_df : Optional[pd.DataFrame]
        Eligible development flight pool used to compute historical congestion and rotation
        history. If None, defaults to df itself.
    include_tail : bool, default False
        Whether to include experimental features derived from retrospective BTS Tail_Number.
        Omitted by default because BTS does not establish live assignment availability at T.

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
    # Group B: Origin Airport Congestion & Delays by T (Strictly Backward-Looking)
    # Target flight itself is strictly excluded even if departed unusually early.
    # =========================================================================
    X = _add_airport_congestion_features(df, cutoff, X, history_df=history_df)

    # =========================================================================
    # Group C: Optional / Experimental Inbound Aircraft Chain (Tail_Number)
    # =========================================================================
    if include_tail:
        X = _add_inbound_chain_features(df, cutoff, X, history_df=history_df)

    # Strict invariant assertion to safeguard evaluation harness iloc indexing
    if not X.index.equals(df.index):
        raise RuntimeError("Feature matrix index does not match input DataFrame index!")

    return X


def _add_airport_congestion_features(
    df: pd.DataFrame,
    cutoff: pd.Series,
    X: pd.DataFrame,
    history_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Compute origin airport departures and delay rate in [cutoff - 3h, cutoff].

    Uses efficient searchsorted logic per origin airport on actual departures.
    Strictly backward looking: only actual departures <= cutoff are counted.
    Guarantees target flight itself is excluded even if it departed unusually early.
    """
    dep_count_series = pd.Series(0, index=df.index, dtype="int32")
    late_rate_series = pd.Series(np.nan, index=df.index, dtype="float32")

    pool = df if history_df is None else history_df
    has_origin_and_observed = "Origin" in pool.columns and OBSERVED in pool.columns and "LATE" in pool.columns
    if not has_origin_and_observed:
        X["origin_departures_3h"] = dep_count_series
        X["origin_late_rate_3h"] = late_rate_series
        return X

    valid_observed = pool[[OBSERVED, "Origin", "LATE"]].dropna(subset=[OBSERVED, "Origin"])
    three_hours = pd.Timedelta(hours=3)

    target_origins = df["Origin"].values if "Origin" in df.columns else np.array([])
    has_target_obs = OBSERVED in df.columns and "LATE" in df.columns

    # Group full development set by Origin
    grouped_observed = valid_observed.groupby("Origin")
    for origin, group in grouped_observed:
        match_idx = df.index[target_origins == origin]
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

        # Self-exclusion check: if target flight departed early and fell inside its own window, exclude it
        if has_target_obs:
            target_actuals = df.loc[match_idx, OBSERVED].values
            target_lates = df.loc[match_idx, "LATE"].values
            is_self = (target_actuals >= window_starts) & (target_actuals <= window_ends)
            counts = counts - is_self.astype(int)
            late_sums = late_sums - (is_self & (target_lates == 1)).astype(int)

        rates = np.full(len(counts), np.nan, dtype="float32")
        valid_counts_mask = counts > 0
        rates[valid_counts_mask] = late_sums[valid_counts_mask] / counts[valid_counts_mask]

        dep_count_series.loc[match_idx] = counts.astype("int32")
        late_rate_series.loc[match_idx] = rates

    X["origin_departures_3h"] = dep_count_series
    X["origin_late_rate_3h"] = late_rate_series
    return X


def _add_inbound_chain_features(
    df: pd.DataFrame,
    cutoff: pd.Series,
    X: pd.DataFrame,
    history_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Compute previous leg status and arrival delay strictly bounded by cutoff."""
    tail_col = "Tail_Number"
    missing_tail_col = "tail_number_missing"
    sched_arr_col = "scheduled_arrival_utc"
    act_arr_col = "actual_arrival_utc"

    pool = df if history_df is None else history_df
    has_tail_data = tail_col in pool.columns and missing_tail_col in pool.columns
    if not has_tail_data:
        X["inbound_missing_no_tail"] = np.uint8(1)
        X["inbound_first_leg_of_day"] = np.uint8(0)
        X["inbound_not_landed_by_T"] = np.uint8(0)
        X["inbound_arrival_delay_min"] = np.float32(0.0)
        X["turnaround_buffer_min"] = np.float32(np.nan)
        X["inbound_prior_late"] = np.float32(0.0)
        return X

    tail_series = pool[tail_col].astype("string").str.strip().replace("", pd.NA)
    is_tail_missing = pool[missing_tail_col].astype(bool) | tail_series.isna()

    chain = pd.DataFrame({
        "tail": tail_series,
        "sched_dep": pool[TIME],
        "sched_arr": pool[sched_arr_col] if sched_arr_col in pool.columns else pool[TIME],
        "act_arr": pool[act_arr_col] if act_arr_col in pool.columns else pd.Series(pd.NaT, index=pool.index),
        "late": pool["LATE"] if "LATE" in pool.columns else pd.Series(0, index=pool.index),
    }, index=pool.index)

    valid_tail_mask = ~is_tail_missing
    valid_chain = chain.loc[valid_tail_mask].sort_values(["tail", "sched_dep"])

    grouped = valid_chain.groupby("tail", observed=True)
    prev_sched_dep = grouped["sched_dep"].shift(1)
    prev_sched_arr = grouped["sched_arr"].shift(1)
    prev_act_arr = grouped["act_arr"].shift(1)
    prev_late = grouped["late"].shift(1)

    target_tails = df[tail_col].astype("string").str.strip().replace("", pd.NA)
    target_tail_missing = df[missing_tail_col].astype(bool) | target_tails.isna()

    aligned_prev_sched_dep = prev_sched_dep.reindex(df.index)
    aligned_prev_sched_arr = prev_sched_arr.reindex(df.index)
    aligned_prev_act_arr = prev_act_arr.reindex(df.index)
    aligned_prev_late = prev_late.reindex(df.index)

    state_no_tail = target_tail_missing
    state_first_leg = (~state_no_tail) & aligned_prev_sched_dep.isna()
    state_not_landed = (
        (~state_no_tail)
        & (~state_first_leg)
        & (aligned_prev_act_arr.isna() | (aligned_prev_act_arr > cutoff))
    )
    state_landed_known = (~state_no_tail) & (~state_first_leg) & (~state_not_landed)

    X["inbound_missing_no_tail"] = state_no_tail.astype("uint8")
    X["inbound_first_leg_of_day"] = state_first_leg.astype("uint8")
    X["inbound_not_landed_by_T"] = state_not_landed.astype("uint8")

    raw_arrival_delay = (
        (aligned_prev_act_arr - aligned_prev_sched_arr).dt.total_seconds() / 60.0
    ).fillna(0.0)
    X["inbound_arrival_delay_min"] = (
        raw_arrival_delay.where(state_landed_known, other=0.0).astype("float32")
    )

    raw_turnaround = (
        (df[TIME] - aligned_prev_act_arr).dt.total_seconds() / 60.0
    )
    X["turnaround_buffer_min"] = (
        raw_turnaround.where(state_landed_known, other=np.nan).astype("float32")
    )

    X["inbound_prior_late"] = (
        aligned_prev_late.where(state_landed_known, other=0.0).fillna(0.0).astype("float32")
    )

    return X
