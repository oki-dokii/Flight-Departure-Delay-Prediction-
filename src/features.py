"""Feature engineering pipeline for Flight Departure Delay Prediction (Phase A & B).

All features are strictly computed relative to a pre-departure checkpoint cutoff:
    cutoff_utc = scheduled_departure_utc - checkpoint_delta

Feature Groups:
    G0: Static schedule, calendar, and route (10 predictors)
    G1: G0 + origin airport 3h congestion (12 predictors, Phase A audited set)
    G2: G1 + origin surface weather via backward asof join <= cutoff (18 predictors, Phase B primary)
    G3: G2 + inbound aircraft chain via Tail_Number (24 predictors, Decision D1 ablation)

Guarantees:
1. Positional index invariance: output X has identical index and row order to input df.
2. Anti-leakage: no target outcomes or events timestamped > cutoff_utc are used.
3. Target self-exclusion: airport congestion excludes target flight even if departed early.
4. Audited predictor subset: features derived from retrospective BTS Tail_Number are omitted
   from G0-G2 and isolated in G3 (or include_tail=True) for Decision D1 ablation.
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

WEATHER_STALENESS_LIMIT = pd.Timedelta(hours=6)

G0_PREDICTORS: List[Dict[str, str]] = [
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
]

CONGESTION_PREDICTORS: List[Dict[str, str]] = [
    {"name": "origin_departures_3h", "role": "numeric"},
    {"name": "origin_late_rate_3h", "role": "numeric"},
]

WEATHER_PREDICTORS: List[Dict[str, str]] = [
    {"name": "wx_temp_c", "role": "numeric"},
    {"name": "wx_wind_speed_kt", "role": "numeric"},
    {"name": "wx_wind_gust_kt", "role": "numeric"},
    {"name": "wx_visibility_mi", "role": "numeric"},
    {"name": "wx_precip_1h_mm", "role": "numeric"},
    {"name": "wx_weather_missing", "role": "indicator"},
]

EXPERIMENTAL_TAIL_PREDICTORS: List[Dict[str, str]] = [
    {"name": "inbound_missing_no_tail", "role": "indicator"},
    {"name": "inbound_first_leg_of_day", "role": "indicator"},
    {"name": "inbound_not_landed_by_T", "role": "indicator"},
    {"name": "inbound_arrival_delay_min", "role": "numeric"},
    {"name": "turnaround_buffer_min", "role": "numeric"},
    {"name": "inbound_prior_late", "role": "indicator"},
]

G1_PREDICTORS: List[Dict[str, str]] = G0_PREDICTORS + CONGESTION_PREDICTORS
G2_PREDICTORS: List[Dict[str, str]] = G1_PREDICTORS + WEATHER_PREDICTORS
G3_PREDICTORS: List[Dict[str, str]] = G2_PREDICTORS + EXPERIMENTAL_TAIL_PREDICTORS

# Backward-compatible aliases
AUDITED_PREDICTORS: List[Dict[str, str]] = G1_PREDICTORS
PHASE_B_AUDITED_PREDICTORS: List[Dict[str, str]] = G2_PREDICTORS

GROUP_PREDICTORS: Dict[str, List[Dict[str, str]]] = {
    "G0": G0_PREDICTORS,
    "G1": G1_PREDICTORS,
    "G2": G2_PREDICTORS,
    "G3": G3_PREDICTORS,
}


def build_features(
    df: pd.DataFrame,
    checkpoint_delta: pd.Timedelta,
    history_df: Optional[pd.DataFrame] = None,
    include_tail: bool = False,
    weather_obs: Optional[pd.DataFrame] = None,
    station_map: Optional[pd.DataFrame] = None,
    feature_group: Optional[str] = None,
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
        Legacy flag for Phase A compatibility. Ignored if feature_group is specified.
    weather_obs : Optional[pd.DataFrame]
        Pre-loaded ASOS weather observations (sorted by obs_time_utc) for G2/G3.
    station_map : Optional[pd.DataFrame]
        Mapping from origin_iata -> station_id. If None, assumes station == Origin.
    feature_group : Optional[str]
        One of 'G0', 'G1', 'G2', 'G3'. If None, defaults to 'G1' (or 'G1'+tail if include_tail).

    Returns
    -------
    pd.DataFrame
        Feature matrix X with identical index and row order as df.
    """
    if TIME not in df.columns:
        raise ValueError(f"Input DataFrame must contain {TIME}")
    if not isinstance(checkpoint_delta, pd.Timedelta):
        raise TypeError("checkpoint_delta must be a pd.Timedelta")

    if feature_group is not None and feature_group not in GROUP_PREDICTORS:
        raise ValueError(f"Unknown feature_group {feature_group!r}; choose from {list(GROUP_PREDICTORS)}")

    use_congestion = feature_group in (None, "G1", "G2", "G3")
    use_weather = feature_group in ("G2", "G3")
    use_tail = (feature_group == "G3") or (feature_group is None and include_tail)

    # Cutoff timestamp per flight
    cutoff = df[TIME] - checkpoint_delta

    # Initialize feature matrix preserving identical index and ordering
    X = pd.DataFrame(index=df.index)

    # =========================================================================
    # Group G0: Calendar, Schedule & Route (Static at booking, zero leak risk)
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
            X[feature_name] = df[col_name].astype("string")

    # =========================================================================
    # Group G1: Origin Airport Congestion & Delays by T (Strictly Backward-Looking)
    # =========================================================================
    if use_congestion:
        X = _add_airport_congestion_features(df, cutoff, X, history_df=history_df)

    # =========================================================================
    # Group G2: Origin Surface Weather by T (Strict Backward asof Join <= T)
    # =========================================================================
    if use_weather:
        X = _add_weather_features(df, cutoff, X, weather_obs=weather_obs, station_map=station_map)

    # =========================================================================
    # Group G3: Optional / Decision D1 Ablation Inbound Aircraft Chain (Tail_Number)
    # =========================================================================
    if use_tail:
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

    target_origins = df["Origin"].astype(str).values if "Origin" in df.columns else np.array([])
    has_target_obs = OBSERVED in df.columns and "LATE" in df.columns

    # Group full development set by Origin
    grouped_observed = valid_observed.groupby("Origin", observed=True)
    for origin, group in grouped_observed:
        match_idx = df.index[target_origins == str(origin)]
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

        # Self-exclusion check: if target flight is in history_df and departed early inside its own window, exclude it
        if has_target_obs:
            in_history = np.asarray(match_idx.isin(pool.index), dtype=bool)
            target_actuals = df.loc[match_idx, OBSERVED].values
            target_lates = df.loc[match_idx, "LATE"].values
            is_self = in_history & (target_actuals >= window_starts) & (target_actuals <= window_ends)
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


def _add_weather_features(
    df: pd.DataFrame,
    cutoff: pd.Series,
    X: pd.DataFrame,
    weather_obs: Optional[pd.DataFrame] = None,
    station_map: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Join latest origin surface weather observation at or before cutoff_utc.

    Strictly backward-looking via pd.merge_asof(..., direction='backward').
    Observations older than WEATHER_STALENESS_LIMIT (6 hours) or unmapped origins
    receive wx_weather_missing = 1 and NaN numeric measurements.
    """
    wx_cols = ["wx_temp_c", "wx_wind_speed_kt", "wx_wind_gust_kt", "wx_visibility_mi", "wx_precip_1h_mm"]
    if weather_obs is None or weather_obs.empty or "Origin" not in df.columns:
        for col in wx_cols:
            X[col] = np.float32(np.nan)
        X["wx_weather_missing"] = np.uint8(1)
        return X

    if station_map is not None and not station_map.empty:
        origin_to_station = dict(
            zip(
                station_map["origin_iata"].astype(str),
                station_map["station_id"].fillna("").astype(str),
            )
        )
        stations = df["Origin"].astype(str).map(origin_to_station).fillna("")
    else:
        stations = df["Origin"].astype(str).fillna("")

    left = pd.DataFrame({
        "_pos": np.arange(len(df), dtype=np.int64),
        "station": stations.values,
        "cutoff": pd.to_datetime(cutoff.values, utc=True),
    })

    valid_left_mask = left["station"].ne("") & left["cutoff"].notna()
    left_valid = left.loc[valid_left_mask].sort_values("cutoff", kind="mergesort")

    wx = weather_obs[["station", "obs_time_utc", *wx_cols]].copy()
    wx["station"] = wx["station"].astype(str)
    wx["obs_time_utc"] = pd.to_datetime(wx["obs_time_utc"], utc=True)
    if not wx["obs_time_utc"].is_monotonic_increasing:
        wx = wx.sort_values("obs_time_utc", kind="mergesort")

    joined = pd.merge_asof(
        left_valid,
        wx,
        left_on="cutoff",
        right_on="obs_time_utc",
        by="station",
        direction="backward",
        allow_exact_matches=True,
    )

    age = joined["cutoff"] - joined["obs_time_utc"]
    is_stale_or_absent = joined["obs_time_utc"].isna() | (age > WEATHER_STALENESS_LIMIT)

    # Restore original positional order into full-length arrays
    out_missing = np.ones(len(df), dtype=np.uint8)
    out_vals = {col: np.full(len(df), np.nan, dtype=np.float32) for col in wx_cols}

    pos = joined["_pos"].to_numpy()
    valid_wx = ~is_stale_or_absent.to_numpy()
    out_missing[pos[valid_wx]] = 0
    for col in wx_cols:
        vals = joined[col].to_numpy(dtype=np.float32)
        out_vals[col][pos[valid_wx]] = vals[valid_wx]

    for col in wx_cols:
        X[col] = out_vals[col]
    X["wx_weather_missing"] = out_missing
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
        X["inbound_prior_late"] = np.uint8(0)
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
        aligned_prev_late.where(state_landed_known, other=0).fillna(0).astype("uint8")
    )

    return X
