"""Generate sample feature matrices for P4 (Classical) and P5 (Boosting) models.

Ensures both checkpoints (t6h, t1h) are generated and validated against the evaluation harness.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from src.evaluate import evaluate
from src.features import CHECKPOINTS, build_features
from src.splits import OBSERVED, TIME

SAMPLE_DIR = Path("data/sample")
DEV_PARQUET = Path("data/processed/flights_development.parquet")


def generate_synthetic_development_cohort(n_records: int = 1200) -> pd.DataFrame:
    """Generate realistic synthetic flight development data covering the 4 folds."""
    np.random.seed(42)
    # Generate timestamps spanning Jan 2025 to Oct 2025 (within frozen development period)
    start_ts = pd.Timestamp("2025-01-01 06:00", tz="UTC").value
    end_ts = pd.Timestamp("2025-10-30 22:00", tz="UTC").value
    random_times = pd.to_datetime(np.random.randint(start_ts, end_ts, size=n_records), utc=True)
    random_times = random_times.sort_values()

    origins = np.random.choice(["JFK", "ORD", "ATL", "LAX", "DFW", "DEN", "SFO", "BOS"], size=n_records)
    dests = np.random.choice(["LAX", "SFO", "MIA", "SEA", "MCO", "LAS", "PHX", "JFK"], size=n_records)
    airlines = np.random.choice(["AA", "DL", "UA", "WN"], size=n_records)
    tail_pool = [f"N{100 + i}AA" for i in range(150)] + [None] * 10
    tails = np.random.choice(tail_pool, size=n_records)

    elapsed_crs = np.random.randint(60, 360, size=n_records)
    actual_dep_delay = np.random.exponential(scale=12.0, size=n_records) - 5.0
    actual_dep_times = random_times + pd.to_timedelta(actual_dep_delay, unit="min")
    late = (actual_dep_delay >= 15.0).astype("uint8")

    sched_arr_times = random_times + pd.to_timedelta(elapsed_crs, unit="min")
    actual_arr_times = actual_dep_times + pd.to_timedelta(elapsed_crs, unit="min")

    df = pd.DataFrame({
        "flight_id": [f"synth_{i:06d}" for i in range(n_records)],
        TIME: random_times,
        OBSERVED: actual_dep_times,
        "LATE": late,
        "Origin": origins,
        "Dest": dests,
        "Reporting_Airline": airlines,
        "Tail_Number": pd.Series(tails, dtype="string"),
        "tail_number_missing": pd.Series(tails).isna(),
        "CRSElapsedTime": elapsed_crs,
        "Distance": np.random.randint(200, 2500, size=n_records),
        "scheduled_arrival_utc": sched_arr_times,
        "actual_arrival_utc": actual_arr_times,
    }, index=range(n_records))

    return df


def build_and_save_sample_features():
    """Build and save sample feature matrices for both checkpoints."""
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    if DEV_PARQUET.exists():
        print(f"Loading real development data from {DEV_PARQUET}...")
        full_df = pd.read_parquet(DEV_PARQUET)
        # Safe slice: Month 1 (Jan 2025)
        cutoff_date = pd.Timestamp("2025-02-01", tz="UTC")
        sample_df = full_df.loc[full_df[TIME] < cutoff_date].copy()
        assert sample_df[TIME].max() < pd.Timestamp("2025-11-01", tz="UTC"), "Sample bleeds into test period!"
    else:
        print(f"{DEV_PARQUET} not found on local disk. Generating synthetic development cohort...")
        sample_df = generate_synthetic_development_cohort(n_records=2000)
        sample_df.to_parquet(SAMPLE_DIR / "flights_development_sample.parquet", compression="zstd")
        print(f"Saved {SAMPLE_DIR / 'flights_development_sample.parquet'} ({len(sample_df)} rows)")

    for cp_name, cp_delta in CHECKPOINTS.items():
        print(f"Building feature matrix for checkpoint {cp_name} (delta={cp_delta})...")
        X = build_features(sample_df, cp_delta)
        assert X.index.equals(sample_df.index), "Index mismatch between X and sample_df!"

        out_path = SAMPLE_DIR / f"features_sample_{cp_name}.parquet"
        X.to_parquet(out_path, compression="zstd")
        print(f"Saved {out_path} with shape {X.shape} and {len(X.columns)} features.")

    print("\nSample feature matrix building completed successfully!")


if __name__ == "__main__":
    build_and_save_sample_features()
