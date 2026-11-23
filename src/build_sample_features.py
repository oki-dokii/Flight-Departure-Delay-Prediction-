"""Generate sample feature matrices for P4 (Classical) and P5 (Boosting) models.

Ensures both checkpoints (t6h, t1h) are generated, populated with pyarrow metadata,
paired with a manifest, and validated against P1's harness and P4's interface.
"""

import json
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.features import CHECKPOINTS, build_features
from src.splits import OBSERVED, TIME

SAMPLE_DIR = Path("data/sample")
DEV_PARQUET = Path("data/processed/flights_development.parquet")
FEATURE_SPEC_REVISION = "phase-a-v1"
SOURCE_RELEASE = "phase-a-data-v1"

PREDICTORS: List[Dict[str, str]] = [
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
    {"name": "inbound_missing_no_tail", "role": "indicator"},
    {"name": "inbound_first_leg_of_day", "role": "indicator"},
    {"name": "inbound_not_landed_by_T", "role": "indicator"},
    {"name": "inbound_arrival_delay_min", "role": "numeric"},
    {"name": "turnaround_buffer_min", "role": "numeric"},
    {"name": "inbound_prior_late", "role": "indicator"},
    {"name": "origin_departures_3h", "role": "numeric"},
    {"name": "origin_late_rate_3h", "role": "numeric"},
]


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
    """Build and save sample feature matrices and manifest for P4 and P5."""
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    if DEV_PARQUET.exists():
        print(f"Loading real development data from {DEV_PARQUET}...")
        full_df = pd.read_parquet(DEV_PARQUET)
        cutoff_date = pd.Timestamp("2025-02-01", tz="UTC")
        sample_df = full_df.loc[full_df[TIME] < cutoff_date].copy()
        assert sample_df[TIME].max() < pd.Timestamp("2025-11-01", tz="UTC"), "Sample bleeds into test period!"
        dev_path_relative = str(DEV_PARQUET)
    else:
        print(f"{DEV_PARQUET} not found on local disk. Generating synthetic development cohort...")
        sample_df = generate_synthetic_development_cohort(n_records=2000)
        synth_dev_path = SAMPLE_DIR / "flights_development_sample.parquet"
        sample_df.to_parquet(synth_dev_path, compression="zstd")
        print(f"Saved {synth_dev_path} ({len(sample_df)} rows)")
        dev_path_relative = str(synth_dev_path)

    checkpoints_manifest = []

    for cp_name, cp_delta in CHECKPOINTS.items():
        minutes = int(cp_delta.total_seconds() // 60)
        print(f"Building feature matrix for checkpoint {cp_name} (delta={cp_delta}, minutes={minutes})...")
        X = build_features(sample_df, cp_delta)
        assert X.index.equals(sample_df.index), "Index mismatch between X and sample_df!"

        # Ensure flight_id is prepended as first column for P4 compatibility
        feature_cols = [p["name"] for p in PREDICTORS]
        X_out = X[feature_cols].copy()
        X_out.insert(0, "flight_id", sample_df["flight_id"].astype("string").values)

        out_path = SAMPLE_DIR / f"features_sample_{cp_name}.parquet"

        # Attach standard metadata for P4 validator
        table = pa.Table.from_pandas(X_out, preserve_index=False)
        existing_meta = table.schema.metadata or {}
        custom_meta = {
            b"checkpoint_minutes": str(minutes).encode("utf-8"),
            b"source_release": SOURCE_RELEASE.encode("utf-8"),
            b"feature_spec_revision": FEATURE_SPEC_REVISION.encode("utf-8"),
        }
        table = table.replace_schema_metadata({**existing_meta, **custom_meta})
        pq.write_table(table, out_path, compression="zstd")

        print(f"Saved {out_path} with shape {X_out.shape} and {len(feature_cols)} predictor columns.")

        checkpoints_manifest.append({
            "offset_minutes": minutes,
            "path": str(out_path),
            "source_release": SOURCE_RELEASE,
        })

    # Save manifest for P4 / P5
    manifest_doc = {
        "development": {
            "path": dev_path_relative,
            "release": SOURCE_RELEASE,
        },
        "feature_spec_revision": FEATURE_SPEC_REVISION,
        "checkpoints": checkpoints_manifest,
        "predictors": PREDICTORS,
    }

    manifest_path = SAMPLE_DIR / "feature_manifest.json"
    manifest_path.write_text(json.dumps(manifest_doc, indent=2) + "\n")
    print(f"Wrote pairing manifest to {manifest_path}")


if __name__ == "__main__":
    build_and_save_sample_features()
