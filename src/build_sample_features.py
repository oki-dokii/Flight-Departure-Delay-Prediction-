"""Generate paired Phase A checkpoint feature matrices and manifest for P4 and P5.

Performs:
1. Full development cohort ingestion (5.7M rows).
2. Sampling of the exact 20,000 flight IDs matching P4's sampler (2,000/month, seed 42).
3. Congestion and history computation from the entire 5.7M development pool prior to cutoff,
   strictly excluding the target flight itself (even if it departed early).
4. Export of paired 360-min and 60-min Parquets with audited predictor columns.
5. Export of sampled_ids.parquet and feature_manifest.json.
"""

import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.features import AUDITED_PREDICTORS, CHECKPOINTS, build_features
from src.splits import DEVELOPMENT_END, GAP, OBSERVED, START, TIME, get_folds, utc_column

ROOT = Path(__file__).resolve().parents[1]
DEV_PARQUET = ROOT / "data/processed/flights_development.parquet"
SAMPLE_DIR = ROOT / "data/sample"
RELEASE = "phase-a-data-v1"
REVISION = "phase-a-v1"
FLIGHT_ID = "flight_id"

SAMPLE_PER_MONTH = 2_000
SAMPLE_SEED = 42


def get_p4_sampled_ids(df: pd.DataFrame) -> pd.Index:
    """Select up to 2,000 targets per scheduled UTC month using P4's exact algorithm."""
    months = df[TIME].dt.month
    rng = np.random.default_rng(SAMPLE_SEED)
    sampled = []
    for _, group in df.groupby(months, sort=True):
        ordered = group.index.sort_values()
        pos = rng.choice(len(ordered), size=min(SAMPLE_PER_MONTH, len(ordered)), replace=False)
        sampled.extend(ordered.take(pos))
    return pd.Index(sampled, name=FLIGHT_ID).sort_values()


def build_and_export():
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    print(f"Reading development dataset from {DEV_PARQUET}...")
    columns = [
        FLIGHT_ID, TIME, OBSERVED, "LATE", "Origin", "Dest",
        "Reporting_Airline", "CRSElapsedTime", "Distance",
    ]
    table = pq.read_table(DEV_PARQUET, columns=columns)
    full_df = table.to_pandas().set_index(FLIGHT_ID)
    print(f"Loaded {len(full_df):,} development records in {time.time() - start_time:.2f}s.")

    # 1. Select the exact 20,000 target cohort
    sampled_ids = get_p4_sampled_ids(full_df)
    target_df = full_df.loc[sampled_ids].copy()
    print(f"Sampled {len(target_df):,} target flights across months {target_df[TIME].dt.month.unique().tolist()}.")

    # Save sampled_ids.parquet
    sampled_ids_path = SAMPLE_DIR / "sampled_ids.parquet"
    sampled_ids.to_frame(index=False).to_parquet(sampled_ids_path, index=False)
    print(f"Wrote {sampled_ids_path}")

    # 2. Build feature matrices for both checkpoints
    checkpoint_configs = [
        {"minutes": 360, "name": "360m", "alias": "t6h", "delta": pd.Timedelta(minutes=360)},
        {"minutes": 60, "name": "60m", "alias": "t1h", "delta": pd.Timedelta(minutes=60)},
    ]

    checkpoints_manifest = []

    for cp in checkpoint_configs:
        cp_start = time.time()
        minutes = cp["minutes"]
        delta = cp["delta"]
        print(f"\nComputing features for checkpoint {minutes} minutes ({delta}) from full development pool...")

        X = build_features(target_df, checkpoint_delta=delta, history_df=full_df, include_tail=False)
        assert X.index.equals(target_df.index), "Index mismatch between X and target_df!"

        # Keep audited predictor columns in consistent order
        feature_cols = [p["name"] for p in AUDITED_PREDICTORS]
        X_out = X[feature_cols].copy()
        X_out.insert(0, FLIGHT_ID, target_df.index.astype("string").values)

        # Output paths
        out_name = f"features_{cp['name']}.parquet"
        out_path = SAMPLE_DIR / out_name
        alias_path = SAMPLE_DIR / f"features_sample_{cp['alias']}.parquet"

        # Construct PyArrow Table with metadata for P4 validator
        arrow_table = pa.Table.from_pandas(X_out, preserve_index=False)
        existing_meta = arrow_table.schema.metadata or {}
        custom_meta = {
            b"checkpoint_minutes": str(minutes).encode("utf-8"),
            b"source_release": RELEASE.encode("utf-8"),
            b"feature_spec_revision": REVISION.encode("utf-8"),
        }
        arrow_table = arrow_table.replace_schema_metadata({**existing_meta, **custom_meta})

        pq.write_table(arrow_table, out_path, compression="zstd")
        pq.write_table(arrow_table, alias_path, compression="zstd")
        print(f"Saved {out_path} ({len(X_out):,} rows, {len(feature_cols)} features) in {time.time() - cp_start:.2f}s.")

        checkpoints_manifest.append({
            "offset_minutes": minutes,
            "path": f"data/sample/{out_name}",
            "source_release": RELEASE,
        })

    # 3. Write manifest
    manifest_doc = {
        "development": {
            "path": "data/processed/flights_development.parquet",
            "release": RELEASE,
        },
        "feature_spec_revision": REVISION,
        "sampled_ids": "data/sample/sampled_ids.parquet",
        "checkpoints": checkpoints_manifest,
        "predictors": AUDITED_PREDICTORS,
    }

    manifest_path = SAMPLE_DIR / "feature_manifest.json"
    manifest_path.write_text(json.dumps(manifest_doc, indent=2) + "\n")
    print(f"\nWrote feature manifest to {manifest_path}")

    # 4. Verify all 4 folds populate and run
    print("\nVerifying 4 validation folds on target cohort...")
    folds = get_folds(target_df)
    print(f"Total folds: {len(folds)}")
    for f_idx, (trn, val) in enumerate(folds, 1):
        trn_late = target_df["LATE"].iloc[trn].mean()
        val_late = target_df["LATE"].iloc[val].mean()
        print(f"Fold {f_idx}: Train {len(trn):,} (late={trn_late:.1%}) -> Val {len(val):,} (late={val_late:.1%})")

    print(f"\nAll operations completed successfully in {time.time() - start_time:.2f}s!")


if __name__ == "__main__":
    build_and_export()
