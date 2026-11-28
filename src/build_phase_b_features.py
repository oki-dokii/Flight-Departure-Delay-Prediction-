"""Build full-scale Phase B checkpoint feature matrices and manifests (G0-G3).

Produces paired 360-minute and 60-minute Parquet files containing all 5,749,041
development flight IDs from flights_development_phase_b.parquet, along with
group-specific manifests compatible with P4's read_manifest()/paired_metadata()
and P1's final_run.py.

Also supports --holdout mode for frozen Nov-Dec 2025 scoring using development
history and Jan-Dec 2025 ASOS weather observations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.features import GROUP_PREDICTORS, build_features
from src.splits import OBSERVED, TIME
from src.weather import load_weather_context

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEV_PARQUET = ROOT / "data/processed/flights_development_phase_b.parquet"
OUT_DIR = ROOT / "data/processed/phase_b"
MANIFEST_DIR = ROOT / "data/manifests"
SHA_PATH = ROOT / "results/feature_SHA256SUMS"

FLIGHT_ID = "flight_id"
RELEASE = "phase-b-data-v1"
REVISION = "phase-b-v1"

BASE_COLUMNS = [
    FLIGHT_ID,
    TIME,
    OBSERVED,
    "LATE",
    "Origin",
    "Dest",
    "Reporting_Airline",
    "CRSElapsedTime",
    "Distance",
]
TAIL_COLUMNS = [
    "Tail_Number",
    "tail_number_missing",
    "scheduled_arrival_utc",
    "actual_arrival_utc",
]
CHECKPOINT_MINUTES = (360, 60)


def _write_checkpoint_parquet(
    X_out: pd.DataFrame,
    out_path: Path,
    minutes: int,
    release: str,
    revision: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arrow_table = pa.Table.from_pandas(X_out, preserve_index=False)
    existing_meta = arrow_table.schema.metadata or {}
    custom_meta = {
        b"checkpoint_minutes": str(minutes).encode("utf-8"),
        b"source_release": release.encode("utf-8"),
        b"feature_spec_revision": revision.encode("utf-8"),
    }
    arrow_table = arrow_table.replace_schema_metadata({**existing_meta, **custom_meta})
    pq.write_table(arrow_table, out_path, compression="zstd")


def build_phase_b_groups(
    groups: List[str],
    dev_parquet: Path = DEFAULT_DEV_PARQUET,
    holdout_parquet: Optional[Path] = None,
) -> None:
    t0 = time.time()
    need_tail = "G3" in groups
    need_weather = any(g in ("G2", "G3") for g in groups)

    cols = list(BASE_COLUMNS) + (TAIL_COLUMNS if need_tail else [])
    print(f"Loading development dataset from {dev_parquet} ({len(cols)} columns)...")
    dev_df = pq.read_table(dev_parquet, columns=cols).to_pandas(ignore_metadata=True)
    dev_df[FLIGHT_ID] = dev_df[FLIGHT_ID].astype("string")
    dev_df = dev_df.set_index(FLIGHT_ID, verify_integrity=True)
    print(f"Loaded {len(dev_df):,} development rows in {time.time() - t0:.2f}s.")

    if holdout_parquet is not None:
        print(f"Loading holdout dataset from {holdout_parquet}...")
        target_df = pq.read_table(holdout_parquet, columns=cols).to_pandas(ignore_metadata=True)
        target_df[FLIGHT_ID] = target_df[FLIGHT_ID].astype("string")
        target_df = target_df.set_index(FLIGHT_ID, verify_integrity=True)
        history_df = pd.concat([dev_df, target_df], axis=0)
        out_base = OUT_DIR / "holdout"
        manifest_prefix = "holdout_manifest"
        source_meta_rel = str(holdout_parquet.relative_to(ROOT))
        print(f"Holdout mode: {len(target_df):,} target rows, {len(history_df):,} total history rows.")
    else:
        target_df = dev_df
        history_df = dev_df
        out_base = OUT_DIR
        manifest_prefix = "manifest"
        source_meta_rel = str(dev_parquet.relative_to(ROOT))

    weather_obs, station_map = (None, None)
    if need_weather:
        weather_obs, station_map = load_weather_context()
        print(f"Loaded {len(weather_obs):,} weather observations across {len(station_map)} origin mappings.")

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    sha_entries = []

    for group in groups:
        predictors = GROUP_PREDICTORS[group]
        feature_cols = [p["name"] for p in predictors]
        checkpoints_manifest = []

        for minutes in CHECKPOINT_MINUTES:
            cp_start = time.time()
            delta = pd.Timedelta(minutes=minutes)
            print(f"\n[{group} @ {minutes}m] Building {len(feature_cols)} features for {len(target_df):,} rows...")

            X = build_features(
                target_df,
                checkpoint_delta=delta,
                history_df=history_df,
                weather_obs=weather_obs,
                station_map=station_map,
                feature_group=group,
            )
            X_out = X[feature_cols].copy()
            X_out.insert(0, FLIGHT_ID, target_df.index.astype("string").values)

            out_path = out_base / f"features_{group}_{minutes}m.parquet"
            _write_checkpoint_parquet(X_out, out_path, minutes, RELEASE, REVISION)
            digest = hashlib.sha256(out_path.read_bytes()).hexdigest()
            rel_out = str(out_path.relative_to(ROOT))
            sha_entries.append(f"{digest}  {rel_out}")
            print(f"  Saved {rel_out} ({len(X_out):,} rows, {X_out.shape[1]} cols) in {time.time() - cp_start:.2f}s.")

            checkpoints_manifest.append({
                "offset_minutes": minutes,
                "path": rel_out,
                "source_release": RELEASE,
            })

        manifest_doc = {
            "development": {
                "path": source_meta_rel,
                "release": RELEASE,
            },
            "feature_spec_revision": REVISION,
            "checkpoints": checkpoints_manifest,
            "predictors": predictors,
        }
        manifest_path = MANIFEST_DIR / f"{manifest_prefix}_{group}.json"
        manifest_path.write_text(json.dumps(manifest_doc, indent=2) + "\n")
        print(f"Wrote manifest {manifest_path.relative_to(ROOT)}")

    if holdout_parquet is None and sha_entries:
        SHA_PATH.parent.mkdir(parents=True, exist_ok=True)
        SHA_PATH.write_text("\n".join(sha_entries) + "\n")
        print(f"\nWrote SHA-256 checksums to {SHA_PATH.relative_to(ROOT)}")

    print(f"\nDone in {time.time() - t0:.2f}s.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--groups",
        nargs="+",
        default=["G0", "G1", "G2", "G3"],
        choices=list(GROUP_PREDICTORS),
        help="Feature groups to build",
    )
    parser.add_argument("--dev-parquet", type=Path, default=DEFAULT_DEV_PARQUET)
    parser.add_argument("--holdout-parquet", type=Path, default=None)
    args = parser.parse_args()
    build_phase_b_groups(
        groups=args.groups,
        dev_parquet=args.dev_parquet,
        holdout_parquet=args.holdout_parquet,
    )


if __name__ == "__main__":
    main()
