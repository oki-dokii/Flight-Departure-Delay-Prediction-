"""Unit tests for leak-safety, self-exclusion, and feature contracts in src.features."""

import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from src.evaluate import evaluate
from src.features import AUDITED_PREDICTORS, CHECKPOINTS, GROUP_PREDICTORS, build_features
from src.models_classical import FeatureSchema
from src.splits import OBSERVED, TIME, get_folds


def _create_synthetic_weather() -> pd.DataFrame:
    """Create synthetic hourly ASOS observations for JFK and BOS."""
    return pd.DataFrame({
        "station": ["JFK", "JFK", "BOS"],
        "obs_time_utc": pd.to_datetime(
            ["2025-07-01 12:51", "2025-07-01 13:51", "2025-07-01 09:51"], utc=True
        ),
        "wx_temp_c": np.array([22.0, 28.0, 18.0], dtype="float32"),
        "wx_wind_speed_kt": np.array([8.0, 18.0, 10.0], dtype="float32"),
        "wx_wind_gust_kt": np.array([0.0, 25.0, 0.0], dtype="float32"),
        "wx_visibility_mi": np.array([10.0, 2.0, 10.0], dtype="float32"),
        "wx_precip_1h_mm": np.array([0.0, 5.0, 0.0], dtype="float32"),
    })


def _create_synthetic_flights() -> pd.DataFrame:
    """Create synthetic flights where flight 11 is the inbound leg to flight 10."""
    return pd.DataFrame({
        TIME: pd.to_datetime(["2025-07-01 14:00", "2025-07-01 11:00"], utc=True),
        OBSERVED: pd.to_datetime(["2025-07-01 14:20", "2025-07-01 11:30"], utc=True),
        "LATE": [1, 0],
        "Tail_Number": ["N123", "N123"],
        "tail_number_missing": [False, False],
        "scheduled_arrival_utc": pd.to_datetime(["2025-07-01 18:00", "2025-07-01 12:45"], utc=True),
        "actual_arrival_utc": pd.to_datetime([pd.NaT, "2025-07-01 13:00"], utc=True),
        "Origin": ["JFK", "BOS"],
        "Dest": ["LAX", "JFK"],
        "Reporting_Airline": ["AA", "AA"],
        "CRSElapsedTime": [240, 105],
        "Distance": [2475, 187],
    }, index=[10, 11])


class FeatureLeakTests(unittest.TestCase):

    def test_negative_leak_post_cutoff_mutation(self):
        """Mutating post-cutoff events must not alter computed features at T."""
        df = _create_synthetic_flights()
        # For T=6h, cutoff for flight 10 is 08:00 UTC (14:00 - 6h).
        # Flight 11's actual departure (11:30) is well after 08:00 UTC.
        X_base = build_features(df.copy(), CHECKPOINTS["t6h"])

        # Mutate flight 11's post-cutoff actual departure and outcome
        df_mutated = df.copy()
        df_mutated.loc[11, OBSERVED] = pd.Timestamp("2025-07-01 13:50", tz="UTC")
        df_mutated.loc[11, "LATE"] = 1
        X_mut = build_features(df_mutated, CHECKPOINTS["t6h"])

        pd.testing.assert_frame_equal(X_base.loc[[10]], X_mut.loc[[10]])

    def test_positive_sensitivity_pre_cutoff_mutation(self):
        """Mutating an observable pre-cutoff event must update downstream features when tail included."""
        df = _create_synthetic_flights()
        # For T=1h, cutoff for flight 10 is 13:00 UTC.
        # Flight 11 scheduled arrival was 12:45, actual arrival was 13:00 (delay = +15 min).
        X_base = build_features(df.copy(), CHECKPOINTS["t1h"], include_tail=True)

        # Change inbound actual arrival to 12:50 (delay = +5 min), still before cutoff
        df_mutated = df.copy()
        df_mutated.loc[11, "actual_arrival_utc"] = pd.Timestamp("2025-07-01 12:50", tz="UTC")
        X_mut = build_features(df_mutated, CHECKPOINTS["t1h"], include_tail=True)

        self.assertAlmostEqual(X_base.loc[10, "inbound_arrival_delay_min"], 15.0, places=1)
        self.assertAlmostEqual(X_mut.loc[10, "inbound_arrival_delay_min"], 5.0, places=1)

    def test_boundary_exact_cutoff_included(self):
        """An event occurring at exactly cutoff_utc is included (<= rule)."""
        df = _create_synthetic_flights()
        # For T=1h, cutoff for flight 10 is 13:00 UTC.
        # Prior leg arrives at exactly 13:00 UTC
        df.loc[11, "actual_arrival_utc"] = pd.Timestamp("2025-07-01 13:00", tz="UTC")
        X = build_features(df, CHECKPOINTS["t1h"], include_tail=True)
        self.assertEqual(X.loc[10, "inbound_not_landed_by_T"], 0)

    def test_diverted_inbound_prevents_leak(self):
        """Diverted prior flight with NaT actual arrival triggers inbound_not_landed_by_T."""
        df = _create_synthetic_flights()
        df.loc[11, "actual_arrival_utc"] = pd.NaT  # Diverted
        X = build_features(df, CHECKPOINTS["t1h"], include_tail=True)

        self.assertEqual(X.loc[10, "inbound_not_landed_by_T"], 1)
        self.assertEqual(X.loc[10, "inbound_arrival_delay_min"], 0.0)
        self.assertTrue(pd.isna(X.loc[10, "turnaround_buffer_min"]))

    def test_target_flight_strictly_excluded_from_own_congestion(self):
        """Airport congestion must exclude target flight itself, even if it departed unusually early."""
        # Flight 10: scheduled at 14:00, cutoff at 13:00 (T=1h).
        # Unusually early departure at 12:45 UTC with LATE=0.
        # Flight 20: another JFK departure at 12:30 UTC with LATE=1.
        test_df = pd.DataFrame({
            TIME: pd.to_datetime(["2025-07-01 14:00", "2025-07-01 12:30"], utc=True),
            OBSERVED: pd.to_datetime(["2025-07-01 12:45", "2025-07-01 12:30"], utc=True),
            "LATE": [0, 1],
            "Origin": ["JFK", "JFK"],
            "Dest": ["LAX", "ORD"],
            "Reporting_Airline": ["AA", "DL"],
            "CRSElapsedTime": [300, 150],
            "Distance": [2475, 740],
        }, index=[10, 20])

        X = build_features(test_df, CHECKPOINTS["t1h"])

        # Flight 10's own departure at 12:45 must NOT be counted.
        # Only Flight 20 (at 12:30) was eligible.
        # A separate history pool can already exclude the target flight.
        separate = build_features(test_df.loc[[10]], pd.Timedelta(hours=1), history_df=test_df.loc[[20]])
        self.assertEqual(separate.loc[10, "origin_departures_3h"], 1)

        self.assertEqual(X.loc[10, "origin_departures_3h"], 1)
        # Late rate must be 1.0 (from flight 20), not 0.5 (which would occur if flight 10 were included)
        self.assertEqual(X.loc[10, "origin_late_rate_3h"], 1.0)

        # Single flight that departed early with no peers: departures must be 0 and rate NaN
        solo_df = test_df.loc[[10]].copy()
        X_solo = build_features(solo_df, CHECKPOINTS["t1h"])
        self.assertEqual(X_solo.loc[10, "origin_departures_3h"], 0)
        self.assertTrue(pd.isna(X_solo.loc[10, "origin_late_rate_3h"]))

    def test_index_and_row_order_invariance(self):
        """Feature matrix index and row ordering must strictly match input DataFrame."""
        df = _create_synthetic_flights()
        X = build_features(df, CHECKPOINTS["t6h"])
        self.assertTrue(X.index.equals(df.index))
        self.assertEqual(len(X), len(df))

    def test_changing_events_after_checkpoint_leaves_features_unchanged(self):
        """Modifying actual departures or outcomes after cutoff leaves features strictly unchanged."""
        base_df = _create_synthetic_flights()
        # Cutoff for flight 10 at T=6h is 08:00 UTC.
        X_before = build_features(base_df, CHECKPOINTS["t6h"])

        mutated_df = base_df.copy()
        # Change flight 10 and 11 events that occur at 11:30 and 14:20
        mutated_df.loc[10, OBSERVED] = pd.Timestamp("2025-07-01 16:00", tz="UTC")
        mutated_df.loc[10, "LATE"] = 1
        mutated_df.loc[11, OBSERVED] = pd.Timestamp("2025-07-01 12:15", tz="UTC")
        mutated_df.loc[11, "LATE"] = 1
        X_after = build_features(mutated_df, CHECKPOINTS["t6h"])

        # Features for flight 10 at 08:00 UTC must be strictly identical
        pd.testing.assert_series_equal(X_before.loc[10], X_after.loc[10])

    def test_all_four_validation_folds_populated_and_run(self):
        """Verify all 4 validation folds (July-Oct) are populated on the 20,000 target cohort."""
        sample_path = Path("data/sample/features_60m.parquet")
        dev_path = Path("data/processed/flights_development.parquet")
        if not sample_path.exists() or not dev_path.exists():
            self.skipTest("Sample or development parquet not found on disk")

        sample_X = pd.read_parquet(sample_path).set_index("flight_id")
        self.assertEqual(len(sample_X), 20000)

        meta = pd.read_parquet(dev_path, columns=["flight_id", TIME, OBSERVED, "LATE"]).set_index("flight_id")
        dev_target = meta.loc[sample_X.index].copy()
        dev_target.index = dev_target.index.astype(sample_X.index.dtype)

        folds = get_folds(dev_target)
        self.assertEqual(len(folds), 4)

        # Check validation sizes
        for fold_idx, (train, val) in enumerate(folds):
            self.assertEqual(len(val), 2000, f"Fold {fold_idx + 1} validation size mismatch")
            self.assertGreater(len(train), 10000, f"Fold {fold_idx + 1} train size too small")

        # Smoke evaluation on numeric subset across all 4 folds
        numeric_cols = [p["name"] for p in AUDITED_PREDICTORS if p["role"] == "numeric"]
        runs, summary = evaluate(
            lambda seed: DecisionTreeClassifier(max_depth=3, random_state=seed),
            sample_X[numeric_cols].fillna(0.0),
            dev_target,
            seeds=(42,),
        )
        self.assertEqual(len(runs), 8)
        self.assertIn("roc_auc", summary.columns.levels[1])
        self.assertIn("pr_auc", summary.columns.levels[1])

    def test_weather_negative_leak_and_exact_boundary(self):
        """Post-cutoff weather observations must not leak; exact cutoff is included."""
        df = _create_synthetic_flights()
        wx = _create_synthetic_weather()
        # Flight 10 cutoff at T=1h is 13:00 UTC. Only 12:51 UTC (temp=22C) is <= 13:00.
        # 13:51 UTC (temp=28C) is strictly post-cutoff.
        X_base = build_features(df, CHECKPOINTS["t1h"], weather_obs=wx, feature_group="G2")
        self.assertEqual(X_base.loc[10, "weather_missing"], 0)
        self.assertAlmostEqual(X_base.loc[10, "wx_temp_c"], 22.0, places=2)

        # Mutate post-cutoff 13:51 observation -> must not change flight 10 features
        wx_mut = wx.copy()
        wx_mut.loc[1, "wx_temp_c"] = 45.0
        wx_mut.loc[1, "wx_precip_1h_mm"] = 99.0
        X_mut = build_features(df, CHECKPOINTS["t1h"], weather_obs=wx_mut, feature_group="G2")
        pd.testing.assert_series_equal(X_base.loc[10], X_mut.loc[10])

        # Exact cutoff (13:00 UTC) must be included
        wx_exact = wx.copy()
        wx_exact.loc[1, "obs_time_utc"] = pd.Timestamp("2025-07-01 13:00", tz="UTC")
        X_exact = build_features(df, CHECKPOINTS["t1h"], weather_obs=wx_exact, feature_group="G2")
        self.assertAlmostEqual(X_exact.loc[10, "wx_temp_c"], 28.0, places=2)

    def test_weather_staleness_and_unmapped_station(self):
        """Observations older than 6h or unmapped airports set weather_missing=1."""
        df = _create_synthetic_flights()
        # Flight 10 at JFK (cutoff 13:00 UTC); make JFK obs > 6h stale (06:30 UTC)
        wx_stale = pd.DataFrame({
            "station": ["JFK"],
            "obs_time_utc": pd.to_datetime(["2025-07-01 06:30"], utc=True),
            "wx_temp_c": np.array([20.0], dtype="float32"),
            "wx_wind_speed_kt": np.array([5.0], dtype="float32"),
            "wx_wind_gust_kt": np.array([0.0], dtype="float32"),
            "wx_visibility_mi": np.array([10.0], dtype="float32"),
            "wx_precip_1h_mm": np.array([0.0], dtype="float32"),
        })
        X = build_features(df, CHECKPOINTS["t1h"], weather_obs=wx_stale, feature_group="G2")
        self.assertEqual(X.loc[10, "weather_missing"], 1)
        self.assertTrue(pd.isna(X.loc[10, "wx_temp_c"]))
        # Flight 11 at BOS has no obs at all -> weather_missing = 1
        self.assertEqual(X.loc[11, "weather_missing"], 1)

    def test_all_feature_groups_pass_p4_schema_normalization(self):
        """Every group G0-G3 must pass FeatureSchema.validate_columns and normalize."""
        df = _create_synthetic_flights()
        wx = _create_synthetic_weather()
        for group, predictors in GROUP_PREDICTORS.items():
            X = build_features(df, CHECKPOINTS["t1h"], weather_obs=wx, feature_group=group)
            schema = FeatureSchema(tuple((p["name"], p["role"]) for p in predictors))
            schema.validate_columns(X.columns)
            norm = schema.normalize(X)
            self.assertEqual(list(norm.columns), [p["name"] for p in predictors])


if __name__ == "__main__":
    unittest.main()
