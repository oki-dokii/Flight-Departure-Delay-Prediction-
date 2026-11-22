"""Unit tests for leak-safety and feature contracts in src.features."""

import unittest
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from src.evaluate import evaluate
from src.features import CHECKPOINTS, build_features
from src.splits import OBSERVED, TIME


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
        """Mutating an observable pre-cutoff event must update downstream features."""
        df = _create_synthetic_flights()
        # For T=1h, cutoff for flight 10 is 13:00 UTC.
        # Flight 11 scheduled arrival was 12:45, actual arrival was 13:00 (delay = +15 min).
        X_base = build_features(df.copy(), CHECKPOINTS["t1h"])

        # Change inbound actual arrival to 12:50 (delay = +5 min), still before cutoff
        df_mutated = df.copy()
        df_mutated.loc[11, "actual_arrival_utc"] = pd.Timestamp("2025-07-01 12:50", tz="UTC")
        X_mut = build_features(df_mutated, CHECKPOINTS["t1h"])

        self.assertAlmostEqual(X_base.loc[10, "inbound_arrival_delay_min"], 15.0, places=1)
        self.assertAlmostEqual(X_mut.loc[10, "inbound_arrival_delay_min"], 5.0, places=1)

    def test_boundary_exact_cutoff_included(self):
        """An event occurring at exactly cutoff_utc is included (<= rule)."""
        df = _create_synthetic_flights()
        # For T=1h, cutoff for flight 10 is 13:00 UTC.
        # Prior leg arrives at exactly 13:00 UTC
        df.loc[11, "actual_arrival_utc"] = pd.Timestamp("2025-07-01 13:00", tz="UTC")
        X = build_features(df, CHECKPOINTS["t1h"])
        self.assertEqual(X.loc[10, "inbound_not_landed_by_T"], 0)

    def test_diverted_inbound_prevents_leak(self):
        """Diverted prior flight with NaT actual arrival triggers inbound_not_landed_by_T."""
        df = _create_synthetic_flights()
        df.loc[11, "actual_arrival_utc"] = pd.NaT  # Diverted
        X = build_features(df, CHECKPOINTS["t1h"])

        self.assertEqual(X.loc[10, "inbound_not_landed_by_T"], 1)
        self.assertEqual(X.loc[10, "inbound_arrival_delay_min"], 0.0)
        self.assertTrue(pd.isna(X.loc[10, "turnaround_buffer_min"]))

    def test_four_state_flags_mutually_exclusive(self):
        """Verify that at most one of the three inbound indicator flags is active."""
        df = _create_synthetic_flights()
        # Add a flight with missing tail
        extra_flight = pd.DataFrame({
            TIME: [pd.Timestamp("2025-07-01 15:00", tz="UTC")],
            OBSERVED: [pd.Timestamp("2025-07-01 15:10", tz="UTC")],
            "LATE": [0],
            "Tail_Number": [pd.NA],
            "tail_number_missing": [True],
            "scheduled_arrival_utc": [pd.Timestamp("2025-07-01 17:00", tz="UTC")],
            "actual_arrival_utc": [pd.Timestamp("2025-07-01 17:10", tz="UTC")],
            "Origin": ["JFK"],
            "Dest": ["MIA"],
            "Reporting_Airline": ["AA"],
            "CRSElapsedTime": [120],
            "Distance": [1089],
        }, index=[12])
        df_all = pd.concat([df, extra_flight])

        for cp_name, cp_delta in CHECKPOINTS.items():
            X = build_features(df_all, cp_delta)
            flag_sum = (
                X["inbound_missing_no_tail"]
                + X["inbound_first_leg_of_day"]
                + X["inbound_not_landed_by_T"]
            )
            self.assertTrue((flag_sum <= 1).all(), f"Indicators not mutually exclusive in {cp_name}")

    def test_index_and_row_order_invariance(self):
        """Feature matrix index and row ordering must strictly match input DataFrame."""
        df = _create_synthetic_flights()
        X = build_features(df, CHECKPOINTS["t6h"])
        self.assertTrue(X.index.equals(df.index))
        self.assertEqual(len(X), len(df))

    def test_evaluation_harness_integration(self):
        """Verify features run smoothly end-to-end through src.evaluate.evaluate."""
        times = pd.to_datetime(
            ["2025-01-01", "2025-06-29", "2025-06-30", "2025-07-01",
             "2025-07-02", "2025-08-01", "2025-08-02", "2025-09-01",
             "2025-09-02", "2025-10-01", "2025-10-02"], utc=True,
        ).repeat(2)
        dev = pd.DataFrame({
            TIME: times,
            OBSERVED: times + pd.to_timedelta([0, 15] * 11, unit="min"),
            "LATE": [0, 1] * 11,
            "Tail_Number": [f"N{i % 4}" for i in range(22)],
            "tail_number_missing": [False] * 22,
            "scheduled_arrival_utc": times + pd.Timedelta(hours=2),
            "actual_arrival_utc": times + pd.Timedelta(hours=2),
            "Origin": ["JFK"] * 22,
            "Dest": ["LAX"] * 22,
            "Reporting_Airline": ["DL"] * 22,
            "CRSElapsedTime": [120] * 22,
            "Distance": [1000] * 22,
        }, index=range(500, 522))

        # Build feature matrix at T=1h
        X = build_features(dev, CHECKPOINTS["t1h"])

        # Numeric subset for decision tree smoke test
        numeric_cols = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month",
                        "inbound_missing_no_tail", "inbound_first_leg_of_day",
                        "inbound_not_landed_by_T", "inbound_arrival_delay_min",
                        "origin_departures_3h"]
        X_num = X[numeric_cols]

        runs, summary = evaluate(
            lambda seed: DecisionTreeClassifier(max_depth=2, random_state=seed),
            X_num,
            dev,
            seeds=(42,),
        )
        self.assertEqual(len(runs), 8)  # 4 folds * (model + majority)
        self.assertIn("roc_auc", summary.columns.levels[1])


if __name__ == "__main__":
    unittest.main()
