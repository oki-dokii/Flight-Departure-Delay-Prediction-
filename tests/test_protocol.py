import unittest

import pandas as pd
from sklearn.dummy import DummyClassifier

from src.evaluate import evaluate
from src.prepare import clean_month, departure_times, hhmm_minutes
from src.splits import OBSERVED, TIME, get_folds, get_test_split


class PreparationTests(unittest.TestCase):
    def test_midnight_dst_and_departure_label(self):
        source = pd.DataFrame({
            "FlightDate": ["2025-03-08", "2025-03-09", "2025-03-09",
                           "2025-07-01", "2025-07-01", "2025-07-01", "2025-07-01"],
            "Origin": ["JFK"] * 7,
            "Tail_Number": ["N1", None, "N2", "N3", "N4", "N5", "N6"],
            "Cancelled": [0, 0, 0, 0, 1, 0, 0],
            "Diverted": [0, 1, 0, 0, 0, 0, 0],
            "CRSDepTime": [2350, 150, 220, 2400, 100, 20, 1200],
            "DepTime": [5, 310, 320, 2350, 115, 2330, 1214],
            "DepDelay": [15, 20, 60, -10, 15, -50, 14],
            "CRSElapsedTime": [60] * 7,
            "ActualElapsedTime": [60] * 7,
        })
        frame, counts = clean_month(source, {"JFK": "America/New_York"})
        self.assertEqual(list(frame.index), [0, 1, 3, 5, 6])
        self.assertEqual(list(frame.LATE), [1, 1, 0, 0, 0])
        self.assertEqual(frame.loc[0, OBSERVED], pd.Timestamp("2025-03-09 05:05Z"))
        self.assertEqual(frame.loc[1, OBSERVED], pd.Timestamp("2025-03-09 07:10Z"))
        self.assertEqual(frame.loc[3, TIME], pd.Timestamp("2025-07-02 04:00Z"))
        self.assertEqual(frame.loc[5, OBSERVED], pd.Timestamp("2025-07-01 03:30Z"))
        self.assertTrue(frame.loc[1, "tail_number_missing"])
        self.assertTrue(pd.isna(frame.loc[1, "actual_arrival_utc"]))
        self.assertEqual(counts["outside_date_window"]["after"], 5)


    def test_fall_back_requires_unique_instant(self):
        frame = pd.DataFrame({"FlightDate": ["2025-11-02"] * 2})
        zones = pd.Series(["America/New_York"] * 2)
        scheduled, actual = departure_times(
            frame, zones, hhmm_minutes(pd.Series([150, 100])),
            hhmm_minutes(pd.Series([110, 105])), pd.Series([20, 5]),
        )
        self.assertTrue(pd.isna(scheduled.iloc[0]))
        self.assertTrue(pd.isna(actual.iloc[0]))
        self.assertTrue(pd.isna(scheduled.iloc[1]))
        self.assertTrue(pd.isna(actual.iloc[1]))


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        times = pd.to_datetime(
            ["2025-01-01", "2025-06-29", "2025-06-30", "2025-07-01",
             "2025-07-02", "2025-08-01", "2025-08-02", "2025-09-01",
             "2025-09-02", "2025-10-01", "2025-10-02"], utc=True,
        ).repeat(2)
        self.development = pd.DataFrame({
            TIME: times,
            OBSERVED: times + pd.to_timedelta([0, 15] * 11, unit="min"),
            "LATE": [0, 1] * 11,
        }, index=range(100, 122))

    def test_folds_and_baseline(self):
        dev = self.development
        dev.loc[103, OBSERVED] = pd.Timestamp("2025-06-30", tz="UTC")
        folds = get_folds(dev)
        self.assertEqual(len(folds), 4)
        self.assertEqual(list(folds[0][0]), [0, 1, 2])
        self.assertEqual(list(folds[0][1]), [6, 7, 8, 9])
        runs, summary = evaluate(
            lambda seed: DummyClassifier(strategy="prior", random_state=seed),
            pd.DataFrame({"schedule_hour": 0}, index=dev.index),
            dev,
        )
        self.assertEqual(len(runs), 24)
        self.assertEqual(summary.loc["majority", ("mean", "roc_auc")], 0.5)
        self.assertEqual(summary.loc["majority", ("mean", "pr_auc")], 0.5)

    def test_holdout_boundaries_on_synthetic_rows(self):
        dates = pd.to_datetime(["2025-10-30", "2025-10-31", "2025-11-01",
                                "2025-11-02", "2026-01-02"], utc=True)
        frame = pd.DataFrame({TIME: dates, OBSERVED: dates})
        train, test = get_test_split(frame)
        self.assertEqual(list(train), [0])
        self.assertEqual(list(test), [3])
        with self.assertRaisesRegex(ValueError, "only the frozen development period"):
            get_folds(frame)


if __name__ == "__main__":
    unittest.main()
