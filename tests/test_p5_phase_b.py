import unittest

import numpy as np
import pandas as pd

from src.phase_b_boosting import NativeCategories, tune_run
from src.splits import OBSERVED, TIME


class TuningTests(unittest.TestCase):
    def test_native_encoding_uses_training_categories(self):
        train = pd.DataFrame({"airport": ["JFK", "LAX"]})
        unseen = pd.DataFrame({"airport": ["DEN", None]})
        light = NativeCategories(["airport"], False).fit(train)
        self.assertTrue(light.transform(unseen).airport.isna().all())
        cat = NativeCategories(["airport"], True).fit(train)
        self.assertEqual(cat.transform(unseen).airport.iloc[0], "DEN")
        self.assertFalse(cat.transform(unseen).airport.isna().any())

    def test_stopping_uses_inner_dates_and_refits_on_outer_draw(self):
        times = pd.date_range("2025-01-01", periods=120, tz="UTC")
        meta = pd.DataFrame({TIME: times, OBSERVED: times, "LATE": np.arange(120) % 2})
        meta.loc[25, OBSERVED] = times[103]
        X = meta[[TIME]]
        fits, probes = [], []

        class Constant:
            def fit(self, frame, y):
                fits.append(frame.index)
                return self

            def predict_proba(self, frame):
                probes.append(frame.index)
                return np.full((len(frame), 2), 0.5)

        model, rounds, sampled = tune_run(X, meta, np.arange(105), np.arange(105,120),
                                          42, lambda n: Constant())
        self.assertEqual(rounds, 10)
        self.assertEqual(len(fits), 3)
        self.assertNotIn(25, fits[0])
        self.assertLess(meta.loc[fits[0], OBSERVED].max() + pd.Timedelta(days=1),
                        meta.loc[probes[0], TIME].min())
        np.testing.assert_array_equal(fits[-1], sampled)
        self.assertLess(max(sampled), 105)
