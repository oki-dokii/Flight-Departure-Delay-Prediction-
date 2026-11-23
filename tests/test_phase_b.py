import unittest

import numpy as np
import pandas as pd

from src.evaluate import evaluate, training_sample
from src.splits import OBSERVED, TEST_START, TIME, get_folds


class PhaseBTests(unittest.TestCase):
    def test_holdout_guard_precedes_fit(self):
        metadata = pd.DataFrame({TIME: [TEST_START], OBSERVED: [TEST_START], "LATE": [0]})
        with self.assertRaisesRegex(ValueError, "Test-period"):
            evaluate(lambda seed: self.fail("fitted on test"), metadata[[TIME]], metadata)

    def test_common_monthly_samples_and_checkpoint_eligibility(self):
        dates = pd.date_range("2025-01-01", "2025-10-30", freq="6h", tz="UTC")
        frame = pd.DataFrame({TIME: dates, OBSERVED: dates})
        frame.loc[1, OBSERVED] -= pd.Timedelta(hours=1)
        train, valid = get_folds(frame)[-1]
        self.assertNotIn(1, train)
        a = training_sample(train, frame, 42)
        b = training_sample(train, frame, 73)
        np.testing.assert_array_equal(a, training_sample(train, frame, 42))
        self.assertFalse(np.array_equal(a, b))
        self.assertTrue(set(a).issubset(train))
        self.assertFalse(set(a).intersection(valid))
        self.assertEqual(set(frame[TIME].iloc[a].dt.month), set(frame[TIME].iloc[train].dt.month))
