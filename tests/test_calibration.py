import unittest
import numpy as np
from src.calibration import calibration_metrics, reliability


class CalibrationTests(unittest.TestCase):
    def test_known_brier_and_ece(self):
        m, b = calibration_metrics([0, 1], [.25, .75], .5, bins=2)
        self.assertAlmostEqual(m['brier'], .0625)
        self.assertAlmostEqual(m['brier_skill'], .75)
        self.assertAlmostEqual(m['ece'], .25)
        self.assertEqual(b.n.sum(), 2)
        self.assertEqual(m['calibration_bias'], 0)

    def test_ties_constant_and_perfect(self):
        m, b = calibration_metrics([0,1,0,1], [.5]*4, .5)
        self.assertEqual(len(b), 1)
        self.assertEqual(m['ece'], 0)
        self.assertEqual(m['brier_skill'], 0)
        m, _ = calibration_metrics([0,1], [0,1], .5)
        self.assertEqual(m['brier'], 0)
        self.assertEqual(m['brier_skill'], 1)

    def test_day_blocks_reproducible(self):
        a = reliability([0,0,1,1], [.5]*4, days=['a','a','b','b'], bootstrap=100)
        b = reliability([0,0,1,1], [.5]*4, days=['a','a','b','b'], bootstrap=100)
        self.assertTrue(a.equals(b))
        self.assertEqual(a.observed_lo.iloc[0], 0)
        self.assertEqual(a.observed_hi.iloc[0], 1)
        for p in ([np.nan], [-.1], [1.1]):
            with self.assertRaises(ValueError): reliability([0],p)
