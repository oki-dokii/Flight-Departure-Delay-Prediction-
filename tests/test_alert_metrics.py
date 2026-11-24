import unittest

import pandas as pd

from src.alert_metrics import alert_metrics


class AlertMetricTests(unittest.TestCase):
    def test_id_breaks_ties_and_cost_counts_missed_delays(self):
        labels = pd.Series([0, 1] + [0] * 8, index=["b", "a", *"cdefghij"])
        scores = [0.9, 0.9] + [0.1] * 8
        metrics = alert_metrics(labels, scores)
        self.assertEqual(metrics["precision_at_10"], 1)
        self.assertEqual(metrics["lift_at_10"], 10)
        self.assertEqual(metrics["cost_at_10"], 0.1)
        self.assertEqual(metrics["precision_at_20"], 0.5)
