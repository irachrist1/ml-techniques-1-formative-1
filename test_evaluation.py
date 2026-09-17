"""Artificial unit fixtures only, never assignment data or empirical results."""
import unittest
from evaluation import score, rolling_examples, milan_midnight_ms


class EvaluationTests(unittest.TestCase):
    def test_metrics_and_zero_policy(self):
        result = score([0, 2, 4], [1, 1, 6])
        self.assertAlmostEqual(result['mae'], 4/3)
        self.assertAlmostEqual(result['rmse'], 2**0.5)
        self.assertEqual(result['mape_percent_nonzero'], 50)
        self.assertEqual(result['mape_excluded_zero_targets'], 1)
        self.assertIsNone(score([0], [1])['mape_percent_nonzero'])

    def test_invalid_metrics(self):
        for a, p in [([], []), ([1], [1, 2]), ([float('nan')], [1])]:
            with self.assertRaises(ValueError):
                score(a, p)

    def test_split_uses_history_but_excludes_future_and_end(self):
        rows = list(rolling_examples([0, 600000, 1200000, 1800000], [1, 2, 3, 4], 2, 1200000, 1800000))
        self.assertEqual(rows, [(1200000, [1, 2], 3)])

    def test_gaps_and_missing_values_are_not_silently_joined(self):
        self.assertEqual(list(rolling_examples([0, 1200000, 1800000], [1, 2, 3], 2, 0, 2400000)), [])
        self.assertEqual(list(rolling_examples([0, 600000, 1200000], [1, None, 3], 2, 0, 1800000)), [])

    def test_duplicate_time_rejected(self):
        with self.assertRaises(ValueError):
            list(rolling_examples([0, 0], [1, 2], 1, 0, 600000))

    def test_evaluation_week_has_1008_intervals(self):
        self.assertEqual((milan_midnight_ms('2013-12-23')-milan_midnight_ms('2013-12-16'))//600000, 1008)


if __name__ == '__main__':
    unittest.main()
