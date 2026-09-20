"""Artificial unit fixtures only, never assignment data or empirical results."""
import unittest

from evaluation import milan_midnight_ms, score


class EvaluationTests(unittest.TestCase):
    def test_metrics_and_zero_policy(self):
        result = score([0, 2, 4], [1, 1, 6])
        self.assertAlmostEqual(result['mae'], 4 / 3)
        self.assertAlmostEqual(result['rmse'], 2 ** 0.5)
        self.assertEqual(result['mape_percent_nonzero'], 50)
        self.assertEqual(result['mape_excluded_zero_targets'], 1)
        self.assertIsNone(score([0], [1])['mape_percent_nonzero'])

    def test_invalid_metrics(self):
        for a, p in [([], []), ([1], [1, 2]), ([float('nan')], [1])]:
            with self.assertRaises(ValueError):
                score(a, p)

    def test_evaluation_week_has_1008_intervals(self):
        self.assertEqual((milan_midnight_ms('2013-12-23') - milan_midnight_ms('2013-12-16')) // 600000, 1008)

    def test_local_midnight_tracks_the_rome_offset(self):
        # Both dates are CET (+01:00) in 2013; the boundary must not drift to UTC.
        self.assertEqual(milan_midnight_ms('2013-12-16') % 600000, 0)
        self.assertEqual(milan_midnight_ms('2013-12-23') - milan_midnight_ms('2013-12-16'), 7 * 86400 * 1000)


if __name__ == '__main__':
    unittest.main()
