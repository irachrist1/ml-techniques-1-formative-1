"""Window construction, leakage boundaries and eligibility, on artificial fixtures."""
import unittest

import numpy as np
import pandas as pd

from experiments import windows, receptive_field, TRAIN_END, TEST_START, TEST_END


class WindowTests(unittest.TestCase):
    def setUp(self):
        times = np.arange(TRAIN_END - 3 * 144 * 600000, TEST_END, 600000, dtype=np.int64)
        self.frame = pd.DataFrame({'1': np.arange(len(times), dtype=float)}, index=times)

    def test_scaling_uses_only_training_and_histories_precede_targets(self):
        data, scale = windows(self.frame, 1, 36)
        changed = self.frame.copy()
        changed.loc[changed.index >= TRAIN_END, '1'] += 1000000
        _, other = windows(changed, 1, 36)
        self.assertEqual(scale, other)
        first = data['test']
        target = first['timestamps'][0]
        i = self.frame.index.get_loc(target)
        reconstructed = first['x'][0, :, 0] * scale['std'] + scale['mean']
        np.testing.assert_allclose(reconstructed, self.frame['1'].iloc[i - 36:i], rtol=1e-6)
        self.assertEqual(first['timestamps'][0], TEST_START)
        self.assertTrue(np.all(first['timestamps'] < TEST_END))
        self.assertEqual(first['seasonal'][0], self.frame['1'].iloc[i - 144])

    def test_corrupting_the_test_week_cannot_touch_training_or_validation(self):
        clean, scale = windows(self.frame, 1, 36)
        poisoned = self.frame.copy()
        poisoned.loc[poisoned.index >= TEST_START, '1'] = 1e9
        dirty, other = windows(poisoned, 1, 36)
        self.assertEqual(scale, other)
        for split in ['train', 'validation']:
            np.testing.assert_array_equal(clean[split]['x'], dirty[split]['x'])
            np.testing.assert_array_equal(clean[split]['delta'], dirty[split]['delta'])
            np.testing.assert_array_equal(clean[split]['y'], dirty[split]['y'])

    def test_all_lookbacks_score_same_targets_and_missing_targets_not_imputed(self):
        self.frame.loc[TEST_START + 600000, '1'] = np.nan
        short, _ = windows(self.frame, 1, 36)
        long, _ = windows(self.frame, 1, 144)
        np.testing.assert_array_equal(short['test']['timestamps'], long['test']['timestamps'])
        self.assertNotIn(TEST_START + 600000, short['test']['timestamps'])
        self.assertEqual(len(short['test']['timestamps']), 1008 - 145)

    def test_split_boundaries_are_half_open_and_exact(self):
        data, _ = windows(self.frame, 1, 36)
        self.assertTrue(np.all(data['train']['timestamps'] < TRAIN_END))
        self.assertEqual(data['validation']['timestamps'][0], TRAIN_END)
        self.assertTrue(np.all(data['validation']['timestamps'] < TEST_START))
        self.assertEqual(data['test']['timestamps'][0], TEST_START)
        self.assertEqual(data['test']['timestamps'][-1], TEST_END - 600000)
        self.assertEqual(len(data['validation']['y']), 1008)
        self.assertEqual(len(data['test']['y']), 1008)

    def test_persistence_column_is_the_immediately_preceding_observation(self):
        data, _ = windows(self.frame, 1, 36)
        d = data['test']
        for j in range(0, len(d['y']), 97):
            i = self.frame.index.get_loc(int(d['timestamps'][j]))
            self.assertEqual(d['last'][j], self.frame['1'].iloc[i - 1])
            self.assertEqual(d['y'][j], self.frame['1'].iloc[i])

    def test_missing_timestamp_rejected(self):
        with self.assertRaises(ValueError):
            windows(self.frame.drop(self.frame.index[10]), 1, 36)

    def test_lookback_outside_the_common_history_is_rejected(self):
        for bad in [0, 145, -1]:
            with self.assertRaises(ValueError):
                windows(self.frame, 1, bad)

    def test_receptive_field_matches_the_reported_formula(self):
        self.assertEqual(receptive_field([1, 2, 4, 8, 16, 32, 64]), 255)
        self.assertEqual(receptive_field([1, 2, 4, 8, 16]), 63)


if __name__ == '__main__':
    unittest.main()
