"""Window construction, leakage boundaries and eligibility, on artificial fixtures.

Every case here is built from a synthetic ramp, never from assignment data. Each
test is written so that it fails if a specific leakage guarantee is broken."""
import unittest

import numpy as np
import pandas as pd

from features import invert, to_forecast, windows
from models import receptive_field
from protocol import TEST_END, TEST_START, TRAIN_END


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
        reconstructed = invert(first['x'][0, :, 0], scale)
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

    def test_scaler_is_exactly_the_training_slice_statistics(self):
        """Fit-on-train-only, asserted directly rather than only by invariance."""
        _, scale = windows(self.frame, 1, 36)
        train_values = self.frame.loc[self.frame.index < TRAIN_END, '1'].to_numpy()
        self.assertAlmostEqual(scale['mean'], float(np.nanmean(train_values)))
        self.assertAlmostEqual(scale['std'], float(np.nanstd(train_values)))
        self.assertEqual(scale['fit_end_exclusive'], TRAIN_END)

    def test_no_lookback_lets_the_test_week_reach_training_or_validation(self):
        """The leakage guarantee must hold at every lookback, not just the one
        the tuning rounds happened to use."""
        poisoned = self.frame.copy()
        poisoned.loc[poisoned.index >= TEST_START, '1'] = 1e9
        for lookback in [1, 6, 36, 143, 144]:
            clean, scale = windows(self.frame, 1, lookback)
            dirty, other = windows(poisoned, 1, lookback)
            self.assertEqual(scale, other, lookback)
            for split in ['train', 'validation']:
                for key in ['x', 'delta', 'y', 'last', 'timestamps']:
                    np.testing.assert_array_equal(clean[split][key], dirty[split][key],
                                                  err_msg=f'{split}/{key} at lookback {lookback}')

    def test_the_correction_target_inverts_exactly_back_to_the_observed_level(self):
        """A model that predicted delta perfectly must score zero error. If this
        round trip is lossy, every model is penalised for the parameterization."""
        # A ramp gives every window the same step, so a lossy delta would round
        # identically everywhere and hide. Use a varied series instead.
        rng = np.random.RandomState(0)
        varied = pd.DataFrame({'1': 500 + 400 * np.sin(np.arange(len(self.frame)) * 2 * np.pi / 144)
                               + rng.gamma(2, 30, len(self.frame))}, index=self.frame.index)
        data, scale = windows(varied, 1, 36)
        for split in ['train', 'validation', 'test']:
            d = data[split]
            # Compare the recovered *correction*, not the level. The level is
            # dominated by persistence, so a lossy delta would still look correct
            # to any tolerance stated relative to y.
            recovered = to_forecast(d['last'], d['delta'], scale) - d['last']
            np.testing.assert_allclose(recovered, d['y'] - d['last'],
                                       rtol=1e-4, atol=1e-9, err_msg=split)

    def test_normalized_history_round_trips_through_invert(self):
        data, scale = windows(self.frame, 1, 36)
        d = data['validation']
        np.testing.assert_allclose(invert(d['x'][:, :, 0], scale)[:, -1], d['last'],
                                   rtol=1e-5, atol=1e-6)

    def test_receptive_field_matches_the_reported_formula(self):
        self.assertEqual(receptive_field([1, 2, 4, 8, 16, 32, 64]), 255)
        self.assertEqual(receptive_field([1, 2, 4, 8, 16]), 63)


if __name__ == '__main__':
    unittest.main()
