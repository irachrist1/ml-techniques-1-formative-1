"""Regression cases for parser, replay, checkpoint and diagnostic failures."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import predict_saved
from prepare_day import read_projected
from summarize_results import failure_neighbors, COLORS
from verify_submission import validate_training_record
from analyze import coverage_robustness


class ReviewRegressions(unittest.TestCase):
    def test_parser_rejects_ids_before_narrowing_in_full_and_chunked_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'data.tsv'
            for bad in ['4294967297', '-4294967295', '1.5', '10001', '0', '18446744073709551617']:
                p.write_text(f'{bad}\t0\t39\t0\t0\t0\t0\t1\n')
                with self.assertRaises((ValueError, OverflowError)):
                    read_projected(p)
                with self.assertRaises((ValueError, OverflowError)):
                    list(read_projected(p, chunksize=1))
            p.write_text('10000\t0\t39\t0\t0\t0\t0\t0\n')
            self.assertEqual(read_projected(p).square.dtype, np.dtype('int32'))
            self.assertEqual(read_projected(p).internet.iloc[0], 0)

    def replay_fixture(self, directory, times):
        root = Path(directory)
        (root / 'results').mkdir()
        pd.DataFrame({'1': [1., 2., 3., 4.]}, index=times).rename_axis('timestamp_ms').to_csv(root / 'results/selected_series.csv')
        run = root / 'run'
        run.mkdir()
        (run / 'summary.json').write_text(json.dumps({'runs': [{'model': 'RidgeAR', 'area': 1,
            'config': {'lookback': 3}, 'scaler': {'mean': 0., 'std': 1.}}]}))
        np.savez(run / 'RidgeAR.npz', coef=np.zeros(3), intercept=0.)
        return root, run

    def test_replay_rejects_off_grid_and_duplicate_compensating_for_gap(self):
        with tempfile.TemporaryDirectory() as directory:
            root, run = self.replay_fixture(directory, [0, 600000, 1200000, 1800000])
            with patch.object(predict_saved, 'ROOT', root):
                self.assertEqual(predict_saved.predict(run, 'RidgeAR', '1970-01-01T00:30:00Z'), 3.)
                for target in ['1970-01-01T00:25:00Z', '1970-01-01T00:30:00.000001Z']:
                    with self.assertRaisesRegex(ValueError, 'grid'):
                        predict_saved.predict(run, 'RidgeAR', target)
                p = root / 'results/selected_series.csv'
                frame = pd.read_csv(p)
                frame['timestamp_ms'] = [0, 0, 1200000, 1800000]
                frame.to_csv(p, index=False)
                with self.assertRaisesRegex(ValueError, 'exactly once'):
                    predict_saved.predict(run, 'RidgeAR', '1970-01-01T00:30:00Z')

    def test_epoch_metadata_follows_min_delta_not_argmin(self):
        history = {'loss': [.2, .1], 'val_loss': [.1, .099995]}
        entry = {'model': 'LSTM', 'epochs_run': 2, 'epoch_budget': 5, 'config': {'epochs': 5},
                 'stopped_early': True, 'best_epoch': 1, 'lowest_val_loss_epoch': 2}
        validate_training_record(entry, history)
        with self.assertRaisesRegex(AssertionError, 'checkpoint'):
            validate_training_record({**entry, 'best_epoch': 2}, history)
        with self.assertRaises(AssertionError):
            validate_training_record({**entry, 'epochs_run': 5}, history)

    def test_failure_neighbors_at_both_boundaries(self):
        table = pd.DataFrame({'actual': [1., 2., 3.], 'prediction': [1.1, 2.1, 3.1]})
        predictions = {(1, kind): table for kind in COLORS}
        previous, following = failure_neighbors(predictions, 1, 0)
        self.assertIsNone(previous)
        self.assertEqual(following['actual'], 2.)
        previous, following = failure_neighbors(predictions, 1, 2)
        self.assertEqual(previous, 2.)
        self.assertIsNone(following)

    def test_coverage_scenario_is_not_an_upper_bound(self):
        rank = pd.DataFrame({'total_internet_activity': [100., 90., 80., 1.], 'coverage': [1., 1., 1., .5]})
        result = coverage_robustness(rank)
        self.assertFalse(result['is_upper_bound'])
        self.assertFalse(result['top3_changes_under_observed_mean_imputation'])
        self.assertGreater(1. + 1000., 80.)  # an unobserved value can reverse the ranking


if __name__ == '__main__':
    unittest.main()
