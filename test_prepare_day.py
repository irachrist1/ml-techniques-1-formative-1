"""Small artificial fixtures test arithmetic; never used as assignment data."""
import unittest

import numpy as np
import pandas as pd

from prepare_day import aggregate_chunk


class AggregationTests(unittest.TestCase):
    def test_country_rows_sum_across_chunks_and_missing_stays_unobserved(self):
        sums = np.zeros((2, 10000))
        counts = np.zeros((2, 10000), dtype=np.uint32)
        frame = pd.DataFrame({'square': [1, 1, 2, 1], 'timestamp': [0, 0, 0, 600000],
                              'internet': [2., 3., np.nan, 0.]})
        for chunk in [frame.iloc[:1], frame.iloc[1:]]:
            aggregate_chunk(chunk, sums, counts, 0)
        self.assertEqual(sums[0, 0], 5)
        self.assertEqual(counts[0, 0], 2)
        self.assertEqual(counts[0, 1], 0)
        self.assertEqual(counts[1, 0], 1)  # observed zero differs from missing

    def test_one_bin_split_across_every_chunk_boundary_sums_the_same(self):
        # A square-interval bin whose country rows straddle chunk edges must not
        # be double counted or dropped, whatever the chunk size happens to be.
        rows = pd.DataFrame({'square': [7] * 9, 'timestamp': [0] * 9, 'internet': [float(i) for i in range(9)]})
        reference = None
        for size in range(1, 10):
            sums = np.zeros((2, 10000))
            counts = np.zeros((2, 10000), dtype=np.uint32)
            for start in range(0, len(rows), size):
                aggregate_chunk(rows.iloc[start:start + size], sums, counts, 0)
            self.assertEqual(counts[0, 6], 9)
            if reference is None:
                reference = sums[0, 6]
            self.assertEqual(sums[0, 6], reference)
        self.assertEqual(reference, sum(range(9)))

    def test_invalid_rows_fail_before_mutation(self):
        bad_rows = [(0, 0, 1), (10001, 0, 1), (1, 1, 1), (1, 1200000, 1), (1, 0, -1), (1, 0, np.inf)]
        for square, timestamp, value in bad_rows:
            sums = np.zeros((2, 10000))
            counts = np.zeros((2, 10000), dtype=np.uint32)
            frame = pd.DataFrame({'square': [square], 'timestamp': [timestamp], 'internet': [value]})
            with self.assertRaises(ValueError):
                aggregate_chunk(frame, sums, counts, 0)
            self.assertEqual(counts.sum(), 0)


if __name__ == '__main__':
    unittest.main()
