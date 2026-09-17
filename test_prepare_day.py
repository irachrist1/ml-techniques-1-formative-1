"""Small artificial fixtures test arithmetic; never used as assignment data."""
import unittest
import numpy as np
import pandas as pd
from prepare_day import aggregate_chunk


class AggregationTests(unittest.TestCase):
    def test_country_rows_sum_across_chunks_and_missing_stays_unobserved(self):
        sums=np.zeros((2,10000)); counts=np.zeros((2,10000),dtype=np.uint32)
        frame=pd.DataFrame({'square':[1,1,2,1], 'timestamp':[0,0,0,600000], 'internet':[2.,3.,np.nan,0.]})
        for chunk in [frame.iloc[:1],frame.iloc[1:]]:
            aggregate_chunk(chunk,sums,counts,0)
        self.assertEqual(sums[0,0],5)
        self.assertEqual(counts[0,0],2)
        self.assertEqual(counts[0,1],0)
        self.assertEqual(counts[1,0],1)  # observed zero differs from missing

    def test_invalid_rows_fail_before_mutation(self):
        for square, timestamp, value in [(0,0,1),(10001,0,1),(1,1,1),(1,1200000,1),(1,0,-1),(1,0,np.inf)]:
            sums=np.zeros((2,10000));counts=np.zeros((2,10000),dtype=np.uint32)
            frame=pd.DataFrame({'square':[square],'timestamp':[timestamp],'internet':[value]})
            with self.assertRaises(ValueError):aggregate_chunk(frame,sums,counts,0)
            self.assertEqual(counts.sum(),0)


if __name__=='__main__':unittest.main()
