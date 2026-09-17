import unittest
import numpy as np
import pandas as pd
from experiments import windows,TRAIN_END,TEST_START,TEST_END

class WindowTests(unittest.TestCase):
    def setUp(self):
        times=np.arange(TRAIN_END-3*144*600000,TEST_END,600000,dtype=np.int64)
        self.frame=pd.DataFrame({'1':np.arange(len(times),dtype=float)},index=times)

    def test_scaling_uses_only_training_and_histories_precede_targets(self):
        data,scale=windows(self.frame,1,36)
        changed=self.frame.copy();changed.loc[changed.index>=TRAIN_END,'1']+=1000000
        _,other=windows(changed,1,36)
        self.assertEqual(scale,other)
        first=data['test'];target=first['timestamps'][0];i=self.frame.index.get_loc(target)
        reconstructed=first['x'][0,:,0]*scale['std']+scale['mean']
        np.testing.assert_allclose(reconstructed,self.frame['1'].iloc[i-36:i],rtol=1e-6)
        self.assertEqual(first['timestamps'][0],TEST_START)
        self.assertTrue(np.all(first['timestamps']<TEST_END))
        self.assertEqual(first['seasonal'][0],self.frame['1'].iloc[i-144])

    def test_all_lookbacks_score_same_targets_and_missing_targets_not_imputed(self):
        self.frame.loc[TEST_START+600000,'1']=np.nan
        short,_=windows(self.frame,1,36);long,_=windows(self.frame,1,144)
        np.testing.assert_array_equal(short['test']['timestamps'],long['test']['timestamps'])
        self.assertNotIn(TEST_START+600000,short['test']['timestamps'])
        self.assertEqual(len(short['test']['timestamps']),1008-145)

    def test_missing_timestamp_rejected(self):
        with self.assertRaises(ValueError):windows(self.frame.drop(self.frame.index[10]),1,36)

if __name__=='__main__':unittest.main()
