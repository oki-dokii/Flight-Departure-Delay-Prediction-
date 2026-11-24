import unittest
import numpy as np
import pandas as pd
from src.drift import psi, ks_effect, kitagawa


class DriftTests(unittest.TestCase):
    def test_training_bins_and_missing_unseen(self):
        a=pd.Series([0.,1.,2.,3.,np.nan])
        score,bins,definition=psi(a,a)
        self.assertAlmostEqual(score,0)
        _,_,changed=psi(a,pd.Series([100.,200.,300.,400.,np.nan]))
        self.assertEqual(definition,changed)
        self.assertIn('__MISSING__',set(bins.bin))
        score,bins,_=psi(pd.Series(['A']*100),pd.Series(['B']*100),True)
        self.assertGreater(score,0)
        self.assertIn('__RARE_UNSEEN__',set(bins.bin))
        self.assertEqual(ks_effect([0,0],[1,1]),1)

    def test_decomposition_exact_with_unsupported_cells(self):
        train=pd.DataFrame({'cell':['a','a','b'], 'LATE':[0,1,0]})
        valid=pd.DataFrame({'cell':['a','c'], 'LATE':[1,1]})
        result=kitagawa(train,valid)
        self.assertAlmostEqual(result.composition.sum()+result.within.sum(),
                               valid.LATE.mean()-train.LATE.mean())
