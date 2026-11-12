import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from analysis_figures import review_v5 as v5,deep_review as v4

ROOT=Path(__file__).resolve().parents[1]

class ReviewV5Tests(unittest.TestCase):
    def test_density_bands_match_paired_records(self):
        records,summary=v5.bands(ROOT)
        self.assertEqual(len(records),11)
        sparse=summary.set_index('density_band').loc['below 0.06']
        dense=summary.set_index('density_band').loc['at least 0.06']
        self.assertEqual((sparse.records,dense.records),(5,6))
        self.assertEqual((sparse.target_er_strictly_closer,dense.global_strictly_closer),(4,5))
        self.assertAlmostEqual(sparse.global_w1,.148408,places=6)
        self.assertAlmostEqual(sparse.target_er_w1,.110316,places=6)
        np.testing.assert_allclose(summary.global_w1,records.groupby('density_band')['global'].mean())
    def test_exact_near_ties_and_unbounded_ratio(self):
        r=v5.table(ROOT,'efficiency_by_density').set_index('cohort')
        self.assertGreater(r.loc['FB02','target_ER']-r.loc['FB02','global'],.005)
        self.assertLess(r.loc['FB05','target_ER']-r.loc['FB05','global'],0)
        school=r.loc['School2013']
        self.assertGreater((school.WS-school.target_ER)/(school.WS-school['global']),1)
    def test_sealed_v4_untouched_and_v5_is_zero_paid(self):
        seal=v5.verify(ROOT)
        self.assertFalse(seal['primary_tests_changed'])
        self.assertEqual(seal['paid_calls'],0)
        v4.verify(ROOT)
