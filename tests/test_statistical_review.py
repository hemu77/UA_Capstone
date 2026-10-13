import unittest

import numpy as np
from scipy.stats import t

from statistical_review import family_summary, holm, OUTPUT


class StatisticalReviewTests(unittest.TestCase):
    def test_new_statistical_fixtures_stay_out_of_v5_evidence_folders(self):
        self.assertEqual(OUTPUT.parts[-3:], ('outputs', 'statistical_review_v6', 'report.json'))

    def test_holm_matches_hand_calculation_and_keeps_family_size(self):
        np.testing.assert_allclose(holm([.01, .04, .03, 1]), [.04, .09, .09, 1])
        self.assertGreater(holm([.01, 1])[0], holm([.01])[0])
        with self.assertRaises(ValueError):
            holm([float('nan')])

    def test_missing_and_zero_variance_never_produce_fake_t_significance(self):
        rows = family_summary([[.3, .3, .3, .3], [0, .1, np.nan, .2]])
        self.assertEqual(rows[0]['status'], 'ZERO_VARIANCE_NO_T_INFERENCE')
        self.assertIsNone(rows[0]['paired_t_holm_p'])
        self.assertIsNone(rows[1]['paired_t_holm_p'])
        self.assertIsNone(rows[1]['bounded_ci'])
        self.assertIsNone(family_summary([[-.05]*32])[0]['paired_t_holm_p'])

    def test_intervals_respect_declared_bounds_and_multiplicity(self):
        values = np.array([-.1, .2, .3, .1, -.2, .4, .1, -.1])
        single = family_summary([values])[0]
        expected = 2*t.sf(abs(values.mean()/(values.std(ddof=1)/np.sqrt(8))), 7)
        self.assertAlmostEqual(single['paired_t_p'], expected)
        large = family_summary([values]*24)[0]
        self.assertLessEqual(large['bounded_ci'][0], single['bounded_ci'][0])
        self.assertGreaterEqual(large['bounded_ci'][1], single['bounded_ci'][1])
        self.assertGreaterEqual(large['paired_t_holm_p'], single['paired_t_holm_p'])
        with self.assertRaisesRegex(ValueError, 'outside'):
            family_summary([[0, 1.01]])
