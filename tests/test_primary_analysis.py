import importlib.util
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

spec=importlib.util.spec_from_file_location('core',Path(__file__).resolve().parents[1]/'analysis_primary/core.py')
c=importlib.util.module_from_spec(spec); spec.loader.exec_module(c)

class PrimaryAnalysis(unittest.TestCase):
    def test_holm_keeps_suppressed_hypotheses_in_family(self):
        adjusted=c.holm_full([.001,None,*([.5]*22)])
        self.assertAlmostEqual(adjusted[0],.024)
        self.assertIsNone(adjusted[1])
        self.assertTrue(all(v is None or 0<=v<=1 for v in adjusted))

    def test_exact_two_sided_eight_block_minimum(self):
        self.assertEqual(c.exact_sign_flip(np.ones(8)),2/256)
        self.assertEqual(c.exact_sign_flip(np.zeros(8)),1.)

    def test_reject_supplement_even_when_count_matches(self):
        frame=pd.DataFrame([dict(cohort='supplement_initial',primary_member=False,valid_graph=True,run_id='supp_x')])
        with self.assertRaises(ValueError): c.validate_primary(frame, {'analysis':{}})

    def test_archived_agreement_allows_roundoff_not_changed_status(self):
        result=c.archived_agreement([{'status':'COMPLETE','p':.5+1e-15}],[{'status':'COMPLETE','p':.5}])
        self.assertLess(result['maximum_absolute_difference'],1e-12)
        with self.assertRaises(ValueError): c.archived_agreement([{'status':'PARTIAL'}],[{'status':'COMPLETE'}])
        with self.assertRaises(ValueError): c.archived_agreement([{'p':.500001}],[{'p':.5}])

    def test_real_primary_and_numeric_regressions(self):
        root=Path(__file__).resolve().parents[1]
        spec,frame,rows=c.load_inputs(root)
        c.validate_primary(frame,spec)
        changed=frame.copy(); changed.loc[0,'cohort']='supplement_initial'
        with self.assertRaises(ValueError): c.validate_primary(changed,spec)
        results=c.independent_contrasts(c.block_table(frame,spec),spec)
        hindi=next(r for r in results if r['hypothesis_id']=='iterative:us:hindi')
        self.assertAlmostEqual(hindi['mean'],-.045278893726652,places=12)
        self.assertAlmostEqual(hindi['holm_p'],.023831222391575,places=11)
        self.assertEqual(hindi['sign_flip_holm_p'],.1875)
        missing=next(r for r in results if r['hypothesis_id']=='global:us:japanese')
        self.assertEqual(missing['n'],5); self.assertIsNone(missing['raw_p'])
        self.assertIsNone(missing['sign_flip_p'])

if __name__=='__main__': unittest.main()
