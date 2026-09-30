"""Small regression check for the offline handoff gate's identity contract."""
import unittest
from verify_viewer_sources import validate_run_ids


class ViewerIdentityTests(unittest.TestCase):
    def test_exact_unique_coverage_not_just_counts(self):
        runs = [dict(run_id='h', study='cultural'), dict(run_id='p', study='engineering_pilot')]
        validate_run_ids(runs, {'h'}, {'p'})
        with self.assertRaises(ValueError):
            validate_run_ids([runs[0], runs[0]], {'h'}, {'p'})
        with self.assertRaises(ValueError):
            validate_run_ids([runs[0], dict(run_id='q', study='engineering_pilot')], {'h'}, {'p'})
        with self.assertRaises(ValueError):
            validate_run_ids([dict(run_id='p', study='cultural'), dict(run_id='h', study='engineering_pilot')], {'h'}, {'p'})


if __name__ == '__main__':
    unittest.main()
