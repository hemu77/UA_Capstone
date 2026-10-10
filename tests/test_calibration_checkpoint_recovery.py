import json
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import recover_calibration_v6_checkpoint as repair


class CheckpointTests(unittest.TestCase):
    def test_failed_replace_preserves_original_and_recovery_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'checkpoint.json'
            recovery = Path(folder)/'recovery.json'
            repair.c.write_json(path, {'old':True})
            original, writer = path.read_bytes(), repair.c.write_json
            def fail_at_replace(destination, value):
                if destination == path:
                    raise PermissionError('offline injected checkpoint failure')
                writer(destination,value)
            with patch.object(repair.c,'write_json',side_effect=fail_at_replace):
                with self.assertRaises(PermissionError):
                    repair.replace_checkpoint(path,recovery,{'status':'VERIFIED'}, {'new':True})
            self.assertEqual(path.read_bytes(),original)
            self.assertEqual((Path(folder)/'checkpoint_original.json').read_bytes(),original)
            self.assertEqual(json.loads(recovery.read_text()),{'status':'VERIFIED'})
            with self.assertRaises(ValueError):
                repair.replace_checkpoint(path,recovery,{}, {})

    def test_only_parse_addition_can_be_reconciled(self):
        response = dict(text='1', finish_reason='stop', usage={'prompt_tokens':10})
        row = ['rid','main',.02,'received','{}',json.dumps(response,ensure_ascii=False)]
        saved = dict(request_id='rid',row_sha256=repair.c.frozen.digest(row))
        with self.assertRaises(ValueError):
            repair.validate_transition(row,saved)
        response['parse'] = dict(valid=True,error=None,duplicate_edges=[])
        row[5] = json.dumps(response,ensure_ascii=False)
        self.assertEqual(repair.validate_transition(row,saved)['row_sha256'],repair.c.frozen.digest(row))
        row[2] = .03
        with self.assertRaises(ValueError):
            repair.validate_transition(row,saved)
        row[2], row[3] = .02, 'uncertain'
        with self.assertRaises(ValueError):
            repair.validate_transition(row,saved)
