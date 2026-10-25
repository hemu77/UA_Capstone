"""Partial audit failures must not leave a stale verified report."""
from contextlib import nullcontext
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import inspect_repair_v8_checkpoint as checkpoint


class CheckpointTests(unittest.TestCase):
    def test_failed_rerun_invalidates_partial_clearance(self):
        v = checkpoint.recovery.v
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(v.r, 'DESTINATION', Path(directory)), \
                patch.object(v.r.base.frozen, 'workflow_lock', return_value=nullcontext()), \
                patch.object(checkpoint, '_inspect', side_effect=ValueError('fixture validation failure')):
            path = Path(directory)/'calibration'/checkpoint.recovery.CONTRACT[:12]/'checkpoint_inspection.json'
            v.r.write_json(path, dict(status='PARTIAL_REPLAY_VERIFIED_PENDING_TRANSPORT'))
            with self.assertRaisesRegex(ValueError, 'fixture validation'):
                checkpoint.inspect()
            record = v.read(path)
            self.assertEqual(record['status'], 'VALIDATION_PENDING')
            self.assertFalse(record['main_authorized'])
            self.assertFalse(record['completion_clearance'])
