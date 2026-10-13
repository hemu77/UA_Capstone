"""Local write retries never repeat generation or mutate frozen source."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import repair_v8_io as recovery


class LocalWriteTests(unittest.TestCase):
    def test_checkpoint_retry_is_identical_and_logged(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            rid = 'a'*64
            path = target/'requests'/(rid+'.json')
            value = {'request_id': rid, 'row_sha256': 'b'*64}
            original = recovery.v.r.base.write_json
            calls = []
            def flaky(path, data):
                if path.parent.name == 'requests':
                    calls.append((path, data))
                    if len(calls) == 1:
                        raise PermissionError('offline simulated file lock')
                return original(path, data)
            with patch.object(recovery.v.r.base, 'write_json', flaky), \
                    patch.object(recovery.v.r.paid, 'OpenAI', side_effect=AssertionError('No API')):
                with recovery.local_writes(target, 'offline-fixture'):
                    recovery.v.r.base.write_json(path, value)
            self.assertEqual(calls, [(path, value), (path, value)])
            self.assertEqual(json.loads(path.read_text()), value)
            audit = recovery.v.read(next((target/'io_adapter').glob('*.json')))
            self.assertEqual(audit['status'], 'COMPLETED')
            self.assertEqual(audit['write_retries'][0]['outcome'], 'write_succeeded')

    def test_unrelated_exception_is_not_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            original = recovery.v.r.base.write_json
            count = []
            def broken(path, data):
                if path.parent.name == 'requests':
                    count.append(path)
                    raise OSError('offline disk full')
                return original(path, data)
            with patch.object(recovery.v.r.base, 'write_json', broken):
                with self.assertRaisesRegex(OSError, 'disk full'), recovery.local_writes(target, 'offline-fixture'):
                    recovery.v.r.base.write_json(target/'requests'/('a'*64+'.json'), {'request_id': 'a'*64})
            self.assertEqual(len(count), 1)


if __name__ == '__main__':
    unittest.main()
