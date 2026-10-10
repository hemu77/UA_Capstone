from pathlib import Path
import unittest
from unittest.mock import Mock

from calibration_v6_io import write_checkpoint


class CheckpointIOTests(unittest.TestCase):
    def test_transient_permission_error_retries_identical_write_only(self):
        journal = Path('fixture-journal')
        path = journal/('a'*64+'.json')
        value = dict(request_id='a'*64,row_sha256='unchanged')
        events, sleep, persist = [], Mock(), Mock()
        writer = Mock(side_effect=[PermissionError(),PermissionError(),None])
        write_checkpoint(path,value,journal,events,writer,sleep,persist)
        self.assertEqual(writer.call_count,3)
        self.assertTrue(all(call.args==(path,value) for call in writer.call_args_list))
        self.assertEqual([c.args[0] for c in sleep.call_args_list],[.05,.1])
        self.assertEqual(len(events),2)
        self.assertEqual(persist.call_count,3)
        self.assertEqual(events[-1]['outcome'],'write_succeeded')

    def test_audit_write_failure_stops_before_another_checkpoint_attempt(self):
        journal = Path('fixture-journal')
        writer, sleep = Mock(side_effect=PermissionError()), Mock()
        with self.assertRaises(PermissionError):
            write_checkpoint(journal/('a'*64+'.json'),dict(request_id='a'*64),journal,[],
                             writer,sleep,Mock(side_effect=PermissionError('offline audit failure')))
        self.assertEqual(writer.call_count,1)
        sleep.assert_not_called()

    def test_permanent_or_non_permission_failure_stops(self):
        journal = Path('fixture-journal')
        path, value = journal/('a'*64+'.json'),dict(request_id='a'*64)
        for error, count in [(PermissionError(),6),(ValueError(),1)]:
            with self.subTest(error=type(error).__name__):
                writer = Mock(side_effect=error)
                with self.assertRaises(type(error)):
                    write_checkpoint(path,value,journal,[],writer,Mock())
                self.assertEqual(writer.call_count,count)

    def test_unrelated_paths_are_never_retried(self):
        writer = Mock(side_effect=PermissionError())
        with self.assertRaises(PermissionError):
            write_checkpoint(Path('other/data.json'),{},Path('fixture-journal'),[],writer,Mock())
        self.assertEqual(writer.call_count,1)
