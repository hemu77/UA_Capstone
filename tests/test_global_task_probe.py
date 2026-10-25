"""Offline checks for the eight-call extension; no provider connections."""
import contextlib
import io
import json
import unittest
from unittest.mock import patch

from tests import test_repair_v7 as helpers
import global_task_probe as probe
import finish_global_task_probe as finish


class TaskProbeTests(unittest.TestCase):
    setup_workspace = helpers.RepairTests.setup_workspace
    client = helpers.RepairTests.client
    response = staticmethod(helpers.RepairTests.response)
    def test_eight_cells_preserve_empty_option_and_match_reference_schedules(self):
        spec = probe.contract()
        self.assertEqual(len(spec['design']), 8)
        self.assertEqual({row['model'] for row in spec['design']}, set(probe.r.MODELS))
        self.assertEqual({row['seed'] for row in spec['design']}, {31000, 31001})
        self.assertTrue(all(row['global_arm'] == 'a0_b0' for row in spec['design']))
        text = probe.catalog()[0]['system']
        self.assertTrue(text.startswith(probe.INSTRUCTION))
        self.assertIn('NONE', text)

    def setup_probe(self):
        r = probe.r
        _, origin = self.setup_workspace()
        original = r.approve('global_probe')
        r.write_json(origin / 'summary.json', dict(status='COMPLETE_REVIEW_REQUIRED',
            contract_sha256=r.digest(r.contract('global_probe')), completed_graphs=64))
        probe.preflight()
        spec = probe.contract()
        target = probe.folder(spec)
        review = json.loads((origin / 'review.json').read_text())
        review['contract_sha256'] = r.digest(spec)
        r.write_json(target / 'review.json', review)
        probe.authorize()
        return original, spec, target

    def test_nonempty_roundtrip_resume_and_original_cap_cannot_reset(self):
        r = probe.r
        original, spec, target = self.setup_probe()
        calls = []
        def handler(request):
            if request.method == 'POST':
                calls.append(request)
                body = json.loads(request.content)
                self.assertTrue(body['messages'][0]['content'].startswith(probe.INSTRUCTION))
            return self.response(request, text='0, 1')
        self.client(handler)
        with contextlib.redirect_stdout(io.StringIO()):
            result = probe.execute()
            self.assertEqual(probe.execute(), result)
        self.assertEqual(result['completed_graphs'], 8)
        self.assertEqual(result['empty_graphs'], 0)
        self.assertEqual(len(calls), 8)
        auth_path = target.parent / 'authorization.json'
        auth = json.loads(auth_path.read_text())
        self.assertEqual(auth['ledger_start_usd'], original['ledger_start_usd'])
        self.assertEqual(auth['cumulative_ceiling_usd'], original['cumulative_ceiling_usd'])
        with self.assertRaises(ValueError):
            probe.authorize()
        auth['ledger_start_usd'] += 1
        auth['cumulative_ceiling_usd'] += 1
        r.write_json(auth_path, auth)
        with self.assertRaisesRegex(ValueError, 'original'):
            probe.execute()
        self.assertEqual(len(calls), 8)
        self.assertEqual(json.loads((target / 'summary.json').read_text())['status'], 'VALIDATION_PENDING')
        auth['ledger_start_usd'] -= 1
        auth['cumulative_ceiling_usd'] -= 1
        r.write_json(auth_path, auth)
        with contextlib.redirect_stdout(io.StringIO()):
            probe.execute()
        origin_summary = probe.origin_folder() / 'summary.json'
        damaged = json.loads(origin_summary.read_text())
        damaged['status'] = 'STOPPED'
        r.write_json(origin_summary, damaged)
        with self.assertRaisesRegex(ValueError, 'original'):
            probe.execute()
        self.assertEqual(len(calls), 8)
        self.assertEqual(json.loads((target / 'summary.json').read_text())['status'], 'VALIDATION_PENDING')

    def test_remaining_budget_is_shared_and_missing_origin_blocks_calls(self):
        r = probe.r
        original, spec, target = self.setup_probe()
        with contextlib.closing(r.paid.Budget(r.LEDGER, require_existing=True).db) as db:
            db.execute('INSERT INTO attempts (id,phase,cost,status,request,response) VALUES (?,?,?,?,?,?)',
                ('other-charge', 'main', original['cumulative_ceiling_usd'] - .000001, 'received', '{}', '{}'))
            db.commit()
        calls = []
        def handler(request):
            if request.method == 'POST':
                calls.append(request)
            return self.response(request)
        self.client(handler)
        with self.assertRaises(r.paid.BudgetExceeded):
            probe.execute()
        self.assertEqual(calls, [])
        (probe.origin_folder().parent / 'authorization.json').unlink()
        with self.assertRaises(FileNotFoundError):
            probe.execute()
        self.assertEqual(calls, [])

    def test_orphan_transport_receipt_blocks_reauthorization(self):
        r = probe.r
        _, _, target = self.setup_probe()
        authorization = target.parent / 'authorization.json'
        authorization.unlink()
        r.write_json(target / 'transport' / 'orphan.json', {'fixture': 'received evidence remains'})
        with patch.object(r.paid, 'OpenAI', side_effect=AssertionError('No client permitted')):
            with self.assertRaisesRegex(ValueError, 'attempts already exist'):
                probe.authorize()
        self.assertFalse(authorization.exists())

    def test_terminal_invalid_response_is_not_a_graph_or_replacement(self):
        r = probe.r
        _, spec, target = self.setup_probe()
        calls = []
        def handler(request):
            if request.method == 'POST':
                calls.append(request)
            return self.response(request, text='0, 1\n1, 0' if len(calls) == 6 else '0, 1')
        self.client(handler)
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(r.DiagnosticInvalid):
                probe.execute()
            self.assertEqual(len(calls), 6)
            with self.assertRaisesRegex(ValueError, 'exactly one'):
                finish.execute(verify_only=True)
            self.assertEqual(len(calls), 6)
            with contextlib.closing(r.paid.Budget(r.LEDGER, require_existing=True).db) as db:
                existing = {row[0] for row in db.execute('SELECT id FROM attempts')}
            original_parse = r.paid.Budget.record_parse
            def new_responses_only(budget, request_id, error=None):
                self.assertNotIn(request_id, existing, 'Cached evidence must be read-only')
                return original_parse(budget, request_id, error)
            with patch.object(r.paid.Budget, 'record_parse', new_responses_only):
                report = finish.execute()
            replayed = finish.execute(verify_only=True)
        self.assertEqual(len(calls), 8)
        self.assertEqual(report['completed_requests'], 8)
        self.assertEqual(report['completed_graphs'], 7)
        self.assertEqual(report['invalid_responses'], 1)
        self.assertEqual(report['status'], 'COMPLETE_WITH_INVALID_RESPONSES')
        self.assertEqual(replayed['stage_charge_usd'], report['stage_charge_usd'])
        self.assertTrue(replayed['cached_replay_ledger_unchanged'])
        self.assertEqual(len(list((target / 'runs').glob('*.adj'))), 7)
        self.assertEqual(len(list((target / 'failures').glob('*.json'))), 1)
        self.assertEqual(json.loads((target / 'summary.json').read_text())['status'], 'STOPPED')
        stopped_path = target / 'summary.json'
        stopped = json.loads(stopped_path.read_text())
        r.write_json(stopped_path, {**stopped, 'completed_graphs': 0})
        with self.assertRaisesRegex(ValueError, 'adapter evidence changed'):
            finish.execute()
        self.assertEqual(len(calls), 8)
        r.write_json(stopped_path, stopped)
        failed = json.loads(next((target / 'failures').glob('*.json')).read_text())
        rid = failed['requests'][0]['request_id']
        with contextlib.closing(r.paid.Budget(r.LEDGER, require_existing=True).db) as db:
            db.execute('DELETE FROM attempts WHERE id=?', (rid,))
            db.commit()
        (target / 'requests' / (rid + '.json')).unlink()
        with self.assertRaisesRegex(ValueError, 'Orphan'):
            finish.execute()
        self.assertEqual(len(calls), 8)
