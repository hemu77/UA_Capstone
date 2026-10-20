"""Prospective stage contracts and fail-closed recovery; no real API calls."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx
from openai import OpenAI

import repair_v8 as v
from tests import test_repair_v7 as fixtures


class ContractTests(unittest.TestCase):
    def test_independent_wording_and_multilingual_calibration(self):
        wording = v.contract('wording_probe')
        calibration = v.contract('calibration')
        self.assertEqual(len(wording['design']), 96)
        self.assertEqual(len(calibration['design']), 104)
        self.assertEqual(wording['dependencies'], [])
        self.assertEqual(calibration['dependencies'], ['wording_probe'])
        self.assertFalse(calibration['main_authorized'])
        self.assertEqual(set(v.CONSTRUCTION), set(v.r.LANGUAGES))
        for row in v.catalog():
            if row['method'] == 'global':
                self.assertTrue(row['system'].startswith(v.CONSTRUCTION[row['language']]))
                self.assertIn('NONE', row['system'])
            else:
                self.assertNotIn(v.CONSTRUCTION[row['language']], row['system'])
        with self.assertRaises(ValueError):
            v.contract('main')


class ExecutionTests(unittest.TestCase):
    response = staticmethod(fixtures.RepairTests.response)

    def setup_workspace(self, methods=('global', 'local', 'sequential', 'iterative'), wordings=('reference',)):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        r = v.r
        self.enterContext(patch.object(r, 'DESTINATION', root/'repair'))
        self.enterContext(patch.object(r, 'LEDGER', root/'ledger.sqlite'))
        self.enterContext(patch.object(r.base.frozen, 'DESTINATION', root/'lock'))
        self.enterContext(patch.object(r, 'design', return_value=[dict(model='gpt-6-luna', method=method,
            culture='us', language='english', repetition=0, seed=32000, wording=wording,
            global_arm='a0_b0' if method == 'global' else None) for method in methods for wording in wordings]))
        with contextlib.closing(r.paid.Budget(r.LEDGER).db):
            pass
        with v.isolated():
            spec = r.prepare('wording_probe')
            target = r.folder(spec)
            r.write_json(target/'preflight.json', dict(status='OFFLINE_PREFLIGHT_PASSED', contract_sha256=r.digest(spec)))
            r.write_json(target/'review.json', dict(contract_sha256=r.digest(spec), prompt_semantics_reviewed=True,
                diagnostic_design_reviewed=True, spending_safeguards_reviewed=True, human_bilingual_validation=False,
                review_reference='offline fixture only', rates=spec['rates'], prices_verified_on=v.date.today().isoformat(),
                price_evidence='offline fixture only'))
            approved = r.approve('wording_probe')
        self.enterContext(patch.object(r.paid, 'credential', return_value='offline-test-key'))
        return spec, target, approved

    def client(self, handler):
        self.enterContext(patch.object(v.r.paid, 'OpenAI', side_effect=lambda **kwargs: OpenAI(
            api_key='offline-test-key', max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(handler)))))

    def test_all_methods_and_terminal_global_resume_without_paid_writes(self):
        spec, target, approved = self.setup_workspace()
        calls = []
        fixtures = {}
        for cell in v.r.cells(spec):
            fixture = v.r.base.FixtureCaller(cell)
            with v.adapter(cell, spec):
                v.r.base.run_cell(cell, fixture)
            for request in fixture.requests:
                fixtures[json.dumps(request['messages'], sort_keys=True)] = request['text']
        def handler(request):
            if request.method == 'GET':
                return self.response(request)
            body = json.loads(request.content)
            calls.append(body)
            text = fixtures[json.dumps(body['messages'], sort_keys=True)]
            if json.loads(body['messages'][1]['content'])['actor'] is None:
                text = '0, 1\n1, 0'
            return self.response(request, text=text)
        self.client(handler)
        with contextlib.redirect_stdout(io.StringIO()):
            result = v.execute('wording_probe')
            n = len(calls)
            with patch.object(v.r.paid.Budget, 'record_parse', side_effect=AssertionError('Cached replay must not write')):
                resumed = v.execute('wording_probe', verify_only=True)
        self.assertEqual(result, resumed)
        self.assertEqual((result['completed_cells'], result['completed_graphs'], result['invalid_cells']), (4, 3, 1))
        self.assertEqual(len(calls), n)
        import inspect_repair_v8 as inspector
        report = inspector.inspect('wording_probe')
        self.assertEqual((report['observations'], report['valid_graphs'], report['invalid_cells']), (4, 3, 1))
        self.assertFalse(report['billing_verified'])
        summary = v.read(target/'summary.json')
        v.r.write_json(target/'summary.json', {**summary, 'completed_graphs': 99})
        with self.assertRaisesRegex(ValueError, 'counts or charge'):
            inspector.inspect('wording_probe')
        self.assertEqual(v.read(target/'inspection.json')['status'], 'VALIDATION_PENDING')
        v.r.write_json(target/'summary.json', summary)
        failure = next((target/'failures').glob('*.json'))
        self.assertFalse((target/'runs'/failure.name).exists())
        self.assertEqual(len(v.read(failure)['requests']), 1)
        with v.isolated(), self.assertRaisesRegex(ValueError, 'Authorization exists'):
            v.r.approve('wording_probe')
        # Frozen V7 cannot grant the same stage another allowance either.
        with patch.object(v.r, 'stage_review', return_value={}), self.assertRaisesRegex(ValueError, 'Authorization exists'):
            v.r.approve('wording_probe')
        item = v.read(failure)['requests'][0]
        rid = item['request_id']
        (target/'requests'/(rid+'.json')).unlink()
        with contextlib.closing(v.r.paid.Budget(v.r.LEDGER, require_existing=True).db) as db:
            db.execute('DELETE FROM attempts WHERE id=?', (rid,))
            db.commit()
        with self.assertRaisesRegex(ValueError, 'Orphan transport'):
            v.execute('wording_probe')
        self.assertEqual(len(calls), n)

    def test_partial_received_prefix_is_read_only_and_never_repurchased(self):
        spec, target, _ = self.setup_workspace(methods=('local',))
        cell = v.r.cells(spec)[0]
        fixture = v.r.base.FixtureCaller(cell)
        with v.adapter(cell, spec):
            v.r.base.run_cell(cell, fixture)
        replies = {json.dumps(q['messages'], sort_keys=True): q['text'] for q in fixture.requests}
        calls = []
        def handler(request):
            if request.method == 'GET':
                return self.response(request)
            body = json.loads(request.content)
            calls.append(body)
            return self.response(request, text=replies[json.dumps(body['messages'], sort_keys=True)])
        self.client(handler)
        original = v.r.Caller.__call__
        def stop_before_second(caller, *args, **kwargs):
            if caller.ordinal == 1:
                raise RuntimeError('fixture interruption before transport')
            return original(caller, *args, **kwargs)
        with patch.object(v.r.Caller, '__call__', stop_before_second), self.assertRaisesRegex(RuntimeError, 'fixture interruption'):
            v.execute('wording_probe')
        with contextlib.closing(v.r.paid.Budget(v.r.LEDGER, require_existing=True).db) as db:
            before = db.execute('SELECT * FROM attempts').fetchall()
        self.assertEqual(len(calls), 1)
        with contextlib.redirect_stdout(io.StringIO()):
            result = v.execute('wording_probe')
        self.assertEqual(len(calls), 50)
        self.assertEqual(result['completed_graphs'], 1)
        with contextlib.closing(v.r.paid.Budget(v.r.LEDGER, require_existing=True).db) as db:
            self.assertEqual(db.execute('SELECT * FROM attempts WHERE id=?', (before[0][0],)).fetchall(), before)

    def test_exhausted_local_is_failure_not_synthetic_graph(self):
        _, target, _ = self.setup_workspace(methods=('local',))
        calls = []
        def handler(request):
            if request.method == 'POST':
                calls.append(request)
            return self.response(request, text='not an id')
        self.client(handler)
        with contextlib.redirect_stdout(io.StringIO()):
            result = v.execute('wording_probe')
            v.execute('wording_probe', verify_only=True)
        self.assertEqual(len(calls), 3)
        self.assertEqual(result['invalid_cells'], 1)
        self.assertFalse((target/'runs').exists())

    def test_altered_cap_and_missing_review_send_no_requests(self):
        _, target, approved = self.setup_workspace(methods=('global',))
        approved_path = target.parent/'authorization.json'
        original = v.read(approved_path)
        v.r.write_json(approved_path, {**original, 'cumulative_ceiling_usd': original['cumulative_ceiling_usd']+1})
        with patch.object(v.r.paid, 'OpenAI', side_effect=AssertionError('No client')):
            with self.assertRaisesRegex(ValueError, 'allowance'):
                v.execute('wording_probe')
        v.r.write_json(approved_path, original)
        (target/'review.json').unlink()
        with patch.object(v.r.paid, 'OpenAI', side_effect=AssertionError('No client')):
            with self.assertRaises(FileNotFoundError):
                v.execute('wording_probe')

    def test_exhausted_allowance_prevents_post(self):
        self.setup_workspace(methods=('global',))
        budget = v.r.paid.Budget(v.r.LEDGER, require_existing=True)
        try:
            budget.reserve('other-stage', 'main', 9.9999, '{}')
            budget.settle('other-stage', 9.9999, {})
        finally:
            budget.db.close()
        calls = []
        def handler(request):
            if request.method == 'POST':
                calls.append(request)
            return self.response(request)
        self.client(handler)
        with self.assertRaises(v.r.paid.BudgetExceeded):
            v.execute('wording_probe')
        self.assertEqual(calls, [])

    def test_old_failure_only_blocks_new_authorization(self):
        _, target, _ = self.setup_workspace(methods=('global',))
        (target.parent/'authorization.json').unlink()
        v.r.write_json(target.parent/'older-contract'/'failures'/'paid-failure.json', {'retained': True})
        with patch.object(v.r.paid, 'OpenAI', side_effect=AssertionError('No client')):
            with self.assertRaisesRegex(ValueError, 'failure evidence'):
                v.approve('wording_probe')
        self.assertFalse((target.parent/'authorization.json').exists())

    def test_wording_pairs_do_not_overwrite_graph_counts(self):
        import inspect_repair_v8 as inspector
        spec, _, _ = self.setup_workspace(methods=('local',), wordings=('reference', 'alternative_1'))
        replies = {}
        for cell in v.r.cells(spec):
            fixture = v.r.base.FixtureCaller(cell)
            with v.adapter(cell, spec):
                v.r.base.run_cell(cell, fixture)
            replies.update({json.dumps(q['messages'], sort_keys=True): q['text'] for q in fixture.requests})
        def handler(request):
            if request.method == 'GET':
                return self.response(request)
            body = json.loads(request.content)
            return self.response(request, text=replies[json.dumps(body['messages'], sort_keys=True)])
        self.client(handler)
        with contextlib.redirect_stdout(io.StringIO()):
            v.execute('wording_probe')
        report = inspector.inspect('wording_probe')
        self.assertIs(type(report['valid_graphs']), int)
        self.assertEqual((report['valid_graphs'], report['invalid_cells']), (2, 0))


if __name__ == '__main__':
    unittest.main()
