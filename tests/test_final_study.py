"""Launch-path checks; fixtures never count as model observations."""
import copy
from collections import Counter
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx
from openai import OpenAI

from final_study import protocol, runner


class FinalContractTests(unittest.TestCase):
    def test_full_collection_and_live_check_are_disjoint(self):
        spec = protocol.build_contract()
        main = protocol.cells(spec, 'main')
        check = protocol.cells(spec, 'hindi_check')
        self.assertEqual(len(main), 896)
        self.assertEqual(spec['schedule_blocks'], 8)
        self.assertEqual(spec['main_budget_ceiling_usd'], 120)
        self.assertEqual(spec['analysis']['schedule_seeds'], list(range(44000, 44008)))
        counts = Counter((c['method'], c['model'], c['culture'], c['language']) for c in main)
        self.assertEqual(len(counts), 112)
        self.assertEqual(set(counts.values()), {8})
        self.assertEqual(len(check), 8)
        self.assertFalse({c['seed'] for c in main} & {c['seed'] for c in check})
        self.assertEqual(len({c['run_id'] for c in main}), len(main))
        self.assertEqual(spec['analysis']['primary_family_size'], 24)
        self.assertFalse(spec['main_authorized'])

    def test_complete_contract_rejects_source_drift(self):
        spec = protocol.build_contract()
        protocol.verify_contract(spec)
        changed = copy.deepcopy(spec)
        changed['sources']['final_study/protocol.py'] = '0' * 64
        with self.assertRaises(ValueError):
            protocol.verify_contract(changed)

    def test_hindi_actual_execution_prompt_matches_catalog(self):
        spec = protocol.build_contract()
        cell = protocol.cells(spec, 'hindi_check')[0]
        fixture = protocol.engine.FixtureCaller(cell)
        with protocol.adapter(cell, spec):
            protocol.engine.run_cell(cell, fixture)
        actual = fixture.requests[0]['messages'][0]['content']
        self.assertIn(protocol.HINDI_NEW, actual)
        self.assertNotIn(protocol.HINDI_OLD, actual)
        self.assertEqual(actual, next(r['system'] for r in protocol.catalog()
            if r['method'] == 'global' and r['country'] == 'us' and r['language'] == 'hindi'))

    def test_historical_evidence_keeps_original_identity_and_is_not_paid_again(self):
        current = protocol.build_contract()
        old = protocol.hindi_evidence_contract()
        self.assertEqual(protocol.legacy.digest(old), protocol.HISTORY_SHA)
        self.assertEqual(protocol.cells(current, 'hindi_check'), protocol.cells(old, 'hindi_check'))
        self.assertEqual(protocol.folder(current, 'hindi_check'), protocol.folder(old, 'hindi_check'))
        with self.assertRaises(ValueError):
            protocol.verify_contract(old)
        with self.assertRaisesRegex(ValueError, 'read-only'):
            runner.check_allowance(current, 'hindi_check', 1, 'must-not-rebuy')
        with self.assertRaisesRegex(ValueError, 'eight-repetition'):
            runner.check_allowance(old, 'main', 120, 'must-not-use-old-plan')

    def test_historical_bridge_rejects_contract_source_or_prompt_tampering(self):
        old = protocol.hindi_evidence_contract()
        changed = copy.deepcopy(old)
        changed['schedule_blocks'] = 8
        with patch.object(protocol, 'read', return_value=changed):
            with self.assertRaisesRegex(ValueError, 'contract changed'):
                protocol.hindi_evidence_contract()
        with patch.object(protocol.engine, 'sha', return_value='0'*64):
            with self.assertRaisesRegex(ValueError, 'source evidence'):
                protocol.hindi_evidence_contract()
        with patch.object(protocol, 'catalog', return_value=[]):
            with self.assertRaisesRegex(ValueError, 'incompatible'):
                protocol.hindi_evidence_contract()

    def test_partial_or_failed_inventory_never_reports_successful_completion(self):
        spec = protocol.build_contract()
        self.assertEqual(runner.summarize([], spec, 'main')['status'], 'PARTIAL')
        cell = protocol.cells(spec, 'main')[0]
        record = dict(cell=cell, requests=[], failure='invalid_global_response', metrics=None)
        with patch.object(protocol, 'cells', return_value=[cell]):
            self.assertEqual(runner.summarize([record], spec, 'main')['status'], 'COMPLETE_WITH_FAILURES')

    def test_analysis_uses_eight_blocks_not_models_as_repetitions(self):
        spec = protocol.analysis.load_spec()
        rows = [dict(c, valid_graph=True, edges=100, density=.1+c['seed']%8*.01,
                     avg_clustering_coef=.2+c['seed']%8*.01) for c in protocol.analysis.design(spec)]
        results = protocol.analysis.primary_contrasts(rows, spec)
        self.assertEqual(len(results), 24)
        self.assertEqual({r['planned_schedule_blocks'] for r in results}, {8})
        self.assertEqual({r['complete_schedule_blocks'] for r in results}, {8})
        with self.assertRaisesRegex(ValueError, 'Every planned cell'):
            protocol.analysis.primary_contrasts(rows[:-1], spec)
        # One failed model is not replaced by a three-model average or zero.
        failed = dict(rows[0], valid_graph=False, density=None, avg_clustering_coef=None)
        missing = protocol.analysis.primary_contrasts([failed]+rows[1:], spec)
        self.assertTrue(any(r['status'] == 'MISSING_NO_INFERENCE' for r in missing))
        changed = copy.deepcopy(spec)
        changed['schedule_seeds'] = list(range(44000, 44032))
        with self.assertRaises(ValueError):
            protocol.analysis.design(changed)
        # The old analysis remains separately pinned to 32, not monkey-patched.
        import precollection_analysis as historical
        self.assertEqual(len(historical.design(historical.load_spec())), 3584)


class LaunchTests(unittest.TestCase):
    def test_ceiling_rejects_oversized_allowance_before_credentials_or_network(self):
        root, spec, cells, fixtures = self.workspace()
        with patch.object(protocol.legacy.paid, 'credential', side_effect=AssertionError('Too early')):
            for cap in [120.01, 520, float('inf'), float('nan')]:
                with self.subTest(cap=cap), self.assertRaises(ValueError):
                    self.launch(cap=cap)
        self.assertFalse(protocol.folder(spec, 'main').exists())

    def test_ceiling_covers_direct_authorization_and_reopening(self):
        root, spec, cells, fixtures = self.workspace()
        with self.assertRaisesRegex(ValueError, 'ceiling'):
            runner.authorize(spec, 'main', 120.01, 'oversized')
        approval = runner.authorize(spec, 'main', 120, 'offline-fixture-only')
        budget = runner.open_budget(spec, 'main', approval)
        try:
            self.assertEqual(budget.total_cap, 120)
            budget.reserve('one', 'main', 119.99, '{}')
            with self.assertRaises(protocol.legacy.paid.BudgetExceeded):
                budget.reserve('two', 'main', .02, '{}')
            self.assertLessEqual(budget.spent(), 120)
        finally:
            budget.db.close()
        with self.assertRaisesRegex(ValueError, 'ceiling'):
            runner.open_budget(spec, 'main', {**approval, 'additional_usd': 520})

    def workspace(self, methods=('global',)):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        spec = protocol.build_contract()
        cells = [c for c in protocol.cells(spec, 'main') if c['seed'] == 44000
                 and c['culture'] == 'us' and c['language'] == 'english'
                 and c['model'] == 'gpt-6-luna' and c['method'] in methods]
        self.enterContext(patch.object(protocol, 'OUTPUT', root))
        self.enterContext(patch.object(protocol, 'load_contract', return_value=spec))
        self.enterContext(patch.object(protocol, 'verify_contract', return_value=spec))
        self.enterContext(patch.object(protocol, 'check_prices'))
        self.enterContext(patch.object(protocol, 'cells', return_value=cells))
        self.enterContext(patch.object(runner, 'readiness', return_value={'status': 'FIXTURE_ONLY'}))
        self.enterContext(patch.object(protocol.legacy.paid, 'credential', return_value='offline-fixture'))
        protocol.write_json(root/'preflight.json', dict(status='OFFLINE_PREFLIGHT_PASSED', cases=112,
            contract_sha256=protocol.legacy.digest(spec)))
        fixtures = {}
        for cell in cells:
            caller = protocol.engine.FixtureCaller(cell)
            with protocol.adapter(cell, spec):
                protocol.engine.run_cell(cell, caller)
            for request in caller.requests:
                fixtures[json.dumps(request['messages'], sort_keys=True)] = request['text']
        return root, spec, cells, fixtures

    def client(self, handler):
        self.enterContext(patch.object(protocol.legacy.paid, 'OpenAI', side_effect=lambda **kwargs:
            OpenAI(api_key='offline-fixture', max_retries=0,
                   http_client=httpx.Client(transport=httpx.MockTransport(handler)))))

    @staticmethod
    def response(request, text='0, 1', finish='stop'):
        if request.method == 'GET':
            return httpx.Response(200, json={'id': request.url.path.rsplit('/', 1)[-1],
                'object': 'model', 'created': 1, 'owned_by': 'fixture'}, request=request)
        body = json.loads(request.content)
        return httpx.Response(200, json=dict(id='chatcmpl-fixture', object='chat.completion',
            created=1, model=body['model'], service_tier='default',
            choices=[dict(index=0, finish_reason=finish, message=dict(role='assistant', content=text))],
            usage=dict(prompt_tokens=20, completion_tokens=10, total_tokens=30)), request=request)

    def launch(self, cap=1):
        with contextlib.redirect_stdout(io.StringIO()):
            return runner.launch('main', cap, 'offline-fixture-only')

    def test_four_methods_received_resume_is_read_only(self):
        root, spec, cells, fixtures = self.workspace(protocol.analysis.METHODS)
        calls = []
        def handler(request):
            if request.method == 'GET':
                return self.response(request)
            calls.append(request)
            body = json.loads(request.content)
            self.assertEqual(body['service_tier'], 'default')
            return self.response(request, fixtures[json.dumps(body['messages'], sort_keys=True)])
        self.client(handler)
        self.enterContext(patch.object(runner, 'analyze'))  # Reduced integration fixture, not 896 observations.
        result = self.launch()
        self.assertEqual(result['valid_graphs'], 4)
        before = len(calls)
        with patch.object(protocol.legacy.paid.Budget, 'record_parse', side_effect=AssertionError('Read only')):
            self.assertEqual(result, self.launch())
        self.assertEqual(len(calls), before)
        target = protocol.folder(spec, 'main')
        self.assertEqual(len(list((target/'runs').glob('*.png'))), 4)
        self.assertEqual(runner.verify_stage(spec, 'main')['valid_graphs'], 4)

    def test_duplicate_pairs_normalize_without_new_model_call(self):
        root, spec, cells, fixtures = self.workspace()
        posts = []
        def handler(request):
            if request.method == 'POST':
                posts.append(request)
            return self.response(request, '0, 1\n1, 0\n0, 1')
        self.client(handler)
        self.enterContext(patch.object(runner, 'analyze'))
        self.assertEqual(self.launch()['valid_graphs'], 1)
        record = protocol.read(next((protocol.folder(spec, 'main')/'runs').glob('*.json')))
        self.assertEqual(record['normalization'][0]['normalization']['repeated_pairs'], 2)
        self.assertEqual(record['requests'][0]['response']['text'], '0, 1\n1, 0\n0, 1')
        self.assertEqual(len(posts), 1)

    def test_invalid_and_truncated_responses_are_terminal_not_rebought(self):
        for text, finish, failure in [('0, 999', 'stop', 'invalid_global_response'),
                                      ('0, 1', 'length', 'incomplete_provider_response')]:
            with self.subTest(failure=failure), contextlib.ExitStack() as stack:
                # Separate TestCase cleanup stack for each paid-state fixture.
                case = LaunchTests()
                stack.callback(case.doCleanups)
                root, spec, cells, fixtures = case.workspace()
                posts = []
                def handler(request):
                    if request.method == 'POST':
                        posts.append(request)
                    return self.response(request, text, finish)
                case.client(handler)
                stack.enter_context(patch.object(runner, 'analyze'))
                self.assertEqual(case.launch()['valid_graphs'], 0)
                self.assertEqual(case.launch()['valid_graphs'], 0)
                record = protocol.read(next((protocol.folder(spec, 'main')/'runs').glob('*.json')))
                self.assertEqual(record['failure'], failure)
                self.assertFalse(list((protocol.folder(spec, 'main')/'runs').glob('*.png')))
                self.assertEqual(len(posts), 1)

    def test_uncertain_transport_stops_without_resend(self):
        root, spec, cells, fixtures = self.workspace()
        posts = []
        def handler(request):
            if request.method == 'POST':
                posts.append(request)
                return httpx.Response(520, json={'error': {'message': 'fixture'}}, request=request)
            return self.response(request)
        self.client(handler)
        with self.assertRaises(RuntimeError):
            self.launch()
        with self.assertRaisesRegex(RuntimeError, 'Unresolved'):
            self.launch()
        self.assertEqual(len(posts), 1)
        approval = protocol.read(protocol.folder(spec, 'main')/'authorization.json')
        budget = runner.open_budget(spec, 'main', approval)
        try:
            self.assertGreater(budget.spent(), 0)
            self.assertEqual(budget.unresolved_count(), 1)
        finally:
            budget.db.close()

    def test_missing_ledger_or_changed_cap_cannot_reanchor(self):
        root, spec, cells, fixtures = self.workspace()
        with runner.workflow_lock(root):
            runner.authorize(spec, 'main', 1, 'offline-fixture-only')
        target = protocol.folder(spec, 'main')
        with self.assertRaisesRegex(ValueError, 'differs'):
            self.launch(cap=2)
        (target/'budget.sqlite').unlink()
        with self.assertRaisesRegex(ValueError, 'missing'):
            self.launch()

    def test_budget_blocks_transport_before_spend(self):
        root, spec, cells, fixtures = self.workspace()
        posts = []
        def handler(request):
            if request.method == 'POST':
                posts.append(request)
            return self.response(request)
        self.client(handler)
        with self.assertRaises(protocol.legacy.paid.BudgetExceeded):
            self.launch(cap=.000001)
        self.assertEqual(posts, [])

    def test_partial_received_prefix_reused_without_mutation(self):
        root, spec, cells, fixtures = self.workspace(('local',))
        posts = []
        def handler(request):
            if request.method == 'GET':
                return self.response(request)
            posts.append(request)
            body = json.loads(request.content)
            return self.response(request, fixtures[json.dumps(body['messages'], sort_keys=True)])
        self.client(handler)
        self.enterContext(patch.object(runner, 'analyze'))
        original = runner.Caller.__call__
        def interrupt(caller, *args, **kwargs):
            if caller.ordinal == 1:
                raise RuntimeError('fixture interruption')
            return original(caller, *args, **kwargs)
        with patch.object(runner.Caller, '__call__', interrupt), self.assertRaisesRegex(RuntimeError, 'fixture interruption'):
            self.launch()
        target = protocol.folder(spec, 'main')
        approval = protocol.read(target/'authorization.json')
        budget = runner.open_budget(spec, 'main', approval)
        before = budget.db.execute('SELECT * FROM attempts').fetchall()
        budget.db.close()
        self.assertEqual(len(before), 1)
        self.assertEqual(self.launch()['valid_graphs'], 1)
        self.assertEqual(len(posts), 50)
        budget = runner.open_budget(spec, 'main', approval)
        self.assertEqual(budget.db.execute('SELECT * FROM attempts WHERE id=?', (before[0][0],)).fetchall(), before)
        budget.db.close()

    def test_received_before_parse_crash_recovers_without_resend(self):
        root, spec, cells, fixtures = self.workspace()
        posts = []
        def handler(request):
            if request.method == 'POST':
                posts.append(request)
            return self.response(request)
        self.client(handler)
        self.enterContext(patch.object(runner, 'analyze'))
        with patch.object(runner.Caller, 'parse_response', side_effect=RuntimeError('crash before parse')):
            with self.assertRaisesRegex(RuntimeError, 'crash before parse'):
                self.launch()
        self.assertEqual(self.launch()['valid_graphs'], 1)
        self.assertEqual(len(posts), 1)

    def test_orphan_evidence_prevents_new_requests(self):
        root, spec, cells, fixtures = self.workspace()
        posts = []
        def handler(request):
            if request.method == 'POST':
                posts.append(request)
            return self.response(request)
        self.client(handler)
        self.enterContext(patch.object(runner, 'analyze'))
        self.launch()
        target = protocol.folder(spec, 'main')
        budget = runner.open_budget(spec, 'main', protocol.read(target/'authorization.json'))
        rid = budget.db.execute('SELECT id FROM attempts').fetchone()[0]
        budget.db.execute('DELETE FROM attempts WHERE id=?', (rid,))
        budget.db.commit()
        budget.db.close()
        (target/'requests'/(rid+'.json')).unlink()
        with self.assertRaisesRegex(ValueError, 'Orphan'):
            self.launch()
        self.assertEqual(len(posts), 1)

    def test_failed_status_invalidates_previous_ready_report(self):
        root, spec, cells, fixtures = self.workspace()
        protocol.write_json(root/'readiness.json', {'status': 'READY_FOR_MAIN_EXECUTION'})
        with patch.object(protocol, 'load_contract', side_effect=ValueError('source changed')):
            with self.assertRaisesRegex(ValueError, 'source changed'):
                runner.status()
        self.assertEqual(protocol.read(root/'readiness.json')['status'], 'VALIDATION_PENDING')

    def test_cached_parse_drift_never_requests_a_correction(self):
        root, spec, cells, fixtures = self.workspace(('local',))
        cell = cells[0]
        fixture = protocol.engine.FixtureCaller(cell)
        with protocol.adapter(cell, spec):
            protocol.engine.run_cell(cell, fixture)
        first = fixture.requests[0]
        cached = [dict(request_id='cached-fixture', request=dict(model=cell['model'], messages=first['messages']),
            response=dict(text=first['text'], finish_reason='stop', parse=dict(valid=False, error='altered')))]
        from unittest.mock import Mock
        live = Mock()
        live.side_effect = AssertionError('Must not purchase a correction for corrupted cache')
        replay = runner.Replay(cached, live)
        with protocol.adapter(cell, spec):
            with self.assertRaisesRegex(RuntimeError, 'Cached parser decision'):
                runner.observe(cell, replay)
        live.assert_not_called()

    def test_fresh_checkout_preflight_creates_output_root(self):
        root, spec, cells, fixtures = self.workspace()
        fresh = root/'previously-absent'
        with patch.object(protocol, 'OUTPUT', fresh):
            report = runner.preflight()
        self.assertEqual(report['status'], 'OFFLINE_PREFLIGHT_PASSED')
        self.assertTrue((fresh/'preflight.json').is_file())

    def test_readiness_requires_private_accounting_and_local_credential(self):
        real_readiness = runner.readiness
        root, spec, cells, fixtures = self.workspace()
        old = protocol.hindi_evidence_contract()
        target = protocol.folder(old, 'hindi_check')
        # Readiness is a consumer of preserved receipts, never a new paid stage.
        # Public replay is exercised separately against all eight real artifacts.
        for index in range(8):
            protocol.write_json(target/'runs'/f'fixture-{index}.json', {})
        protocol.write_json(target/'authorization.json', {'fixture_only': True})
        from unittest.mock import Mock
        budget = Mock()
        with patch.object(runner, 'verify_stage', return_value={'valid_graphs': 8}), \
                patch.object(runner, 'open_budget', return_value=budget) as opened, \
                patch.object(runner, 'accounting') as checked, \
                patch.object(runner, 'readiness', real_readiness):
            self.assertEqual(runner.status()['status'], 'READY_FOR_MAIN_EXECUTION')
            self.assertEqual(opened.call_args.args[0], old)
            self.assertEqual(checked.call_args.args[2], old)
            budget.db.close.assert_called()
            with patch.object(protocol.legacy.paid, 'credential', side_effect=RuntimeError('Key absent')):
                self.assertEqual(runner.status()['status'], 'LOCAL_CREDENTIAL_REQUIRED')
                with self.assertRaisesRegex(RuntimeError, 'Key absent'):
                    runner.authorize(spec, 'main', 120, 'must-not-authorize')
            with patch.object(runner, 'open_budget', side_effect=ValueError('Missing private ledger')):
                with self.assertRaisesRegex(ValueError, 'Missing private ledger'):
                    runner.status()
            self.assertEqual(protocol.read(root/'readiness.json')['status'], 'VALIDATION_PENDING')
            self.assertFalse((protocol.folder(spec, 'main')/'authorization.json').exists())


if __name__ == '__main__':
    unittest.main()
