"""Launch-path checks; fixtures never count as model observations."""
import copy
from collections import Counter
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx
from openai import OpenAI

from final_study import protocol, runner


class FinalContractTests(unittest.TestCase):
    def test_published_contract_matches_actual_checkout_without_hash_overrides(self):
        # Unlike generated fixture contracts, this catches missing/normalized
        # Git blobs and stale committed source pins in an ordinary clone.
        spec = protocol.load_contract()
        self.assertEqual(protocol.legacy.digest(protocol.hindi_evidence_contract()), protocol.HISTORY_SHA)
        self.assertEqual(spec['catalog_sha256'], protocol.legacy.digest(protocol.catalog()))

    def test_freeze_never_migrates_an_attempted_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root/'contract.json'
            previous = {'fixture_only': 'previous contract'}
            protocol.write_json(path, previous)
            fingerprint = protocol.legacy.digest(previous)
            (root/fingerprint[:12]/'main').mkdir(parents=True)
            before = path.read_bytes()
            for version in ('UNEXECUTED_V2_SHA', 'UNEXECUTED_V3_SHA', 'UNEXECUTED_V4_SHA'):
                with self.subTest(version=version), patch.object(protocol, 'CONTRACT', path), \
                        patch.object(protocol, 'OUTPUT', root), patch.object(protocol, version, fingerprint):
                    with self.assertRaisesRegex(ValueError, 'existing freeze'):
                        protocol.freeze()
            self.assertEqual(path.read_bytes(), before)

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
    def test_local_write_retries_are_bounded_and_atomic(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'checkpoint.json'
            runner.write_json(path, {'before': True})
            original = path.read_bytes()
            error = PermissionError('Persistent lock fixture')
            with patch.object(os, 'replace', side_effect=error) as replace, \
                    patch.object(runner.time, 'sleep') as sleep, \
                    self.assertLogs('final_study.runner', level='WARNING') as logs:
                with self.assertRaises(PermissionError) as raised:
                    runner.write_json(path, {'after': True})
            self.assertIs(raised.exception, error)
            self.assertEqual(replace.call_count, 6)
            self.assertEqual(sleep.call_count, 5)
            self.assertEqual(len(logs.output), 5)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(folder).iterdir()), [path])

    def test_non_lock_errors_are_not_retried(self):
        for error in (OSError(28, 'Disk full fixture'), ValueError('Bad JSON fixture')):
            with self.subTest(error=type(error).__name__), \
                    patch.object(runner, 'atomic_write_json', side_effect=error) as writer, \
                    patch.object(runner.time, 'sleep') as sleep:
                with self.assertRaises(type(error)) as raised:
                    runner.write_json(Path('unused.json'), {})
                self.assertIs(raised.exception, error)
                self.assertEqual(writer.call_count, 1)
                sleep.assert_not_called()

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

    def test_transient_file_locks_never_resend_received_reply(self):
        for phase in ('intent', 'transport', 'received', 'parsed'):
            with self.subTest(phase=phase), contextlib.ExitStack() as scope:
                # Keep each injected failure in its own ledger and journal.
                case = LaunchTests()
                scope.callback(case.doCleanups)
                root, spec, cells, fixtures = case.workspace()
                case.enterContext(patch.object(runner, 'analyze'))
                calls, failures, request_writes = [], [], []
                def handler(request):
                    if request.method == 'POST':
                        calls.append(request)
                    return case.response(request)
                case.client(handler)
                replace = os.replace
                def locked_replace(source, destination):
                    destination = Path(destination)
                    at_boundary = (destination.parent.name == 'transport' if phase == 'transport'
                        else destination.parent.name == 'requests'
                        and len(request_writes) == {'intent': 0, 'received': 1, 'parsed': 2}[phase])
                    if at_boundary and len(failures) < 5:
                        failures.append(Path(source).read_bytes())
                        raise PermissionError('Offline Windows sharing-lock fixture')
                    result = replace(source, destination)
                    if destination.parent.name == 'requests':
                        request_writes.append(destination)
                    return result
                original_writers = protocol.engine.write_json, protocol.legacy.write_json
                with patch.object(os, 'replace', side_effect=locked_replace), \
                        patch.object(runner.time, 'sleep') as sleep, \
                        self.assertLogs('final_study.runner', level='WARNING'):
                    self.assertEqual(case.launch()['status'], 'COMPLETE')
                self.assertEqual(len(failures), 5)
                self.assertEqual(len(set(failures)), 1)  # Identical local bytes, not new responses.
                self.assertEqual([c.args[0] for c in sleep.call_args_list], [.05, .1, .25, .5, 1.])
                self.assertEqual(len(calls), 1)
                self.assertEqual((protocol.engine.write_json, protocol.legacy.write_json), original_writers)
                target = protocol.folder(spec, 'main')
                budget = case.ledger(spec)
                try:
                    rows = budget.db.execute('SELECT id,phase,cost,status,request,response FROM attempts').fetchall()
                    self.assertEqual(len(rows), 1)
                    self.assertEqual(rows[0][3], 'received')
                    self.assertEqual(protocol.read(request_writes[-1]),
                        dict(request_id=rows[0][0], row_sha256=protocol.legacy.digest(rows[0])))
                finally:
                    budget.db.close()
                before = {p: p.read_bytes() for folder in ('requests', 'transport', 'runs')
                          for p in (target/folder).glob('*.json')}
                self.assertEqual(case.launch()['status'], 'COMPLETE')
                self.assertEqual(len(calls), 1)
                self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_exhausted_checkpoint_lock_stops_without_rebuy(self):
        root, spec, cells, fixtures = self.workspace()
        posts = self.failing_client(0)
        original_writers = protocol.engine.write_json, protocol.legacy.write_json
        replace, failures = os.replace, []
        def locked_checkpoint(source, destination):
            if Path(destination).parent.name == 'requests' and posts:
                failures.append(Path(source).read_bytes())
                raise PermissionError('Persistent received-checkpoint lock fixture')
            return replace(source, destination)
        with patch.object(os, 'replace', side_effect=locked_checkpoint), \
                patch.object(runner.time, 'sleep'), self.assertLogs('final_study.runner', level='WARNING'):
            with self.assertRaises(PermissionError):
                self.launch()
        self.assertEqual((protocol.engine.write_json, protocol.legacy.write_json), original_writers)
        self.assertEqual(len(failures), 6)
        self.assertEqual(len(posts), 1)
        budget = self.ledger(spec)
        try:
            before = budget.db.execute('SELECT * FROM attempts').fetchall()
            self.assertEqual(before[0][3], 'received')
        finally:
            budget.db.close()
        with self.assertRaisesRegex(ValueError, 'Journal/ledger mismatch'):
            self.launch()
        with self.assertRaises(runner.RecoveryRefused), contextlib.redirect_stdout(io.StringIO()):
            runner.run('main', 1, 'offline-fixture-only', True)
        self.assertEqual(len(posts), 1)
        budget = self.ledger(spec)
        try:
            self.assertEqual(budget.db.execute('SELECT * FROM attempts').fetchall(), before)
        finally:
            budget.db.close()

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


    # --- Lost-transport recovery: one replacement per request, never a rebuy ---
    def failing_client(self, failures, status=520, text='0, 1'):
        """First `failures` POSTs fail with `status`; later POSTs succeed."""
        posts = []
        def handler(request):
            if request.method == 'GET':
                return self.response(request)
            posts.append(request)
            if len(posts) <= failures:
                return httpx.Response(status, json={'error': {'message': 'fixture'}}, request=request)
            return self.response(request, text)
        self.client(handler)
        return posts

    def ledger(self, spec):
        approval = protocol.read(protocol.folder(spec, 'main')/'authorization.json')
        return runner.open_budget(spec, 'main', approval)

    def test_lost_transport_recovers_once_and_keeps_reservation(self):
        root, spec, cells, fixtures = self.workspace()
        posts = self.failing_client(1)
        self.enterContext(patch.object(runner, 'analyze'))
        with self.assertRaises(RuntimeError):
            self.launch()
        budget = self.ledger(spec)
        try:
            reserved = budget.spent()
        finally:
            budget.db.close()
        with self.assertRaises(runner.RecoveryRefused):
            runner.recover('main', 'a-different-reference')
        info = runner.recover('main', 'offline-fixture-only')
        self.assertEqual(info['status'], 'ONE_REPLACEMENT_AUTHORIZED')
        self.assertEqual(info['paid_calls'], 0)
        self.assertEqual(len(posts), 1)
        with self.assertRaisesRegex(runner.RecoveryRefused, '0 unresolved'):
            runner.recover('main', 'offline-fixture-only')
        result = self.launch()
        self.assertEqual(result['valid_graphs'], 1)
        self.assertEqual(len(posts), 2)
        summary = protocol.read(protocol.folder(spec, 'main')/'summary.json')
        self.assertEqual(summary['transport_recoveries'], 1)
        self.assertAlmostEqual(summary['retained_retired_reservations_usd'], reserved)
        self.assertGreater(summary['ledger_charge_or_reservation_usd'], summary['conservative_usd'])
        record = protocol.read(next((protocol.folder(spec, 'main')/'runs').glob('*.json')))
        self.assertEqual(len(record['requests']), 1)
        self.assertEqual(record['requests'][0]['request_id'], info['replacement_request_id'])
        self.assertEqual(record['requests'][0]['request']['replaces_request_id'], info['original_request_id'])
        # Public replay verifies replacement provenance without the private ledger.
        self.assertEqual(runner.verify_stage(spec, 'main')['valid_graphs'], 1)
        # Resume is read-only and never re-buys either attempt.
        self.assertEqual(self.launch()['valid_graphs'], 1)
        self.assertEqual(len(posts), 2)

    def test_failed_replacement_stops_for_human_review(self):
        root, spec, cells, fixtures = self.workspace()
        posts = self.failing_client(2)
        with self.assertRaises(RuntimeError):
            self.launch()
        runner.recover('main', 'offline-fixture-only')
        with self.assertRaises(RuntimeError):
            self.launch()
        with self.assertRaisesRegex(runner.RecoveryRefused, 'replacement also failed'):
            runner.recover('main', 'offline-fixture-only')
        with self.assertRaisesRegex(RuntimeError, 'Unresolved|Abandoned|non-received'):
            self.launch()
        self.assertEqual(len(posts), 2)

    def test_client_errors_and_saved_replies_are_not_recoverable(self):
        for status in [400, 401, 404]:
            with self.subTest(status=status), contextlib.ExitStack() as stack:
                case = LaunchTests()
                stack.callback(case.doCleanups)
                root, spec, cells, fixtures = case.workspace()
                posts = case.failing_client(1, status)
                with self.assertRaises(RuntimeError):
                    case.launch()
                with self.assertRaisesRegex(runner.RecoveryRefused, 'lost-transport'):
                    runner.recover('main', 'offline-fixture-only')
                self.assertEqual(len(posts), 1)
        with contextlib.ExitStack() as stack:
            case = LaunchTests()
            stack.callback(case.doCleanups)
            root, spec, cells, fixtures = case.workspace()
            def handler(request):
                response = self.response(request)
                if request.method == 'POST':
                    body = json.loads(response.content)
                    body['service_tier'] = 'flex'  # Received, but not the budgeted tier.
                    return httpx.Response(200, json=body, request=request)
                return response
            case.client(handler)
            with self.assertRaises(RuntimeError):
                case.launch()
            with self.assertRaisesRegex(runner.RecoveryRefused, 'reply was saved'):
                runner.recover('main', 'offline-fixture-only')

    def test_recovery_limit_is_enforced(self):
        root, spec, cells, fixtures = self.workspace()
        self.failing_client(1)
        with self.assertRaises(RuntimeError):
            self.launch()
        with patch.object(runner, 'MAX_RECOVERIES', 0), \
                self.assertRaisesRegex(runner.RecoveryRefused, 'limit'):
            runner.recover('main', 'offline-fixture-only')

    def test_auto_recover_resumes_under_the_same_cap(self):
        root, spec, cells, fixtures = self.workspace()
        posts = self.failing_client(1)
        self.enterContext(patch.object(runner, 'analyze'))
        waits = []
        with contextlib.redirect_stdout(io.StringIO()):
            result = runner.run('main', 1, 'offline-fixture-only', auto_recover=True, sleep=waits.append)
        self.assertEqual(result['valid_graphs'], 1)
        self.assertEqual(len(posts), 2)
        self.assertEqual(waits, [30])
        log = protocol.read(protocol.folder(spec, 'main')/'transport_recoveries.json')
        self.assertEqual([entry['automatic'] for entry in log], [True])
        # Without the flag, the original stop-and-review behavior is unchanged.
        case = LaunchTests()
        self.addCleanup(case.doCleanups)
        case.workspace()
        case.failing_client(1)
        with self.assertRaises(RuntimeError), contextlib.redirect_stdout(io.StringIO()):
            runner.run('main', 1, 'offline-fixture-only')

    def test_auto_recover_waits_on_free_probe_outage_without_paying(self):
        root, spec, cells, fixtures = self.workspace()
        gets, posts = [], []
        def handler(request):
            if request.method == 'GET':
                gets.append(request)
                if len(gets) == 1:
                    raise httpx.ConnectError('fixture outage', request=request)
                return self.response(request)
            posts.append(request)
            return self.response(request)
        self.client(handler)
        self.enterContext(patch.object(runner, 'analyze'))
        waits = []
        with contextlib.redirect_stdout(io.StringIO()):
            result = runner.run('main', 1, 'offline-fixture-only', auto_recover=True, sleep=waits.append)
        self.assertEqual(result['valid_graphs'], 1)
        self.assertEqual(waits, [60])
        self.assertEqual(len(posts), 1)
        budget = self.ledger(spec)
        try:
            self.assertEqual(budget.db.execute("SELECT COUNT(*) FROM attempts WHERE status='abandoned'").fetchone()[0], 0)
        finally:
            budget.db.close()


    def test_mid_cell_recovery_reuses_received_prefix(self):
        root, spec, cells, fixtures = self.workspace(('local',))
        posts = []
        def handler(request):
            if request.method == 'GET':
                return self.response(request)
            posts.append(request)
            if len(posts) == 3:
                return httpx.Response(503, json={'error': {'message': 'fixture'}}, request=request)
            body = json.loads(request.content)
            return self.response(request, fixtures[json.dumps(body['messages'], sort_keys=True)])
        self.client(handler)
        self.enterContext(patch.object(runner, 'analyze'))
        with contextlib.redirect_stdout(io.StringIO()):
            result = runner.run('main', 1, 'offline-fixture-only', auto_recover=True, sleep=lambda s: None)
        self.assertEqual(result['valid_graphs'], 1)
        record = protocol.read(next((protocol.folder(spec, 'main')/'runs').glob('*.json')))
        # Every received reply bought exactly once; only the lost one was replaced.
        self.assertEqual(len(posts), len(record['requests'])+1)
        self.assertEqual(sum('replaces_request_id' in q['request'] for q in record['requests']), 1)
        self.assertEqual(runner.verify_stage(spec, 'main')['valid_graphs'], 1)

    def test_recovery_restarts_after_each_durable_write(self):
        # A process can die after the ledger commit, or after either file write.
        # Neither a second recovery nor direct launch may lose that provenance.
        for point in ['journal_before', 'journal_after', 'log_before', 'log_after']:
            for resume in ['recover', 'launch']:
                with self.subTest(point=point, resume=resume), contextlib.ExitStack() as stack:
                    case = LaunchTests()
                    stack.callback(case.doCleanups)
                    root, spec, cells, fixtures = case.workspace()
                    posts = case.failing_client(1)
                    stack.enter_context(patch.object(runner, 'analyze'))
                    with self.assertRaises(RuntimeError):
                        case.launch()
                    real_write = runner.write_json

                    def crash(path, value):
                        matches = (point.startswith('journal') and path.parent.name == 'requests'
                                   or point.startswith('log') and path.name == 'transport_recoveries.json')
                        if matches and point.endswith('before'):
                            raise RuntimeError('injected recovery crash')
                        real_write(path, value)
                        if matches and point.endswith('after'):
                            raise RuntimeError('injected recovery crash')

                    with patch.object(runner, 'write_json', side_effect=crash), \
                            self.assertRaisesRegex(RuntimeError, 'injected recovery crash'):
                        runner.recover('main', 'offline-fixture-only')
                    budget = case.ledger(spec)
                    before = budget.db.execute('SELECT * FROM attempts').fetchall()
                    budget.db.close()
                    self.assertEqual(before[0][3], 'abandoned')
                    if resume == 'recover':
                        try:
                            runner.recover('main', 'offline-fixture-only')
                        except runner.RecoveryRefused:
                            # A crash after the final write already completed recovery.
                            self.assertEqual(point, 'log_after')
                    self.assertEqual(len(posts), 1)
                    self.assertEqual(case.launch()['valid_graphs'], 1)
                    self.assertEqual(len(posts), 2)
                    target = protocol.folder(spec, 'main')
                    entries = protocol.read(target/'transport_recoveries.json')
                    self.assertEqual(len(entries), 1)
                    self.assertEqual(entries[0]['original_request_id'], before[0][0])
                    budget = case.ledger(spec)
                    self.assertEqual(budget.db.execute('SELECT * FROM attempts WHERE id=?',
                                     (before[0][0],)).fetchall(), before)
                    budget.db.close()
                    self.assertEqual(case.launch()['valid_graphs'], 1)
                    self.assertEqual(protocol.read(target/'transport_recoveries.json'), entries)
                    self.assertEqual(len(posts), 2)

    def test_recovery_rejects_conflicting_journal_or_log(self):
        for damaged in ['journal', 'log']:
            with self.subTest(damaged=damaged), contextlib.ExitStack() as stack:
                case = LaunchTests()
                stack.callback(case.doCleanups)
                root, spec, cells, fixtures = case.workspace()
                posts = case.failing_client(1)
                with self.assertRaises(RuntimeError):
                    case.launch()
                info = runner.recover('main', 'offline-fixture-only')
                target = protocol.folder(spec, 'main')
                path = (target/'requests'/(info['original_request_id']+'.json')
                        if damaged == 'journal' else target/'transport_recoveries.json')
                value = protocol.read(path)
                if damaged == 'journal':
                    value['row_sha256'] = '0'*64
                else:
                    value[0]['retained_reservation_usd'] = 0
                protocol.write_json(path, value)
                original_bytes = path.read_bytes()
                with self.assertRaises((ValueError, runner.RecoveryRefused)):
                    case.launch()
                self.assertEqual(path.read_bytes(), original_bytes)
                self.assertEqual(len(posts), 1)

    def test_reconciliation_can_crash_repeatedly_at_recovery_limit(self):
        root, spec, cells, fixtures = self.workspace()
        posts = self.failing_client(1)
        self.enterContext(patch.object(runner, 'analyze'))
        with self.assertRaises(RuntimeError):
            self.launch()
        real_write = runner.write_json

        def crash(path, value):
            if path.name == 'transport_recoveries.json':
                raise RuntimeError('interrupted reconciliation')
            return real_write(path, value)

        with patch.object(runner, 'MAX_RECOVERIES', 1):
            for _ in range(2):
                with patch.object(runner, 'write_json', side_effect=crash), \
                        self.assertRaisesRegex(RuntimeError, 'interrupted reconciliation'):
                    runner.recover('main', 'offline-fixture-only')
            self.assertEqual(self.launch()['valid_graphs'], 1)
            self.assertEqual(len(protocol.read(protocol.folder(spec, 'main')/'transport_recoveries.json')), 1)
        self.assertEqual(len(posts), 2)


if __name__ == '__main__':
    unittest.main()
