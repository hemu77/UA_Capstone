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
            for version in ('UNEXECUTED_V2_SHA', 'UNEXECUTED_V3_SHA', 'UNEXECUTED_V4_SHA',
                            'UNEXECUTED_V5_SHA', 'UNEXECUTED_V6_SHA'):
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
        # One failed model is not replaced by a three-model average or zero: its
        # whole schedule block leaves the affected contrasts, which are labelled partial.
        failed = dict(rows[0], valid_graph=False, density=None, avg_clustering_coef=None)
        missing = protocol.analysis.primary_contrasts([failed]+rows[1:], spec)
        affected = [r for r in missing if r['method'] == rows[0]['method']]
        self.assertTrue(affected)
        for r in affected:
            # This fixture has identical arms, so the partial result is zero-variance.
            self.assertEqual(r['status'], 'PARTIAL_ZERO_VARIANCE_NO_T_INFERENCE')
            self.assertEqual(r['complete_schedule_blocks'], 7)
            self.assertEqual(r['excluded_schedule_seeds'], [rows[0]['seed']])
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
            self.assertEqual(replace.call_count, len(runner.LOCAL_WRITE_DELAYS) + 1)
            self.assertEqual([c.args[0] for c in sleep.call_args_list], list(runner.LOCAL_WRITE_DELAYS))
            self.assertEqual(len(logs.output), len(runner.LOCAL_WRITE_DELAYS))
            self.assertGreaterEqual(sum(runner.LOCAL_WRITE_DELAYS), 30)
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
        self.assertEqual(len(failures), len(runner.LOCAL_WRITE_DELAYS) + 1)
        self.assertEqual(len(posts), 1)
        budget = self.ledger(spec)
        try:
            before = budget.db.execute('SELECT * FROM attempts').fetchall()
            before_ids = [r[0] for r in budget.db.execute('SELECT id,status FROM attempts').fetchall()]
            self.assertEqual(budget.db.execute('SELECT status FROM attempts').fetchall(), [('received',)])
        finally:
            budget.db.close()
        # A plain relaunch still refuses: repair is explicit or opt-in, never silent.
        with self.assertRaisesRegex(ValueError, 'Journal/ledger mismatch'):
            self.launch()
        self.enterContext(patch.object(runner, 'analyze'))
        with contextlib.redirect_stdout(io.StringIO()):
            result = runner.run('main', 1, 'offline-fixture-only', True, sleep=lambda seconds: None)
        self.assertEqual(result['status'], 'COMPLETE')
        self.assertEqual(len(posts), 1)  # The received reply was reused, never bought again.
        log = protocol.read(protocol.folder(spec, 'main')/'journal_reconciliations.json')
        self.assertEqual([e['stale_state'] for e in log], ['intent_before_received_checkpoint'])
        budget = self.ledger(spec)
        try:
            after = budget.db.execute('SELECT id,phase,cost,status,request FROM attempts').fetchall()
        finally:
            budget.db.close()
        # Same single paid row, charge and request; only the parser decision was recorded on resume.
        self.assertEqual(len(after), 1)
        self.assertEqual(after[0][0], before_ids[0])
        self.assertEqual(after[0][3], 'received')

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


    # --- v6: persistent local locks leave a stale journal; repair it from the ledger ---
    def _persistent_journal_lock(self, after_writes):
        """Lock every request-journal write after `after_writes` successful ones."""
        replace, state = os.replace, dict(done=0)
        def locked(source, destination):
            if Path(destination).parent.name == 'requests':
                if state['done'] >= after_writes:
                    raise PermissionError('Persistent journal lock fixture')
                result = replace(source, destination)
                state['done'] += 1
                return result
            return replace(source, destination)
        return locked

    def test_persistent_post_parse_lock_is_reconciled_without_rebuy(self):
        root, spec, cells, fixtures = self.workspace()
        posts = self.failing_client(0)
        self.enterContext(patch.object(runner, 'analyze'))
        # Intent and post-reply writes succeed; the post-parse checkpoint stays locked.
        with patch.object(os, 'replace', side_effect=self._persistent_journal_lock(2)), \
                patch.object(runner.time, 'sleep'), self.assertLogs('final_study.runner', level='WARNING'):
            with self.assertRaises(PermissionError):
                self.launch()
        self.assertEqual(len(posts), 1)
        with self.assertRaisesRegex(ValueError, 'Journal/ledger mismatch'):
            self.launch()
        info = runner.recover('main', 'offline-fixture-only')
        self.assertEqual(info['status'], 'JOURNAL_RECONCILED')
        self.assertEqual(info['stale_state'], 'received_before_parse_checkpoint')
        self.assertEqual(info['paid_calls'], 0)
        self.assertEqual(self.launch()['status'], 'COMPLETE')
        self.assertEqual(len(posts), 1)
        # A second recover has nothing to repair and nothing unresolved.
        with self.assertRaisesRegex(runner.RecoveryRefused, '0 unresolved'):
            runner.recover('main', 'offline-fixture-only')

    def _stale_after_received_lock(self):
        root, spec, cells, fixtures = self.workspace()
        posts = self.failing_client(0)
        with patch.object(os, 'replace', side_effect=self._persistent_journal_lock(1)), \
                patch.object(runner.time, 'sleep'), self.assertLogs('final_study.runner', level='WARNING'):
            with self.assertRaises(PermissionError):
                self.launch()
        target = protocol.folder(spec, 'main')
        journal = next((target/'requests').glob('*.json'))
        self.assertEqual(protocol.read(journal)['status'], 'INTENT_BEFORE_TRANSPORT')
        return spec, target, journal, posts

    def test_unrecognised_journal_mismatch_is_never_rewritten(self):
        spec, target, journal, posts = self._stale_after_received_lock()
        protocol.write_json(journal, dict(request_id=journal.stem, row_sha256='0'*64))
        tampered = journal.read_bytes()
        with self.assertRaisesRegex(runner.RecoveryRefused, 'unrecognised'):
            runner.recover('main', 'offline-fixture-only')
        self.assertEqual(journal.read_bytes(), tampered)
        self.assertFalse((target/'journal_reconciliations.json').exists())
        self.assertEqual(len(posts), 1)

    def test_null_journal_hash_is_not_a_preparse_checkpoint(self):
        spec, target, journal, posts = self._stale_after_received_lock()
        protocol.write_json(journal, dict(request_id=journal.stem, row_sha256=None))
        before = journal.read_bytes()
        with self.assertRaisesRegex(runner.RecoveryRefused, 'unrecognised'):
            runner.recover('main', 'offline-fixture-only')
        self.assertEqual(journal.read_bytes(), before)
        self.assertFalse((target/'journal_reconciliations.json').exists())
        self.assertEqual(len(posts), 1)

    def test_tampered_journal_reconciliation_log_blocks_relaunch(self):
        spec, target, journal, posts = self._stale_after_received_lock()
        runner.recover('main', 'offline-fixture-only')
        log = target/'journal_reconciliations.json'
        entries = protocol.read(log)
        entries[0]['reconciled_row_sha256'] = '0'*64
        protocol.write_json(log, entries)
        before = log.read_bytes()
        self.enterContext(patch.object(runner, 'analyze'))
        with self.assertRaisesRegex(runner.RecoveryRefused, 'reconciliation log'):
            self.launch()
        self.assertEqual(log.read_bytes(), before)
        self.assertEqual(len(posts), 1)

    def test_every_reconciliation_provenance_field_is_checked(self):
        spec, target, journal, posts = self._stale_after_received_lock()
        runner.recover('main', 'offline-fixture-only')
        log = target/'journal_reconciliations.json'
        original = protocol.read(log)
        changes = dict(request_id='unknown', run_id='other-run', ordinal=99,
            stale_state='unknown', previous_journal={}, reconciled_row_sha256='0'*64,
            date_utc='not-a-date')
        for field, value in changes.items():
            with self.subTest(field=field):
                altered = copy.deepcopy(original)
                altered[0][field] = value
                protocol.write_json(log, altered)
                with self.assertRaisesRegex(runner.RecoveryRefused, 'reconciliation log'):
                    runner.recover('main', 'offline-fixture-only')
                self.assertEqual(protocol.read(log), altered)
        protocol.write_json(log, original * 2)
        with self.assertRaisesRegex(runner.RecoveryRefused, 'reconciliation log'):
            runner.recover('main', 'offline-fixture-only')
        self.assertEqual(len(posts), 1)

    def test_journal_repair_is_restart_safe_at_each_write_boundary(self):
        for state in (1, 2):  # Intent checkpoint, then post-parse checkpoint.
            for boundary in ('log_before', 'log_after', 'journal_before', 'journal_after'):
                with self.subTest(state=state, boundary=boundary), contextlib.ExitStack() as scope:
                    case = LaunchTests()
                    scope.callback(case.doCleanups)
                    root, spec, cells, fixtures = case.workspace()
                    posts = case.failing_client(0)
                    scope.enter_context(patch.object(runner, 'analyze'))
                    with patch.object(os, 'replace', side_effect=case._persistent_journal_lock(state)), \
                            patch.object(runner.time, 'sleep'), case.assertLogs('final_study.runner', level='WARNING'):
                        with self.assertRaises(PermissionError):
                            case.launch()
                    target = protocol.folder(spec, 'main')
                    ledger = case.ledger(spec)
                    before = ledger.db.execute('SELECT * FROM attempts').fetchall()
                    ledger.db.close()
                    original_write = runner.write_json
                    def interrupted(path, value):
                        chosen = (Path(path).name == 'journal_reconciliations.json' if boundary.startswith('log')
                                  else Path(path).parent.name == 'requests')
                        if chosen and boundary.endswith('before'):
                            raise OSError('Repair interrupted fixture')
                        original_write(path, value)
                        if chosen and boundary.endswith('after'):
                            raise OSError('Repair interrupted fixture')
                    # Repeat the interruption to test recovery of recovery.
                    for _ in range(2):
                        with patch.object(runner, 'write_json', side_effect=interrupted):
                            try:
                                runner.recover('main', 'offline-fixture-only')
                            except (OSError, runner.RecoveryRefused):
                                pass
                    try:
                        runner.recover('main', 'offline-fixture-only')
                    except runner.RecoveryRefused as error:
                        self.assertIn('0 unresolved', str(error))
                    ledger = case.ledger(spec)
                    self.assertEqual(ledger.db.execute('SELECT * FROM attempts').fetchall(), before)
                    ledger.db.close()
                    self.assertEqual(len(protocol.read(target/'journal_reconciliations.json')), 1)
                    self.assertEqual(case.launch()['status'], 'COMPLETE')
                    self.assertEqual(len(posts), 1)

    def test_stale_journal_needs_matching_transport_receipt(self):
        spec, target, journal, posts = self._stale_after_received_lock()
        receipt = target/'transport'/(journal.stem+'.json')
        audit = protocol.read(receipt)
        audit['response']['choices'][0]['message']['content'] = 'altered after receipt'
        protocol.write_json(receipt, audit)
        with self.assertRaisesRegex(runner.RecoveryRefused, 'transport evidence'):
            runner.recover('main', 'offline-fixture-only')
        self.assertEqual(protocol.read(journal)['status'], 'INTENT_BEFORE_TRANSPORT')
        self.assertEqual(len(posts), 1)

    def test_journal_reconciliation_limit_is_enforced(self):
        spec, target, journal, posts = self._stale_after_received_lock()
        with patch.object(runner, 'MAX_JOURNAL_RECONCILIATIONS', 0):
            with self.assertRaisesRegex(runner.RecoveryRefused, 'limit'):
                runner.recover('main', 'offline-fixture-only')
        self.assertEqual(protocol.read(journal)['status'], 'INTENT_BEFORE_TRANSPORT')


class V6OrderAndAnalysisTests(unittest.TestCase):
    """Block-major execution and the predeclared partial-study rule."""

    def test_main_runs_block_major_and_hindi_order_is_unchanged(self):
        spec = protocol.build_contract()
        main = protocol.cells(spec, 'main')
        seeds = [c['seed'] for c in main]
        self.assertEqual(seeds, sorted(seeds))
        self.assertEqual(Counter(seeds), Counter({s: 112 for s in range(44000, 44008)}))
        design = {(c['method'], c['model'], c['culture'], c['language'], c['seed'])
                  for c in protocol.analysis.design(spec['analysis'])}
        self.assertEqual({(c['method'], c['model'], c['culture'], c['language'], c['seed']) for c in main}, design)
        first = [c['run_id'] for c in main[:112]]
        self.assertNotEqual(first, sorted(first))  # Shuffled within the block, not sorted.
        self.assertEqual(protocol.cells(spec, 'main'), main)  # Deterministic.
        old = protocol.hindi_evidence_contract()
        self.assertEqual([c['run_id'] for c in protocol.cells(spec, 'hindi_check')],
                         [c['run_id'] for c in protocol.cells(old, 'hindi_check')])

    @staticmethod
    def rows(spec, drop=(), unattempted_seeds=()):
        result = []
        for i, c in enumerate(protocol.analysis.design(spec)):
            value = .1 + ((c['seed']*7 + i*3) % 11)*.005
            row = dict(c, valid_graph=True, edges=100, density=value, avg_clustering_coef=value+.2)
            if c['seed'] in unattempted_seeds:
                row.update(terminal_recorded=False, valid_graph=False, density=None, avg_clustering_coef=None)
            elif (c['method'], c['model'], c['culture'], c['language'], c['seed']) in drop:
                row.update(valid_graph=False, density=None, avg_clustering_coef=None)
            result.append(row)
        return result

    def test_complete_study_matches_analysis_v1_exactly(self):
        spec = protocol.analysis.load_spec()
        rows = self.rows(spec)
        new = protocol.analysis.primary_contrasts(rows, spec)
        old = protocol.analysis._math.primary_contrasts(rows, spec)
        for a, b in zip(new, old):
            for k in b:
                self.assertEqual(a[k], b[k], k)
            self.assertEqual(a['excluded_schedule_seeds'], [])
            self.assertEqual(a['missing_cells'], [])
        self.assertEqual({r['status'] for r in new}, {'COMPLETE_ASSUMPTIONS_REQUIRED'})

    def test_one_failed_baseline_cell_drops_one_block_not_six_hypotheses(self):
        spec = protocol.analysis.load_spec()
        drop = {('global', 'gpt-6-sol', 'us', 'english', 44003)}
        results = protocol.analysis.primary_contrasts(self.rows(spec, drop=drop), spec)
        global_rows = [r for r in results if r['method'] == 'global']
        self.assertEqual(len(global_rows), 6)
        for r in global_rows:
            self.assertEqual(r['status'], 'PARTIAL_BLOCKS_ASSUMPTIONS_REQUIRED')
            self.assertEqual(r['complete_schedule_blocks'], 7)
            self.assertEqual(r['excluded_schedule_seeds'], [44003])
            self.assertEqual(r['missing_cells'], [dict(model='gpt-6-sol', culture='us', language='english',
                                                       seed=44003, terminal_recorded=True)])
            self.assertLessEqual(r['missingness_lower'], r['complete_block_mean'])
            self.assertGreaterEqual(r['missingness_upper'], r['complete_block_mean'])
        self.assertTrue(all(r['status'] == 'COMPLETE_ASSUMPTIONS_REQUIRED' for r in results if r['method'] != 'global'))
        # The interval uses the complete blocks' own degrees of freedom.
        import numpy as np
        from scipy.stats import t
        r = global_rows[0]
        values = {(x['model'], x['culture'], x['language'], x['seed']): x['density']
                  for x in self.rows(spec, drop=drop) if x['method'] == 'global'}
        diffs = np.array([np.mean([values[(m, r['culture'], r['language'], s)] for m in protocol.analysis.MODELS])
                          - np.mean([values[(m, 'us', 'english', s)] for m in protocol.analysis.MODELS])
                          for s in range(44000, 44008) if s != 44003])
        half = t.isf(.05/48, 6)*diffs.std(ddof=1)/np.sqrt(7)
        self.assertAlmostEqual(r['simultaneous_ci'][1] - r['simultaneous_ci'][0], 2*half, places=12)

    def test_fewer_than_six_complete_blocks_suppresses_inference(self):
        spec = protocol.analysis.load_spec()
        drop = {('local', 'gpt-4.1', 'india', 'english', s) for s in (44000, 44001, 44002)}
        results = protocol.analysis.primary_contrasts(self.rows(spec, drop=drop), spec)
        india = next(r for r in results if r['hypothesis_id'] == 'local:india:english')
        self.assertEqual(india['status'], 'MISSING_NO_INFERENCE')
        self.assertIsNone(india['holm_p'])
        self.assertEqual(india['complete_schedule_blocks'], 5)
        self.assertEqual(len(results), 24)  # The family keeps its suppressed member.

    def test_stopped_run_reports_partial_blocks_and_missing_terminal_receipts(self):
        spec = protocol.analysis.load_spec()
        rows = self.rows(spec, unattempted_seeds=(44006, 44007))
        results = protocol.analysis.primary_contrasts(rows, spec)
        self.assertEqual({r['status'] for r in results}, {'PARTIAL_BLOCKS_ASSUMPTIONS_REQUIRED'})
        self.assertEqual({r['complete_schedule_blocks'] for r in results}, {6})
        self.assertTrue(all(not cell['terminal_recorded'] for r in results for cell in r['missing_cells']))
        report = protocol.analysis.missingness_report(rows, spec)
        self.assertEqual(len(report), 4*4*7)
        self.assertEqual({(r['planned'], r['terminal_results'], r['without_terminal_result'],
                           r['terminal_failures'], r['valid_without_endpoint'])
                          for r in report}, {(8, 6, 2, 0, 0)})
        bad = [dict(r, valid_graph=True) if not r.get('terminal_recorded', True) else r for r in rows]
        with self.assertRaisesRegex(ValueError, 'without a terminal receipt'):
            protocol.analysis.primary_contrasts(bad, spec)

    def test_runner_analyze_writes_labelled_partial_study(self):
        spec = protocol.build_contract()
        planned = protocol.cells(spec, 'main')
        records = []
        for i, cell in enumerate(planned[:6*112]):  # A stop after six complete blocks.
            value = .1 + ((cell['seed']*7 + i*3) % 11)*.005
            records.append(dict(cell=cell, failure=None,
                                metrics=dict(density=value, avg_clustering_coef=value+.2, edges=100)))
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)
            self.assertEqual(runner.analyze(records, spec, target), 'PARTIAL_STUDY')
            summary = protocol.read(target/'analysis_summary.json')
            self.assertEqual((summary['planned'], summary['terminal_results']), (896, 672))
            self.assertEqual(summary['hypothesis_status'], {'PARTIAL_BLOCKS_ASSUMPTIONS_REQUIRED': 24})
            rows = protocol.read(target/'analysis_rows.json')
            self.assertEqual(sum(not r['terminal_recorded'] for r in rows), 224)
            self.assertTrue((target/'missingness_report.json').exists())

    def test_analyze_stage_replays_receipts_offline_and_handles_empty_inventory(self):
        spec = protocol.build_contract()
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(protocol, 'OUTPUT', Path(folder)), \
                patch.object(protocol, 'load_contract', return_value=spec), \
                patch.object(protocol.legacy.paid, 'OpenAI', side_effect=AssertionError('no client')), \
                patch.object(protocol.legacy.paid, 'credential', side_effect=AssertionError('no key')):
            result = runner.analyze_stage('main')
            self.assertFalse(protocol.folder(protocol.load_contract(), 'main').exists())
            with patch.object(runner, 'readiness'), patch.object(runner, 'require_preflight'), \
                    patch.object(protocol, 'check_prices'):
                approval = runner.authorize(spec, 'main', 120, 'offline-fixture-only')
            self.assertEqual(approval['additional_usd'], 120)
        self.assertEqual(result['status'], 'NOT_STARTED')
        self.assertEqual(result['terminal_results'], 0)

    def test_analysis_does_not_call_unfinished_paid_cells_unattempted(self):
        spec = protocol.build_contract()
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)
            # A stopped run may have paid replies without any terminal graph.
            # Public analysis has no private ledger, so it must report unknown
            # execution status rather than inventing a zero-attempt count.
            runner.analyze([], spec, target)
            summary = protocol.read(target/'analysis_summary.json')
            self.assertNotIn('attempted', summary)
            self.assertEqual(summary['terminal_results'], 0)
            self.assertEqual(summary['without_terminal_result'], 896)
            rows = protocol.read(target/'analysis_rows.json')
            self.assertTrue(all(r['terminal_recorded'] is False for r in rows))
            self.assertTrue(all('attempted' not in r for r in rows))
            coverage = protocol.read(target/'missingness_report.json')
            self.assertTrue(all(r['terminal_failures'] == 0 for r in coverage))

    def test_failed_analysis_invalidates_old_success_before_replay(self):
        spec = protocol.build_contract()
        cell = protocol.cells(spec, 'main')[0]
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(protocol, 'OUTPUT', Path(folder)), \
                patch.object(protocol, 'load_contract', return_value=spec):
            target = protocol.folder(spec, 'main')
            protocol.write_json(target/'analysis_summary.json', dict(status='COMPLETE'))
            protocol.write_json(target/'primary_contrasts.json', [dict(raw_p=.001)])
            protocol.write_json(target/'runs'/(cell['run_id']+'.json'), dict(cell='tampered'))
            with self.assertRaisesRegex(ValueError, 'receipt'):
                runner.analyze_stage()
            self.assertEqual(protocol.read(target/'analysis_summary.json')['status'], 'VALIDATION_PENDING')
            self.assertEqual(protocol.read(target/'primary_contrasts.json')['status'], 'VALIDATION_PENDING')

    def test_analysis_source_failure_invalidates_prior_success(self):
        spec = protocol.read(protocol.CONTRACT)
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(protocol, 'OUTPUT', Path(folder)), \
                patch.object(protocol, 'load_contract', side_effect=ValueError('Frozen source differs')):
            target = protocol.folder(spec, 'main')
            protocol.write_json(target/'analysis_summary.json', dict(status='COMPLETE'))
            with self.assertRaisesRegex(ValueError, 'Frozen source'):
                runner.analyze_stage()
            self.assertEqual(protocol.read(target/'analysis_summary.json')['status'], 'VALIDATION_PENDING')

    def test_failed_export_cannot_publish_analysis_success(self):
        spec = protocol.build_contract()
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)
            with patch.object(protocol.engine, 'write_csv', side_effect=OSError('Disk full fixture')):
                with self.assertRaisesRegex(OSError, 'Disk full'):
                    runner.analyze([], spec, target)
            self.assertEqual(protocol.read(target/'analysis_summary.json')['status'], 'VALIDATION_PENDING')

    def test_duplicate_terminal_records_are_rejected(self):
        spec = protocol.build_contract()
        record = dict(cell=protocol.cells(spec, 'main')[0], failure='invalid_global_response', metrics=None)
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, 'Duplicate terminal'):
                runner.analyze([record, record], spec, Path(folder))

    def test_coverage_separates_failed_graphs_from_valid_empty_graphs(self):
        spec = protocol.analysis.load_spec()
        rows = self.rows(spec)
        for row in rows:
            if row['seed'] == 44000:
                row.update(edges=0, density=0, avg_clustering_coef=None)
            elif row['seed'] == 44001:
                row.update(valid_graph=False, density=None, avg_clustering_coef=None)
        report = protocol.analysis.missingness_report(rows, spec)
        for row in report:
            self.assertEqual(row['terminal_results'], 8)
            self.assertEqual(row['terminal_failures'], 1)
            self.assertEqual(row['valid_without_endpoint'], 0 if row['method'] == 'global' else 1)
            self.assertEqual(row['without_terminal_result'], 0)


if __name__ == '__main__':
    unittest.main()
