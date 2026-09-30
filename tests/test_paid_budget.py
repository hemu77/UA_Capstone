"""Money controls must survive failures and process restarts, without API calls."""
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path

from paid_study import Budget, BudgetExceeded, build_cells, PaidCaller, verify_completed, ROOT, compute_network_metrics
from types import SimpleNamespace
from unittest.mock import Mock
from unittest.mock import patch
import constants_and_utils as shared
import json
import hashlib
import networkx as nx
from PIL import Image


def raw_client():
    # Existing fixtures describe parsed replies; mirror the SDK's raw envelope.
    client = Mock()
    client.chat.completions.with_raw_response.create.side_effect = lambda **kwargs: SimpleNamespace(
        headers={}, status_code=200,
        parse=Mock(return_value=client.chat.completions.create(**kwargs)))
    return client


class PaidBudgetTests(unittest.TestCase):
    def test_full_roster_friend_list_fits_output_cap_and_cached_reply_is_not_repaid(self):
        # Numeric replies use the same schema in every instruction language.
        for method in ['global', 'local', 'sequential', 'iterative']:
            reply = ('\n'.join(f'{a}, {b}' for a in range(50) for b in range(a + 1, 50))
                     if method == 'global' else ', '.join(str(i) for i in range(1, 50)))
            with self.subTest(method=method), tempfile.TemporaryDirectory() as folder:
                budget = Budget(Path(folder) / 'budget.sqlite')
                try:
                    cell = {**build_cells('pilot')[0], 'method': method}
                    client = raw_client()
                    def provider(**kwargs):
                        self.assertGreaterEqual(kwargs['extra_body']['max_completion_tokens'], len(reply.encode('utf-8')))
                        usage = Mock()
                        usage.model_dump.return_value = {'prompt_tokens': 100, 'completion_tokens': 150}
                        return SimpleNamespace(usage=usage, choices=[SimpleNamespace(
                            message=SimpleNamespace(content=reply), finish_reason='stop')], id='offline-cap-fixture', model=cell['model'])
                    client.chat.completions.create.side_effect = provider
                    messages = [{'role': 'user', 'content': 'offline full-roster fixture'}]
                    self.assertEqual(PaidCaller(client, budget, cell)(cell['model'], messages), reply)
                    charged = budget.spent()
                    self.assertEqual(PaidCaller(client, budget, cell)(cell['model'], messages), reply)
                    self.assertEqual(client.chat.completions.create.call_count, 1)
                    self.assertEqual(budget.spent(), charged)
                finally:
                    budget.db.close()

    def test_higher_fresh_cap_is_explicit_finite_and_still_enforced(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'budget.sqlite'
            with self.assertRaises(ValueError):
                Budget(path, total_cap=160)
            for invalid in [float('inf'), float('nan'), True, 0]:
                with self.assertRaises(ValueError):
                    Budget(path, total_cap=invalid, allow_higher_cap=True)
            budget = Budget(path, total_cap=160, allow_higher_cap=True)
            try:
                budget.reserve('spent', 'main', 159, '{}')
                with self.assertRaises(BudgetExceeded):
                    budget.reserve('too-much', 'main', 2, '{}')
                self.assertEqual(budget.spent(), 159)
            finally:
                budget.db.close()
    def test_required_existing_ledger_cannot_silently_reset_budget(self):
        import sqlite3
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'ledger.sqlite'
            with self.assertRaises(sqlite3.OperationalError):
                Budget(path, require_existing=True)
            self.assertFalse(path.exists())
            original = Budget(path)
            original.reserve('old-spend', 'pilot', 2, '{}')
            original.db.close()
            reopened = Budget(path, require_existing=True)
            try:
                self.assertEqual(reopened.spent(), 2)
            finally:
                reopened.db.close()
    def test_uncertain_provider_error_stops_shared_retry_loop(self):
        import httpx
        from openai import APIConnectionError
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            budget = Budget(Path(folder) / 'budget.sqlite')
            stack.callback(budget.db.close)
            cell = build_cells('pilot')[0]
            client = raw_client()
            client.chat.completions.create.side_effect = APIConnectionError(request=httpx.Request('POST', 'https://api.openai.com/v1/chat/completions'))
            caller = PaidCaller(client, budget, cell)
            graph = nx.empty_graph(['0', '1'])
            with patch.object(shared, 'get_llm_response', caller), patch.object(shared.time, 'sleep'):
                with self.assertRaises(RuntimeError):
                    shared.repeat_prompt_until_parsed(cell['model'], 'system', 'user',
                        __import__('generate_networks').update_graph_from_response,
                        {'method': 'global', 'G': graph})
            self.assertEqual(client.chat.completions.create.call_count, 1)
            self.assertEqual(budget.db.execute('SELECT COUNT(*) FROM attempts').fetchone()[0], 1)
            self.assertGreater(budget.spent(), 0)

    def test_luna56_extension_shares_pilot_budget_without_replacing_old_runs(self):
        cells = build_cells('luna56-pilot')
        self.assertEqual(len(cells), 4)
        self.assertEqual({c['model'] for c in cells}, {'gpt-5.6-luna'})
        self.assertEqual({c['phase'] for c in cells}, {'pilot'})
        self.assertTrue(all(c['personas'] == 50 and c['seed'] == 1000 for c in cells))
        self.assertFalse({c['run_id'] for c in cells} & {c['run_id'] for c in build_cells('pilot') + build_cells('sol-pilot')})

    def test_reservations_enforce_both_caps_and_survive_restart(self):
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            path = Path(folder) / 'budget.sqlite'
            budget = Budget(path, 50, 5)
            stack.callback(budget.db.close)
            budget.reserve('one', 'pilot', 4, '{}')
            with self.assertRaises(BudgetExceeded):
                budget.reserve('two', 'pilot', 2, '{}')
            reopened = Budget(path, 50, 5)
            stack.callback(reopened.db.close)
            self.assertEqual(reopened.spent('pilot'), 4)
            budget.settle('one', 1, {'text': '1', 'usage': {'prompt_tokens': 2}})
            self.assertEqual(budget.spent('pilot'), 1)
            self.assertEqual(budget.cached('one')['text'], '1')
            budget.reserve('main', 'main', 49, '{}')
            with self.assertRaises(BudgetExceeded):
                budget.reserve('extra', 'main', .01, '{}')

    def test_uncertain_attempt_is_never_refunded_or_resent(self):
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            budget = Budget(Path(folder) / 'budget.sqlite', 50, 5)
            stack.callback(budget.db.close)
            budget.reserve('one', 'pilot', 2, '{}')
            budget.fail('one', 'APIConnectionError')
            self.assertEqual(budget.spent('pilot'), 2)
            with self.assertRaises(RuntimeError):
                budget.cached('one')
            with self.assertRaises(ValueError):
                budget.settle('one', -1, {})

    def test_explicit_abandonment_keeps_cost_and_permanently_blocks_replay(self):
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            path = Path(folder) / 'budget.sqlite'
            budget = Budget(path)
            stack.callback(budget.db.close)
            budget.reserve('interrupted', 'main', .02, '{"original":true}')
            self.assertEqual(budget.unresolved_count(), 1)
            budget.abandon('interrupted', 'Author approved retaining the full reservation; no replay.')
            self.assertEqual(budget.unresolved_count(), 0)
            self.assertEqual(budget.spent(), .02)
            self.assertEqual(budget.db.execute('SELECT request FROM attempts').fetchone()[0], '{"original":true}')
            reopened = Budget(path, require_existing=True)
            stack.callback(reopened.db.close)
            with self.assertRaisesRegex(RuntimeError, 'Abandoned'):
                reopened.cached('interrupted')
            with self.assertRaises(ValueError):
                reopened.settle('interrupted', 0, {})
            with self.assertRaises(ValueError):
                reopened.abandon('interrupted', 'Cannot rewrite the decision.')
            record = json.loads(reopened.db.execute('SELECT response FROM attempts').fetchone()[0])
            self.assertFalse(record['billing_verified'])
            self.assertNotIn('usage', record)
            self.assertEqual(record['retained_reservation_usd'], .02)

    def test_settlement_lock_prevents_abandonment_between_check_and_update(self):
        import sqlite3
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            path = Path(folder) / 'budget.sqlite'
            budget, other = Budget(path), Budget(path)
            original = budget.db
            stack.callback(original.close)
            stack.callback(other.db.close)
            other.db.execute('PRAGMA busy_timeout=0')
            budget.reserve('race', 'main', .02, '{}')
            blocked = []

            class InterleavedConnection:
                def __enter__(self):
                    original.__enter__()
                    return self

                def __exit__(self, *args):
                    return original.__exit__(*args)

                def __getattr__(self, name):
                    return getattr(original, name)

                def execute(self, sql, *args):
                    cursor = original.execute(sql, *args)
                    if sql.startswith('SELECT cost,status'):
                        row = cursor.fetchone()
                        try:
                            other.abandon('race', 'Concurrent operator action.')
                            blocked.append(False)
                        except sqlite3.OperationalError:
                            blocked.append(True)
                        return SimpleNamespace(fetchone=lambda: row)
                    return cursor

            budget.db = InterleavedConnection()
            budget.settle('race', .01, {'text': 'received'})
            self.assertEqual(blocked, [True])
            self.assertEqual(original.execute('SELECT status,cost FROM attempts').fetchone(), ('received', .01))

    def test_contract_has_unique_twenty_pilot_and_640_main_cells(self):
        pilot = build_cells('pilot')
        main = build_cells('main')
        self.assertEqual(len(pilot), 20)
        self.assertEqual(len(main), 640)
        self.assertEqual(len({c['run_id'] for c in pilot + main}), 660)
        self.assertTrue(all(c['personas'] == 50 for c in pilot + main))
        self.assertEqual({c['method'] for c in pilot}, {'global','local','sequential','iterative'})
        self.assertEqual({c['roster'] for c in main}, set(range(5)))

    def test_sol_extension_is_four_distinct_full_roster_cells_sharing_pilot_cap(self):
        cells = build_cells('sol-pilot')
        self.assertEqual(len(cells), 4)
        self.assertEqual({c['model'] for c in cells}, {'gpt-6-sol'})
        self.assertEqual({c['phase'] for c in cells}, {'pilot'})
        self.assertTrue(all(c['personas'] == 50 and c['seed'] == 1000 for c in cells))
        self.assertFalse({c['run_id'] for c in cells} & {c['run_id'] for c in build_cells('pilot')})

    def test_api_usage_and_response_are_durable_and_resume_does_not_pay_twice(self):
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            budget = Budget(Path(folder) / 'budget.sqlite', 50, 5)
            stack.callback(budget.db.close)
            cell = build_cells('pilot')[0]
            usage = Mock()
            usage.model_dump.return_value = {'prompt_tokens': 20, 'completion_tokens': 3}
            response = SimpleNamespace(usage=usage, choices=[SimpleNamespace(
                message=SimpleNamespace(content='0, 1'), finish_reason='stop')],
                id='response-test', model='resolved-test', system_fingerprint='fp-test')
            client = raw_client()
            client.chat.completions.create.return_value = response
            messages = [{'role': 'user', 'content': 'Choose an eligible ID.'}]
            with self.assertRaisesRegex(ValueError, 'approved cell'):
                PaidCaller(client, budget, cell)('gpt-6-sol', messages)
            client.chat.completions.create.assert_not_called()
            self.assertEqual(budget.spent(), 0)
            self.assertEqual(PaidCaller(client, budget, cell)(cell['model'], messages), '0, 1')
            self.assertEqual(PaidCaller(client, budget, cell)(cell['model'], messages), '0, 1')
            self.assertEqual(client.chat.completions.create.call_count, 1)
            self.assertGreater(budget.spent(), 0)
            self.assertEqual(client.chat.completions.create.call_args.kwargs['extra_body']['reasoning_effort'], 'none')
            saved = json.loads(budget.db.execute('SELECT response FROM attempts').fetchone()[0])
            self.assertEqual(saved['resolved_model'], 'resolved-test')
            self.assertEqual(saved['usage']['prompt_tokens'], 20)

    def test_completed_graph_or_image_cannot_disappear_silently(self):
        with tempfile.TemporaryDirectory() as folder:
            cell = build_cells('pilot')[0]
            root = Path(folder)
            destination = root / 'outputs/revision_budget_v1'
            destination.mkdir(parents=True)
            result = destination / (cell['run_id'] + '.json')
            graph = nx.path_graph([str(i) for i in range(50)])
            nx.write_adjlist(graph, result.with_suffix('.adj'))
            Image.new('RGB', (4,4)).save(result.with_suffix('.png'))
            record = dict(cell=cell, status='ENGINEERING_PILOT_NOT_CONFIRMATORY',
                          homophily={demo: .99 for demo in ['gender','race/ethnicity','religion','political affiliation','age']},
                          events=[{'added':list(graph.edges()),'removed':[]}],
                          metrics=compute_network_metrics(graph),
                          roster_sha256=hashlib.sha256((ROOT / 'text-files/us_50_gpt4o_w_interests.json').read_bytes()).hexdigest(), artifact_sha256={
                suffix: hashlib.sha256(result.with_suffix(suffix).read_bytes()).hexdigest() for suffix in ['.adj','.png']})
            result.write_text(json.dumps(record))
            verified = verify_completed(result, cell)
            self.assertEqual(verified['verified_source_variant'], 'legacy_pre_source_hash')
            from export_research_viewer import pilot_records
            exported = pilot_records(root)[0]
            self.assertEqual(exported['homophily']['gender'], verified['verified_homophily']['gender'])
            self.assertNotEqual(exported['homophily']['gender'], .99)
            self.assertEqual(exported['age_assortativity'], verified['verified_homophily']['age'])
            record['generation_source_sha256'] = '0' * 64
            result.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                verify_completed(result, cell)
            del record['generation_source_sha256']
            record['metrics']['density'] = 1
            result.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, 'measurements'):
                verify_completed(result, cell)
            record['metrics'] = compute_network_metrics(graph)
            result.write_text(json.dumps(record))
            result.with_suffix('.adj').unlink()
            with self.assertRaisesRegex(ValueError, 'missing or changed'):
                verify_completed(result, cell)

    def test_reversed_duplicates_retry_without_mutation_or_refunding_usage(self):
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            budget = Budget(Path(folder) / 'budget.sqlite')
            stack.callback(budget.db.close)
            cell = build_cells('pilot')[4]
            usage = Mock()
            usage.model_dump.return_value = {'prompt_tokens': 20, 'completion_tokens': 8}
            def response(text):
                return SimpleNamespace(usage=usage, choices=[SimpleNamespace(
                    message=SimpleNamespace(content=text), finish_reason='stop')],
                    id='test', model=cell['model'])
            client = raw_client()
            graph = nx.empty_graph(['2', '7'])
            client.chat.completions.create.side_effect = [response('2, 7\n7, 2'), response('2, 7')]
            caller = PaidCaller(client, budget, cell)
            with patch.object(shared, 'get_llm_response', caller), patch.object(shared.time, 'sleep'):
                _, _, attempts = shared.repeat_prompt_until_parsed(cell['model'], 'system', 'user',
                    caller.parse_response, {'method': 'global', 'G': graph})
            self.assertEqual(attempts, 2)
            self.assertEqual(graph.number_of_edges(), 1)
            rows = [json.loads(row[0]) for row in budget.db.execute('SELECT response FROM attempts ORDER BY rowid')]
            self.assertEqual([row['parse']['valid'] for row in rows], [False, True])
            self.assertIn('Duplicate', rows[0]['parse']['error'])
            self.assertAlmostEqual(budget.spent(), sum(row['conservative_charge_usd'] for row in rows))
            correction = client.chat.completions.create.call_args.kwargs['messages'][-1]['content']
            self.assertIn('smaller numeric ID first', correction)

    def test_offline_verification_cannot_dispatch_paid_calls(self):
        import paid_study
        import shutil
        import contextlib
        import io
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            shutil.copy(ROOT / 'study_protocol.json', root / 'study_protocol.json')
            with patch.object(paid_study, 'ROOT', root), patch.object(paid_study, 'OpenAI') as client, \
                    contextlib.redirect_stdout(io.StringIO()):
                report = paid_study.verify_pilot()
            client.assert_not_called()
            self.assertEqual(report['verified'], 0)
            self.assertEqual(report['target'], 28)
            self.assertEqual(report['original_target'], 20)
            self.assertEqual(report['by_model']['gpt-5.6-luna']['verified_graphs'], 0)
            self.assertEqual(report['by_model']['gpt-5.6-luna']['conservative_charge_or_reservation_usd'], 0)
            self.assertEqual(report['conservative_charge_or_reservation_usd'], 0)
            self.assertTrue((root / 'outputs/revision_budget_v1/verification_summary.csv').exists())


if __name__ == '__main__':
    unittest.main()
