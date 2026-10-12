"""Offline SDK transport fixtures: never real paid responses or real credentials."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx
from openai import OpenAI

import repair_v7 as r


class RepairTests(unittest.TestCase):
    def setup_workspace(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        self.enterContext(patch.object(r, 'DESTINATION', root / 'repair'))
        self.enterContext(patch.object(r, 'LEDGER', root / 'ledger.sqlite'))
        self.enterContext(patch.object(r.base.frozen, 'DESTINATION', root / 'lock'))
        with contextlib.closing(r.paid.Budget(r.LEDGER).db):
            pass
        spec = r.prepare('global_probe')
        target = r.folder(spec)
        r.write_json(target / 'preflight.json', dict(status='OFFLINE_PREFLIGHT_PASSED', contract_sha256=r.digest(spec)))
        r.write_json(target / 'review.json', dict(contract_sha256=r.digest(spec), prompt_semantics_reviewed=True,
            diagnostic_design_reviewed=True, spending_safeguards_reviewed=True, human_bilingual_validation=False,
            review_reference='offline fixture only', rates=spec['rates'], prices_verified_on=r.date.today().isoformat(),
            price_evidence='offline fixture; not actual reviewed pricing'))
        return spec, target

    def client(self, handler):
        self.enterContext(patch.object(r.paid, 'credential', return_value='offline-test-key'))
        self.enterContext(patch.object(r.paid, 'OpenAI', side_effect=lambda **kwargs: OpenAI(
            api_key='offline-test-key', max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(handler)))))

    @staticmethod
    def response(request, text='NONE', finish='stop', tier='default'):
        if request.method == 'GET':
            return httpx.Response(200, json={'id': request.url.path.rsplit('/', 1)[-1], 'object': 'model', 'created': 0, 'owned_by': 'fixture'})
        body = json.loads(request.content)
        return httpx.Response(200, json={'id': 'offline-response', 'object': 'chat.completion', 'created': 0,
            'model': body['model'], 'service_tier': tier, 'choices': [{'index': 0, 'finish_reason': finish,
                'message': {'role': 'assistant', 'content': text}}],
            'usage': {'prompt_tokens': 100, 'completion_tokens': 1, 'total_tokens': 101}})

    def test_scope_counts_seeds_and_same_grammar_in_all_global_arms(self):
        for stage, expected in [('global_probe', 64), ('wording_probe', 96)]:
            rows = r.design(stage)
            self.assertEqual(len(rows), expected)
            self.assertEqual(rows, r.design(stage))
        self.assertEqual(len(r.design('calibration', {'global_arm': 'a0_b0', 'wording': 'reference'})), 104)
        for language, clauses in r.prompts.GLOBAL_CLAUSES.items():
            for arm in ['a0_b0', 'a0_b1', 'a1_b0', 'a1_b1']:
                text = r.prompts.treatment_prompt('global', language, 'us', global_arm=arm)
                self.assertIn(clauses[3], text)
                self.assertEqual(clauses[0] in text, arm.startswith('a1'))
                self.assertEqual(clauses[2] in text, arm.endswith('b1'))
        self.assertEqual(len(r.catalog()), 288)
        self.assertFalse(r.contract('global_probe')['main_authorized'])
        with self.assertRaises(ValueError):
            r.design('main')

    def test_preflight_prices_exact_messages_without_network_or_paid_evidence(self):
        self.setup_workspace()
        result = r.preflight('global_probe')
        self.assertEqual(result['fixture_count'], 64)
        self.assertEqual(result['paid_requests'], 0)
        self.assertGreater(result['first_attempt_reservation_sum_usd'], 0)
        spec = r.contract('global_probe')
        report = json.loads((r.folder(spec) / 'preflight.json').read_text())
        self.assertEqual(sum(row['request_count'] for row in report['fixture_cells']), 64)
        self.assertTrue(all(row['nodes'] == 50 for row in report['fixture_cells']))
        self.assertFalse((r.folder(spec) / 'runs').exists())
        self.assertEqual(list((r.folder(spec) / 'requests').glob('*.json')), [])

    def test_wording_fixtures_keep_full_roster_and_variant_on_corrections(self):
        spec = r.contract('wording_probe')
        before = r.v6.system_prompt
        for language in r.LANGUAGES:
            cell = next(c for c in r.cells(spec) if c['language'] == language and c['wording'] == 'alternative_1')
            caller = r.base.FixtureCaller(cell)
            original_call = caller.__class__.__call__
            def first_invalid(self, model, messages):
                text = original_call(self, model, messages)
                return 'invalid fixture' if len(self.requests) == 1 else text
            with patch.object(r.base.FixtureCaller, '__call__', first_invalid), r.adapter(cell, spec):
                result = r.base.run_cell(cell, caller)
            self.assertEqual(len(result[0]), 50)
            self.assertEqual(len(result[1]), 50)
            self.assertEqual(len(result[2]), 51)
            second = caller.requests[1]['messages'][0]['content']
            self.assertIn(r.prompts.LOCAL_VARIANTS[language]['alternative_1'][0].split('{actor}')[0], second)
        self.assertIs(r.v6.system_prompt, before)

    def test_missing_review_fails_before_client_and_repeat_approval_cannot_reset_cap(self):
        spec, target = self.setup_workspace()
        (target / 'review.json').unlink()
        with patch.object(r.paid, 'OpenAI', side_effect=AssertionError('Must not open client')):
            with self.assertRaises(FileNotFoundError):
                r.approve('global_probe')
        self.assertFalse((target.parent / 'authorization.json').exists())

    def test_64_empty_graphs_are_valid_and_resume_sends_zero_new_requests(self):
        spec, target = self.setup_workspace()
        approved = r.approve('global_probe')
        with self.assertRaises(ValueError):
            r.approve('global_probe')
        calls = []
        def handler(request):
            if request.method == 'POST':
                calls.append(request)
            return self.response(request)
        self.client(handler)
        with contextlib.redirect_stdout(io.StringIO()):
            result = r.execute('global_probe')
            resumed = r.execute('global_probe')
        self.assertEqual(result, resumed)
        self.assertEqual(result['completed_graphs'], 64)
        self.assertEqual(result['empty_graphs'], 64)
        self.assertEqual(len(calls), 64)
        self.assertLess(result['stage_charge_usd'], approved['additional_usd'])
        # Public graph replay needs no private transport log; paid resume does.
        receipt = next((target / 'runs').glob('*.json'))
        saved = json.loads(receipt.read_text())
        audit_path = target / 'transport' / (saved['requests'][0]['request_id'] + '.json')
        audit = json.loads(audit_path.read_text())
        audit_path.unlink()
        r.verify_run(receipt, saved['cell'], spec)
        with self.assertRaises(FileNotFoundError):
            r.execute('global_probe')
        self.assertEqual(len(calls), 64)
        r.write_json(audit_path, audit)
        # A missing charge cannot silently purchase a previously completed request.
        with contextlib.closing(r.paid.Budget(r.LEDGER, require_existing=True).db) as db:
            db.execute('DELETE FROM attempts WHERE id=(SELECT id FROM attempts LIMIT 1)')
            db.commit()
        with self.assertRaises(ValueError):
            r.execute('global_probe')
        self.assertEqual(len(calls), 64)
        self.assertEqual(json.loads((target / 'summary.json').read_text())['status'], 'STOPPED')
        # Losing the permission file must not reset the stage allowance.
        (target.parent / 'authorization.json').unlink()
        with self.assertRaisesRegex(ValueError, 're-anchor'):
            r.approve('global_probe')

    def test_timeout_is_not_resent_and_invalid_global_is_not_corrected(self):
        for kind in ['timeout', 'invalid', 'truncated']:
            with self.subTest(kind=kind):
                spec, target = self.setup_workspace()
                r.approve('global_probe')
                calls = []
                def handler(request):
                    if request.method == 'POST':
                        calls.append(request)
                        if kind == 'timeout':
                            raise httpx.ReadTimeout('offline fixture', request=request)
                    return self.response(request, text='invalid' if kind == 'invalid' else 'NONE',
                                         finish='length' if kind == 'truncated' else 'stop')
                self.client(handler)
                for _ in range(2):
                    with self.assertRaises((RuntimeError, ValueError)):
                        r.execute('global_probe')
                self.assertEqual(len(calls), 1)
                self.assertEqual(json.loads((target / 'summary.json').read_text())['status'], 'STOPPED')

    def test_budget_reserves_before_transport_and_prepared_artifacts_are_bound(self):
        spec, target = self.setup_workspace()
        approved = r.approve('global_probe')
        with contextlib.closing(r.paid.Budget(r.LEDGER, require_existing=True).db) as db:
            db.execute('INSERT INTO attempts (id,phase,cost,status,request,response) VALUES (?,?,?,?,?,?)',
                       ('unrelated-received', 'main', approved['additional_usd'] - .000001,
                        'received', '{}', '{}'))
            db.commit()
        calls = []
        def handler(request):
            if request.method == 'POST':
                calls.append(request)
            return self.response(request)
        self.client(handler)
        with self.assertRaises(r.paid.BudgetExceeded):
            r.execute('global_probe')
        self.assertEqual(len(calls), 0)
        self.assertEqual(list((target / 'requests').glob('*.json')), [])
        r.write_json(target / 'planned_cells.json', [])
        with self.assertRaisesRegex(ValueError, 'Prepared contract'):
            r.execute('global_probe')
        self.assertEqual(json.loads((target / 'summary.json').read_text())['status'], 'VALIDATION_PENDING')

    def test_stage_caps_survive_version_changes_and_upstream_paths_are_validated(self):
        spec, target = self.setup_workspace()
        r.approve('global_probe')
        changed = {**spec, 'version': 'changed-for-offline-test'}
        with patch.object(r, 'contract', return_value=changed), patch.object(r, 'stage_review'):
            with self.assertRaisesRegex(ValueError, 'never reset'):
                r.approve('global_probe')
        next_spec = r.prepare('wording_probe')
        next_target = r.folder(next_spec)
        review = json.loads((target / 'review.json').read_text())
        review['contract_sha256'] = r.digest(next_spec)
        r.write_json(next_target / 'review.json', review)
        r.write_json(next_target / 'preflight.json', dict(status='OFFLINE_PREFLIGHT_PASSED',
                                                       contract_sha256=r.digest(next_spec)))
        upstream = dict(next_contract_sha256=r.digest(next_spec), accepted=True, rationale='fixture only',
                        evidence={'global_probe': {'contract_id': '../outside'}})
        r.write_json(next_target / 'upstream_review.json', upstream)
        with self.assertRaisesRegex(ValueError, 'Invalid upstream'):
            r.stage_review(next_spec)
        with self.assertRaisesRegex(ValueError, 'Global clauses'):
            r.prompts.treatment_prompt('local', 'english', 'us', '0', 1, global_arm='a0_b0')

    def test_source_changes_after_import_require_restart(self):
        original = r.base.sha
        with patch.object(r.base, 'sha', side_effect=lambda path:
                '0' * 64 if Path(path).name == 'repair_v7.py' else original(path)):
            with self.assertRaisesRegex(ValueError, 'Source changed after import'):
                r.contract('global_probe')

    def test_standard_tier_is_sent_logged_and_missing_audit_blocks_cached_run(self):
        spec, target = self.setup_workspace()
        cell = r.cells(spec)[0]
        calls = []
        def handler(request):
            calls.append(json.loads(request.content))
            return self.response(request)
        self.client(handler)
        budget = r.paid.Budget(r.LEDGER, require_existing=True)
        with contextlib.closing(budget.db):
            with r.paid.OpenAI() as client, r.adapter(cell, spec):
                caller = r.Caller(client, budget, cell, spec, target)
                r.base.run_cell(cell, caller)
                self.assertEqual(calls[0].get('service_tier'), 'default')
                audit_path = target / 'transport' / (caller.last_request_id + '.json')
                self.assertEqual(json.loads(audit_path.read_text())['response']['service_tier'], 'default')
                r.base.run_cell(cell, r.Caller(client, budget, cell, spec, target))
                self.assertEqual(len(calls), 1)
                audit_path.unlink()
                with self.assertRaises(FileNotFoundError):
                    r.base.run_cell(cell, r.Caller(client, budget, cell, spec, target))
                self.assertEqual(len(calls), 1)

    def test_unknown_or_expensive_tier_is_retained_once_and_not_rebought(self):
        for tier in [None, 'priority']:
            with self.subTest(tier=tier):
                spec, target = self.setup_workspace()
                cell = r.cells(spec)[0]
                calls = []
                def handler(request):
                    calls.append(request)
                    return self.response(request, tier=tier)
                self.client(handler)
                budget = r.paid.Budget(r.LEDGER, require_existing=True)
                self.addCleanup(budget.db.close)
                with r.paid.OpenAI() as client, r.adapter(cell, spec):
                    for _ in range(2):
                        with self.assertRaises(RuntimeError):
                            r.base.run_cell(cell, r.Caller(client, budget, cell, spec, target))
                self.assertEqual(len(calls), 1)
                self.assertEqual(budget.unresolved_count(), 1)
                self.assertGreater(budget.spent(), 0)
                audit = next((target / 'transport').glob('*.json'))
                self.assertEqual(json.loads(audit.read_text())['response']['service_tier'], tier)


if __name__ == '__main__':
    unittest.main()
