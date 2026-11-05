"""Offline supplement identity, spending, and duplicate-send guard regressions."""
import copy
import io
import importlib.util
from pathlib import Path
import unittest
import contextlib
import json
import socket
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

path = Path(__file__).with_name('runner.py')
spec = importlib.util.spec_from_file_location('supplement', path)
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)

class Guards(unittest.TestCase):
    def test_allowance_cannot_reset_original_spending(self):
        remaining = s.remaining_allowance(101.98695215000001)
        self.assertLessEqual(101.98695215000001 + remaining, 120)
        self.assertGreater(remaining, 18.013)
        for value in (-1, 120, 121, float('nan'), float('inf'), True):
            with self.assertRaises(ValueError):
                s.remaining_allowance(value)

    def test_exact_six_cells_get_new_ids_without_mutating_originals(self):
        originals = {rid: {'cell': {'run_id': rid, 'final_contract_sha256': s.CONTRACT}, 'failure': failure}
                     for rid, failure in s.FAILURES.items()}
        saved = copy.deepcopy(originals)
        cells = s.supplemental_cells(originals, 'a' * 64)
        self.assertEqual(originals, saved)
        self.assertEqual(len(cells), 6)
        self.assertFalse(set(originals) & {c['run_id'] for c in cells})
        self.assertEqual({c['original_run_id'] for c in cells}, set(originals))
        self.assertTrue(all(c['supplement_policy_sha256'] == 'a' * 64 for c in cells))
        for changed in ({}, {**originals, 'extra': next(iter(originals.values()))}):
            with self.assertRaises(ValueError):
                s.supplemental_cells(changed, 'a' * 64)
        first = next(iter(originals))
        originals[first]['failure'] = None
        with self.assertRaises(ValueError):
            s.supplemental_cells(originals, 'a' * 64)

    def actual_caller(self, *, text='0, 1', cap=.1, transport_error=False):
        r, a, p = s.engine()
        contract = p.load_contract()
        rid = next(iter(s.FAILURES))
        original = next(c for c in p.cells(contract, 'main') if c['run_id'] == rid)
        cell = {**original, 'run_id': 'supplement_fixture_' + rid, 'original_run_id': rid,
                'supplement_id': s.NAME, 'supplement_policy_sha256': 'a' * 64}
        target = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        budget = p.legacy.paid.Budget(target / 'budget.sqlite', total_cap=cap, pilot_cap=cap, allow_higher_cap=True)
        self.addCleanup(budget.db.close)
        self.enterContext(patch.object(socket.socket, 'connect', side_effect=AssertionError('Offline test')))
        calls = []
        dump = dict(id='offline-response', model=a.expected_models()[cell['model']], service_tier='default',
                    usage=dict(prompt_tokens=1, completion_tokens=1),
                    choices=[dict(message=dict(content=text), finish_reason='stop')])
        response = SimpleNamespace(id=dump['id'], model=dump['model'], service_tier='default',
                    usage=SimpleNamespace(model_dump=lambda: dump['usage']),
                    choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason='stop')],
                    model_dump=lambda: dump)
        def create(**kwargs):
            calls.append(kwargs)
            if transport_error:
                raise RuntimeError('Offline simulated unknown transport outcome')
            return SimpleNamespace(parse=lambda: response, headers={})
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(with_raw_response=SimpleNamespace(create=create))))
        return r, p, contract, cell, target, budget, client, calls

    def test_actual_caller_sends_once_then_replays_without_paid_call(self):
        r, p, spec, cell, target, budget, client, calls = self.actual_caller()
        with p.adapter(cell, spec):
            result, failure = r.observe(cell, r.Replay([], r.Caller(client, budget, cell, spec, target)))
        self.assertIsNone(failure)
        self.assertEqual(len(calls), 1)
        items = r.requests_from_ledger(budget, cell)
        with p.adapter(cell, spec):
            replayed, failure = r.observe(cell, r.Replay(items, r.Caller(client, budget, cell, spec, target)))
        self.assertIsNone(failure)
        self.assertEqual(set(result[0].edges), set(replayed[0].edges))
        self.assertEqual(len(calls), 1)
        with patch.object(s, 'TARGET', target):
            s.accounting(budget, [cell], spec, r, p)
        (target / 'transport' / 'foreign.json').write_text('{}')
        with patch.object(s, 'TARGET', target), self.assertRaisesRegex(ValueError, 'orphan transport'):
            s.accounting(budget, [cell], spec, r, p)

    def test_invalid_global_is_terminal_not_retried_until_valid(self):
        r, p, spec, cell, target, budget, client, calls = self.actual_caller(text='not a graph')
        with p.adapter(cell, spec):
            result, failure = r.observe(cell, r.Replay([], r.Caller(client, budget, cell, spec, target)))
        self.assertEqual(failure, 'invalid_global_response')
        self.assertEqual(len(calls), 1)

    def test_uncertain_transport_blocks_repurchase(self):
        r, p, spec, cell, target, budget, client, calls = self.actual_caller(transport_error=True)
        for _ in range(2):
            with p.adapter(cell, spec), self.assertRaises(RuntimeError):
                r.observe(cell, r.Replay([], r.Caller(client, budget, cell, spec, target)))
        self.assertEqual(len(calls), 1)
        self.assertGreater(budget.spent(), 0)

    def test_reservation_cap_blocks_before_transport(self):
        r, p, spec, cell, target, budget, client, calls = self.actual_caller(cap=1e-9)
        with p.adapter(cell, spec), self.assertRaises(p.legacy.paid.BudgetExceeded):
            r.observe(cell, r.Replay([], r.Caller(client, budget, cell, spec, target)))
        self.assertEqual(calls, [])
        self.assertEqual(budget.spent(), 0)

    def launch_fixture(self, *, transport_error=False, unavailable=False):
        r, a, p = s.engine()
        root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        bundle, target = root / 'bundle', root / 'results'
        bundle.mkdir()
        (bundle / 'test_supplement.py').write_bytes(Path(__file__).read_bytes())
        audit = dict(ledger_sha256='f' * 64, ledger_charge_or_reservation_usd=101.98695215000001)
        policy = dict(supplement_id=s.NAME, contract_sha256=s.CONTRACT,
                      runner_sha256=s.sha(s.__file__), tests_sha256=s.sha(bundle / 'test_supplement.py'),
                      original_ledger_sha256=audit['ledger_sha256'], original_spent_usd=audit['ledger_charge_or_reservation_usd'],
                      additional_cap_usd=s.remaining_allowance(audit['ledger_charge_or_reservation_usd']), adapter_sha256={})
        r.write_json(bundle / 'policy.json', policy)
        digest = s.sha(bundle / 'policy.json')
        r.write_json(bundle / 'offline-verified.json', dict(status='SUPPLEMENT_OFFLINE_VERIFIED', policy_sha256=digest,
                      original_failure_replays=6, fixture_cases=6, paid_calls=0))
        r.write_json(bundle / 'review.json', dict(status='CONDITIONAL_PAID_GO', policy_sha256=digest))
        for name, value in (('BUNDLE', bundle), ('TARGET', target)):
            self.enterContext(patch.object(s, name, value))
        self.enterContext(patch.object(s, 'original_anchor', return_value=audit))
        self.enterContext(patch.object(r, 'output_lock', side_effect=lambda: r.workflow_lock(root)))
        self.enterContext(patch.object(socket.socket, 'connect', side_effect=AssertionError('Offline launch test')))
        calls, access = [], []
        def create(**kwargs):
            calls.append(kwargs)
            if transport_error:
                raise RuntimeError('PRIVATE_PROVIDER_SENTINEL')
            text = '0, 1' if len(calls) == 1 else 'not a graph'
            model = a.expected_models()[kwargs['model']]
            dump = dict(id='offline-launch', model=model, service_tier='default',
                        usage=dict(prompt_tokens=1, completion_tokens=1),
                        choices=[dict(message=dict(content=text), finish_reason='stop')])
            response = SimpleNamespace(id=dump['id'], model=model, service_tier='default',
                        usage=SimpleNamespace(model_dump=lambda: dump['usage']),
                        choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason='stop')], model_dump=lambda: dump)
            return SimpleNamespace(parse=lambda: response, headers={})
        def retrieve(model):
            access.append(model)
            return SimpleNamespace(id='unavailable' if unavailable else model)
        client = SimpleNamespace(models=SimpleNamespace(retrieve=retrieve), chat=SimpleNamespace(
            completions=SimpleNamespace(with_raw_response=SimpleNamespace(create=create))))
        credential = self.enterContext(patch.object(p.legacy.paid, 'credential', return_value='offline-placeholder'))
        self.enterContext(patch.object(p.legacy.paid, 'OpenAI', side_effect=lambda **kwargs: contextlib.nullcontext(client)))
        return r, p, bundle, target, calls, access, credential

    def test_launch_reopens_cached_success_failure_and_interrupted_artifact(self):
        r, p, bundle, target, calls, access, credential = self.launch_fixture()
        with patch.object(p.engine, 'save_run', side_effect=OSError('Offline artifact interruption')), self.assertRaises(OSError):
            s.launch()
        self.assertEqual(len(calls), 1)
        s.launch()
        summary = s.read(target / 'summary.json')
        self.assertEqual((summary['supplemental_attempts'], summary['supplemental_valid_graphs'], summary['supplemental_failed_attempts']), (6, 1, 5))
        self.assertEqual(len(calls), 8)
        self.assertEqual(set(access), {'gpt-4.1', 'gpt-6-sol', 'gpt-6-luna', 'gpt-5.6-luna'})
        credential.reset_mock()
        ledger_sha = s.sha(target / 'budget.sqlite')
        s.launch()
        self.assertEqual(len(calls), 8)
        credential.assert_not_called()
        self.assertEqual(s.sha(target / 'budget.sqlite'), ledger_sha)
        r.write_json(bundle / 'review.json', dict(status='CONDITIONAL_PAID_GO', policy_sha256='0' * 64))
        with self.assertRaisesRegex(ValueError, 'clearance'):
            s.launch()
        r.write_json(bundle / 'review.json', dict(status='CONDITIONAL_PAID_GO', policy_sha256=s.sha(bundle / 'policy.json')))
        (target / 'budget.sqlite').rename(target / 'saved.sqlite')
        with self.assertRaisesRegex(ValueError, 'cannot be reset'):
            s.launch()
        self.assertFalse((target / 'budget.sqlite').exists())
        self.assertEqual(len(calls), 8)

    def test_launch_uncertain_request_stops_after_reopen(self):
        r, p, bundle, target, calls, access, credential = self.launch_fixture(transport_error=True)
        with self.assertRaises(RuntimeError):
            s.launch()
        credential.reset_mock()
        ledger_sha = s.sha(target / 'budget.sqlite')
        with self.assertRaisesRegex(ValueError, 'transport'):
            s.launch()
        self.assertEqual(len(calls), 1)
        credential.assert_not_called()
        self.assertEqual(s.sha(target / 'budget.sqlite'), ledger_sha)

    def test_unavailable_model_stops_before_reservation(self):
        r, p, bundle, target, calls, access, credential = self.launch_fixture(unavailable=True)
        with self.assertRaisesRegex(ValueError, 'unavailable'):
            s.launch()
        self.assertEqual(calls, [])

    def test_cli_never_prints_provider_exception_or_chain(self):
        output = io.StringIO()
        with patch.object(s.sys, 'argv', ['runner.py', 'launch']), patch.object(s, 'launch', side_effect=RuntimeError('PRIVATE_PROVIDER_SENTINEL')), contextlib.redirect_stdout(output):
            self.assertEqual(s.main(), 1)
        self.assertNotIn('PRIVATE_PROVIDER_SENTINEL', output.getvalue())
        self.assertIn('SUPPLEMENT_STOPPED', output.getvalue())

if __name__ == '__main__':
    unittest.main()
