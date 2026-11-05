import contextlib
import importlib.util
import io
import json
from pathlib import Path
import socket
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('completion', Path(__file__).with_name('runner.py'))
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)

class Completion(unittest.TestCase):
    def setUp(self):
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))

    def prior(self):
        _, cells = c.s.load_policy()
        return {v['original_run_id']: c.read(c.PRIOR / 'runs' / (v['run_id'] + '.json')) for v in cells}

    def test_unique_bounded_ids_and_stop_on_first_success(self):
        prior = self.prior()
        cells = c.plan(prior, 'a' * 64)
        self.assertEqual(len(cells), 6)
        self.assertEqual({v['completion_attempt'] for v in cells}, {1,2,3})
        self.assertEqual(len({v['run_id'] for v in cells}), 6)
        saved, sent = {}, []
        for cell in c.eligible(cells, saved):
            sent.append(cell)
            saved[cell['run_id']] = {'failure': None}
        self.assertEqual(len(sent), 2)
        saved, sent = {}, []
        for cell in c.eligible(cells, saved):
            sent.append(cell)
            saved[cell['run_id']] = {'failure': 'invalid_global_response'}
        self.assertEqual(len(sent), 6)

    def test_received_prefix_after_success_is_not_silently_ignored(self):
        cells = c.plan(self.prior(), 'a' * 64)
        saved = {cells[0]['run_id']: {'failure': None}}
        with self.assertRaisesRegex(ValueError, 'after successful'):
            list(c.eligible(cells, saved, {cells[1]['run_id']}))

    def test_cross_cell_conflict_is_checked_before_any_credentials(self):
        bundle, target, calls, credential = self.fixture()
        cells = c.plan(self.prior(), c.sha(bundle / 'policy.json'))
        saved = {cells[3]['run_id']: {'failure': None}}
        paid = {cells[4]['run_id']}
        validate = c.validate_chains
        with patch.object(c, 'validate_chains', side_effect=lambda *args: validate(cells, saved, paid)), self.assertRaisesRegex(ValueError, 'Existing chain'):
            c.launch()
        credential.assert_not_called()
        self.assertEqual(calls, [])

    def fixture(self, *, fail_all=False, uncertain=False, anchor=102.63348622500001):
        root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        bundle, target = root / 'bundle', root / 'target'
        bundle.mkdir()
        (bundle / 'test_completion.py').write_bytes(Path(__file__).read_bytes())
        prior, spec = self.prior(), c.p.load_contract()
        self.enterContext(patch.object(c, 'BUNDLE', bundle))
        self.enterContext(patch.object(c, 'TARGET', target))
        self.enterContext(patch.object(c, 'baseline', return_value=(spec, prior, anchor)))
        self.enterContext(patch.object(c, 'manifest', return_value={'fixture': 'a'*64}))
        self.enterContext(patch.object(c.r, 'output_lock', side_effect=lambda: c.r.workflow_lock(root)))
        self.enterContext(patch.object(socket.socket, 'connect', side_effect=AssertionError('Offline fixture')))
        policy = dict(supplement_id=c.NAME, contract_sha256=c.s.CONTRACT, upstream_sha256=c.UPSTREAM_SHA,
                      runner_sha256=c.sha(c.__file__), tests_sha256=c.sha(bundle / 'test_completion.py'),
                      original_and_prior_usd=anchor, additional_cap_usd=c.s.remaining_allowance(anchor), cumulative_cap_usd=120,
                      prior_manifest_sha256=c.p.legacy.digest(c.manifest(c.COPY)),
                      missing_originals=sorted(k for k,v in prior.items() if v['failure'] is not None),
                      max_additional_per_cell=3, target_unique_valid_cells=6)
        c.r.write_json(bundle / 'policy.json', policy)
        digest = c.sha(bundle / 'policy.json')
        cells = c.plan(prior, digest)
        c.r.write_json(bundle / 'offline-verified.json', dict(status='COMPLETION_OFFLINE_VERIFIED', policy_sha256=digest,
                      prior_replays=6, fixture_cases=len(cells), paid_calls=0))
        c.r.write_json(bundle / 'review.json', dict(status='CONDITIONAL_PAID_GO', policy_sha256=digest))
        calls = []
        def create(**kwargs):
            calls.append(kwargs)
            if uncertain:
                raise RuntimeError('PRIVATE_SENTINEL')
            text = 'not a graph' if fail_all or len(calls) == 1 else '0, 1'
            model = c.a.expected_models()[kwargs['model']]
            dump = dict(id='completion-fixture', model=model, service_tier='default',
                        usage=dict(prompt_tokens=1, completion_tokens=1),
                        choices=[dict(message=dict(content=text), finish_reason='stop')])
            response = SimpleNamespace(id=dump['id'], model=model, service_tier='default',
                        usage=SimpleNamespace(model_dump=lambda: dump['usage']),
                        choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason='stop')], model_dump=lambda: dump)
            return SimpleNamespace(parse=lambda: response, headers={})
        client = SimpleNamespace(models=SimpleNamespace(retrieve=lambda m: SimpleNamespace(id=m)), chat=SimpleNamespace(
            completions=SimpleNamespace(with_raw_response=SimpleNamespace(create=create))))
        credential = self.enterContext(patch.object(c.p.legacy.paid, 'credential', return_value='offline-placeholder'))
        self.enterContext(patch.object(c.p.legacy.paid, 'OpenAI', side_effect=lambda **kw: contextlib.nullcontext(client)))
        return bundle, target, calls, credential

    def test_actual_launch_cache_reopen_stops_after_two_successes(self):
        bundle, target, calls, credential = self.fixture()
        with patch.object(c.p.engine, 'save_run', side_effect=OSError('Offline interrupted plot')), self.assertRaises(OSError):
            c.launch()
        self.assertEqual(len(calls), 2)
        c.launch()
        summary = c.read(target / 'summary.json')
        self.assertEqual((summary['valid_supplemental_cells'], summary['additional_attempts'], summary['all_attempts'], summary['all_valid_graphs']), (6,3,905,896))
        self.assertEqual(summary['status'], 'COMPLETION_TARGET_REACHED')
        self.assertEqual(len(calls), 3)
        credential.reset_mock()
        digest = c.sha(target / 'budget.sqlite')
        c.launch()
        credential.assert_not_called()
        self.assertEqual(len(calls), 3)
        self.assertEqual(c.sha(target / 'budget.sqlite'), digest)
        c.r.write_json(bundle / 'review.json', {'status':'CONDITIONAL_PAID_GO','policy_sha256':'0'*64})
        with self.assertRaisesRegex(ValueError, 'Review required'):
            c.launch()
        c.r.write_json(bundle / 'review.json', {'status':'CONDITIONAL_PAID_GO','policy_sha256':c.sha(bundle / 'policy.json')})
        (target / 'budget.sqlite').rename(target / 'retained.sqlite')
        with self.assertRaisesRegex(ValueError, 'Never reset'):
            c.launch()
        self.assertFalse((target / 'budget.sqlite').exists())

    def test_exhaustion_has_finite_limit_and_no_automatic_extra_draw(self):
        bundle, target, calls, credential = self.fixture(fail_all=True)
        c.launch()
        self.assertEqual(len(calls), 6)
        self.assertEqual(c.read(target / 'summary.json')['status'], 'BOUNDED_LIMIT_REACHED')
        credential.reset_mock()
        c.launch()
        self.assertEqual(len(calls), 6)
        credential.assert_not_called()

    def test_uncertain_transport_preserved_without_repurchase(self):
        bundle, target, calls, credential = self.fixture(uncertain=True)
        with self.assertRaises(RuntimeError):
            c.launch()
        digest = c.sha(target / 'budget.sqlite')
        credential.reset_mock()
        with self.assertRaisesRegex(ValueError, 'transport'):
            c.launch()
        credential.assert_not_called()
        self.assertEqual(len(calls), 1)
        self.assertEqual(c.sha(target / 'budget.sqlite'), digest)

    def test_cumulative_reservation_cap_blocks_before_transport(self):
        bundle, target, calls, credential = self.fixture(anchor=119.999999)
        with self.assertRaises(c.p.legacy.paid.BudgetExceeded):
            c.launch()
        self.assertEqual(calls, [])

    def test_zero_rounded_allowance_fails_closed(self):
        bundle, target, calls, credential = self.fixture(anchor=119.99999999)
        with self.assertRaises(ValueError):
            c.launch()
        credential.assert_not_called()
        self.assertEqual(calls, [])

    def test_changed_runner_clearance_refuses_before_credentials(self):
        bundle, target, calls, credential = self.fixture()
        policy = c.read(bundle / 'policy.json')
        policy['runner_sha256'] = '0' * 64
        c.r.write_json(bundle / 'policy.json', policy)
        with self.assertRaisesRegex(ValueError, 'Code changed'):
            c.launch()
        credential.assert_not_called()
        self.assertFalse(target.exists())

    def test_redirected_checkpoint_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(Path, 'is_symlink', return_value=True), self.assertRaisesRegex(ValueError, 'redirected'):
                c.manifest(root)

    def test_cli_does_not_expose_exception_body(self):
        output = io.StringIO()
        with patch.object(c.sys, 'argv', ['runner.py','launch']), patch.object(c, 'launch', side_effect=RuntimeError('PRIVATE_SENTINEL')), contextlib.redirect_stdout(output):
            self.assertEqual(c.main(), 1)
        self.assertNotIn('PRIVATE_SENTINEL', output.getvalue())

if __name__ == '__main__':
    unittest.main()
