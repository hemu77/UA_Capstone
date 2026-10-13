"""Isolated SDK fixtures for the extra $1 probe; no network or paid calls."""
import contextlib
import io
import json
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

import httpx
from openai import OpenAI

import probe_hindi_candidate as p
from tests import test_repair_v7 as fixtures


class HindiProbeTests(unittest.TestCase):
    def test_contract_and_original_are_isolated(self):
        before = p.v.contract('wording_probe')
        with p.isolated():
            spec = p.contract()
            self.assertEqual(spec['additional_cap_usd'], 1.)
            self.assertFalse(spec['main_authorized'])
            self.assertEqual(len(spec['design']), 4)
            self.assertEqual([r['wording'] for r in spec['design']],
                             ['original', 'clarified', 'clarified', 'original'])
            self.assertEqual({r['seed'] for r in spec['design']}, {34000, 34001})
            self.assertIn('revision_hindi_candidate.py', spec['sources'])
            for row in p.catalog():
                expected = p.candidate.frozen.treatment_prompt(
                    'local', 'hindi', 'us', '0', row['count'], wording='alternative_2')
                self.assertEqual(row['system'], expected if row['wording'] == 'original'
                                 else expected+' '+p.candidate.FORMAT_CLARIFICATION)
            with self.assertRaises(ValueError):
                p.contract('calibration')
        self.assertEqual(p.v.contract('wording_probe'), before)

    def test_preflight_cap_resume_failure_and_public_replay(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        r = p.v.r
        self.enterContext(patch.object(r, 'DESTINATION', root/'repair'))
        self.enterContext(patch.object(r, 'LEDGER', root/'ledger.sqlite'))
        self.enterContext(patch.object(r.base.frozen, 'DESTINATION', root/'lock'))
        self.enterContext(patch.object(socket.socket, 'connect', side_effect=AssertionError('No network')))
        budget = r.paid.Budget(r.LEDGER)
        with contextlib.closing(budget.db):
            # Model the existing ledger; the frozen runner retains a $5 pilot bucket.
            budget.reserve('prior-history-fixture', 'main', 5., '{}')
            budget.settle('prior-history-fixture', 5., {})
        self.assertEqual(p.run('preflight')['fixture_count'], 4)
        spec = p.contract()
        target = r.folder(spec)
        r.write_json(target/'review.json', dict(contract_sha256=r.digest(spec),
            prompt_semantics_reviewed=True, diagnostic_design_reviewed=True,
            spending_safeguards_reviewed=True, human_bilingual_validation=False,
            review_reference='offline test only', rates=spec['rates'],
            prices_verified_on=p.v.date.today().isoformat(), price_evidence='offline fixture'))
        with self.assertRaisesRegex(ValueError, 'Verified completed V8 calibration required'):
            p.run('approve')
        self.assertFalse((target.parent/'authorization.json').exists())
        calibration = p.BASE_CONTRACT('calibration')
        previous = r.folder(calibration)
        completed = dict(status='COMPLETE_REVIEW_REQUIRED', contract_sha256=r.digest(calibration),
                         completed_cells=104)
        # Explicit fixtures for the prior-stage gate, not published calibration evidence.
        for partial in ({**completed, 'status': 'STOPPED'}, {**completed, 'completed_cells': 17}):
            r.write_json(previous/'summary.json', partial)
            r.write_json(previous/'inspection.json', dict(status='PUBLIC_REPLAY_VERIFIED',
                contract_sha256=r.digest(calibration), summary_sha256=r.digest(partial)))
            with self.assertRaisesRegex(ValueError, 'Verified completed V8 calibration required'):
                p.run('approve')
            self.assertFalse((target.parent/'authorization.json').exists())
        r.write_json(previous/'summary.json', completed)
        r.write_json(previous/'inspection.json', dict(status='PUBLIC_REPLAY_VERIFIED',
            contract_sha256=r.digest(calibration), summary_sha256='stale'))
        with self.assertRaisesRegex(ValueError, 'Verified completed V8 calibration required'):
            p.run('approve')
        self.assertFalse((target.parent/'authorization.json').exists())
        r.write_json(previous/'inspection.json', dict(status='PUBLIC_REPLAY_VERIFIED',
            contract_sha256=r.digest(calibration), summary_sha256=r.digest(completed)))
        self.assertEqual(p.run('approve')['additional_usd'], 1.)
        auth_path = target.parent/'authorization.json'
        authorization = p.v.read(auth_path)
        replies = {}
        for cell in r.cells(spec):
            fixture = r.base.FixtureCaller(cell)
            with p.isolated(), p.adapter(cell, spec):
                r.base.run_cell(cell, fixture)
            for request in fixture.requests:
                replies[json.dumps(request['messages'], sort_keys=True)] = request['text']
        calls = []
        def handler(request):
            if request.method == 'GET':
                return fixtures.RepairTests.response(request)
            body = json.loads(request.content)
            calls.append(body)
            # Force both clarified fixtures to fail; never hide or replace failures.
            if p.candidate.FORMAT_CLARIFICATION in body['messages'][0]['content']:
                text = '6,29\n6,41'
            else:
                text = replies[json.dumps(body['messages'], sort_keys=True)]
            return fixtures.RepairTests.response(request, text=text)
        self.enterContext(patch.object(r.paid, 'credential', return_value='offline-test-key'))
        self.enterContext(patch.object(r.paid, 'OpenAI', side_effect=lambda **kwargs: OpenAI(
            api_key='offline-test-key', max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(handler)))))
        r.write_json(auth_path, {**authorization, 'additional_usd': 2.})
        with self.assertRaisesRegex(ValueError, 'allowance changed'):
            p.run('execute')
        self.assertEqual(calls, [])
        r.write_json(auth_path, authorization)
        with contextlib.redirect_stdout(io.StringIO()):
            result = p.run('execute')
            self.assertEqual((result['completed_cells'], result['completed_graphs'], result['invalid_cells']), (4, 2, 2))
            count = len(calls)
            self.assertEqual(p.run('verify'), result)
            report = p.run('inspect')
        self.assertEqual(len(calls), count)
        self.assertEqual((report['observations'], report['valid_graphs'], report['invalid_cells']), (4, 2, 2))
        auth_path.unlink()
        with self.assertRaisesRegex(ValueError, 'Prior failure evidence'):
            p.run('approve')


if __name__ == '__main__':
    unittest.main()
