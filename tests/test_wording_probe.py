"""A stopped/restarted wording screen must not pay for a reply twice."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from openai import OpenAI

import wording_probe as probe


class WordingProbeTests(unittest.TestCase):
    def test_manifest_is_fixed_randomized_and_pairs_have_identical_data(self):
        rows = probe.manifest()
        self.assertEqual(len(rows), 72)
        self.assertEqual(rows, probe.manifest())
        self.assertEqual(len({r['probe_id'] for r in rows}), 72)
        self.assertNotEqual([r['wording'] for r in rows], sorted(r['wording'] for r in rows))
        groups = {}
        for row in rows:
            groups.setdefault((row['language'], row['count'], row['actor']), []).append(row)
        self.assertTrue(all(len(pair) == 2 and pair[0]['user'] == pair[1]['user'] for pair in groups.values()))

    def setup_probe(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        path = Path(folder.name)
        self.enterContext(patch.object(probe, 'OUTPUT', path / 'probe'))
        self.enterContext(patch.object(probe, 'LEDGER', path / 'ledger.sqlite'))
        self.enterContext(patch.object(probe.revised.frozen, 'DESTINATION', path / 'workflow'))
        initial = probe.paid.Budget(probe.LEDGER)
        initial.reserve('old_cost', 'main', .4, '{}')
        initial.settle('old_cost', .4, {'text': 'old'})
        initial.db.close()
        return path, probe.authorize()

    def test_single_calls_resume_cached_and_keep_original_cap(self):
        _, approved = self.setup_probe()
        calls = []
        rows = probe.manifest()
        def handler(request):
            body = json.loads(request.content)
            self.assertEqual(body['max_completion_tokens'], 512)
            self.assertEqual(body['reasoning_effort'], 'none')
            index = len(calls)
            calls.append(body)
            ids = [r[0] for r in json.loads(body['messages'][1]['content'])['candidates']]
            return httpx.Response(200, json={'id': f'fixture-{index}', 'object': 'chat.completion',
                'created': 1, 'model': probe.MODEL, 'choices': [{'index': 0, 'finish_reason': 'stop',
                'message': {'role': 'assistant', 'content': ', '.join(ids[:rows[index]['count']])}}],
                'usage': {'prompt_tokens': 100, 'completion_tokens': 5, 'total_tokens': 105}})
        with OpenAI(api_key='offline-key-not-real', max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(handler))) as client:
            budget = probe.paid.Budget(probe.LEDGER, total_cap=approved['cumulative_ceiling_usd'], pilot_cap=1, require_existing=True)
            with contextlib.closing(budget.db), contextlib.redirect_stdout(io.StringIO()):
                first = probe.collect(client, budget, approved)
                cost = budget.spent()
                self.assertEqual(probe.collect(client, budget, approved), first)
                self.assertEqual(budget.spent(), cost)
        self.assertEqual(len(calls), 72, 'Resume must not purchase replacement replies.')
        self.assertEqual(probe.authorize(), approved, 'Authorization must not re-anchor the $1 allowance.')
        self.assertEqual(probe.summarize(first)['engineering_screen'], 'NO_FAILURES_OBSERVED')
        self.assertGreater(probe.summarize(first)['groups'][0]['descriptive_binomial_p975'], .4)

    def test_uncertain_reply_stops_and_reserves_without_resending(self):
        _, approved = self.setup_probe()
        calls = []
        def handler(request):
            calls.append(1)
            raise httpx.ReadTimeout('offline timeout', request=request)
        with OpenAI(api_key='offline-key-not-real', max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(handler))) as client:
            budget = probe.paid.Budget(probe.LEDGER, total_cap=approved['cumulative_ceiling_usd'], pilot_cap=1, require_existing=True)
            with contextlib.closing(budget.db):
                with self.assertRaisesRegex(RuntimeError, 'unresolved'):
                    probe.collect(client, budget, approved)
                cost = budget.spent()
                with self.assertRaisesRegex(RuntimeError, 'Unresolved'):
                    probe.collect(client, budget, approved)
                self.assertEqual(cost, budget.spent())
                self.assertGreater(cost, approved['ledger_start_usd'])
        self.assertEqual(len(calls), 1)

    def test_exhausted_cap_cannot_dispatch_and_partial_cannot_pass(self):
        _, approved = self.setup_probe()
        budget = probe.paid.Budget(probe.LEDGER, total_cap=approved['cumulative_ceiling_usd'], pilot_cap=1, require_existing=True)
        with contextlib.closing(budget.db):
            budget.reserve('other_allowed_work', 'main', 1, '{}')
            budget.settle('other_allowed_work', 1, {})
            with self.assertRaises(probe.paid.BudgetExceeded):
                probe.collect(None, budget, approved)
        with self.assertRaisesRegex(ValueError, 'exact completed'):
            probe.summarize([])

    def test_saved_manifest_tampering_stops_authorization_before_spending(self):
        self.setup_probe()
        (probe.OUTPUT / 'manifest.json').write_text('[]', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Saved probe manifest'):
            probe.authorization()


if __name__ == '__main__':
    unittest.main()
