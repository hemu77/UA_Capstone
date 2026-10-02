"""A received truncated reply is a failed trial, never a replacement request."""
import contextlib
import io
import json
import unittest
from unittest.mock import patch

import httpx
from openai import OpenAI

import resume_wording_probe as recovery
import test_wording_probe

probe = recovery.probe


class ResumeProbeTests(unittest.TestCase):
    setup_probe = test_wording_probe.WordingProbeTests.setup_probe

    def test_missing_environment_uses_hidden_input_not_shell_arguments(self):
        with patch.object(probe.paid, 'credential', side_effect=RuntimeError('No local key')):
            with patch.object(recovery.getpass, 'getpass', return_value='offline-test-key') as hidden:
                self.assertEqual(recovery.read_key(), 'offline-test-key')
                hidden.assert_called_once()

    def test_truncated_valid_prefix_is_failed_and_never_rebought(self):
        _, approved = self.setup_probe()
        calls, rows = [], probe.manifest()
        def handler(request):
            body = json.loads(request.content)
            index = len(calls)
            calls.append(body)
            ids = [r[0] for r in json.loads(body['messages'][1]['content'])['candidates']]
            return httpx.Response(200, json={'id': f'fixture-{index}', 'object': 'chat.completion',
                'created': 1, 'model': probe.MODEL, 'choices': [{'index': 0,
                'finish_reason': 'length' if index == 60 else 'stop',
                'message': {'role': 'assistant', 'content': ', '.join(ids[:rows[index]['count']])}}],
                'usage': {'prompt_tokens': 100, 'completion_tokens': 5, 'total_tokens': 105}})
        with OpenAI(api_key='offline-key-not-real', max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(handler))) as client:
            budget = probe.paid.Budget(probe.LEDGER, total_cap=approved['cumulative_ceiling_usd'], pilot_cap=1, require_existing=True)
            with contextlib.closing(budget.db), contextlib.redirect_stdout(io.StringIO()):
                # Reproduce a pre-existing settled failure before the resume.
                with self.assertRaisesRegex(ValueError, 'finish normally'):
                    probe.collect(client, budget, approved)
                partition = recovery.verify_partition(budget, approved)
                self.assertEqual(len(partition['received_ids']), 61)
                self.assertEqual(len(partition['missing_ids']), 11)
                # A lost historical receipt must fail BEFORE any API client opens.
                historical = budget.db.execute('SELECT * FROM attempts WHERE id=?', (partition['received_ids'][0],)).fetchone()
                with budget.db:
                    budget.db.execute('DELETE FROM attempts WHERE id=?', (historical[0],))
                with self.assertRaisesRegex(ValueError, '61 received'):
                    recovery.verify_partition(budget, approved)
                with budget.db:
                    budget.db.execute('INSERT INTO attempts VALUES (?,?,?,?,?,?)', historical)
                results = recovery.collect(client, budget, approved)
                cost = budget.spent()
                self.assertEqual(recovery.collect(client, budget, approved), results)
                self.assertEqual(cost, budget.spent())
        self.assertEqual(len(calls), 72)
        self.assertEqual(len(results), 72)
        self.assertFalse(results[60]['valid'], 'Even a parseable truncated prefix is not a successful reply.')
        self.assertEqual(results[60]['response']['finish_reason'], 'length')
        self.assertEqual(sum(not r['valid'] for r in results), 1)
        self.assertEqual(probe.authorize(), approved)

    def test_transport_error_still_aborts_without_retry(self):
        _, approved = self.setup_probe()
        calls = []
        def handler(request):
            calls.append(1)
            raise httpx.ReadTimeout('offline timeout', request=request)
        with OpenAI(api_key='offline-key-not-real', max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(handler))) as client:
            budget = probe.paid.Budget(probe.LEDGER, total_cap=approved['cumulative_ceiling_usd'], pilot_cap=1, require_existing=True)
            with contextlib.closing(budget.db):
                with self.assertRaisesRegex(RuntimeError, 'unresolved'):
                    recovery.collect(client, budget, approved)
                with self.assertRaisesRegex(RuntimeError, 'Unresolved'):
                    recovery.collect(client, budget, approved)
        self.assertEqual(len(calls), 1)


if __name__ == '__main__':
    unittest.main()
