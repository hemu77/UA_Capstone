"""Exercise the installed SDK over an offline transport, never the paid API."""
import json
import tempfile
import unittest
from pathlib import Path

import httpx
from openai import OpenAI

from paid_study import Budget, PaidCaller, build_cells, diagnostic_rows


class ApiDiagnosticsTests(unittest.TestCase):
    def test_raw_logging_transport_preserves_request_body_and_parsed_reply(self):
        seen = []
        def handler(request):
            seen.append(json.loads(request.content))
            return httpx.Response(200, json={'id': 'offline', 'object': 'chat.completion',
                'created': 1, 'model': 'fixture', 'choices': [{'index': 0, 'finish_reason': 'stop',
                'message': {'role': 'assistant', 'content': '0, 1'}}],
                'usage': {'prompt_tokens': 20, 'completion_tokens': 3, 'total_tokens': 23}})
        budget, caller, cell = self.exercise(handler)
        for model in ['gpt-4.1', 'gpt-5.6-luna', 'gpt-6-luna', 'gpt-6-sol']:
            settings = {'max_completion_tokens': 512,
                        **({'temperature': .8} if model == 'gpt-4.1' else {'reasoning_effort': 'none'})}
            kwargs = dict(model=model, messages=[{'role': 'user', 'content': 'offline fixture'}], extra_body=settings)
            parsed = caller.client.chat.completions.create(**kwargs)
            raw = caller.client.chat.completions.with_raw_response.create(**kwargs,
                extra_headers={'X-Client-Request-Id': 'offline-trace'})
            self.assertEqual(parsed.model_dump(), raw.parse().model_dump())
            self.assertEqual(seen[-2], seen[-1])

    def exercise(self, handler):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        budget = Budget(Path(folder.name) / 'budget.sqlite')
        self.addCleanup(budget.db.close)
        client = OpenAI(api_key='offline-test-not-a-key', max_retries=0,
                        http_client=httpx.Client(transport=httpx.MockTransport(handler)))
        self.addCleanup(client.close)
        cell = build_cells('pilot')[0]
        return budget, PaidCaller(client, budget, cell), cell

    def test_success_correlates_ids_and_filters_headers_without_changing_usage(self):
        seen = []
        def handler(request):
            seen.append(request)
            return httpx.Response(200, headers={'x-request-id': 'req_offline_success',
                'x-ratelimit-remaining-tokens': '1200', 'openai-processing-ms': '12.5',
                'set-cookie': 'private-cookie', 'x-private-header': 'private-header'}, json={
                'id': 'chatcmpl-offline', 'object': 'chat.completion', 'created': 1,
                'model': 'gpt-6-luna', 'choices': [{'index': 0, 'finish_reason': 'stop',
                'message': {'role': 'assistant', 'content': '0, 1'}}],
                'usage': {'prompt_tokens': 20, 'completion_tokens': 3, 'total_tokens': 23}})
        budget, caller, cell = self.exercise(handler)
        messages = [{'role': 'user', 'content': 'private-prompt-canary'}]
        self.assertEqual(caller(cell['model'], messages), '0, 1')
        rid, request, response = budget.db.execute('SELECT id,request,response FROM attempts').fetchone()
        saved = json.loads(response)
        self.assertEqual(seen[0].headers.get('x-client-request-id'), rid)
        self.assertEqual(json.loads(request)['client_request_id'], rid)
        diagnostics = saved['diagnostics']
        self.assertEqual(diagnostics['server_request_id'], 'req_offline_success')
        self.assertEqual(diagnostics['http_status'], 200)
        self.assertGreaterEqual(diagnostics['elapsed_ms'], 0)
        self.assertEqual(diagnostics['headers']['x-ratelimit-remaining-tokens'], '1200')
        self.assertNotIn('private', json.dumps(diagnostics))
        self.assertEqual(saved['usage']['prompt_tokens'], 20)
        self.assertEqual(PaidCaller(caller.client, budget, cell)(cell['model'], messages), '0, 1')
        self.assertEqual(len(seen), 1)

    def test_http_error_records_safe_cause_and_keeps_reservation_without_retry(self):
        calls = []
        def handler(request):
            calls.append(request)
            return httpx.Response(429, headers={'x-request-id': 'req_offline_limit', 'retry-after': '2',
                'x-ratelimit-remaining-tokens': 'sk-proj-private-canary'},
                json={'error': {'message': 'Authorization: private-message-canary',
                                'type': 'tokens', 'code': 'rate_limit_exceeded'}})
        budget, caller, cell = self.exercise(handler)
        with self.assertRaisesRegex(RuntimeError, 'unresolved'):
            caller(cell['model'], [{'role': 'user', 'content': 'private-prompt-canary'}])
        row = budget.db.execute('SELECT status,cost,response FROM attempts').fetchone()
        record = json.loads(row[2])
        self.assertEqual(row[0], 'uncertain')
        self.assertGreater(row[1], 0)
        self.assertEqual(record['diagnostics']['http_status'], 429)
        self.assertEqual(record['diagnostics']['server_request_id'], 'req_offline_limit')
        self.assertEqual(record['diagnostics']['error_code'], 'rate_limit_exceeded')
        self.assertNotIn('private', row[2])
        self.assertEqual(len(calls), 1)
        path = budget.db.execute('PRAGMA database_list').fetchone()[2]
        exported = diagnostic_rows(path)
        self.assertEqual(len(exported), 1)
        self.assertNotIn('private', json.dumps(exported))
        self.assertEqual(diagnostic_rows(path, caller.last_request_id), exported)

    def test_timeout_and_missing_usage_keep_transport_metadata_without_refunding(self):
        def timeout(request):
            raise httpx.ReadTimeout('private-timeout-canary', request=request)
        def missing_usage(request):
            return httpx.Response(200, headers={'x-request-id': 'req_missing_usage'}, json={
                'id': 'chatcmpl-offline', 'object': 'chat.completion', 'created': 1,
                'model': 'gpt-6-luna', 'choices': []})
        for handler, error_type, server_id in [(timeout, 'APITimeoutError', None),
                                               (missing_usage, 'RuntimeError', 'req_missing_usage')]:
            with self.subTest(error=error_type):
                budget, caller, cell = self.exercise(handler)
                with self.assertRaises(RuntimeError):
                    caller(cell['model'], [{'role': 'user', 'content': 'offline'}])
                status, cost, raw = budget.db.execute('SELECT status,cost,response FROM attempts').fetchone()
                record = json.loads(raw)
                self.assertEqual(status, 'uncertain')
                self.assertGreater(cost, 0)
                self.assertEqual(record['error_type'], error_type)
                self.assertEqual(record['diagnostics']['server_request_id'], server_id)
                self.assertNotIn('private', raw)

    def test_legacy_export_does_not_invent_a_client_id_or_expose_private_request(self):
        budget, _, _ = self.exercise(lambda request: self.fail('No transport call expected'))
        budget.reserve('legacy', 'pilot', .01, json.dumps({'messages': ['private-canary']}))
        budget.fail('legacy', 'APIConnectionError')
        path = budget.db.execute('PRAGMA database_list').fetchone()[2]
        exported = diagnostic_rows(path)
        self.assertIsNone(exported[0]['client_request_id'])
        self.assertIsNone(exported[0]['diagnostics'])
        self.assertNotIn('private', json.dumps(exported))

    def test_connection_failure_keeps_client_id_and_cause_without_claiming_receipt(self):
        calls = []
        def handler(request):
            calls.append(request)
            raise httpx.ConnectError('private-url-and-key-canary', request=request)
        budget, caller, cell = self.exercise(handler)
        messages = [{'role': 'user', 'content': 'private-prompt-canary'}]
        with self.assertRaises(RuntimeError):
            caller(cell['model'], messages)
        record = json.loads(budget.db.execute('SELECT response FROM attempts').fetchone()[0])
        diagnostics = record['diagnostics']
        self.assertEqual(diagnostics['client_request_id'], caller.last_request_id)
        self.assertIsNone(diagnostics['server_request_id'])
        self.assertIsNone(diagnostics['http_status'])
        self.assertEqual(diagnostics['cause_types'], ['ConnectError'])
        self.assertNotIn('private', json.dumps(record))
        with self.assertRaises(RuntimeError):
            PaidCaller(caller.client, budget, cell)(cell['model'], messages)
        self.assertEqual(len(calls), 1)


if __name__ == '__main__':
    unittest.main()
