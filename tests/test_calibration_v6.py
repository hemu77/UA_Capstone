"""Offline integration tests use temporary ledgers and intercepted SDK transport."""
import contextlib
import copy
import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import httpx
import networkx as nx
from openai import OpenAI

import calibration_v6 as runner


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.home = Path(temp.name)
        self.enterContext(patch.object(runner, 'BASE', self.home / 'results'))
        self.enterContext(patch.object(runner, 'LEDGER', self.home / 'ledger.sqlite'))
        self.enterContext(patch.object(runner.frozen, 'DESTINATION', self.home / 'workflow'))
        self.spec = runner.contract()
        self.target = runner.folder(self.spec)

    def budget(self, cap=10):
        value = runner.paid.Budget(runner.LEDGER, total_cap=cap, pilot_cap=min(5,cap))
        self.addCleanup(value.db.close)
        return value

    def cell(self, method='global', model='gpt-6-luna', language='english'):
        return next(c for c in runner.cells(self.spec) if (c['method'],c['model'],c['language'],c['culture'],c['repetition']) == (method,model,language,'us',0))

    def client(self, responder):
        calls = []
        def handle(request):
            body = json.loads(request.content)
            calls.append(body)
            text, finish = responder(body, len(calls))
            return httpx.Response(200, json={'id': f'fixture-{len(calls)}', 'object':'chat.completion',
                'created':1, 'model':body['model'], 'choices':[{'index':0,'finish_reason':finish,
                'message':{'role':'assistant','content':text}}],
                'usage':{'prompt_tokens':100,'completion_tokens':5,'total_tokens':105}})
        client = OpenAI(api_key='offline-not-real', max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(handle)))
        self.addCleanup(client.close)
        return client, calls

    def approved(self, scope='core68'):
        runner.prepare()
        runner.write_json(self.target/'preflight.json', dict(status='OFFLINE_PREFLIGHT_PASSED',contract_sha256=runner.frozen.digest(self.spec)))
        budget = self.budget()
        budget.reserve('historical', 'main', .2, '{}')
        budget.settle('historical', .2, dict(text='historical'))
        # Authorization unit fixtures do not copy the user's private billing DB.
        with patch.object(runner, 'verify_probe'):
            return runner.approve(1, scope, 'offline-test-only', date.today().isoformat())

    def test_design_has_core_68_plus_36_distinct_coverage_cells(self):
        rows = runner.cells(self.spec)
        self.assertEqual(len(rows), 104)
        self.assertEqual(len({r['run_id'] for r in rows}), 104)
        self.assertEqual(sum(r['stage']=='core68' for r in rows), 68)
        for model in runner.revised.config()['models']:
            for language in runner.revised.config()['languages']:
                self.assertEqual({r['method'] for r in rows if r['model']==model and r['language']==language},
                    set(runner.revised.config()['methods']))
        self.assertTrue(all(r['personas']==50 for r in rows))

    def test_unapproved_execution_cannot_read_credentials_or_construct_client(self):
        runner.prepare()
        with patch.object(runner.paid,'credential',side_effect=AssertionError('Key must not be read')), \
                patch.object(runner.paid,'OpenAI',side_effect=AssertionError('API must not open')):
            with self.assertRaisesRegex(ValueError, 'not authorized'):
                runner.execute()
        self.assertFalse((self.target/'authorization.json').exists())

    def test_all_models_and_methods_use_real_sdk_cache_and_exact_replay(self):
        budget = self.budget()
        for model in runner.revised.config()['models']:
            for method in runner.revised.config()['methods']:
                cell = self.cell(method, model)
                fixture = runner.FixtureCaller(cell)
                def respond(body, index):
                    self.assertEqual({k:body[k] for k in runner.settings(model,method)}, runner.settings(model,method))
                    return fixture(body['model'],body['messages']), 'stop'
                client, calls = self.client(respond)
                caller = runner.CalibrationCaller(client,budget,cell,self.spec,self.target)
                result = runner.run_cell(cell,caller)
                requests = []
                for rid in caller.request_ids:
                    row = runner.ledger_row(budget,rid)
                    requests.append(dict(request_id=rid,request=json.loads(row[4]),response=json.loads(row[5])))
                path = self.target/'runs'/(cell['run_id']+'.json')
                runner.save_run(path,cell,result,requests,'received_revised_calibration')
                runner.verify_receipt(path,cell,budget)
                count, cost = len(calls), budget.spent()
                again = runner.CalibrationCaller(None,budget,cell,self.spec,self.target)
                rerun = runner.run_cell(cell,again)
                self.assertEqual(rerun[1:],result[1:])
                self.assertEqual(len(calls),count)
                self.assertEqual(budget.spent(),cost)
                if method!='global':
                    self.assertEqual([e['persona'] for e in result[1][:50]],result[3]['actor_order'])
                    self.assertEqual({d['actor']:d['requested_count'] for d in result[2][:50]},result[3]['nomination_quotas'])

    def test_global_repairs_never_shrink_tie_set_and_none_is_valid(self):
        budget = self.budget()
        cell = self.cell()
        replies = ['0, 1\n1, 0\n2, 3','0, 1','0, 1\n2, 3']
        def respond(body, index):
            if index>1:
                self.assertIn('required_pairs',body['messages'][0]['content'])
            return replies[index-1],'stop'
        client,calls = self.client(respond)
        result = runner.run_cell(cell,runner.CalibrationCaller(client,budget,cell,self.spec,self.target))
        self.assertEqual(result[0].number_of_edges(),2)
        self.assertEqual([d['valid'] for d in result[2]],[False,False,True])
        other = self.cell(language='portuguese')
        client,_ = self.client(lambda *_:('NONE','stop'))
        result = runner.run_cell(other,runner.CalibrationCaller(client,budget,other,self.spec,self.target))
        self.assertEqual(len(result[0]),50)
        self.assertEqual(result[0].number_of_edges(),0)
        self.assertIsNone(runner.measures(result[0],runner.revised.previous.adult_roster())['modularity'])

    def test_public_inspection_replays_empty_graph_without_key_or_private_ledger(self):
        from inspect_calibration_v6 import inspect
        runner.prepare()
        cell, budget = self.cell(), self.budget()
        client, calls = self.client(lambda *_: ('NONE','stop'))
        caller = runner.CalibrationCaller(client,budget,cell,self.spec,self.target)
        result = runner.run_cell(cell,caller)
        requests = []
        for rid in caller.request_ids:
            row = runner.ledger_row(budget,rid)
            requests.append(dict(request_id=rid,request=json.loads(row[4]),response=json.loads(row[5])))
        runner.save_run(self.target/'runs'/(cell['run_id']+'.json'),cell,result,requests,'received_revised_calibration')
        with patch.object(runner.paid,'credential',side_effect=AssertionError('No key allowed')), \
                patch.object(runner.paid,'Budget',side_effect=AssertionError('No private ledger allowed')), \
                patch.object(runner.paid,'OpenAI',side_effect=AssertionError('No API client allowed')):
            report = inspect()
        self.assertEqual(len(calls),1)
        self.assertEqual(report['verified_networks'],1)
        self.assertEqual(report['status'],'PARTIAL_CALIBRATION')
        self.assertEqual(report['empty_networks'],[cell['run_id']])
        self.assertEqual(report['undefined_metrics']['modularity'],1)
        self.assertEqual(report['first_decisions'],1)
        self.assertEqual(report['hypothetical_main_forecasts'],[])
        # Missing receipts must not make old CSV contents look newly verified.
        (self.target/'runs'/(cell['run_id']+'.json')).unlink()
        with self.assertRaisesRegex(ValueError, 'Existing table conflicts with empty'):
            inspect()

    def test_sequential_early_correction_remains_local(self):
        cell, budget = self.cell('sequential',language='portuguese'), self.budget()
        fixture = runner.FixtureCaller(cell)
        def respond(body,index):
            if index == 1:
                return '999','stop'
            if index == 2:
                self.assertNotIn('degree',body['messages'][0]['content'])
                self.assertNotIn('degree',json.loads(body['messages'][1]['content'])['fields'])
            return fixture(body['model'],body['messages']),'stop'
        client,_ = self.client(respond)
        result = runner.run_cell(cell,runner.CalibrationCaller(client,budget,cell,self.spec,self.target))
        self.assertEqual([d['prompt_method'] for d in result[2][:4]],['local']*4)

    def test_uncertain_and_truncated_replies_stop_without_repurchase(self):
        budget,cell = self.budget(),self.cell()
        client,calls = self.client(lambda *_:('0, 1','length'))
        with self.assertRaises(ValueError):
            runner.run_cell(cell,runner.CalibrationCaller(client,budget,cell,self.spec,self.target))
        with self.assertRaises(ValueError):
            runner.run_cell(cell,runner.CalibrationCaller(client,budget,cell,self.spec,self.target))
        self.assertEqual(len(calls),1)
        cell = self.cell(language='hindi')
        def lost(*_):
            raise httpx.ReadTimeout('offline timeout')
        client,calls = self.client(lost)
        with self.assertRaises(RuntimeError):
            runner.run_cell(cell,runner.CalibrationCaller(client,budget,cell,self.spec,self.target))
        with self.assertRaises(RuntimeError):
            runner.run_cell(cell,runner.CalibrationCaller(client,budget,cell,self.spec,self.target))
        self.assertEqual(len(calls),1)
        self.assertEqual(budget.unresolved_count(),1)

    def test_lost_historical_or_inflight_receipt_blocks_before_client(self):
        approved = self.approved()
        budget = runner.paid.Budget(runner.LEDGER,require_existing=True)
        self.addCleanup(budget.db.close)
        with budget.db:
            budget.db.execute("DELETE FROM attempts WHERE id='historical'")
        with patch.object(runner.paid,'OpenAI',side_effect=AssertionError('Must not open')):
            with self.assertRaisesRegex(ValueError,'Historical ledger'):
                runner.execute(1)

    def test_cap_and_reauthorization_and_changed_contract_fail_closed(self):
        approved = self.approved()
        with self.assertRaisesRegex(ValueError,'already exists'):
            runner.approve(2,'core68','second-test',date.today().isoformat())
        budget = self.budget(cap=1.2)
        budget.reserve('other_cost','main',1,'{}')
        budget.settle('other_cost',1,{})
        cell = self.cell()
        client,calls = self.client(lambda *_:('0, 1','stop'))
        with self.assertRaises(runner.paid.BudgetExceeded):
            runner.run_cell(cell,runner.CalibrationCaller(client,budget,cell,self.spec,self.target))
        self.assertEqual(len(calls),0)
        self.assertFalse(list((self.target/'requests').glob('*.json')))
        changed = copy.deepcopy(self.spec)
        changed['correction_attempts'] = 4
        with self.assertRaisesRegex(ValueError,'not authorized'):
            runner.review(changed)

    def test_prior_evidence_and_imported_source_guards(self):
        with self.assertRaisesRegex(ValueError,'below verified prior spending'):
            runner.verify_probe(self.budget())
        with patch.object(runner,'IMPORTED_SOURCES',{}):
            with self.assertRaisesRegex(ValueError,'after import'):
                runner.contract()

    def test_complete_execute_prefix_is_cached_and_reports_real_receipt_usage(self):
        self.approved()
        cell = runner.cells(self.spec)[0]
        fixture = runner.FixtureCaller(cell)
        client,calls = self.client(lambda body,index:(fixture(body['model'],body['messages']),'stop'))
        with patch.object(runner,'verify_probe'), patch.object(runner.paid,'credential',return_value='offline-not-real'), \
                patch.object(runner.paid,'OpenAI',return_value=client) as factory, patch.object(client.models,'retrieve'):
            first = runner.execute(1)
            self.assertEqual(first['completed_networks'],1)
            self.assertEqual(first['planned_networks'],68)
            self.assertEqual(first['status'],'PARTIAL')
            self.assertEqual(len(calls),1)
            self.assertEqual(factory.call_args.kwargs['base_url'],'https://api.openai.com/v1')
        with patch.object(runner,'verify_probe'), patch.object(runner.paid,'OpenAI',side_effect=AssertionError('Cached run needs no client')):
            self.assertEqual(runner.execute(1),first)
            with self.assertRaisesRegex(ValueError,'scope'):
                runner.execute(69)
        self.assertTrue((self.target/'cost_stats.csv').exists())
        path = self.target/'runs'/(cell['run_id']+'.json')
        record = json.loads(path.read_text(encoding='utf-8'))
        record['metrics']['density'] += .1
        runner.write_json(path,record)
        with patch.object(runner,'verify_probe'), patch.object(runner.paid,'OpenAI',side_effect=AssertionError('Must stop before API')):
            with self.assertRaisesRegex(ValueError,'metrics'):
                runner.execute(2)

    def test_deleted_partial_request_and_unsettled_intent_cannot_be_rebought(self):
        approved = self.approved()
        budget = self.budget()
        cell = self.cell()
        client,calls = self.client(lambda *_:('0, 1','stop'))
        caller = runner.CalibrationCaller(client,budget,cell,self.spec,self.target)
        runner.run_cell(cell,caller)
        rid = caller.request_ids[0]
        with budget.db:
            budget.db.execute('DELETE FROM attempts WHERE id=?',(rid,))
        with patch.object(runner,'verify_probe'), patch.object(runner.paid,'OpenAI',side_effect=AssertionError('Must not open')):
            with self.assertRaisesRegex(ValueError,'journaled'):
                runner.execute()
        with self.assertRaisesRegex(ValueError,'journaled'):
            runner.run_cell(cell,runner.CalibrationCaller(client,budget,cell,self.spec,self.target))
        self.assertEqual(len(calls),1)
        runner.write_json(self.target/'requests'/(rid+'.json'),dict(request_id=rid,status='INTENT_BEFORE_TRANSPORT'))
        with patch.object(runner,'verify_probe'):
            with self.assertRaisesRegex(ValueError,'journaled'):
                runner.verify_accounting(budget,approved,self.target,{c['run_id']:c for c in runner.cells(self.spec)})

    def test_abandoned_request_cannot_trigger_shared_caller_replacement(self):
        self.approved()
        budget,cell = self.budget(),self.cell()
        def lost(*_):
            raise httpx.ReadTimeout('offline timeout')
        client,calls = self.client(lost)
        caller = runner.CalibrationCaller(client,budget,cell,self.spec,self.target)
        with self.assertRaises(RuntimeError):
            runner.run_cell(cell,caller)
        budget.abandon(caller.last_request_id,'offline-test-retirement','offline-replacement-test')
        caller.checkpoint()
        with patch.object(runner,'verify_probe'), patch.object(runner.paid,'credential',return_value='offline-not-real'), \
                patch.object(runner.paid,'OpenAI',side_effect=AssertionError('Must not open replacement-capable client')):
            with self.assertRaisesRegex(ValueError,'non-received'):
                runner.execute(1)
        with self.assertRaisesRegex(RuntimeError,'non-received'):
            runner.run_cell(cell,runner.CalibrationCaller(client,budget,cell,self.spec,self.target))
        self.assertEqual(len(calls),1)


if __name__ == '__main__':
    unittest.main()
