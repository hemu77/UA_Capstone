"""Real fresh-run persistence with a fake provider: no credentials or API use."""
import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import networkx as nx
import pandas as pd

import revision224 as study
from paid_study import Budget


class ExecutionTests(unittest.TestCase):
    def test_unresolved_other_protocol_blocks_new_client(self):
        with self.workspace() as (config, root, ledger, client, factory):
            budget = Budget(ledger)
            budget.reserve('older-interrupted-protocol', 'main', .02,
                           json.dumps({'cell': {'run_id': 'older-interrupted-protocol'}}))
            budget.db.close()
            with self.assertRaisesRegex(RuntimeError, 'Unresolved'):
                study.execute(config, 1, 'calibration')
            factory.assert_not_called()
            report = study.calibration_report(config)
            self.assertEqual(report['unresolved_attempts'], 0)
            self.assertEqual(report['shared_ledger_unresolved_attempts'], 1)

    @contextlib.contextmanager
    def workspace(self):
        config = study.load_config()
        frozen = dict(source_sha256=study.source_hashes(),
                      roster_sha256=study.digest(study.prompts.adult_roster()),
                      runtime_versions=study.runtime_versions(), spend_ceiling_usd=5)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = root / 'outputs/revision_budget_v1/budget.sqlite'
            budget = Budget(ledger)
            budget.db.close()
            study.write_json(ledger.parent / 'pilot_summary.json',
                             {'conservative_charge_or_reservation_usd': 0})
            study.write_json(root / config['persona_file'], study.prompts.adult_roster())
            client = Mock()
            client.chat.completions.with_raw_response.create.side_effect = lambda **kwargs: SimpleNamespace(
                headers={}, status_code=200,
                parse=Mock(return_value=client.chat.completions.create(**kwargs)))
            usage = Mock()
            usage.model_dump.return_value = {'prompt_tokens': 100, 'completion_tokens': 4}
            client.chat.completions.create.return_value = SimpleNamespace(
                usage=usage, choices=[SimpleNamespace(message=SimpleNamespace(content='0, 1\n1, 2'),
                                                     finish_reason='stop')], id='offline-fixture', model='gpt-4.1')
            with patch.object(study, 'ROOT', root), patch.object(study, 'DESTINATION', root / 'preflight'), \
                    patch.object(study, 'source_hashes', return_value=frozen['source_sha256']), \
                    patch.object(study, 'execution_review', return_value=frozen), \
                    patch.object(study, 'credential', return_value='offline-test-not-a-key'), \
                    patch.object(study, 'OpenAI', return_value=client) as factory, \
                    contextlib.redirect_stdout(io.StringIO()):
                yield config, root, ledger, client, factory

    def test_execute_resume_and_lost_ledger_stop_before_client(self):
        with self.workspace() as (config, root, ledger, client, factory):
            study.execute(config, 1, 'calibration')
            self.assertEqual(client.chat.completions.create.call_count, 1)
            path = next((root / study.RESULTS).glob('*.json'))
            record = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(record['metrics']['density'], 2 / 1225)
            factory.reset_mock()
            study.execute(config, 1, 'calibration')
            factory.assert_not_called()
            self.assertEqual(client.chat.completions.create.call_count, 0)
            budget = Budget(ledger, require_existing=True)
            with budget.db:
                budget.db.execute('DELETE FROM attempts')
            budget.db.close()
            with self.assertRaisesRegex(ValueError, 'ledger|Ledger|spending'):
                study.execute(config, 2, 'calibration')
            factory.assert_not_called()

    def test_receipt_usage_mismatch_stops_before_client(self):
        with self.workspace() as (config, root, ledger, client, factory):
            study.execute(config, 1, 'calibration')
            path = next((root / study.RESULTS).glob('*.json'))
            record = json.loads(path.read_text(encoding='utf-8'))
            record['requests'][0]['usage']['prompt_tokens'] += 1
            study.write_json(path, record)
            factory.reset_mock()
            with self.assertRaisesRegex(ValueError, 'ledger|Ledger|receipt'):
                study.execute(config, 2, 'calibration')
            factory.assert_not_called()

    def test_interrupted_artifact_write_reuses_paid_response(self):
        with self.workspace() as (config, root, ledger, client, factory):
            with patch.object(study.nx, 'write_adjlist', side_effect=OSError('fixture disk failure')):
                with self.assertRaises(OSError):
                    study.execute(config, 1, 'calibration')
            self.assertEqual(client.chat.completions.create.call_count, 1)
            self.assertEqual(list((root / study.RESULTS).glob('*.json')), [])
            study.execute(config, 1, 'calibration')
            self.assertEqual(client.chat.completions.create.call_count, 1)
            self.assertEqual(len(list((root / study.RESULTS).glob('*.json'))), 1)

    def test_omitted_retry_receipt_is_rejected(self):
        with self.workspace() as (config, root, ledger, client, factory):
            valid = client.chat.completions.create.return_value
            invalid = copy.deepcopy(valid)
            # A recoverable pair list exercises a real correction. An answer
            # containing no valid pairs now stops instead of resampling ties.
            invalid.choices[0].message.content = '0, 1\n1, 2\n2, 1'
            client.chat.completions.create.side_effect = [invalid, valid]
            with patch.object(study.shared.time, 'sleep'):
                study.execute(config, 1, 'calibration')
            path = next((root / study.RESULTS).glob('*.json'))
            record = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(record['events'][0]['attempts'], 2)
            self.assertEqual(len(record['requests']), 2)
            record['requests'].pop()
            study.write_json(path, record)
            factory.reset_mock()
            with self.assertRaisesRegex(ValueError, 'incomplete or inconsistent'):
                study.execute(config, 2, 'calibration')
            factory.assert_not_called()

    def test_workflow_lock_prevents_concurrent_publication_and_releases(self):
        with self.workspace() as (config, root, ledger, client, factory):
            with study.workflow_lock():
                with self.assertRaisesRegex(RuntimeError, 'already running'):
                    study.execute(config, 1, 'calibration')
            factory.assert_not_called()
            study.execute(config, 1, 'calibration')
            self.assertEqual(client.chat.completions.create.call_count, 1)

    def test_preflight_rejects_source_changed_during_check(self):
        with tempfile.TemporaryDirectory() as directory:
            before = study.source_hashes()
            after = {**before, 'revision224_prompts.py': 'changed'}
            with patch.object(study, 'DESTINATION', Path(directory)), \
                    patch.object(study, 'source_hashes', side_effect=[before, after]), \
                    patch.object(study, 'prepare', return_value=(study.prompts.adult_roster(), [])):
                with self.assertRaisesRegex(ValueError, 'changed'):
                    study.preflight(study.load_config())
            report = json.loads((Path(directory) / 'report.json').read_text())
            self.assertEqual(report['status'], 'FAILED')

    def test_sequential_events_record_actual_prompt_method(self):
        roster = study.prompts.adult_roster()
        captured, events = [], []
        def fake_request(model, system, user, parse, args, **kwargs):
            payload = json.loads(user)
            captured.append('sequential' if 'degree' in payload['fields'] else 'local')
            self.assertEqual(len(payload['actor']), len(payload['actor_fields']))
            reply = ', '.join(row[0] for row in payload['candidates'][:args['num_choices']])
            return parse(response=reply, **args), reply, 1
        with patch.object(study.generation, 'get_system_prompt', study.prompts.system_prompt), \
                patch.object(study.generation, 'get_user_prompt', study.prompts.user_prompt), \
                patch.object(study.generation, 'repeat_prompt_until_parsed', fake_request), \
                contextlib.redirect_stdout(io.StringIO()):
            study.generation.generate_network('sequential', study.prompts.DEMOS, roster,
                list(roster), 'gpt-4.1', mean_choices=5, culture_context='us', events=events)
        self.assertEqual(captured, ['local'] * 3 + ['sequential'] * 47)
        self.assertEqual([event['prompt_method'] for event in events], captured)
        self.assertEqual({event['run_method'] for event in events}, {'sequential'})

    def test_fresh_analysis_keeps_matched_roster_and_explicit_paired_contrasts(self):
        with self.workspace() as (config, root, ledger, client, factory):
            selected = [cell for cell in study.cells(config) if cell['model'] == 'gpt-4.1'
                        and cell['method'] == 'global' and cell['repetition'] == 0
                        and (cell['culture'], cell['language']) in
                        {('us', 'english'), ('india', 'english'), ('us', 'portuguese')}]
            with patch.object(study, 'cells', return_value=selected), patch.object(study, 'calibration_cells', return_value=selected):
                study.execute(config, 3, 'calibration')
            factory.reset_mock()
            result = study.analyze(config)
            factory.assert_not_called()
            self.assertEqual(result['status'], 'PARTIAL')
            self.assertEqual(result['analyzed_networks'], 3)
            self.assertEqual(result['control_graphs'], 9)
            folder = root / study.STATS
            controls = pd.read_csv(folder / 'matched_controls.csv')
            self.assertEqual(set(controls['run_id']), {c['run_id'] for c in selected})
            self.assertTrue((controls['nodes'] == 50).all())
            self.assertTrue((controls['edges'] == 2).all())
            self.assertTrue(controls['roster_sha256'].eq(study.digest(study.prompts.adult_roster())).all())
            for path in (folder / 'baselines').glob('*.adj'):
                graph = nx.read_adjlist(path)
                self.assertEqual(set(graph), set(study.prompts.adult_roster()))
            contrasts = pd.read_csv(folder / 'paired_contrasts.csv')
            summary = pd.read_csv(folder / 'contrast_summary.csv')
            self.assertTrue(summary['n_pairs_present'].isin([0, 1]).all())
            self.assertTrue(summary['n_pairs_present'].eq(0).any())
            self.assertTrue(summary.loc[summary['n_pairs_present'].eq(0), 'mean'].isna().all())
            self.assertTrue(summary['planned_repetitions'].eq(8).all())
            self.assertTrue(summary['sample_sd'].isna().all())
            self.assertEqual(set(contrasts['dimension']), {'culture', 'language'})
            self.assertTrue(contrasts['comparison_minus_reference'].dropna().eq(0).all())
            self.assertTrue(contrasts.loc[contrasts['dimension'] == 'culture', 'reference_level'].eq('us').all())
            self.assertTrue(contrasts.loc[contrasts['dimension'] == 'language', 'reference_level'].eq('english').all())

    def test_fresh_analysis_never_substitutes_the_old_archive(self):
        with self.workspace() as (config, root, ledger, client, factory):
            result = study.analyze(config)
            self.assertEqual(result['status'], 'NOT_COLLECTED')
            self.assertEqual(result['analyzed_networks'], 0)
            factory.assert_not_called()

    def test_calibration_forecast_does_not_rebill_completed_main_cells(self):
        with self.workspace() as (config, root, ledger, client, factory):
            selected = study.calibration_cells(config)
            selected_ids = {cell['run_id'] for cell in selected}
            extra = next(cell for cell in study.cells(config) if cell['run_id'] not in selected_ids)
            budget = Budget(ledger, require_existing=True)
            for cell in [*selected, extra]:
                study.write_json(root / study.RESULTS / (cell['run_id'] + '.json'), {})
                budget.reserve(cell['run_id'], 'main', .01, json.dumps({'cell': cell}))
                budget.settle(cell['run_id'], .01, {})
            budget.db.close()
            record = {'requests': [{'usage': {'prompt_tokens': 100, 'completion_tokens': 2},
                'parse': {'valid': True}, 'finish_reason': 'stop', 'conservative_charge_usd': .01}]}
            with patch.object(study, 'verify_receipt', return_value=(record, None)):
                report = study.calibration_report(config)
            factory.assert_not_called()
            self.assertEqual(report['status'], 'COMPLETE')
            self.assertEqual(report['verified_completed_study_networks'], 69)
            self.assertAlmostEqual(report['full_study_forecast_usd'], 8.96)
            self.assertAlmostEqual(report['forecast_remaining_usd'], 8.27)
            self.assertAlmostEqual(report['forecast_cumulative_with_20_percent_remaining_contingency_usd'], .69 + 8.27 * 1.2)

    def test_main_refuses_missing_calibration_receipt_before_client(self):
        with self.workspace() as (config, root, ledger, client, factory):
            selected = study.calibration_cells(config)
            for cell in selected:
                study.write_json(root / study.RESULTS / (cell['run_id'] + '.json'), {})
            (root / study.RESULTS / (selected[-1]['run_id'] + '.json')).unlink()
            with patch.object(study, 'verify_receipt', return_value=None), self.assertRaisesRegex(ValueError, 'every calibration receipt'):
                study.execute(config, 896, 'main')
            factory.assert_not_called()

    def test_planned_contrast_orientation_is_independent_of_row_order(self):
        cells = [c for c in study.cells(study.load_config()) if c['repetition'] == 0]
        signature = lambda rows: {(a['run_id'], b['run_id'], dimension)
                                  for a, b, dimension in study.contrast_pairs(rows)}
        self.assertEqual(signature(cells), signature(list(reversed(cells))))

    def test_calibration_report_does_not_forecast_from_incomplete_coverage(self):
        with self.workspace() as (config, root, ledger, client, factory):
            report = study.calibration_report(config)
            self.assertEqual(report['status'], 'NOT_COLLECTED')
            self.assertEqual(report['planned_calibration_networks'], 68)
            self.assertIsNone(report['full_study_forecast_usd'])
            factory.assert_not_called()
            study.execute(config, 1, 'calibration')
            factory.reset_mock()
            report = study.calibration_report(config)
            self.assertEqual(report['status'], 'PARTIAL')
            self.assertEqual(report['completed_calibration_networks'], 1)
            self.assertIsNone(report['full_study_forecast_usd'])
            self.assertFalse(report['generation_authorized'])
            factory.assert_not_called()


if __name__ == '__main__':
    unittest.main()
