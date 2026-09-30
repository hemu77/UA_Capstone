"""Treatment separation and offline preparation checks; never paid API calls."""
import copy
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import networkx as nx

import revision224 as study
import revision224_prompts as prompts
from make_matched_baselines import matched_baselines


class FreshStudyTests(unittest.TestCase):
    def test_adult_roster_has_declared_marginals_and_no_party_or_race_labels(self):
        roster = prompts.adult_roster()
        self.assertEqual(roster, prompts.adult_roster())
        self.assertEqual(set(roster), {str(i) for i in range(50)})
        self.assertEqual(sorted(p['age'] for p in roster.values()), list(range(18, 68)))
        self.assertTrue(all(set(p) == set(prompts.DEMOS) for p in roster.values()))
        self.assertNotIn('Democrat', json.dumps(roster))
        self.assertNotIn('Republican', json.dumps(roster))

    def test_candidate_bytes_do_not_change_when_instruction_language_changes(self):
        roster = prompts.adult_roster()
        before = copy.deepcopy(roster)
        graph = nx.path_graph(list(roster))
        for method in ['global', 'local', 'sequential', 'iterative-add', 'iterative-drop']:
            payloads = []
            for language in prompts.TEXT:
                random.seed(224)
                payload = prompts.user_prompt(method, roster, list(roster), prompts.DEMOS,
                    curr_pid=None if method == 'global' else '0', G=graph, prompt_language=language)
                payloads.append(payload)
                ids = {p[0] for p in json.loads(payload)['candidates']}
                if method == 'global':
                    self.assertEqual(ids, set(roster))
                elif method == 'iterative-add':
                    self.assertEqual(ids, set(roster) - set(graph['0']) - {'0'})
                elif method == 'iterative-drop':
                    self.assertEqual(ids, set(graph['0']))
                else:
                    self.assertEqual(ids, set(roster) - {'0'})
            self.assertEqual(len(set(payloads)), 1)
        self.assertEqual(roster, before)

    def test_catalog_covers_all_languages_countries_and_prompt_methods(self):
        roster = prompts.adult_roster()
        for language, countries in prompts.COUNTRIES.items():
            for country in countries:
                for method in ['global', 'local', 'sequential', 'iterative-add', 'iterative-drop']:
                    system = prompts.system_prompt(method, roster, prompts.DEMOS,
                        curr_pid=None if method == 'global' else '0', culture_context=country,
                        prompt_language=language, num_choices=5 if method in {'local', 'sequential'} else None)
                    self.assertIn(countries[country], system)
                    self.assertNotIn('{', system)
        for country, language in [('portuguese', 'english'), ('brazli', 'english'), ('us', 'spanish')]:
            with self.assertRaises(ValueError):
                prompts.system_prompt('global', roster, prompts.DEMOS, culture_context=country, prompt_language=language)

    def test_iterative_instructions_define_mutual_relative_to_actor(self):
        # These phrases freeze reviewed wording, not proof of linguistic quality.
        meanings = {'english': 'friends shared with you', 'hindi': 'आपके और उम्मीदवार के साझा मित्रों',
                    'japanese': 'あなたと候補者に共通する友人', 'portuguese': 'amigos em comum com você'}
        for language, meaning in meanings.items():
            for method in ['iterative-add', 'iterative-drop']:
                self.assertIn(meaning, prompts.TEXT[language][method])
        roster = prompts.adult_roster()
        graph = nx.empty_graph(roster)
        graph.add_edges_from([('0', '1'), ('0', '2'), ('1', '2'), ('1', '3')])
        for method in ['iterative-add', 'iterative-drop']:
            payload = json.loads(prompts.user_prompt(method, roster, list(roster), prompts.DEMOS, curr_pid='0', G=graph))
            for values in payload['candidates']:
                row = dict(zip(payload['fields'], values))
                self.assertEqual(row['degree'], graph.degree(row['id']))
                self.assertEqual(row['mutual'], len(set(graph[row['id']]) & set(graph['0'])))

    def test_preparation_exports_actual_localized_retry_messages_without_api(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            with patch.object(study, 'DESTINATION', destination), patch.object(study.shared, 'OpenAI') as client:
                study.prepare(study.load_config())
            client.assert_not_called()
            catalog = json.loads((destination / 'retry_catalog.json').read_text(encoding='utf-8'))
        self.assertEqual(len(catalog), 20)
        for language, prefix in {'english': 'Invalid response.', 'hindi': 'अमान्य उत्तर।',
                                 'japanese': '無効な回答です。', 'portuguese': 'Resposta inválida.'}.items():
            rows = [r for r in catalog if r['language'] == language]
            self.assertEqual({r['method'] for r in rows}, {'global', 'local', 'sequential', 'iterative-add', 'iterative-drop'})
            for row in rows:
                self.assertTrue(row['correction'].startswith(prefix))
                self.assertEqual(row['attempts'], 2)
                self.assertEqual(row['evidence_type'], 'translation_review_fixture')
                if row['method'] in {'local', 'sequential'}:
                    self.assertIn('5', row['correction'])
                if row['method'] == 'global':
                    self.assertNotIn('2, 7', row['correction'])
                    self.assertNotIn('7, 2', row['correction'])
                    self.assertIn('1, 0', row['correction'])
                    self.assertTrue(row['nonresponse_correction'])
                    self.assertNotIn('2, 7', row['nonresponse_correction'])
                if language != 'english':
                    self.assertNotIn('Invalid response', row['correction'])
                    self.assertNotIn('Duplicate friendship', row['correction'])

    def test_protocol_cells_are_balanced_and_config_changes_are_rejected(self):
        config = study.load_config()
        cells = study.cells(config)
        self.assertEqual(len(cells), 896)
        for model in config['models']:
            self.assertEqual(sum(c['model'] == model for c in cells), 224)
        self.assertEqual({c['seed'] for c in cells}, set(range(11000, 11008)))
        calibration = study.calibration_cells(config)
        self.assertEqual(len(calibration), 68)
        self.assertEqual(sum(c['model'] == 'gpt-6-luna' for c in calibration), 56)
        self.assertTrue({c['run_id'] for c in calibration} <= {c['run_id'] for c in cells})
        changed = {**config, 'languages': ['english', 'spanish']}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            path.write_text(json.dumps(changed))
            with patch.object(study, 'CONFIG', path), self.assertRaises(ValueError):
                study.load_config()

    def test_prepare_refuses_to_replace_an_edited_roster(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = study.load_config()
            path = root / config['persona_file']
            path.parent.mkdir(parents=True)
            path.write_text('{}')
            with patch.object(study, 'ROOT', root), self.assertRaisesRegex(ValueError, 'refusing to overwrite'):
                study.prepare(config)
            self.assertEqual(path.read_text(), '{}')

    def test_bad_event_trace_is_rejected_and_optional_metrics_remain_explicit(self):
        roster = prompts.adult_roster()
        graph = nx.empty_graph(roster)
        graph.add_edge('0', '1')
        with self.assertRaises(ValueError):
            study.verify_graph(graph, [], roster)
        metrics, homophily = study.verify_graph(graph, [{'persona': '0', 'added': [['0', '1']], 'removed': []}], roster)
        self.assertAlmostEqual(metrics['density'], 1 / 1225)
        self.assertAlmostEqual(metrics['prop_nodes_lcc'], 2 / 50)
        self.assertIsNone(homophily['religion'][0])

    def test_fresh_baselines_match_counts_with_the_actual_visible_attributes(self):
        roster = prompts.adult_roster()
        graph = nx.path_graph(list(roster))
        controls = matched_baselines(graph, roster, seed=224, categorical=['gender', 'religion', 'political orientation'])
        self.assertEqual(len(controls), 3)
        for _, control, _ in controls:
            self.assertEqual(set(control), set(roster))
            self.assertEqual(control.number_of_edges(), graph.number_of_edges())
            self.assertEqual(nx.number_of_selfloops(control), 0)
        self.assertEqual(dict(controls[2][1].degree()), dict(graph.degree()))


    def test_failed_preflight_does_not_leave_a_success_report(self):
        config = study.load_config()
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            (destination / 'report.json').write_text('{"status":"READY_FOR_ASTRA_CODE_REVIEW"}')
            with patch.object(study, 'DESTINATION', destination), patch.object(study, 'prepare', side_effect=ValueError('injected failure')):
                with self.assertRaises(ValueError):
                    study.preflight(config)
            report = json.loads((destination / 'report.json').read_text())
            self.assertEqual(report['status'], 'FAILED')
            self.assertEqual(report['protocol_sha256'], study.digest(config))

    def test_execution_cannot_create_client_before_review(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(study, 'DESTINATION', Path(directory)), patch.object(study, 'OpenAI') as client:
                with self.assertRaises(FileNotFoundError):
                    study.execute(study.load_config(), 224)
                client.assert_not_called()

    def test_calibration_cannot_authorize_main_and_main_requires_finite_ceiling(self):
        config = study.load_config()
        fingerprint = dict(protocol_sha256=study.digest(config), source_sha256=study.source_hashes(),
                           roster_sha256=study.digest(prompts.adult_roster()), runtime_versions=study.runtime_versions())
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            study.write_json(destination / 'report.json', {**fingerprint, 'status': 'READY_FOR_ASTRA_CODE_REVIEW', 'offline_fixture_cells_checked': 896})
            review = {**fingerprint, 'astra_review': True, 'translation_review': True,
                      'code_review': True, 'price_review': True, 'scope': 'calibration',
                      'cost_review': False, 'generation_authorized': True, 'approved_limit': 68, 'spend_ceiling_usd': 12,
                      'starting_ledger_usd': 7, 'additional_allowance_usd': 5}
            study.write_json(destination / 'calibration_review.json', review)
            study.write_json(destination / 'review.json', review)
            with patch.object(study, 'DESTINATION', destination):
                self.assertEqual(study.execution_review(config, 68, 'calibration')['spend_ceiling_usd'], 12)
                with self.assertRaisesRegex(ValueError, 'scope'):
                    study.execution_review(config, 69, 'calibration')
                with self.assertRaisesRegex(ValueError, 'scope'):
                    study.execution_review(config, 896)
                main = {**review, 'scope': 'main', 'cost_review': True, 'study_design_review': True,
                        'approved_limit': 896, 'spend_ceiling_usd': 160}
                study.write_json(destination / 'review.json', main)
                calibration_report = {**fingerprint, 'status': 'COMPLETE', 'completed_calibration_networks': 68, 'unresolved_attempts': 0}
                study.write_json(destination / 'calibration_report.json', calibration_report)
                self.assertEqual(study.execution_review(config, 896)['spend_ceiling_usd'], 160)
                for bad in [None, True, -1, float('inf')]:
                    # JSON disallows infinity: use a decoded review fixture for that case.
                    with patch.object(study.json, 'loads', side_effect=[{**fingerprint, 'status': 'READY_FOR_ASTRA_CODE_REVIEW', 'offline_fixture_cells_checked': 896}, {**main, 'spend_ceiling_usd': bad}, calibration_report]):
                        with self.assertRaisesRegex(ValueError, 'ceiling'):
                            study.execution_review(config, 896)
                study.write_json(destination / 'calibration_report.json', {**calibration_report, 'status': 'PARTIAL'})
                with self.assertRaisesRegex(ValueError, 'complete calibration'):
                    study.execution_review(config, 896)

    def test_paid_cache_identity_is_bound_to_frozen_source_hashes(self):
        from paid_study import PaidCaller, Budget
        from unittest.mock import Mock
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as directory:
            budget = Budget(Path(directory) / 'ledger.sqlite')
            try:
                cell = {**study.cells(study.load_config())[0], 'phase': 'main', 'frozen_source_sha256': {'prompt.py': 'a'}}
                usage = Mock()
                usage.model_dump.return_value = {'prompt_tokens': 10, 'completion_tokens': 2}
                client = Mock()
                client.chat.completions.with_raw_response.create.side_effect = lambda **kwargs: SimpleNamespace(
                    headers={}, status_code=200,
                    parse=Mock(return_value=client.chat.completions.create(**kwargs)))
                client.chat.completions.create.return_value = SimpleNamespace(usage=usage, choices=[SimpleNamespace(
                    message=SimpleNamespace(content='0, 1'), finish_reason='stop')], id='fixture', model='gpt-4.1')
                message = [{'role': 'user', 'content': 'fixture'}]
                PaidCaller(client, budget, cell)(cell['model'], message)
                PaidCaller(client, budget, cell)(cell['model'], message)
                self.assertEqual(client.chat.completions.create.call_count, 1)
                changed = {**cell, 'frozen_source_sha256': {'prompt.py': 'b'}}
                PaidCaller(client, budget, changed)(cell['model'], message)
                self.assertEqual(client.chat.completions.create.call_count, 2)
            finally:
                budget.db.close()


if __name__ == '__main__':
    unittest.main()
