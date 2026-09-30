"""Small offline examples make research-affecting bugs visible without API spend."""
import unittest
import contextlib
import io
import json
import hashlib
import tempfile
from pathlib import Path
from unittest.mock import patch

import networkx as nx

import analyze_networks as analysis
import constants_and_utils as api
from generate_networks import get_culture_statement, update_graph_from_response
import generate_networks as generation
from plan_revision_study import build_plan
from study_runner_utils import build_pairwise_graph_divergence


class ResearchRegressionTests(unittest.TestCase):
    def test_224_plan_is_balanced_and_not_authorized(self):
        config = json.loads(Path('study_protocol_224.json').read_text())
        plan = build_plan(config)
        self.assertEqual(plan['confirmatory_networks_draft'], 224)
        self.assertEqual(len(config['main_conditions']), 7)
        self.assertEqual(config['main_conditions'].count(['us', 'english']), 1)
        self.assertFalse(config['generation_authorized'])
        self.assertEqual(plan['confirmatory_requests_before_retries_upper_bound'], 25256)

    def test_country_and_language_are_validated_independently(self):
        for country in ['us', 'india', 'japan', 'brazil']:
            for language in ['english', 'hindi', 'japanese', 'portuguese']:
                statement = get_culture_statement(country, language)
                self.assertIn(generation.COUNTRY_NAMES[language][country], statement)
                self.assertNotIn('{', statement)
        for country, language in [('portuguese', 'english'), ('brazli', 'english'),
                                  ('neutral', 'invalid'), (None, 'invalid')]:
            with self.subTest(country=country, language=language), self.assertRaises(ValueError):
                get_culture_statement(country, language)

    def test_decimal_is_not_silently_parsed_as_two_personas(self):
        for method in ['local', 'sequential']:
            graph = nx.empty_graph(['0', '1', '2'])
            with self.subTest(method=method), self.assertRaises(ValueError):
                update_graph_from_response(method, '1.2', graph, curr_pid='0')
            self.assertEqual(graph.number_of_edges(), 0)

    def test_iterative_punctuation_and_blank_global_are_not_repaired(self):
        for method in ['iterative-add', 'iterative-drop']:
            for response in ['.1', '1.', '1.2']:
                graph = nx.empty_graph(['0', '1', '2'])
                if method == 'iterative-drop':
                    graph.add_edge('0', '1')
                before = set(graph.edges())
                with self.subTest(method=method, response=response), self.assertRaises(ValueError):
                    update_graph_from_response(method, response, graph, curr_pid='0')
                self.assertEqual(set(graph.edges()), before)
        for response in ['', ' \n\t ']:
            with self.assertRaises(ValueError):
                update_graph_from_response('global', response, nx.empty_graph(['0', '1']))

    def test_completely_different_undirected_graphs_have_distance_one(self):
        self.assertEqual(analysis.compute_edge_distance(nx.empty_graph(3), nx.complete_graph(3)), 1)

    def test_reversed_edge_is_the_same_undirected_edge(self):
        left, right = nx.Graph(), nx.Graph()
        left.add_nodes_from(['0', '1', '2'])
        right.add_nodes_from(['2', '1', '0'])
        left.add_edge('0', '1')
        right.add_edge('1', '0')
        self.assertEqual(analysis.compute_edge_distance(left, right), 0)

    def test_neutral_country_label_omits_country_instruction(self):
        for language in generation.SUPPORTED_PROMPT_LANGUAGES:
            with self.subTest(language=language):
                self.assertEqual(get_culture_statement('neutral', language), '')

    def test_global_contract_and_retry_identify_reversed_duplicates(self):
        personas = json.loads(Path('text-files/us_50_gpt4o_w_interests.json').read_text())
        prompt = generation.get_system_prompt('global', personas, ['age'])
        self.assertIn('ID1 < ID2', prompt)
        graph = nx.empty_graph(['2', '7', '9'])
        with self.assertRaises(ValueError) as failure:
            update_graph_from_response('global', '2, 7\n7, 2\n2, 9\n9, 2', graph)
        self.assertEqual(failure.exception.duplicate_edges, [('7', '2'), ('9', '2')])
        self.assertEqual(graph.number_of_edges(), 0)

    def test_edge_frequency_has_no_self_edges_or_orientation_duplicates(self):
        edges, props = analysis.get_edge_proportions([nx.complete_graph(3)])
        self.assertEqual(len(edges), 3)
        self.assertEqual(props, [1, 1, 1])

    def test_global_retry_cannot_replace_network_with_only_corrected_pairs(self):
        for language in ['english', 'hindi', 'japanese', 'portuguese', 'spanish']:
            graph = nx.empty_graph(['0', '1', '2'])
            replies = ['0, 1\n0, 2\n2, 0', '0, 2', '0, 1\n0, 2']
            with self.subTest(language=language), patch.object(api, 'get_llm_response', side_effect=replies), patch.object(api.time, 'sleep'):
                _, _, attempts = api.repeat_prompt_until_parsed('test', 's', 'u', update_graph_from_response,
                    {'method': 'global', 'G': graph}, prompt_language=language)
            self.assertEqual(attempts, 3)
            self.assertEqual(set(graph.edges()), {('0', '1'), ('0', '2')})

    def test_global_retry_rejects_changed_ties_without_mutating_graph(self):
        for changed in ['0, 2', '0, 2\n1, 2', '0, 1\n0, 2\n1, 2']:
            graph = nx.empty_graph(['0', '1', '2'])
            with patch.object(api, 'get_llm_response', side_effect=['0, 1\n0, 2\n2, 0', changed]), patch.object(api.time, 'sleep'):
                with self.assertRaises(RuntimeError):
                    api.repeat_prompt_until_parsed('test', 's', 'u', update_graph_from_response,
                        {'method': 'global', 'G': graph}, max_tries=2)
            self.assertEqual(graph.number_of_edges(), 0)

    def test_global_correction_target_survives_second_invalid_reply(self):
        graph = nx.empty_graph(['0', '1', '2'])
        with patch.object(api, 'get_llm_response', side_effect=['0, 1\n0, 2\n2, 0', '0, 2\n2, 0', '0, 2']), patch.object(api.time, 'sleep'):
            with self.assertRaises(RuntimeError):
                api.repeat_prompt_until_parsed('test', 's', 'u', update_graph_from_response, {'method': 'global', 'G': graph})
        self.assertEqual(graph.number_of_edges(), 0)

    def test_global_repair_keeps_only_explicit_valid_pairs_and_stops_when_none(self):
        graph = nx.empty_graph(['0', '1', '2'])
        with patch.object(api, 'get_llm_response', side_effect=['0, 1\n1, 0\n0, 0\n0, 99\nMaybe 1 and 2', '0, 1']), patch.object(api.time, 'sleep'):
            _, _, attempts = api.repeat_prompt_until_parsed('test', 's', 'u', update_graph_from_response, {'method': 'global', 'G': graph})
        self.assertEqual(attempts, 2)
        self.assertEqual(set(graph.edges()), {('0', '1')})
        graph = nx.empty_graph(['0', '1', '2'])
        with patch.object(api, 'get_llm_response', return_value='Maybe 1 and 2') as call:
            with self.assertRaisesRegex(RuntimeError, 'no recoverable'):
                api.repeat_prompt_until_parsed('test', 's', 'u', update_graph_from_response, {'method': 'global', 'G': graph})
        self.assertEqual(call.call_count, 1)
        self.assertEqual(graph.number_of_edges(), 0)

    def test_empty_global_answer_is_bounded_nonresponse_retry_not_empty_graph(self):
        for language in ['english', 'hindi', 'japanese', 'portuguese']:
            graph = nx.empty_graph(['0', '1'])
            with patch.object(api, 'get_llm_response', side_effect=['', '0, 1']) as call, patch.object(api.time, 'sleep'):
                _, _, attempts = api.repeat_prompt_until_parsed('test', 's', 'u', update_graph_from_response,
                    {'method': 'global', 'G': graph}, prompt_language=language)
            self.assertEqual(attempts, 2)
            self.assertEqual(graph.number_of_edges(), 1)
            self.assertNotIn('original network response', call.call_args.args[1][-1]['content'])
        graph = nx.empty_graph(['0', '1'])
        with patch.object(api, 'get_llm_response', return_value='') as call, patch.object(api.time, 'sleep'):
            with self.assertRaisesRegex(RuntimeError, 'Exhausted 3'):
                api.repeat_prompt_until_parsed('test', 's', 'u', update_graph_from_response, {'method': 'global', 'G': graph})
        self.assertEqual(call.call_count, 3)
        self.assertEqual(graph.number_of_edges(), 0)

    def test_invalid_responses_do_not_mutate_graph(self):
        for response in ['0', '1, 1', '1, 99']:
            with self.subTest(response=response):
                graph = nx.empty_graph(['0', '1', '2'])
                with self.assertRaises(ValueError):
                    update_graph_from_response('local', response, graph, curr_pid='0')
                self.assertEqual(graph.number_of_edges(), 0)

    def test_iterative_cannot_add_existing_friend(self):
        graph = nx.Graph([('0', '1')])
        graph.add_node('2')
        with self.assertRaises(ValueError):
            update_graph_from_response('iterative-add', '1', graph, curr_pid='0')

    def test_iterative_reason_is_recorded(self):
        graph = nx.empty_graph(['0', '1'])
        graph, reasons = update_graph_from_response(
            'iterative-add', '{"new friend": "1", "reason": "shared activity"}',
            graph, curr_pid='0', include_reason=True)
        self.assertEqual(reasons[('1', 'add')], 'shared activity')

    def test_english_context_does_not_assign_participant_language(self):
        self.assertNotIn('communicates in', get_culture_statement('us', 'english'))

    def test_configuration_failure_is_not_retried(self):
        with patch.object(api, 'get_llm_response', side_effect=ValueError('missing key')) as call:
            with patch.object(api.time, 'sleep'):
                with self.assertRaises(ValueError):
                    api.repeat_prompt_until_parsed('gpt-test', 'system', 'user', lambda response: response, {})
        self.assertEqual(call.call_count, 1)

    def test_live_dispatch_is_locked_before_client_creation(self):
        with patch.object(api, 'OpenAI') as client:
            with self.assertRaisesRegex(RuntimeError, 'Paid generation is locked'):
                api.get_llm_response('gpt-4.1-mini', [{'role': 'user', 'content': 'test'}])
        client.assert_not_called()

    def test_missing_plot_does_not_invalidate_graph_or_enter_quarantine(self):
        import export_research_viewer as exporter
        import tempfile
        import shutil
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'text-files').mkdir()
            shutil.copy('text-files/us_50_gpt4o_w_interests.json', root / 'text-files')
            name = 'sequential_gpt-4.1-mini_n5_culture_us_0'
            shutil.copy(f'text-files/{name}.adj', root / 'text-files')
            with patch.object(exporter, 'expected_runs', return_value={name}), patch.object(exporter, 'render_figures'):
                manifest = exporter.export(root)
            self.assertEqual(manifest['quarantined'], 0)
            self.assertEqual(manifest['exported_runs'], 1)
            self.assertEqual(manifest['png_missing_or_corrupt_among_valid_graphs'], 1)

    def test_directed_distance_and_small_graphs(self):
        self.assertEqual(analysis.compute_edge_distance(nx.DiGraph([(0, 1)]), nx.DiGraph([(1, 0)])), 1)
        self.assertEqual(analysis.compute_edge_distance(nx.empty_graph(1), nx.empty_graph(1)), 0)
        with self.assertRaises(ValueError):
            analysis.compute_edge_distance(nx.Graph([(0, 1)]), nx.DiGraph([(0, 1)]))
        with self.assertRaises(ValueError):
            analysis.compute_edge_distance(nx.Graph([(0, 0)]), nx.Graph([(0, 0)]))

    def test_coleman_group_weights_and_numeric_age(self):
        personas = {str(i): {'group': 'A' if i < 2 else 'B', 'age': 20 if i < 2 else 60} for i in range(4)}
        graph = nx.Graph([('0', '1'), ('2', '3')])
        score, rows = analysis.compute_coleman_homophily(graph, personas, 'group')
        self.assertEqual(score, 1)
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(analysis.compute_age_assortativity(graph, personas), 1)
        cross = nx.Graph([('0', '2'), ('1', '3')])
        self.assertEqual(analysis.compute_coleman_homophily(cross, personas, 'group')[0], -1)

    def test_disconnected_and_empty_scalars_are_explicit(self):
        graph = nx.Graph([(0, 1)])
        graph.add_node(2)
        values = analysis.compute_network_metrics(graph)
        self.assertAlmostEqual(values['prop_nodes_lcc'], 2 / 3)
        self.assertEqual(values['avg_shortest_path_lcc'], 1)
        self.assertTrue(analysis.np.isnan(analysis.compute_network_metrics(nx.Graph())['modularity']))

    def test_metric_results_do_not_depend_on_edge_insertion_order(self):
        graph = nx.gnp_random_graph(20, .25, seed=1)
        reverse = nx.Graph(list(graph.edges())[::-1])
        self.assertEqual(set(graph), set(reverse))
        self.assertEqual(analysis.compute_network_metrics(graph), analysis.compute_network_metrics(reverse))

    def test_matched_divergence_retains_real_seeds(self):
        a, b = nx.path_graph(3), nx.path_graph(3)
        a.graph['seed'] = b.graph['seed'] = 42
        records = [{'save_prefix': 'a', 'model': 'a'}, {'save_prefix': 'b', 'model': 'b'}]
        table = build_pairwise_graph_divergence(records, {'a': [a], 'b': [b]}, [], 'model', 'pair')
        self.assertEqual(table.iloc[0]['seed'], 42)
        with self.assertRaises(ValueError):
            build_pairwise_graph_divergence(records, {'a': [a], 'b': []}, [], 'model', 'pair')

    def test_invalid_format_retries_are_bounded_and_localized(self):
        graph = nx.empty_graph(['0', '1'])
        with patch.object(api, 'get_llm_response', return_value='0') as call:
            with patch.object(api.time, 'sleep'):
                with self.assertRaises(RuntimeError):
                    api.repeat_prompt_until_parsed('gpt-test', 's', 'u', update_graph_from_response,
                        {'method': 'local', 'G': graph, 'curr_pid': '0'}, prompt_language='portuguese')
        self.assertEqual(call.call_count, 3)
        self.assertIn('Resposta', call.call_args.args[1][-1]['content'])
        self.assertEqual(graph.number_of_edges(), 0)

    def test_wrong_choice_count_retry_includes_exact_required_count(self):
        graph = nx.empty_graph(['0', '1', '2'])
        with patch.object(api, 'get_llm_response', side_effect=['1, 2', '1']) as call, patch.object(api.time, 'sleep'):
            _, _, attempts = api.repeat_prompt_until_parsed('test', 's', 'u', update_graph_from_response,
                {'method': 'sequential', 'G': graph, 'curr_pid': '0', 'num_choices': 1}, prompt_language='hindi')
        self.assertEqual(attempts, 2)
        self.assertIn('ठीक 1', call.call_args.args[1][-1]['content'])
        self.assertEqual(graph.number_of_edges(), 1)

    def test_full_fifty_person_roster_all_methods_and_languages_without_api(self):
        personas = json.loads(Path('text-files/us_50_gpt4o_w_interests.json').read_text())
        self.assertEqual(len(personas), 50)
        demos = ['gender', 'age', 'race/ethnicity', 'religion', 'political affiliation']
        def fake_request(model, system, user, parse, parse_args, **kwargs):
            self.assertEqual(kwargs['prompt_language'], language)
            self.assertTrue(system and user)
            graph, method = parse_args['G'], parse_args['method']
            node = parse_args.get('curr_pid')
            if method == 'global':
                ids = list(graph)
                response = '\n'.join(f'{a}, {b}' for a, b in zip(ids[:-1], ids[1:]))
            else:
                candidates = [n for n in graph if n != node]
                if method == 'iterative-add':
                    candidates = [n for n in candidates if not graph.has_edge(node, n)]
                elif method == 'iterative-drop':
                    candidates = list(graph[node])
                count = parse_args.get('num_choices') or 1
                response = ', '.join(candidates[:count])
            return parse(**parse_args, response=response), response, 1
        for language in sorted(generation.SUPPORTED_PROMPT_LANGUAGES):
            for method in ['global', 'local', 'sequential', 'iterative']:
                with self.subTest(language=language, method=method):
                    events = []
                    with patch.object(generation, 'repeat_prompt_until_parsed', side_effect=fake_request):
                        with contextlib.redirect_stdout(io.StringIO()):
                            graph, *_ = generation.generate_network(method, demos, personas, list(personas),
                                'gpt-test', mean_choices=5, culture_context='us', prompt_language=language, events=events)
                    replay = nx.empty_graph(list(personas))
                    for event in events:
                        replay.remove_edges_from(event['removed'])
                        replay.add_edges_from(event['added'])
                    self.assertEqual(analysis.compute_edge_distance(graph, replay), 0)
                    self.assertEqual(len(graph), 50)
                    self.assertGreater(graph.number_of_edges(), 0)
                    self.assertEqual(nx.number_of_selfloops(graph), 0)

    def test_pilot_plan_never_authorizes_generation(self):
        plan = build_plan(json.loads(Path('study_protocol.json').read_text()))
        self.assertEqual(plan['pilot_networks'], 20)
        self.assertEqual(plan['pilot_requests_before_retries_upper_bound'], 2255)
        self.assertEqual(plan['confirmatory_networks_draft'], 640)
        self.assertEqual(plan['status'], 'NOT_AUTHORIZED_NO_API_CALLS')

    def test_export_matches_original_graphs_and_keeps_quarantine_visible(self):
        import export_research_viewer as exporter
        data = json.loads(Path('viewer/public/data/networks.json').read_text(encoding='utf-8'))
        manifest = json.loads(Path('stats/revision_v1/manifest.json').read_text())
        historical = [run for run in data['runs'] if run['status'] == 'historical_reanalyzed']
        self.assertEqual({run['run_id'] for run in historical} | set(data['quarantined_runs']), exporter.expected_runs())
        self.assertEqual(len(data['quarantined_runs']), 16)
        self.assertEqual(manifest['self_links'], 23)
        for filename, expected_hash in manifest['source_sha256'].items():
            self.assertEqual(hashlib.sha256(Path(filename).read_bytes()).hexdigest(), expected_hash)
        personas = json.loads(Path('text-files/us_50_gpt4o_w_interests.json').read_text())
        for run in data['runs']:
            original = analysis.nx.read_adjlist(run['source'])
            exported = analysis.nx.empty_graph(list(personas))
            exported.add_edges_from(run['edges'])
            self.assertEqual(analysis.compute_edge_distance(original, exported), 0)
            for metric, value in analysis.compute_network_metrics(original).items():
                if analysis.np.isfinite(value):
                    self.assertAlmostEqual(value, run['metrics'][metric])
                else:
                    self.assertIsNone(run['metrics'][metric])
            for demo, stored in {**run['homophily'], 'age': run['age_assortativity']}.items():
                value = analysis.compute_age_assortativity(original, personas) if demo == 'age' else analysis.compute_coleman_homophily(original, personas, demo)[0]
                if analysis.np.isfinite(value):
                    self.assertAlmostEqual(value, stored)
                else:
                    self.assertIsNone(stored)
        for run_id in data['quarantined_runs']:
            self.assertGreater(analysis.nx.number_of_selfloops(analysis.nx.read_adjlist(f'text-files/{run_id}.adj')), 0)

    def test_invalid_plan_does_not_quote_negative_or_duplicate_work(self):
        config = json.loads(Path('study_protocol.json').read_text())
        for field, invalid in [('methods', []), ('models', []), ('languages', ['english', 'english']),
                               ('confirmatory_rosters', -1), ('iterative_rounds', 1.5),
                               ('main_conditions', [['unknown', 'english']]), ('main_conditions', ['us']),
                               ('main_conditions', [['us', 'unknown']])]:
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    build_plan(dict(config, **{field: invalid}))

    def test_partial_model_figure_replaces_old_png_with_explicit_missing_panel(self):
        import export_research_viewer as exporter
        from PIL import Image
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            png = root / 'pilot_network_comparison.png'
            png.write_bytes(b'old stale figure')
            graph = nx.path_graph([str(i) for i in range(50)])
            nx.write_adjlist(graph, root / 'graph.adj')
            personas = {node: {'political affiliation': 'Democrat'} for node in graph}
            rows = [dict(run_id=f'{model}_{method}', model=model, method=method,
                         source='graph.adj', language='english')
                    for model in ['gpt-4.1-mini', 'gpt-6-luna', 'gpt-6-sol']
                    for method in ['global', 'local', 'sequential', 'iterative']
                    if (model, method) != ('gpt-6-sol', 'iterative')]
            real_subplots = exporter.plt.subplots
            created = []
            def capture(*args, **kwargs):
                result = real_subplots(*args, **kwargs)
                created.append(result)
                return result
            with patch.object(exporter.plt, 'subplots', side_effect=capture):
                exporter.render_pilot_comparison(root, root, rows, personas)
            self.assertEqual(created[0][1].shape, (3, 4))
            self.assertIn('NOT COLLECTED', [text.get_text() for text in created[0][1][2, 3].texts])
            with Image.open(png) as image:
                image.verify()


if __name__ == '__main__':
    unittest.main()
