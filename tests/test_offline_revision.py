"""Offline regression checks for the revision, not evidence of model behavior."""
import copy
import itertools
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import networkx as nx
import pandas as pd
from scipy.stats import nct, t

import revision_next as revised
import analyze_saved_study as saved


class RevisedPromptTests(unittest.TestCase):
    def test_single_friend_has_single_id_instructions_in_both_affected_languages(self):
        english = revised.system_prompt('local', 'english', 'us', '0', 1)
        portuguese = revised.system_prompt('local', 'portuguese', 'us', '0', 1)
        self.assertIn('exactly one friend', english)
        self.assertIn('only one integer ID', english)
        self.assertNotIn('commas', english)
        self.assertIn('exatamente um amigo', portuguese)
        self.assertNotIn('vírgulas', portuguese)

    def test_prompt_catalog_covers_all_conditions_without_mutating_v5(self):
        before = copy.deepcopy(revised.previous.TEXT)
        with tempfile.TemporaryDirectory() as folder, patch.object(revised, 'DESTINATION', Path(folder)):
            report = revised.prepare()
            catalog = json.loads((Path(folder) / 'prompt_catalog.json').read_text(encoding='utf-8'))
            probe = json.loads((Path(folder) / 'probe_manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(len(catalog), 144)
        self.assertEqual(len(probe), 72)
        self.assertEqual(report['paid_calls'], 0)
        self.assertFalse(report['main_authorized'])
        self.assertEqual(revised.previous.TEXT, before)
        self.assertEqual({row['language'] for row in catalog}, set(revised.config()['languages']))
        for row in catalog:
            self.assertNotIn('{country}', row['system'])
            self.assertNotIn('{actor}', row['system'])
            self.assertNotIn('{count}', row['system'])
            self.assertEqual(len(json.loads(row['user'])['candidates']) if row['method'] == 'global' else 50, 50)
        for old, new in zip(probe[::2], probe[1::2]):
            self.assertEqual(old['user'], new['user'], 'Probe arms must change only wording.')

    def test_schedule_is_reproducible_and_not_id_order(self):
        plan = revised.schedule(21000)
        self.assertEqual(plan, revised.schedule(21000))
        self.assertNotEqual(plan['display_order'], list(revised.previous.adult_roster()))
        self.assertNotEqual(plan['display_order'], revised.schedule(21001)['display_order'])
        graph = nx.path_graph(list(revised.previous.adult_roster()))
        payloads = [revised.candidate_payload('sequential', lang, '0', graph, plan['display_order']) for lang in revised.config()['languages']]
        self.assertEqual(len(set(payloads)), 1, 'Only instructions, not persona data, change with language.')
        for method in ['iterative-add', 'iterative-drop']:
            rows = json.loads(revised.candidate_payload(method, 'english', '0', graph, plan['display_order']))['candidates']
            self.assertTrue(all((row[0] in graph['0']) == (method == 'iterative-drop') for row in rows))
        self.assertEqual(plan, revised.schedule(21000), 'Payload construction must not consume scheduling RNG.')

    def test_all_56_method_setting_seed_fixtures_replay_on_full_roster(self):
        schedules = {}
        for method, (country, language), seed in itertools.product(revised.config()['methods'], revised.config()['settings'], [21000, 21001]):
            with self.subTest(method=method, country=country, language=language, seed=seed):
                graph, events, requests, plan = revised.dry_run(method, country, language, seed)
                self.assertEqual(set(graph), set(revised.previous.adult_roster()))
                self.assertFalse(list(nx.selfloop_edges(graph)))
                self.assertTrue(all(request['valid'] for request in requests))
                if method != 'global':
                    self.assertEqual([event['persona'] for event in events[:50]], plan['actor_order'])
                    self.assertEqual({r['actor']: r['count'] for r in requests[:50]}, plan['nomination_quotas'])
                schedules.setdefault(seed, plan)
                self.assertEqual(schedules[seed], plan, 'Schedules must match across conditions.')

    def test_global_none_is_valid_but_cannot_erase_repair_target(self):
        graph, events, _, _ = revised.dry_run('global', 'us', 'english', 21000, reply=lambda *args: 'NONE')
        self.assertEqual(graph.number_of_edges(), 0)
        self.assertEqual(events[0]['added'], [])
        self.assertIsNone(saved.measures(graph, revised.previous.adult_roster())['modularity'])
        with self.assertRaisesRegex(ValueError, 'cannot discard'):
            revised.parse_response('global', 'NONE', graph, required_global_edges=[['0', '1']])

    def test_retries_preserve_global_ties_and_explain_single_id(self):
        def global_reply(system, user, args, attempt):
            if attempt > 1:
                self.assertIn('required_pairs', system)
            return ['0, 1\n1, 0\n2, 3', '0, 1', '0, 1\n2, 3'][attempt-1]
        graph, _, requests, _ = revised.dry_run('global', 'us', 'portuguese', 21000, reply=global_reply)
        self.assertEqual({frozenset(e) for e in graph.edges()}, {frozenset(['0', '1']), frozenset(['2', '3'])})
        self.assertEqual([r['valid'] for r in requests], [False, False, True])
        def local_reply(system, user, args, attempt):
            ids = [row[0] for row in json.loads(user)['candidates']]
            if args['num_choices'] == 1:
                if attempt == 1:
                    return ', '.join(ids[:2])
                self.assertIn('exatamente um amigo', system)
            return ', '.join(ids[:args['num_choices']])
        _, _, requests, _ = revised.dry_run('local', 'us', 'portuguese', 21000, reply=local_reply)
        self.assertTrue(any(not r['valid'] for r in requests))

    def test_invalid_condition_and_unauthorized_mode_fail_closed(self):
        for method, language, country in [('local', 'spanish', 'us'), ('local', 'english', 'mexico'), ('unknown', 'english', 'us')]:
            with self.assertRaises(ValueError):
                revised.system_prompt(method, language, country, '0', 1)
        with self.assertRaisesRegex(ValueError, 'nonnegative integer'):
            revised.schedule(True)
        with self.assertRaisesRegex(ValueError, 'three-attempt'):
            revised.dry_run('local', 'us', 'english', 21000, reply=lambda *args: 'not IDs')

    def test_preparation_rejects_authorization_and_roster_drift(self):
        protocol = revised.config()
        with tempfile.TemporaryDirectory() as folder, patch.object(revised, 'ROOT', Path(folder)):
            protocol_file = Path(folder) / 'study_protocol_next.json'
            protocol['generation_authorized'] = True
            protocol_file.write_text(json.dumps(protocol), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'cannot grant'):
                revised.config()
            protocol['generation_authorized'] = False
            protocol['persona_file'] = 'changed_roster.json'
            protocol_file.write_text(json.dumps(protocol), encoding='utf-8')
            (Path(folder) / 'changed_roster.json').write_text('{}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Declared roster differs'):
                revised.config()


class SavedAnalysisTests(unittest.TestCase):
    def test_detectable_effect_matches_noncentral_t_and_corrects_review_table(self):
        effect = saved.detectable_effect(8, 288)
        critical = t.isf(.05 / 288 / 2, 7)
        self.assertAlmostEqual(nct.sf(critical, 7, effect * 8**.5) + nct.cdf(-critical, 7, effect * 8**.5), .8)
        self.assertAlmostEqual(effect, 3.0970834945)
        self.assertLess(saved.detectable_effect(32, 24), saved.detectable_effect(8, 24))
        with self.assertRaises(ValueError):
            saved.detectable_effect(1)

    def test_real_receipts_reanalyze_without_client_or_private_ledger(self):
        with patch.object(saved.study, 'Budget', side_effect=AssertionError('No private ledger')), \
                patch.object(saved.study.shared, 'OpenAI', side_effect=AssertionError('No API')):
            _, _, verified = saved.load_calibration()
            self.assertEqual(len(verified), 68)
            decisions = [row for _, _, record, _ in verified for row in saved.decision_rows(record)]
        df = pd.DataFrame(decisions)
        focus = df[(df.model == 'gpt-6-luna') & (df.method == 'local') & (df.culture == 'us')]
        retries = focus.groupby('language').first_attempt_failed.sum().to_dict()
        self.assertEqual(retries, dict(english=3, hindi=4, japanese=0, portuguese=25))
        self.assertEqual(focus.groupby('language').size().to_dict(), dict.fromkeys(retries, 100))

    def test_undefined_scores_and_failed_rewires_are_not_zero_or_success(self):
        roster = revised.previous.adult_roster()
        values = saved.measures(nx.empty_graph(roster), roster)
        self.assertEqual(values['density'], 0)
        self.assertEqual(values['isolate_share'], 1)
        self.assertIsNone(values['degree_gini'])
        observed = pd.DataFrame([dict(run_id='empty', **values)]).set_index('run_id')
        controls = pd.DataFrame([dict(run_id='empty', baseline='degree_preserving_rewire', rewiring_target_reached=False, **values)])
        summary = saved.reference_summary(observed, controls)
        self.assertTrue((summary['attempted_draws'] == 1).all())
        self.assertTrue((summary['completed_draws'] == 0).all())
        self.assertTrue(summary['reference_mean'].isna().all())


if __name__ == '__main__':
    unittest.main()
