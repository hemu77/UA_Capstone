import unittest

import networkx as nx
import numpy as np
import pandas as pd

from analysis_extension.core import ATTRIBUTES, SCORES, categorical_mixing, paired_cells, method_summary, rank_cells
from analysis_extension.sampling import strict_pairs, graph_metrics, observed_nodes, candidate_samples, compare_sizes
from analysis_heldout.benchmark import METRICS, candidate_weights, sample_nodes
from analysis_figures.review_sensitivity import nested_metrics
from scripts.fetch_copenhagen_public import allowed


class MixingTests(unittest.TestCase):
    def graph(self, edges):
        g = nx.Graph()
        g.add_nodes_from(range(4))
        g.add_edges_from(edges)
        return g

    def test_algebraic_fixtures(self):
        cases = [([(0, 1), (2, 3)], 2/3, 1, 1),
                 ([(0, 2)], -1/3, -1, -1),
                 (list(nx.complete_graph(4).edges()), 0, -1/3, -1/3)]
        for edges, excess, newman, coleman in cases:
            with self.subTest(edges=edges):
                result = categorical_mixing(self.graph(edges), dict(enumerate('AABB')))
                for name, value in [('excess', excess), ('newman', newman), ('coleman', coleman)]:
                    self.assertAlmostEqual(result[name], value)
                e = result['matrix']
                np.testing.assert_allclose(e, e.T)
                self.assertAlmostEqual(e.sum(), 1)
                np.testing.assert_allclose(e.sum(axis=1), result['marginals'])

    def test_undefined_and_isolates(self):
        result = categorical_mixing(self.graph([(0, 1)]), dict(enumerate('AABB')))
        self.assertAlmostEqual(result['excess'], 2/3)
        self.assertTrue(np.isnan(result['newman']))
        self.assertTrue(np.isnan(result['coleman']))
        result = categorical_mixing(self.graph([]), dict(enumerate('AABB')))
        self.assertTrue(np.isnan(result['matrix']).all())
        self.assertTrue(all(np.isnan(result[k]) for k in ('excess', 'newman', 'coleman')))

    def test_roster_and_graph_invariants(self):
        g = self.graph([(0, 1), (2, 3)])
        with self.assertRaises(ValueError):
            categorical_mixing(g, {0: 'A', 1: 'A'})
        g.add_edge(0, 0)
        with self.assertRaises(ValueError):
            categorical_mixing(g, dict(enumerate('AABB')))

    def fixture(self):
        return pd.DataFrame([
            dict(method=m, model='model', culture='us', language='english', seed=s,
                 attribute='gender', excess=x, newman=x, coleman=x)
            for m, s, x in [('global', 1, 0.), ('global', 2, 10.),
                            ('local', 1, 1.), ('local', 2, np.nan)]])

    def test_pair_before_averaging(self):
        cells = paired_cells(self.fixture(), 'excess')
        r = cells[(cells['first'] == 'global') & (cells['second'] == 'local')].iloc[0]
        self.assertEqual((r.defined_seeds, r.difference), (1, 1.))
        r = paired_cells(self.fixture(), 'excess', omitted=1).iloc[0]
        self.assertEqual(r.defined_seeds, 0)
        self.assertTrue(np.isnan(r.difference))

    def test_duplicate_pairs_rejected(self):
        f = self.fixture()
        with self.assertRaises(ValueError):
            paired_cells(pd.concat([f, f.iloc[:1]]), 'excess')

    def test_common_score_support(self):
        f = self.fixture()
        f.loc[f.method.eq('local') & f.seed.eq(1), 'newman'] = np.nan
        self.assertEqual(paired_cells(f, 'excess').iloc[0].defined_seeds, 1)
        self.assertEqual(paired_cells(f, 'excess', common=True).iloc[0].defined_seeds, 0)

    def test_missing_cell_suppresses_28_cell_mean(self):
        cells = pd.DataFrame([dict(first='global', second='local', attribute='gender', score='excess',
            support='score_specific', model=str(i), culture='us', language='english',
            difference=float(i == 0), defined_seeds=1 if i == 0 else 8) for i in range(28)])
        self.assertAlmostEqual(method_summary(cells, -1).mean_difference.iloc[0], 1/28)
        missing = method_summary(cells.iloc[1:], -1).iloc[0]
        self.assertEqual((missing.observed_cells, missing.planned_cells), (27, 28))
        self.assertTrue(np.isnan(missing.mean_difference))
        with self.assertRaises(ValueError):
            method_summary(pd.concat([cells, cells.iloc[:1]]), -1)

    def test_rank_uses_attribute_and_score_intersection(self):
        rows = [dict(method='global', model='model', culture='us', language='english', seed=seed,
                     attribute=attr, **{score: float(attr == 'political orientation') for score in SCORES})
                for seed in (1, 2) for attr in ATTRIBUTES]
        frame = pd.DataFrame(rows)
        frame.loc[frame.seed.eq(2) & frame.attribute.eq('religion'), 'excess'] = np.nan
        frame.loc[frame.seed.eq(1) & frame.attribute.eq('gender'), 'newman'] = np.nan
        ranks = rank_cells(frame, -1)
        specific = ranks[ranks.support.eq('score_specific') & ranks.score.eq('excess')].iloc[0]
        self.assertEqual((specific.defined_seeds, specific.winner), (1, 'political orientation'))
        self.assertTrue(ranks[ranks.support.eq('all_three_scores')].winner.eq('undefined').all())
        self.assertTrue(rank_cells(frame, 1).query("score == 'excess'").winner.eq('undefined').all())

    def test_newman_agrees_with_networkx(self):
        g = self.graph([(0, 1), (0, 2), (1, 3)])
        labels = dict(enumerate('AABB'))
        nx.set_node_attributes(g, labels, 'group')
        self.assertAlmostEqual(categorical_mixing(g, labels)['newman'], nx.attribute_assortativity_coefficient(g, 'group'))


class SamplingTests(unittest.TestCase):
    def test_original_ids_nested_and_historical_metrics(self):
        g = nx.Graph()
        g.add_nodes_from(range(100, 700, 5))
        g.add_edges_from(zip(range(100, 250, 5), range(105, 255, 5)))
        g.graph.update(source='fixture', cohort='original_ids')
        sets = {n: observed_nodes(g, 0, n) for n in (25, 40, 50)}
        self.assertTrue(sets[25] < sets[40] < sets[50] <= set(g))
        self.assertEqual([len(sets[n]) for n in (25, 40, 50)], [25, 40, 50])
        self.assertLess(len(sets[50] & observed_nodes(g, 1, 50)), 50)
        for n, reference in nested_metrics(sample_nodes(g, 0), ('fixture', 'original_ids', 0)):
            recovered = graph_metrics(g.subgraph(sets[n]))
            self.assertGreater(recovered['isolates'], 0)
            for metric, value in reference.items():
                self.assertAlmostEqual(recovered[metric], value)

    def test_thinning_recomputes_density_and_keeps_isolates(self):
        empty = candidate_samples(nx.empty_graph(50), ('empty',))
        for n, values in empty.items():
            self.assertEqual(values['isolates'], n)
            self.assertAlmostEqual(values['prop_nodes_lcc'], 1/n)
        samples = candidate_samples(nx.star_graph(49), ('star',))
        for n, values in samples.items():
            self.assertAlmostEqual(values['density'], values['edges']/(n*(n-1)/2))
        self.assertNotEqual(samples[25]['density'], samples[50]['density'])

    def test_instruction_weights_and_missing_condition(self):
        frame = pd.DataFrame([dict(kind='LLM', condition=str(c), valid=True, density=float(c))
                              for c in range(7) for _ in range(c+1)])
        _, weights = candidate_weights(frame, 'density')
        for c in range(7):
            self.assertAlmostEqual(weights[frame.condition.eq(str(c))].sum(), 1/7)
        with self.assertRaises(ValueError):
            candidate_weights(frame[frame.condition.ne('6')], 'density')

    def test_average_model_distances_not_pooled_distribution(self):
        candidates, targets = [], []
        for n in (25, 40, 50):
            for model, value in enumerate((0., 0., 1., 1.)):
                for condition in range(7):
                    candidates.append(dict(n=n, candidate=f'LLM:global:{model}', kind='LLM', bank=0,
                        condition=str(condition), valid=True, **dict.fromkeys(METRICS, value)))
            for value in (0., 1.):
                targets.append(dict(n=n, source='fixture', cohort='a', **dict.fromkeys(METRICS, value)))
        _, records = compare_sizes(pd.DataFrame(candidates), pd.DataFrame(targets))
        self.assertTrue(records.models_or_banks.eq(4).all())
        np.testing.assert_allclose(records.mean_w1, .5)  # Pooled-model W1 would be zero.

    def test_untrusted_csv_and_redirect_boundaries(self):
        self.assertEqual(strict_pairs(b'# user_a,user_b\n1,2\n', '# user_a,user_b'), [(1, 2)])
        for raw in (b'wrong\n1,2', b'# h\n-1,2', b'# h\n1,2,3', b'# h\n1,2\n\n', b'x'*60001):
            with self.assertRaises(ValueError):
                strict_pairs(raw, '# h')
        allowed('https://ndownloader.figshare.com/files/13389320')
        for url in ('http://ndownloader.figshare.com/x', 'https://localhost/x',
                    'https://user@ndownloader.figshare.com/x', 'https://ndownloader.figshare.com:8080/x',
                    'https://ndownloader.figshare.com.attacker.test/x'):
            with self.assertRaises(ValueError):
                allowed(url)


if __name__ == '__main__':
    unittest.main()
