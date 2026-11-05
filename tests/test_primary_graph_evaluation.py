import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import networkx as nx
import pandas as pd


spec = importlib.util.spec_from_file_location(
    'primary_graph_evaluation',
    Path(__file__).resolve().parents[1] / 'analysis_primary/graph_evaluation.py',
)
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


class GraphEvaluationTests(unittest.TestCase):
    def fixture(self, root):
        rows = []
        for index, edges in enumerate([[(0, 1), (1, 2), (2, 0)], [], None]):
            row = dict(cohort='primary', run_id=f'original_{index}',
                       original_run_id=f'original_{index}', primary_member=True,
                       eligible_primary_graph=edges is not None, valid_graph=edges is not None,
                       model='fixture', method='global', culture='test', language='english',
                       wording='reference', failure='fixture_failure' if edges is None else None,
                       record_path=f'record_{index}.json', record_sha256='a' * 64,
                       adj_path=None, adj_sha256=None, nodes=None, edges=None,
                       density=None, avg_clustering_coef=None, global_efficiency=None)
            if edges is not None:
                graph = nx.empty_graph([str(i) for i in range(50)])
                graph.add_edges_from((str(a), str(b)) for a, b in edges)
                buffer = io.BytesIO()
                nx.write_adjlist(graph, buffer)
                data = buffer.getvalue()
                path = root / f'graph_{index}.adj'
                path.write_bytes(data)
                row.update(adj_path=path.name, adj_sha256=hashlib.sha256(data).hexdigest(),
                           nodes=50, edges=len(edges), density=nx.density(graph),
                           avg_clustering_coef=nx.average_clustering(graph),
                           global_efficiency=nx.global_efficiency(graph))
            rows.append(row)
        primary = pd.DataFrame(rows)
        source = root / evaluation.PRIMARY_CSV
        source.parent.mkdir(parents=True)
        primary.to_csv(source, index=False)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        return primary, digest

    def test_outputs_retain_failures_and_empty_graphs_and_repeat_exactly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            primary, digest = self.fixture(root)
            with patch.object(evaluation, 'PRIMARY_CSV_SHA256', digest):
                result = evaluation.run_graph_evaluation(root, primary, root / 'first')
                evaluation.run_graph_evaluation(root, primary, root / 'second')
            self.assertEqual(result['verified_graphs'], 2)
            self.assertEqual(result['failed_attempts'], 1)
            self.assertEqual(result['synthetic_graphs'], 2 * evaluation.REFERENCE_REPLICATES)
            self.assertEqual(result['reference_MC_reps'], 8)
            self.assertEqual(result['valid_empty_graphs'], 1)
            self.assertEqual(result['reference_empty_graphs'], 8)
            self.assertEqual(result['reference_failures'], 0)
            self.assertFalse(result['reference_SD_is_CI'])
            for name in result['files']:
                self.assertEqual((root / 'first' / name).read_bytes(),
                                 (root / 'second' / name).read_bytes())
            quality = pd.read_csv(root / 'first/graph_quality.csv')
            self.assertEqual(quality.loc[2, 'status'], 'failure')
            self.assertEqual(quality.loc[2, 'failure'], 'fixture_failure')
            self.assertTrue(pd.isna(quality.loc[2, 'recomputed_density']))
            reference = pd.read_csv(root / 'first/synthetic_reference_per_run.csv')
            empty = reference[(reference.run_id == 'original_1') & (reference.metric == 'density')]
            self.assertEqual(empty.iloc[0]['reference_mean'], 0)
            self.assertEqual(empty.iloc[0]['reference_sd'], 0)
            failed = reference[reference.run_id == 'original_2']
            self.assertTrue(failed.reference_mean.isna().all())
            self.assertTrue(failed.observed_value.isna().all())
            aggregate = pd.read_csv(root / 'first/synthetic_reference_descriptive.csv')
            self.assertTrue((aggregate.attempts == 3).all())
            self.assertTrue((aggregate.valid_graphs == 2).all())
            manifest = json.loads((root / 'first/graph_evaluation.json').read_text())
            self.assertFalse(manifest['empirical_ground_truth'])
            self.assertFalse(manifest['primary_hypothesis_test'])

    def test_rejects_supplemental_unknown_duplicate_and_incomplete_cohorts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            primary, digest = self.fixture(root)
            mutations = []
            for column, value in [('cohort', 'supplemental'), ('primary_member', False),
                                  ('primary_member', 'False'), ('model', 'changed'),
                                  ('record_sha256', 'b' * 64),
                                  ('original_run_id', 'other'), ('run_id', 'other')]:
                changed = primary.copy()
                changed.loc[0, column] = value
                mutations.append(changed)
            mutations.extend([primary.iloc[:2], pd.concat([primary, primary.iloc[:1]]),
                              primary.drop(columns='cohort')])
            with patch.object(evaluation, 'PRIMARY_CSV_SHA256', digest):
                for changed in mutations:
                    with self.subTest(rows=len(changed)), self.assertRaises(ValueError):
                        evaluation.run_graph_evaluation(root, changed, root / 'out')
            self.assertFalse((root / 'out').exists())

    def test_unbound_source_csv_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            primary, digest = self.fixture(root)
            source = root / evaluation.PRIMARY_CSV
            source.write_bytes(source.read_bytes() + b'\n')
            with patch.object(evaluation, 'PRIMARY_CSV_SHA256', digest):
                with self.assertRaisesRegex(ValueError, 'SHA-256'):
                    evaluation.run_graph_evaluation(root, primary, root / 'out')
            self.assertFalse((root / 'out').exists())

    def test_reference_failure_aborts_without_zero_filling_or_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            primary, digest = self.fixture(root)
            with patch.object(evaluation, 'PRIMARY_CSV_SHA256', digest), patch.object(
                    evaluation.nx, 'gnm_random_graph', side_effect=RuntimeError('fixture failure')):
                with self.assertRaisesRegex(RuntimeError, 'fixture failure'):
                    evaluation.run_graph_evaluation(root, primary, root / 'out')
            self.assertFalse((root / 'out').exists())

    def test_rejects_hash_metric_path_roster_loop_and_edge_corruption(self):
        for corruption in ['hash', 'density', 'clustering', 'path', 'roster', 'loop', 'edges']:
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                primary, digest = self.fixture(root)
                if corruption == 'hash':
                    (root / 'graph_0.adj').write_bytes(b'changed')
                elif corruption in ('density', 'clustering', 'path', 'edges'):
                    column, value = {'density': ('density', .9),
                                     'clustering': ('avg_clustering_coef', .9),
                                     'path': ('adj_path', '../outside.adj'),
                                     'edges': ('edges', 400)}[corruption]
                    primary.loc[0, column] = value
                else:
                    path = root / 'graph_0.adj'
                    data = path.read_bytes() + (b'50\n' if corruption == 'roster' else b'0 0\n')
                    path.write_bytes(data)
                    primary.loc[0, 'adj_sha256'] = hashlib.sha256(data).hexdigest()
                # Rebind the fixture CSV so graph-level checks, not table checks, catch corruption.
                source = root / evaluation.PRIMARY_CSV
                primary.to_csv(source, index=False)
                digest = hashlib.sha256(source.read_bytes()).hexdigest()
                with patch.object(evaluation, 'PRIMARY_CSV_SHA256', digest):
                    with self.assertRaises(ValueError):
                        evaluation.run_graph_evaluation(root, primary, root / 'out')
                self.assertFalse((root / 'out').exists())

    def test_undefined_observations_are_not_zero_filled(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            primary, _ = self.fixture(root)
            primary.loc[0, ['density', 'avg_clustering_coef', 'global_efficiency']] = float('nan')
            source = root / evaluation.PRIMARY_CSV
            primary.to_csv(source, index=False)
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            with patch.object(evaluation, 'PRIMARY_CSV_SHA256', digest):
                evaluation.run_graph_evaluation(root, primary, root / 'out')
            reference = pd.read_csv(root / 'out/synthetic_reference_per_run.csv')
            row = reference[(reference.run_id == 'original_0') &
                            (reference.metric == 'global_efficiency')].iloc[0]
            self.assertTrue(pd.isna(row.observed_value))
            quality = pd.read_csv(root / 'out/graph_quality.csv')
            self.assertEqual(quality.loc[0, 'density_check'], 'undefined')
            self.assertTrue(pd.isna(quality.loc[0, 'recorded_avg_clustering_coef']))

    def test_trivial_reference_graphs(self):
        for n in [0, 1, 2]:
            for m in sorted({0, n * (n - 1) // 2}):
                first = evaluation._reference_metrics('run', n, m)
                pd.testing.assert_frame_equal(pd.DataFrame(first), pd.DataFrame(
                    evaluation._reference_metrics('run', n, m)))
                self.assertTrue(all(len(values) == evaluation.REFERENCE_REPLICATES
                                    for values in first.values()))
                density = 2 * m / (n * (n - 1)) if n > 1 else 0
                self.assertTrue(all(value == density for value in first['density']))
                self.assertTrue(all(math.isnan(value) if n == 0 else value == 0
                                    for value in first['avg_clustering_coef']))


if __name__ == '__main__':
    unittest.main()
