"""Offline plot-contract checks; no receipt scan or pipeline execution."""
import itertools
import hashlib
import ast
import json
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd

from analysis_primary.plots import render_figures


class PrimaryPlotsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        spec = json.loads((Path(__file__).resolve().parents[1] / 'final_study/contract.json').read_text())
        (self.root / 'final_study').mkdir()
        (self.root / 'final_study/contract.json').write_text(json.dumps(spec))
        a = spec['analysis']
        records = []
        for i, (model, method, (culture, language), seed) in enumerate(itertools.product(
                a['models'], a['methods'], a['conditions'], a['schedule_seeds'])):
            valid = i >= 6
            records.append(dict(run_id=f'final_060dcd3a86d5_main_fixture_{i}', cohort='primary',
                primary_member=True, model=model, method=method, culture=culture, language=language,
                seed=seed, valid_graph=valid, empty_graph=6 <= i < 18,
                density=.2 if valid else np.nan, avg_clustering_coef=.3 if valid else np.nan))
        self.primary = pd.DataFrame(records)
        self.output = self.root / 'output'
        self.output.mkdir()
        contrasts, blocks, descriptions, resources = [], [], [], []
        for method, (culture, language) in itertools.product(a['methods'], a['conditions'][1:]):
            suppressed = method == 'global' and language == 'japanese'
            hid = f'{method}:{culture}:{language}'
            contrasts.append(dict(hypothesis_id=hid, method=method, culture=culture, language=language,
                endpoint=a['primary_endpoints'][method], complete_schedule_blocks=5 if suppressed else 8,
                status='SUPPRESSED' if suppressed else 'COMPLETE_ASSUMPTIONS_REQUIRED',
                complete_block_mean=.01, holm_p=np.nan if suppressed else .8,
                ci_lower=np.nan if suppressed else -.02, ci_upper=np.nan if suppressed else .04,
                missingness_lower=-.1, missingness_upper=.1))
            for i, seed in enumerate(a['schedule_seeds']):
                complete = not suppressed or i < 5
                blocks.append(dict(hypothesis_id=hid, method=method, culture=culture, language=language,
                    endpoint=a['primary_endpoints'][method], seed=seed, complete=complete,
                    difference=(i - 3) / 100 if complete else np.nan, lower=-.1, upper=.1))
        for model, method, (culture, language) in itertools.product(a['models'], a['methods'], a['conditions']):
            descriptions.append(dict(model=model, method=method, culture=culture, language=language,
                metric=a['primary_endpoints'][method], mean=.2, endpoint_defined=8))
        for model, method in itertools.product(a['models'], a['methods']):
            resources.append(dict(model=model, method=method, attempts=56, requests=112,
                received_conservative_usd=2.4, api_seconds=90))
        for name, rows in [('primary_contrasts', contrasts), ('paired_schedule_blocks', blocks),
                           ('descriptive_metrics', descriptions), ('resources_by_method_model', resources)]:
            pd.DataFrame(rows).to_csv(self.output / f'{name}.csv', index=False)

    def test_render_pairs_and_close_only_owned_figures(self):
        existing = plt.figure()
        self.addCleanup(plt.close, existing)
        paths = render_figures(self.root, self.primary, self.output)
        self.assertEqual(len(paths), 12)
        self.assertEqual(len(set(paths)), 12)
        for name in paths:
            path = Path(name)
            self.assertTrue(path.is_file())
            self.assertEqual(path.parent, self.output / 'figures')
            self.assertGreater(path.stat().st_size, 1000)
        self.assertEqual(plt.get_fignums(), [existing.number])
        forest = (self.output / 'figures/01_primary_forest.svg').read_text()
        for text in ['95% simultaneous', 'Bonferroni', '24', 'no CI', 'n=5/8', 'US-English']:
            self.assertIn(text, forest)
        means = (self.output / 'figures/03_model_endpoint_means.svg').read_text()
        for model in self.primary.model.unique():
            self.assertIn(model, means)

    def test_reject_supplement_before_writing(self):
        self.primary.loc[0, 'cohort'] = 'supplement'
        with self.assertRaisesRegex(ValueError, 'Supplemental or mixed'):
            render_figures(self.root, self.primary, self.output)
        self.assertFalse((self.output / 'figures').exists())

    def test_failures_with_endpoints_rejected(self):
        self.primary.loc[0, 'density'] = 0
        with self.assertRaisesRegex(ValueError, 'Failures cannot carry endpoints'):
            render_figures(self.root, self.primary, self.output)

    def test_incomplete_blocks_not_zero_filled(self):
        path = self.output / 'paired_schedule_blocks.csv'
        blocks = pd.read_csv(path)
        blocks.loc[~blocks.complete, 'difference'] = 0
        blocks.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, 'Incomplete blocks cannot carry differences'):
            render_figures(self.root, self.primary, self.output)

    def test_save_failure_closes_figure(self):
        before = plt.get_fignums()
        with patch('matplotlib.figure.Figure.savefig', side_effect=OSError('disk full')):
            with self.assertRaisesRegex(OSError, 'disk full'):
                render_figures(self.root, self.primary, self.output)
        self.assertEqual(plt.get_fignums(), before)

    def test_notebook_code_and_outputs_are_valid(self):
        path = Path(__file__).resolve().parents[1] / 'analysis/02_primary_comparison_plots.ipynb'
        notebook = json.loads(path.read_text())
        nbformat.validate(nbformat.from_dict(notebook))
        self.assertEqual(notebook['metadata']['kernelspec']['name'], 'ua-capstone-final060d')
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                count = cell['execution_count']
                self.assertTrue(count is None or (type(count) is int and count > 0))
                self.assertFalse(any(output['output_type'] == 'error' for output in cell['outputs']))
                compile(''.join(cell['source']), str(path), 'exec')

    def test_notebook_copies_checks_hashes_and_preserves_sealed_output(self):
        path = Path(__file__).resolve().parents[1] / 'analysis/02_primary_comparison_plots.ipynb'
        notebook = json.loads(path.read_text())
        code = ''.join(next(cell for cell in notebook['cells'] if cell['id'] == 'validated-render')['source'])
        (self.root / 'analysis_primary').mkdir()
        (self.root / 'analysis_primary/core.py').touch()
        sealed = self.root / 'outputs/primary_analysis/20261105'
        sealed.parent.mkdir(parents=True)
        self.output.rename(sealed)
        sentinel = sealed / 'figures/01_primary_forest.svg'
        sentinel.parent.mkdir()
        sentinel.write_bytes(b'sealed figure must not change')
        optional = sealed / 'synthetic_reference_per_run.csv'
        optional.write_bytes(b'optional reference fixture')
        for name in ('empirical_reference_metrics.csv', 'empirical_comparisons_descriptive.csv'):
            (sealed / name).write_bytes(b'optional empirical fixture')
        inputs = list(sealed.glob('*.csv'))
        manifest = {'status': 'COMPLETE_PRIMARY_ANALYSIS_WITH_CAVEATS', 'artifacts': {
            p.name: {'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in inputs}}
        driver = path.parents[1] / 'scripts/run_primary_evaluation.py'
        driver_statuses = [node.value.value for node in ast.walk(ast.parse(driver.read_text()))
                           if isinstance(node, ast.keyword) and node.arg == 'status'
                           and isinstance(node.value, ast.Constant)]
        self.assertIn(manifest['status'], driver_statuses)
        (sealed / 'evaluation_manifest.json').write_text(json.dumps(manifest))
        before = {p.relative_to(sealed): p.read_bytes() for p in sealed.rglob('*') if p.is_file()}
        namespace = {}

        def render(root, primary, output):
            self.assertEqual(root, self.root)
            self.assertIs(primary, self.primary)
            self.assertNotEqual(output, sealed)
            for source in inputs:
                self.assertEqual((output / source.name).read_bytes(), source.read_bytes())
            return []

        with patch('sys.path', list(sys.path)), \
                patch('pathlib.Path.cwd', return_value=self.root), \
                patch('analysis_primary.core.load_inputs', return_value=({}, self.primary, [])) as loader, \
                patch('analysis_primary.plots.render_figures', side_effect=render) as renderer:
            exec(code, namespace)
            self.addCleanup(namespace['plot_workspace'].cleanup)
            renderer.assert_called_once()
            self.assertTrue(namespace['PLOT_OUTPUT'].is_dir())
            self.assertEqual(before, {p.relative_to(sealed): p.read_bytes() for p in sealed.rglob('*') if p.is_file()})
            (sealed / 'primary_contrasts.csv').write_bytes(b'tampered')
            rejected = {}
            try:
                with self.assertRaisesRegex(ValueError, 'Evaluation manifest hash mismatch'):
                    exec(code, rejected)
            finally:
                if 'plot_workspace' in rejected:
                    rejected['plot_workspace'].cleanup()
            renderer.assert_called_once()
            calls_before_status_check = loader.call_count
            for status in ('IN_PROGRESS', 'FAILED_INCOMPLETE', 'COMPLETED', None):
                (sealed / 'evaluation_manifest.json').write_text(json.dumps({'status': status}))
                with self.subTest(status=status):
                    with self.assertRaisesRegex(ValueError, 'Evaluation manifest must be COMPLETE_PRIMARY_ANALYSIS_WITH_CAVEATS'):
                        exec(code, {})
            self.assertEqual(loader.call_count, calls_before_status_check)
            renderer.assert_called_once()

    def test_notebook_refuses_missing_manifest_before_loading_inputs(self):
        path = Path(__file__).resolve().parents[1] / 'analysis/02_primary_comparison_plots.ipynb'
        notebook = json.loads(path.read_text())
        code = ''.join(next(cell for cell in notebook['cells'] if cell['id'] == 'validated-render')['source'])
        (self.root / 'analysis_primary').mkdir()
        (self.root / 'analysis_primary/core.py').touch()
        with patch('sys.path', list(sys.path)), \
                patch('pathlib.Path.cwd', return_value=self.root), \
                patch('analysis_primary.core.load_inputs') as loader, \
                patch('analysis_primary.plots.render_figures') as renderer:
            with self.assertRaisesRegex(FileNotFoundError, 'Completed evaluation manifest required'):
                exec(code, {})
            loader.assert_not_called()
            renderer.assert_not_called()

    def test_optional_reference_supplement_rejected_before_writing(self):
        pd.DataFrame([dict(run_id='supplement', metric='avg_clustering_coef')]).to_csv(
            self.output / 'synthetic_reference_per_run.csv', index=False)
        with self.assertRaisesRegex(ValueError, 'Supplemental reference rows refused'):
            render_figures(self.root, self.primary, self.output)
        self.assertFalse((self.output / 'figures').exists())

    def test_optional_gnm_context_and_deterministic_svg(self):
        sample = self.primary.groupby(['method', 'model'], sort=False).head(8).copy()
        sample['metric'] = 'avg_clustering_coef'
        sample['observed_value'] = sample.avg_clustering_coef
        sample['reference_mean'] = np.where(sample.valid_graph, .1, np.nan)
        sample['reference_sd'] = np.where(sample.valid_graph, .02, np.nan)
        sample['reference_replicates'] = np.where(sample.valid_graph, 8, 0)
        sample.to_csv(self.output / 'synthetic_reference_per_run.csv', index=False)
        before = plt.get_fignums()
        paths = render_figures(self.root, self.primary, self.output)
        self.assertEqual(len(paths), 14)
        svg = self.output / 'figures/07_gnm_clustering_context.svg'
        original = svg.read_bytes()
        self.assertIn(b'G(n,m)', original)
        self.assertIn(b'not realism', original)
        self.assertIn(b'n=2', original)
        self.assertNotIn(b'<dc:date>', original)
        originals = {Path(p).name: Path(p).read_bytes() for p in paths if p.endswith('.svg')}
        render_figures(self.root, self.primary, self.output)
        for name, data in originals.items():
            self.assertEqual((self.output / 'figures' / name).read_bytes(), data)
        self.assertEqual(plt.get_fignums(), before)

    def test_optional_empirical_distributions_use_primary_metrics(self):
        names = ['Galesburg', 'Hi-tech', 'Karate', 'Prison', 'Tailor 1', 'Tailor 2',
                 'Moreno freshmen', 'Moreno high school']
        pd.DataFrame(dict(reference=names, nodes=[31, 36, 34, 67, 39, 39, 31, 70],
                          edges=[63, 91, 78, 142, 158, 223, 218, 274],
                          density=np.linspace(.1, .8, 8), avg_clustering_coef=np.linspace(.2, .9, 8),
                          global_efficiency=np.linspace(.15, .85, 8))).to_csv(
                              self.output / 'empirical_reference_metrics.csv', index=False)
        self.primary['global_efficiency'] = np.where(self.primary.valid_graph, .4, np.nan)
        self.primary.loc[self.primary.valid_graph, 'global_efficiency'] = np.nan
        before = plt.get_fignums()
        paths = render_figures(self.root, self.primary, self.output)
        self.assertEqual(len(paths), 14)
        svg = (self.output / 'figures/08_empirical_context.svg').read_text()
        for text in names + ['31-70', '50 nodes', 'not independent', 'not a realism',
                             'Global efficiency', 'n=0/224', 'Primary CSV']:
            self.assertIn(text, svg)
        self.assertEqual(plt.get_fignums(), before)


if __name__ == '__main__':
    unittest.main()
