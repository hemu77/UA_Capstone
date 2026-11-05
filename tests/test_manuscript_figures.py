"""Bounded manuscript rendering checks against copied, sealed CSV fixtures."""
import hashlib
import importlib.util
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
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from analysis_figures.manuscript import render_manuscript


class ManuscriptFigureTests(unittest.TestCase):
    def test_notebook_helper_checks_primary_dependency_hash_before_kernel(self):
        spec=importlib.util.spec_from_file_location('check_manuscript',ROOT/'scripts/check_manuscript_notebook.py')
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        original_digest=module.digest
        target=ROOT/'analysis_primary/core.py'
        with patch.object(module,'digest',side_effect=lambda path:'0'*64 if path==target else original_digest(path)), \
             patch.object(sys,'argv',['check_manuscript','--root',str(ROOT)]), \
             patch.object(module,'NotebookClient') as client, patch.object(module.nbformat,'write'):
            with self.assertRaisesRegex(ValueError,'Protected artifact differs'):
                module.main()
            client.assert_not_called()

    @classmethod
    def setUpClass(cls):
        cls.spec = json.loads((ROOT / 'final_study/contract.json').read_text())
        cls.primary = pd.read_csv(ROOT / 'outputs/analysis_ready/060dcd3a86d5/primary_attempts.csv')

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.sealed = self.root / 'outputs/primary_analysis/20261105'
        self.sealed.mkdir(parents=True)
        names = ['primary_contrasts.csv', 'paired_schedule_blocks.csv', 'leave_one_seed_out.csv',
                 'descriptive_metrics.csv', 'sensitivity.csv']
        artifacts = {}
        for name in names:
            data = (ROOT / 'outputs/primary_analysis/20261105' / name).read_bytes()
            (self.sealed / name).write_bytes(data)
            artifacts[name] = {'sha256': hashlib.sha256(data).hexdigest()}
        self.manifest = {'status': 'COMPLETE_PRIMARY_ANALYSIS_WITH_CAVEATS', 'artifacts': artifacts}
        self.seal = self.sealed / 'evaluation_manifest.json'
        self.seal.write_text(json.dumps(self.manifest))
        self.output = self.root / 'manuscript'
        self.loader = patch('analysis_figures.manuscript.load_inputs',
                            return_value=(self.spec, self.primary.copy(), []))
        self.loader.start()
        self.addCleanup(self.loader.stop)

    def test_nine_files_determinism_and_sealed_bytes(self):
        before = {p.name: p.read_bytes() for p in self.sealed.iterdir()}
        existing = plt.figure()
        self.addCleanup(plt.close, existing)
        paths = render_manuscript(self.root, self.output)
        self.assertEqual(len(set(paths)), 9)
        self.assertTrue(all(isinstance(p, Path) and p.is_absolute() and p.stat().st_size > 1000 for p in paths))
        self.assertEqual({p.suffix for p in paths}, {'.png', '.svg', '.pdf'})
        self.assertEqual(plt.get_fignums(), [existing.number])
        originals = {p.name: p.read_bytes() for p in paths if p.suffix == '.svg'}
        second = render_manuscript(self.root, self.root / 'another_workspace')
        for path in second:
            if path.suffix == '.svg':
                self.assertEqual(originals[path.name], path.read_bytes())
                self.assertNotIn(b'<dc:date>', path.read_bytes())
        forest = (self.output / '01_primary_contrasts.svg').read_text()
        for label in ('Global', 'Local', 'Sequential', 'Iterative', 'no CI', 'n=5', 'US-English'):
            self.assertIn(label, forest)
        robust = (self.output / '02_iterative_hindi_robustness.svg').read_text()
        for label in ('44000', '44007', 'Omitted schedule seed'):
            self.assertIn(label, robust)
        means = (self.output / '03_model_endpoint_means.svg').read_text()
        for model in self.spec['analysis']['models']:
            self.assertIn(model, means)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.sealed.iterdir()})

    def test_missing_incomplete_and_changed_seals_rejected(self):
        self.seal.unlink()
        with self.assertRaises(FileNotFoundError):
            render_manuscript(self.root, self.output)
        self.manifest['status'] = 'IN_PROGRESS'
        self.seal.write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError, 'COMPLETE_PRIMARY_ANALYSIS_WITH_CAVEATS'):
            render_manuscript(self.root, self.output)
        self.manifest['status'] = 'COMPLETE_PRIMARY_ANALYSIS_WITH_CAVEATS'
        self.seal.write_text(json.dumps(self.manifest))
        (self.sealed / 'primary_contrasts.csv').write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            render_manuscript(self.root, self.output)
        self.assertFalse(self.output.exists())

    def test_supplement_and_protected_output_refused(self):
        primary = self.primary.copy()
        primary.loc[0, 'cohort'] = 'supplement'
        with patch('analysis_figures.manuscript.load_inputs', return_value=(self.spec, primary, [])):
            with self.assertRaisesRegex(ValueError, 'Supplemental or mixed'):
                render_manuscript(self.root, self.output)
        with self.assertRaisesRegex(ValueError, 'protected'):
            render_manuscript(self.root, self.sealed / 'figures')
        self.assertFalse(self.output.exists())

    def test_save_error_closes_only_owned_figure(self):
        existing = plt.figure()
        self.addCleanup(plt.close, existing)
        with patch('matplotlib.figure.Figure.savefig', side_effect=OSError('disk full')):
            with self.assertRaisesRegex(OSError, 'disk full'):
                render_manuscript(self.root, self.output)
        self.assertEqual(plt.get_fignums(), [existing.number])

    def test_panel_scales_and_loo_marker_counts(self):
        figures = []
        def capture(figure, *args, **kwargs):
            if figure not in figures:
                figures.append(figure)
        with patch('matplotlib.figure.Figure.savefig', autospec=True, side_effect=capture):
            render_manuscript(self.root, self.output)
        forest, robustness, means = figures
        self.assertEqual(forest.axes[1].get_xlim(), forest.axes[2].get_xlim())
        self.assertEqual(forest.axes[1].get_xlim(), forest.axes[3].get_xlim())
        raw_count = sum(len(group.get_offsets()) for ax in forest.axes for group in ax.collections)
        blocks = pd.read_csv(self.sealed / 'paired_schedule_blocks.csv')
        self.assertEqual(raw_count, int(blocks.complete.sum()))
        for ax in forest.axes:
            self.assertEqual(ax.lines[0].get_linestyle(), '--')
            for group in ax.collections:
                values = group.get_offsets()[:, 0]
                self.assertTrue(((values >= ax.get_xlim()[0]) & (values <= ax.get_xlim()[1])).all())
        suppressed = [line for line in forest.axes[0].lines if line.get_marker() == 'x']
        self.assertEqual(len(suppressed), 1)
        self.assertEqual(list(suppressed[0].get_ydata()), [4])
        loo_points = [line for line in robustness.axes[1].lines if line.get_marker() == 'o']
        self.assertEqual(len(loo_points), 8)
        self.assertEqual(sum(line.get_markerfacecolor() != 'white' for line in loo_points), 3)
        self.assertEqual(len(robustness.axes[1].collections), 0)
        self.assertEqual(means.axes[1].get_xlim(), means.axes[3].get_xlim())
        self.assertEqual(len(means.legends), 1)

    def test_notebook_renders_only_to_live_temporary_workspace(self):
        notebook = nbformat.read(ROOT / 'analysis/04_manuscript_figures.ipynb', as_version=4)
        code = next(cell.source for cell in notebook.cells if cell.id == 'render-temporary')
        (self.root / 'analysis_figures').mkdir()
        (self.root / 'analysis_figures/manuscript.py').touch()
        namespace = {}
        with patch('sys.path', list(sys.path)), patch('pathlib.Path.cwd', return_value=self.root), \
                patch('analysis_figures.manuscript.render_manuscript', return_value=[]) as renderer:
            exec(code, namespace)
        self.addCleanup(namespace['manuscript_workspace'].cleanup)
        renderer.assert_called_once_with(self.root, namespace['TEMP_OUTPUT'])
        self.assertTrue(namespace['TEMP_OUTPUT'].is_dir())
        self.assertFalse(namespace['TEMP_OUTPUT'].is_relative_to(self.root))
        self.assertFalse((self.root / 'outputs/manuscript_figures').exists())

    def test_notebook_schema_code_and_outputs(self):
        path = ROOT / 'analysis/04_manuscript_figures.ipynb'
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        self.assertEqual(notebook.metadata.kernelspec.name, 'ua-capstone-final060d')
        for cell in notebook.cells:
            if cell.cell_type == 'code':
                count = cell.execution_count
                self.assertTrue(count is None or (type(count) is int and count > 0))
                self.assertFalse(any(output.output_type == 'error' for output in cell.outputs))
                compile(cell.source, str(path), 'exec')


if __name__ == '__main__':
    unittest.main()
