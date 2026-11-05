"""Presentation regression: the notebooks people open must show the replacement."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis_figures import publication as pub


class NotebookPresentationTests(unittest.TestCase):
    def test_main_notebooks_use_current_gallery_not_archived_renderer(self):
        for number in ('01', '02', '03', '04'):
            path = next((ROOT / 'analysis').glob(number + '_*.ipynb'))
            notebook = nbformat.read(path, as_version=4)
            source = '\n'.join(cell.source for cell in notebook.cells)
            self.assertIn('analysis_figures.publication', source, path.name)
            self.assertNotIn('from analysis_primary.plots import render_figures', source)
            self.assertNotIn("OUT / 'figures/01_primary_forest.png'", source)
            self.assertNotIn("OUT/'figures'/(stem+'.png')", source)

    def test_current_display_rejects_missing_or_incomplete_manifest(self):
        with tempfile.TemporaryDirectory() as tmp, patch('IPython.display.display') as display:
            root = Path(tmp)
            with self.assertRaises(FileNotFoundError):
                pub.show(root, ['01_primary_contrasts'])
            folder = root / pub.OUTPUT
            folder.mkdir(parents=True)
            for status in ('IN_PROGRESS', 'FAILED_INCOMPLETE', 'COMPLETE'):
                (folder / 'figure_manifest.json').write_text(json.dumps({'status': status}))
                with self.assertRaisesRegex(ValueError, 'Incomplete publication'):
                    pub.show(root, ['01_primary_contrasts'])
            display.assert_not_called()

    def test_current_display_rejects_changed_figure_before_display(self):
        with tempfile.TemporaryDirectory() as tmp, patch('IPython.display.display') as display:
            root = Path(tmp)
            folder = root / pub.OUTPUT
            folder.mkdir(parents=True)
            name = '01_primary_contrasts.png'
            (folder / name).write_bytes(b'changed')
            manifest = dict(status='COMPLETE_PUBLICATION_FIGURES_V2', protected_sha256={}, code_sha256={},
                            artifacts={name: {'sha256': '0' * 64}})
            (folder / 'figure_manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'Figure differs'):
                pub.show(root, ['01_primary_contrasts'])
            display.assert_not_called()


class PublicationDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a, cls.primary, cls.tables, cls.held, cls.protected = pub.load_tables(ROOT)

    def draw(self, name):
        with plt.rc_context(pub.STYLE):
            fig = pub.DRAWERS[name](self.a, self.primary, self.tables, self.held)
        self.addCleanup(plt.close, fig)
        return fig

    def test_full_family_values_intervals_and_raw_blocks_unchanged(self):
        fig = self.draw('01_primary_contrasts')
        points, intervals, suppressed, blocks = [], [], [], []
        for ax in fig.axes:
            for line in ax.lines:
                if line.get_marker() in ('o', 'x'):
                    points.extend(line.get_xdata())
                    suppressed.extend(line.get_xdata() if line.get_marker() == 'x' else [])
                elif line.get_transform() == ax.transData:
                    intervals.append(list(line.get_xdata()))
            for marks in ax.collections:
                blocks.extend(marks.get_offsets()[:, 0])
        original = self.tables['primary_contrasts']
        np.testing.assert_allclose(sorted(points), sorted(original.complete_block_mean), rtol=0, atol=1e-12)
        expected = sorted(original[['ci_lower', 'ci_upper']].dropna().to_numpy().tolist())
        np.testing.assert_allclose(sorted(intervals), expected, rtol=0, atol=1e-12)
        self.assertEqual(len(suppressed), 1)
        self.assertEqual(len(blocks), 186)
        np.testing.assert_allclose(sorted(blocks), sorted(self.tables['paired_schedule_blocks'].difference.dropna()))

    def test_transfer_panels_show_every_model_and_exact_baseline_ranges(self):
        fig = self.draw('08_heldout_distances')
        summary = self.held['source_distance_summary']
        for i, ax in enumerate(fig.axes):
            source, metric = pub.SOURCES[i // 6], pub.METRICS[i % 6]
            rows = summary[summary.source.eq(source) & summary.metric.eq(metric)]
            dots = [float(line.get_xdata()[0]) for line in ax.lines if line.get_marker() in pub.MARKERS]
            np.testing.assert_allclose(sorted(dots), sorted(rows.mean_w1), rtol=0, atol=1e-12)
            ranges = [list(line.get_xdata()) for line in ax.lines if line.get_transform() == ax.transData and len(line.get_xdata()) == 2]
            expected = rows[rows.kind.eq('baseline')][['min_bank_w1', 'max_bank_w1']].to_numpy().tolist()
            np.testing.assert_allclose(sorted(ranges), sorted(expected), rtol=0, atol=1e-12)
            self.assertEqual(ax.get_xlim(), fig.axes[i % 6].get_xlim())

    def test_weighted_ecdf_zeros_ties_and_failure_affected_groups(self):
        x, y = pub.weighted_ecdf([.5, 0, .5], [.25, .5, .25])
        np.testing.assert_allclose(x, [0, 0, .5, .5, 1])
        np.testing.assert_allclose(y, [0, .5, .75, 1, 1])
        candidate = self.held['candidate_metrics']
        group = candidate[candidate.candidate.eq('LLM:global:gpt-5.6-luna')]
        valid, weights = pub.candidate_weights(group, 'density')
        self.assertEqual(len(group), 56)
        self.assertEqual(len(valid), 53)
        for condition in valid.condition.unique():
            self.assertAlmostEqual(float(weights[valid.condition.eq(condition)].sum()), 1 / 7)
        for values, weight in (([], []), ([2], [1]), ([0], [-1]), ([float('nan')], [1]), ([0], [0])):
            with self.assertRaises(ValueError):
                pub.weighted_ecdf(values, weight)

    def test_all_panels_keep_data_and_text_inside_figure(self):
        for name in pub.DRAWERS:
            with self.subTest(figure=name):
                fig = self.draw(name)
                fig.canvas.draw()
                for ax in fig.axes:
                    low, high = sorted(ax.get_xlim())
                    for line in ax.lines:
                        if line.get_transform() == ax.transData:
                            values = np.asarray(line.get_xdata(), float)
                            values = values[np.isfinite(values)]
                            self.assertTrue(((values >= low - 1e-10) & (values <= high + 1e-10)).all(), name)
                    for marks in ax.collections:
                        values = np.asarray(marks.get_offsets()[:, 0], float)
                        values = values[np.isfinite(values)]
                        self.assertTrue(((values >= low - 1e-10) & (values <= high + 1e-10)).all(), name)
                    # Locators may return off-axis ticks that Matplotlib never draws.
                    bottom, top = sorted(ax.get_ylim())
                    xticks = [label for tick, label in zip(ax.get_xticks(), ax.get_xticklabels())
                              if low <= tick <= high]
                    yticks = [label for tick, label in zip(ax.get_yticks(), ax.get_yticklabels())
                              if bottom <= tick <= top]
                    texts = [ax.title, ax._left_title, ax.xaxis.label, ax.yaxis.label,
                             *xticks, *yticks, *ax.texts]
                    for text in texts:
                        if text.get_visible() and text.get_text():
                            box = text.get_window_extent(fig.canvas.get_renderer())
                            self.assertGreaterEqual(box.x0, -1, (name, text.get_text()))
                            self.assertGreaterEqual(box.y0, -1, (name, text.get_text()))
                            self.assertLessEqual(box.x1, fig.bbox.x1 + 1, (name, text.get_text()))
                            self.assertLessEqual(box.y1, fig.bbox.y1 + 1, (name, text.get_text()))
                plt.close(fig)

    def test_renderer_refuses_completed_release_before_reads_or_writes(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(pub, 'load_tables') as loader:
            output = Path(tmp)
            seal = output / 'figure_manifest.json'
            seal.write_text(json.dumps({'status': 'COMPLETE_PUBLICATION_FIGURES_V2'}))
            sentinel = output / '01_primary_contrasts.png'
            sentinel.write_bytes(b'published')
            with self.assertRaisesRegex(ValueError, 'immutable'):
                pub.render(ROOT, output)
            loader.assert_not_called()
            self.assertEqual(sentinel.read_bytes(), b'published')

    def test_protected_output_refused_before_read_or_write(self):
        with patch.object(pub, 'load_tables') as loader:
            for folder in (pub.PRIMARY, pub.HELDOUT, Path('outputs/final_study/new')):
                with self.assertRaisesRegex(ValueError, 'Cannot overwrite'):
                    pub.render(ROOT, ROOT / folder)
            loader.assert_not_called()

    def test_changed_protected_input_rejected(self):
        target = ROOT / 'analysis_primary/core.py'
        original = pub.sha
        with patch.object(pub, 'sha', side_effect=lambda path: '0' * 64 if path == target else original(path)):
            with self.assertRaisesRegex(ValueError, 'Protected input differs'):
                pub.protected_files(ROOT)

    def test_export_formats_determinism_and_figure_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            before = plt.get_fignums()
            paths = pub.render(ROOT, output, names=['01_primary_contrasts'])
            self.assertEqual({p.suffix for p in paths}, {'.png', '.svg', '.pdf'})
            original = (output / '01_primary_contrasts.svg').read_bytes()
            pub.render(ROOT, output, names=['01_primary_contrasts'])
            self.assertEqual(original, (output / '01_primary_contrasts.svg').read_bytes())
            self.assertEqual(plt.get_fignums(), before)
            pub.verify_hashes(self.protected)


if __name__ == '__main__':
    unittest.main()
