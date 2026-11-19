"""Review regressions. Offline only; never modify a released input."""
from pathlib import Path
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from analysis_figures.v6 import data
from analysis_figures import reviewed_v6 as reviewed
from analysis_figures.reviewed_v6 import fig16
from analysis_figures.v6 import release as original

ROOT = Path(__file__).resolve().parents[1]


class FigureReviewFixes(unittest.TestCase):
    def test_sign_count_agrees_with_mean_not_majority(self):
        fig, table = fig16(data.load(ROOT))
        try:
            row = table[(table.attribute == 'religion') & (table['first'] == 'local') &
                        (table.second == 'sequential')].iloc[0]
            self.assertLess(row.mean_difference, 0)
            self.assertEqual(row.cells_same_sign, 12)
            self.assertNotIn('failed Global graph', ' '.join(t.get_text() for t in fig.texts))
        finally:
            plt.close(fig)

    def test_each_sign_count_uses_its_own_mean(self):
        D = data.load(ROOT)
        fig, table = fig16(D)
        try:
            cells = D['ext']['mixing_paired_cells']
            cells = cells[cells.score.eq('excess') & cells.support.eq('score_specific')]
            for row in table.itertuples():
                values = cells[cells.attribute.eq(row.attribute) & cells['first'].eq(row.first) &
                               cells.second.eq(row.second)].difference
                expected = sum(v > 0 if values.mean() > 0 else v < 0 if values.mean() < 0 else v == 0 for v in values)
                self.assertEqual(row.cells_same_sign, expected)
        finally:
            plt.close(fig)

    def test_observed_only_figures_do_not_claim_primary_cohort(self):
        for name in ('10_sampling_effect', '11_conditional_controls', '18_sampling_distortion'):
            self.assertNotIn('Primary cohort only:', reviewed.caption(name))
        self.assertIn('Primary cohort only:', reviewed.caption('16_rq2_paired_mixing'))
        self.assertNotIn('agree only', reviewed.caption('01_primary_contrasts'))

    def test_historical_v2_and_extension_code_mutations_rejected(self):
        sha = original.sha
        for relative in ('outputs/publication_figures/20261105_v2/01_primary_contrasts.png',
                         'analysis_extension/figures.py', 'scripts/check_publication_notebooks.py'):
            target = ROOT / relative
            with self.subTest(path=relative), patch.object(original, 'sha', side_effect=lambda p:
                    '0' * 64 if Path(p) == target else sha(p)):
                with self.assertRaisesRegex(ValueError, 'Historical input differs'):
                    reviewed.verify_history(ROOT)

    def test_extension_generator_imports_from_notebook_directory(self):
        from scripts.check_research_extension import notebook
        nb = notebook()
        setup = next(c.source for c in nb.cells if c.id == 'integrity')
        self.assertLess(setup.index('sys.path.insert(0, str(root))'),
                        setup.index('from analysis_figures.reviewed_v62 import show'))
        self.assertIn('ratios do NOT sum to one', next(c.source for c in nb.cells if c.id == 'matrices-intro'))

    def test_no_overwrite_of_published_or_outside_directory(self):
        for folder in (ROOT / original.OUT, ROOT / 'outside-release'):
            with self.assertRaisesRegex(ValueError, 'NEW directory'):
                reviewed.build(ROOT, folder)

    def test_matrix_cells_are_not_interpolated_and_values_do_not_change(self):
        import pandas as pd
        import numpy as np
        fig, table = reviewed.fig17(data.load(ROOT))
        try:
            self.assertFalse(any(ax.images for ax in fig.axes))
            cells = [mesh for ax in fig.axes for mesh in ax.collections
                     if mesh.get_gid() == 'categorical-mixing-cells']
            self.assertEqual(len(cells), 12)
            self.assertTrue(all(not mesh.get_rasterized() for mesh in cells))
            old = pd.read_csv(ROOT / original.OUT / '17_rq2_mixing_matrices_plotted.csv')
            np.testing.assert_allclose(table.ratio_to_roster_chance, old.ratio_to_roster_chance, atol=1e-12)
        finally:
            plt.close(fig)

    def test_published_svgs_are_font_independent(self):
        for name in reviewed.CAPTIONS:
            text = (ROOT / reviewed.OUT / (name + '.svg')).read_text(encoding='utf8')
            self.assertNotIn('<text', text, name)
            self.assertIn('DejaVuSans-', text, name)


if __name__ == '__main__':
    unittest.main()
