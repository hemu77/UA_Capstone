"""Focused checks for figure release v6: cohort guards, seals and display arithmetic."""
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from analysis_figures.v6 import data, release, style
from analysis_figures.v6.primary import _fmt, _p

ROOT = Path(__file__).resolve().parents[1]


class FigureReleaseV6(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.D = data.load(ROOT)
        cls.manifest = release.verify(ROOT)

    def test_release_is_exactly_the_19_current_figures(self):
        self.assertEqual(len(release.figures()), 19)
        self.assertEqual(sorted(release.CAPTIONS), sorted(release.figures()))
        self.assertEqual(len(self.manifest['artifacts']), 19 * 4)
        self.assertTrue(set(release.MAIN) <= set(release.CAPTIONS))

    def test_manifest_records_primary_cohort_and_no_generation(self):
        m = self.manifest
        self.assertEqual([m['primary_attempts'], m['primary_valid'], m['primary_empty'], m['primary_failed']],
                         [896, 890, 12, 6])
        self.assertEqual([m['supplemental_graphs'], m['paid_calls'], m['ledger_queries'], m['new_generation'],
                          m['new_hypothesis_tests'], m['refitted_baselines']], [0] * 6)

    def test_supplement_ids_are_refused(self):
        primary = pd.read_csv(ROOT / data.PRIMARY_IDS)
        supplement = pd.read_csv(ROOT / data.SUPPLEMENT_IDS)
        data.assert_primary_only(ROOT, primary.run_id)
        with self.assertRaises(ValueError):
            data.assert_primary_only(ROOT, [*primary.run_id[:-1], supplement.run_id.iloc[0]])
        with self.assertRaises(ValueError):
            data.assert_primary_only(ROOT, primary.run_id[:-1])

    def test_no_plotted_table_contains_a_supplement(self):
        banned = set(pd.read_csv(ROOT / data.SUPPLEMENT_IDS).run_id)
        for name in release.CAPTIONS:
            text = (ROOT / release.OUT / (name + '_plotted.csv')).read_text(encoding='utf8')
            self.assertFalse(any(run in text for run in banned), name)

    def test_attempt_map_counts(self):
        table = pd.read_csv(ROOT / release.OUT / '04_coverage_missingness_plotted.csv')
        self.assertEqual(table.outcome.value_counts().to_dict(), {'nonempty': 878, 'empty': 12, 'failed': 6})

    def test_forest_rows_equal_archived_contrasts(self):
        plotted = pd.read_csv(ROOT / release.OUT / '01_primary_contrasts_plotted.csv').set_index('figure_row')
        archived = self.D['tables']['primary_contrasts'].set_index('hypothesis_id')
        self.assertEqual(len(plotted), 24)
        for a, b in (('mean', 'complete_block_mean'), ('ci_lower', 'ci_lower'), ('holm_p', 'holm_p')):
            np.testing.assert_allclose(plotted[a], archived.loc[plotted.index, b], rtol=0, atol=1e-12)
        self.assertEqual(list(plotted.index[plotted.holm_p < .05]), ['iterative:us:hindi'])

    def test_sign_flip_floor_exceeds_first_holm_threshold(self):
        table = pd.read_csv(ROOT / release.OUT / '02_primary_robustness_plotted.csv')
        self.assertTrue(np.allclose(table.sign_flip_raw_p, 2 / 256))
        self.assertTrue((table.sign_flip_raw_p > table.first_holm_threshold).all())
        self.assertTrue(np.allclose(table.sign_flip_holm_p, 24 * 2 / 256))
        self.assertEqual(int((table.loo_holm_p < .05).sum()), 3)

    def test_mixing_ratio_is_fraction_over_roster_chance(self):
        table = pd.read_csv(ROOT / release.OUT / '17_rq2_mixing_matrices_plotted.csv')
        np.testing.assert_allclose(table.ratio_to_roster_chance,
                                   table.mean_oriented_edge_fraction / table.roster_chance_fraction, rtol=1e-12)
        sums = table.groupby(['attribute', 'method'])[['mean_oriented_edge_fraction', 'roster_chance_fraction']].sum()
        np.testing.assert_allclose(sums, 1, atol=1e-9)
        row = table[(table.attribute == 'religion') & (table.row_category == table.column_category)].iloc[0]
        self.assertAlmostEqual(row.roster_chance_fraction, 10 * 9 / (50 * 49))

    def test_politics_first_counts(self):
        table = pd.read_csv(ROOT / release.OUT / '14_demographic_mixing_plotted.csv')
        b = table[table.panel == 'b'].set_index(['method', 'row']).politics_first_cells
        self.assertEqual([b[('global', s)] for s in ('excess', 'coleman', 'newman')], [1, 6, 6])
        self.assertTrue((b.drop('global', level=0) == 28).all())

    def test_diagnostic_is_a_separate_tier(self):
        table = pd.read_csv(ROOT / release.OUT / '19_copenhagen_transfer_plotted.csv')
        diagnostic = table[table.tier != 'transfer']
        self.assertEqual(len(diagnostic), 6)
        self.assertEqual(set(diagnostic.procedure), {'ER given target density'})
        self.assertEqual(len(table[table.tier == 'transfer']), 42)

    def test_weighted_quantile_and_number_format(self):
        np.testing.assert_allclose(style.weighted_quantile([3, 1, 2], [1, 1, 1], [.5]), [2])
        np.testing.assert_allclose(style.weighted_quantile([1, 10], [3, 1], [.5, .9]), [1, 10])
        with self.assertRaises(ValueError):
            style.weighted_quantile([1, 2], [1, -1], .5)
        self.assertEqual(_fmt(-0.00004), '0.000')
        self.assertEqual(_fmt(-0.0453), '−0.045')
        self.assertEqual(_p(1.0), '1')
        self.assertEqual(_p(float('nan')), 'n/a')

    def test_build_refuses_existing_or_outside_destination(self):
        with self.assertRaises(ValueError):
            release.build(ROOT, ROOT / release.OUT)
        with self.assertRaises(ValueError):
            release.build(ROOT, ROOT / 'outputs/somewhere_else')

    def test_historical_manifests_unchanged(self):
        extension = ROOT / data.EXTENSION / 'manifest.json'
        self.assertEqual(data.sha(extension), data.EXTENSION_MANIFEST_SHA256)
        self.assertEqual(json.loads(extension.read_text())['supplemental_graphs'], 0)


if __name__ == '__main__':
    unittest.main()
