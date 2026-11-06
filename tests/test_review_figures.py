from pathlib import Path
import sys
import unittest
import json
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis_figures import publication_v3 as figures
from analysis_figures import publication as v2
from analysis_figures.review_sensitivity import nested_metrics, evaluate
from scripts.review_publication import build, inputs


class ReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a,cls.primary,cls.tables,cls.held,_=figures.load_tables(ROOT)

    def test_fragility_is_on_primary_figure(self):
        fig = figures.primary_contrasts(self.a,self.primary,self.tables,self.held)
        self.addCleanup(plt.close, fig)
        self.assertTrue(any('3/8' in t.get_text() for t in fig.axes[-1].texts),
                        'Iterative panel omits leave-one-block-out sensitivity.')

    def test_record_ranges_and_means(self):
        frame=self.held['per_cohort_distances']
        _,scores=figures.cohort_scores(frame)
        self.assertEqual(len(scores),11*7*6)
        fig=figures.distances(self.a,self.primary,self.tables,self.held)
        self.addCleanup(plt.close,fig)
        self.assertEqual(len(fig.axes),5)
        self.assertNotIn('Largest component',[a.get_title(loc='left') for a in fig.axes])
        for ax,metric in zip(fig.axes,v2.METRICS):
            for j,proc in enumerate(figures.PROCEDURES):
                source=frame[(frame.metric==metric)&frame.candidate.str.split(':').str[1].eq(proc)]
                fb=source[source.source.eq(v2.SOURCES[0])].groupby('cohort').w1.mean()
                school=source[source.source.eq(v2.SOURCES[1])].w1.mean()
                lines=ax.lines[3*j:3*j+3]
                np.testing.assert_allclose(lines[0].get_xdata(),[fb.min(),fb.max()])
                np.testing.assert_allclose(lines[1].get_xdata(),[fb.mean()])
                np.testing.assert_allclose(lines[2].get_xdata(),[school])
        with self.assertRaisesRegex(ValueError,'Incomplete candidate banks'):
            figures.cohort_scores(frame.drop(frame[frame.kind.eq('baseline')].index[0]))
        bad=frame.copy()
        bad.loc[0,'w1']=np.nan
        with self.assertRaisesRegex(ValueError,'Undefined'):
            figures.cohort_scores(bad)

    def test_wrong_original_size_metric_stops_sensitivity(self):
        bad=self.held['candidate_metrics'].copy()
        first=bad[bad.valid].index[0]
        bad.loc[first,figures.M]+=.1
        with self.assertRaisesRegex(ValueError,'Reconstructed 50-node metric differs'):
            evaluate(ROOT,None,self.primary,dict(self.held,candidate_metrics=bad))

    def test_residuals_use_all_paired_draws(self):
        draws=figures.residual_draws(self.held)
        self.assertEqual(len(draws),11*32*2)
        c=self.held['conditional_control_metrics']
        t=self.held['heldout_sample_metrics']
        for metric in (figures.M,'transitivity'):
            for r in draws.itertuples():
                ref=c[(c.cohort==r.cohort)&(c.draw==r.draw)&(c.control==r.control)][metric]
                target=t[(t.cohort==r.cohort)&(t.draw==r.draw)][metric].item()
                self.assertAlmostEqual(getattr(r,metric),target-ref.mean(),places=12)
        altered=c.copy()
        altered.loc[0,figures.M]=np.nan
        with self.assertRaisesRegex(ValueError,'Incomplete conditional controls'):
            figures.residual_draws(dict(self.held,conditional_control_metrics=altered))

    def test_paired_thinning_keeps_isolates_and_empty_graphs(self):
        g=nx.empty_graph(50)
        values=dict(nested_metrics(g,('empty',)))
        self.assertEqual(set(values),{25,40,50})
        for n,row in values.items():
            self.assertEqual(row[figures.M],0)
            self.assertEqual(row['global_efficiency'],0)
            self.assertEqual(row['prop_nodes_lcc'],1/n)
        graph=nx.path_graph(50)
        self.assertEqual(dict(nested_metrics(graph,('repeat',))),dict(nested_metrics(graph,('repeat',))))

    def test_distribution_panel_composition(self):
        fig=figures.distributions(self.a,self.primary,self.tables,self.held)
        self.addCleanup(plt.close,fig)
        self.assertEqual([len(a.lines) for a in fig.axes],[6,6,5,5])
        self.assertLess(fig.axes[0].get_xlim()[1],.5)
        for ax in fig.axes:
            for line in ax.lines:
                self.assertAlmostEqual(line.get_ydata()[-1],1)
                self.assertTrue(np.all(np.diff(line.get_ydata())>=-1e-12))
            self.assertEqual([line.get_label() for line in ax.lines[:2]],['Facebook','School'])
        fb=self.held['heldout_sample_metrics'].query("source == 'snap_facebook'")
        cutoff=.1
        x,y=fig.axes[0].lines[0].get_data()
        self.assertAlmostEqual(y[np.searchsorted(x,cutoff,side='right')-1],
                               fb.groupby('cohort').density.apply(lambda g:g.le(cutoff).mean()).mean())
        gen=self.held['candidate_metrics'].query("kind == 'LLM'")
        gen=gen[gen.candidate.str.startswith('LLM:global:') & gen.valid]
        expected=gen.assign(under=gen.density.le(cutoff)).groupby(['candidate','condition']).under.mean().mean()
        x,y=fig.axes[0].lines[2].get_data()
        self.assertAlmostEqual(y[np.searchsorted(x,cutoff,side='right')-1],expected)

    def test_shipped_sizes_and_complete_exports(self):
        folder=ROOT/figures.OUTPUT
        manifest=json.loads((folder/'figure_manifest.json').read_text())
        self.assertEqual(manifest['inputs'],inputs(ROOT)[0])
        self.assertEqual(len(manifest['artifacts']),13*3+2)
        for name in figures.DRAWERS:
            self.assertTrue(ET.parse(folder/(name+'.svg')).getroot().tag.endswith('}svg'))
            self.assertTrue((folder/(name+'.pdf')).read_bytes().startswith(b'%PDF'))
            if name in figures.KEEP:
                for suffix in ('.png','.pdf','.svg'):
                    self.assertEqual((folder/(name+suffix)).read_bytes(),(ROOT/v2.OUTPUT/(name+suffix)).read_bytes())
        raw=pd.read_csv(folder/'sensitivity_bank_distances.csv')
        self.assertEqual(len(raw),4488)
        base=raw[(raw.n==50) & raw.scope.eq('original weighting')]
        on=['source','cohort','candidate','bank','metric']
        original=self.held['per_cohort_distances']
        ref=original[original.metric.isin(base.metric.unique())]
        self.assertEqual(len(base),len(ref))
        joined=base.merge(ref,on=on,validate='one_to_one')
        self.assertEqual(len(joined),len(ref))
        np.testing.assert_allclose(joined.w1_x,joined.w1_y,rtol=0,atol=1e-12)
        summaries=pd.read_csv(folder/'sensitivity_source_summary.csv')
        self.assertEqual(len(summaries),168)
        self.assertTrue((summaries['count']==np.where(summaries.source.eq(v2.SOURCES[0]),10,1)).all())
        subset=summaries[summaries.metric.eq(figures.M)&summaries.scope.eq('original weighting')]
        winners=subset.loc[subset.groupby(['n','source'])['mean'].idxmin()].set_index(['n','source']).procedure
        self.assertEqual([winners[n,s] for n in [25,40,50] for s in v2.SOURCES],
                         ['BA','ER','WS','BA','WS','BA'])
        with self.assertRaises(FileExistsError):
            build(ROOT,folder,None)
