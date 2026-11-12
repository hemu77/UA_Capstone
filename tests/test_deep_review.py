import unittest
from pathlib import Path
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from analysis_figures import deep_review as v

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/v.OUT

class ReviewTests(unittest.TestCase):
    def test_efficiency_definition(self):
        for g in (nx.empty_graph(5),nx.complete_graph(5),nx.disjoint_union(nx.path_graph(3),nx.empty_graph(2))):
            self.assertAlmostEqual(v.efficiency(nx.to_numpy_array(g,dtype=bool)),nx.global_efficiency(g))

    def test_all_banks_and_gap(self):
        sims=pd.read_csv(DATA/'efficiency_simulations.csv'); banks=pd.read_csv(DATA/'efficiency_banks.csv').set_index(['source','cohort','bank'])
        targets=pd.read_csv(ROOT/'outputs/heldout_evaluation/20261105_v1/heldout_sample_metrics.csv')
        self.assertEqual((len(sims),len(banks)),(4928,88))
        for key,sim in sims.groupby(['source','cohort','bank']):
            obs=targets[(targets.source==key[0])&(targets.cohort==key[1])]
            self.assertEqual((sim.draw.nunique(),len(obs)),(56,32))
            self.assertAlmostEqual(wasserstein_distance(sim[v.EFF],obs[v.EFF]),banks.loc[key,'target_ER'])
        per=pd.read_csv(DATA/'efficiency_records.csv').set_index(['source','cohort'])
        np.testing.assert_allclose(per.target_ER,banks.target_ER.groupby(level=[0,1]).mean().loc[per.index])
        result=v.summarize(per.reset_index()).iloc[-1]
        self.assertAlmostEqual(result.ws_gap_closed,.86265,places=4)
        self.assertEqual((result.target_strictly_closer,result.target_closer_or_within_015),(5,6))

    def test_ranks_and_margins(self):
        r=pd.read_csv(DATA/'attribute_ranks.csv')
        self.assertEqual((len(r),r.coleman.eq(v.ATTR[0]).sum(),r.excess.eq(v.ATTR[0]).sum()),(112,90,85))
        switched=r[r.excess!=r.coleman]
        self.assertEqual((len(switched),set(switched.method)),(5,{'global'}))
        cells=pd.read_csv(DATA/'attribute_cells.csv')
        self.assertEqual((len(cells),cells.graphs.sum()),(336,878*3))
        m=pd.read_csv(DATA/'clustering_margins.csv')
        for n,first,second,delta in [(40,'WS','BA',.003714),(25,'ER','global',.017139)]:
            row=m[(m.n==n)&(m.closest==first)&(m.runner_up==second)].iloc[0]
            self.assertAlmostEqual(row.margin,delta,places=6)
