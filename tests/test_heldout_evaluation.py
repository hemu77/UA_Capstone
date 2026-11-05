import gzip
import importlib.util
import io
from pathlib import Path
import sys
import tarfile
import unittest
from unittest.mock import patch
import tempfile
import shutil

import networkx as nx
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from analysis_heldout.data import unpack_gzip, facebook_graphs, school_graph
from analysis_heldout.benchmark import metrics, sample_nodes, degree_control, candidate_weights, conditional_residuals, fit_baselines
from analysis_heldout.figures import render
import analysis_heldout.data as data_module


class HeldoutTests(unittest.TestCase):
    def test_gzip_decompression_bound(self):
        with patch.object(data_module,'MAX_EXPANDED',10):
            with self.assertRaisesRegex(ValueError,'Expanded'):
                unpack_gzip(gzip.compress(b'x'*11))

    def test_development_fit_balances_repeated_tailor_waves(self):
        names=['Galesburg','Hi-tech','Karate','Prison','Tailor 1','Tailor 2','Moreno freshmen','Moreno high school']
        frame=pd.DataFrame(dict(reference=names,density=[.1,.1,.1,.1,.9,.9,.1,.1],avg_clustering_coef=[.4]*8))
        fit=fit_baselines(frame,{'transfer_baselines':{'WS_p_grid':[0,1],'WS_fit_draws':2}})
        self.assertAlmostEqual(fit['density_target'],(6*.1+.9)/7)
        self.assertFalse(fit['fitted_from_heldout'])

    def test_empty_graph_conventions(self):
        result=metrics(nx.empty_graph(50))
        self.assertEqual(result['degree_gini'],0)
        self.assertEqual(result['prop_nodes_lcc'],.02)
        self.assertEqual(result['avg_clustering_coef'],0)
        self.assertEqual(result['global_efficiency'],0)

    def test_node_sampling_keeps_isolates_and_is_deterministic(self):
        graph=nx.empty_graph(80)
        graph.graph.update(source='fixture',cohort='test')
        first=sample_nodes(graph,0)
        self.assertEqual(len(first),50)
        self.assertEqual(first.number_of_edges(),0)
        self.assertEqual(list(first.edges()),list(sample_nodes(graph,0).edges()))

    def test_school_roster_does_not_invent_nonresponse_zeros(self):
        graph=school_graph(b'1 2\n2 1\n1 99\n3 99\n')
        self.assertEqual(set(graph),{1,2,3})
        self.assertEqual(set(graph.edges()),{(1,2)})
        self.assertEqual(graph.graph['dropped_external_nominations'],2)

    def test_public_tar_paths_and_links_rejected_without_extraction(self):
        for name,kind in (('../escape',tarfile.REGTYPE),('facebook/link',tarfile.SYMTYPE)):
            with self.subTest(name=name):
                memory=io.BytesIO()
                with tarfile.open(fileobj=memory,mode='w') as archive:
                    item=tarfile.TarInfo(name)
                    item.type=kind
                    archive.addfile(item)
                with self.assertRaisesRegex(ValueError,'Unsafe'):
                    facebook_graphs(memory.getvalue())

    def test_degree_controls_preserve_degrees_or_report_missing(self):
        graph=nx.cycle_graph(50)
        result,status=degree_control(graph,4)
        self.assertIsNotNone(result)
        self.assertEqual(sorted(dict(result.degree()).values()),[2]*50)
        self.assertEqual(graph.number_of_edges(),50)
        impossible,status=degree_control(nx.complete_graph(50),4)
        self.assertIsNone(impossible)
        self.assertEqual(status,'swap_failed_missing')

    def test_failure_not_zero_and_conditions_have_equal_weight(self):
        rows=[dict(condition=str(c),valid=True,kind='LLM',density=.1) for c in range(7) for _ in range(8)]
        rows[0].update(valid=False,density=np.nan)
        valid,weight=candidate_weights(pd.DataFrame(rows),'density')
        self.assertEqual(len(valid),55)
        for condition in range(7):
            self.assertAlmostEqual(weight[valid.condition.eq(str(condition))].sum(),1/7)
        broken=pd.DataFrame(rows)
        broken.loc[broken.condition.eq('1'),'valid']=False
        with self.assertRaises(ValueError): candidate_weights(broken,'density')

    def test_partial_controls_do_not_overweight_easy_target(self):
        targets=pd.DataFrame([dict(source='s',cohort='c',draw=i,**metrics(nx.cycle_graph(50))) for i in range(2)])
        controls=pd.DataFrame([dict(source='s',cohort='c',control='degree_swap',draw=i,success=not(i==1 and j==7),
            **metrics(nx.cycle_graph(50))) for i in range(2) for j in range(8)])
        result=conditional_residuals(targets,controls)
        self.assertTrue(result.mean_target_minus_control.isna().all())
        self.assertTrue(result.complete_target_draws.eq(1).all())

    def test_rendered_figures_close_and_have_deterministic_svg(self):
        import matplotlib.pyplot as plt
        source=ROOT/'outputs/heldout_evaluation/20261105_v1'
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)
            for name in ('source_distance_summary.csv','per_cohort_distances.csv','dataset_quality.csv',
                         'heldout_sample_metrics.csv','conditional_residuals.csv'):
                shutil.copyfile(source/name,output/name)
            before=plt.get_fignums()
            paths=render(output)
            self.assertEqual(len(paths),12)
            first={p.name:p.read_bytes() for p in paths if p.suffix=='.svg'}
            render(output)
            self.assertEqual(first,{p.name:p.read_bytes() for p in paths if p.suffix=='.svg'})
            self.assertEqual(plt.get_fignums(),before)

    def test_fetch_redirect_cannot_downgrade_or_change_host(self):
        spec=importlib.util.spec_from_file_location('fetch_public',ROOT/'scripts/fetch_heldout_public.py')
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        from urllib.request import Request
        handler=module.SourceRedirects()
        request=Request('https://snap.stanford.edu/data/facebook.tar.gz')
        for target in ('http://snap.stanford.edu/file','https://untrusted.example/file'):
            with self.subTest(target=target), self.assertRaises(ValueError):
                handler.redirect_request(request,None,302,'',{},target)


if __name__=='__main__': unittest.main()
