"""Post-hoc structural diagnostics only; no acquisition, fit, or provider calls."""
import json
import random

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from analysis_heldout.benchmark import candidate_weights, generate_baseline, sample_nodes, seed
from analysis_heldout.data import load_public
from analysis_primary.graph_evaluation import _read_graph
from analysis_figures import publication as v2

METRICS = ('avg_clustering_coef','global_efficiency','prop_nodes_lcc')
SIZES = (25,40,50)


def nested_metrics(graph, key):
    if len(graph)!=50:
        raise ValueError('Expected an original 50-node graph')
    order = random.Random(seed('v3-induced-sensitivity', *key)).sample(sorted(graph), 50)
    for n in SIZES:
        g = graph if n==50 else graph.subgraph(order[:n]).copy()
        yield n, dict(avg_clustering_coef=nx.average_clustering(g),
                      global_efficiency=nx.global_efficiency(g),
                      prop_nodes_lcc=max(map(len,nx.connected_components(g)))/n)


def evaluate(root, cache, primary, held):
    slots = {n: held['candidate_metrics'].copy() for n in SIZES}
    targets = {n: held['heldout_sample_metrics'].copy() for n in SIZES}
    fit = json.loads((root/v2.HELDOUT/'development_fit.json').read_text())
    records = {r.original_run_id:r for r in primary.itertuples()}
    def store(frames, i, graph, key):
        for n, values in nested_metrics(graph,key):
            for metric, value in values.items():
                if n==50 and not np.isclose(value,frames[n].at[i,metric],rtol=0,atol=1e-12):
                    raise ValueError('Reconstructed 50-node metric differs: '+str(key))
                frames[n].at[i,metric] = value
    for r in held['candidate_metrics'].itertuples():
        if not r.valid:
            continue
        if r.kind=='LLM':
            graph = _read_graph(root,records[r.run_id])
            key = (r.run_id,)
        else:
            procedure = r.candidate.split(':')[1]
            graph = generate_baseline(procedure,fit,r.bank,r.draw)
            key = (procedure,r.bank,r.draw)
        store(slots,r.Index,graph,key)
    graphs = {(g.graph['source'],g.graph['cohort']):g for g in load_public(cache)}
    for r in held['heldout_sample_metrics'].itertuples():
        store(targets,r.Index,sample_nodes(graphs[r.source,r.cohort],r.draw),(r.source,r.cohort,r.draw))
    reference = held['per_cohort_distances'].set_index(['source','cohort','candidate','bank','metric']).w1
    rows=[]
    for n in SIZES:
        for (cand,bank), group in slots[n].groupby(['candidate','bank']):
            for metric in METRICS:
                valid, weights = candidate_weights(group,metric)
                for (src,cohort),t in targets[n].groupby(['source','cohort']):
                    w=wasserstein_distance(valid[metric],t[metric],u_weights=weights)
                    if n==50 and not np.isclose(w, reference.loc[src,cohort,cand,bank,metric],rtol=0,atol=1e-12):
                        raise ValueError('Original W1 not reproduced')
                    rows.append((n,'original weighting',src,cohort,cand,bank,metric,w))
                    if n==50 and group.kind.iloc[0]=='LLM':
                        sub=valid[valid.condition.eq('us:english')]
                        if sub.empty:
                            raise ValueError('Empty US-English group')
                        rows.append((n,'US-English only',src,cohort,cand,bank,metric,
                                     wasserstein_distance(sub[metric],t[metric])))
    return pd.DataFrame(rows,columns=['n','scope','source','cohort','candidate','bank','metric','w1'])
