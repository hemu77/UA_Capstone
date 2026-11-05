"""Fixed-protocol structural transfer against evaluation-held-out public graphs."""
import hashlib
import json
import random
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from analysis_primary.core import load_inputs, sha, write_json
from analysis_primary.graph_evaluation import _read_graph
from analysis_heldout.data import PROTOCOL_SHA256, SOURCES, load_public

CLARIFICATIONS_SHA256 = '90a73e2a35b18c763e04143a48dd18928be80dfd70bb99ad1789797bbedb797a'
METRICS = ('density','avg_clustering_coef','global_efficiency','transitivity','degree_gini','prop_nodes_lcc')
VERSION = 'heldout-structural-transfer-20261105-v1'


def seed(*keys):
    return int.from_bytes(hashlib.sha256(json.dumps([VERSION,*keys]).encode()).digest()[:8],'big')


def metrics(graph):
    n=len(graph)
    if n != 50 or nx.is_directed(graph) or nx.number_of_selfloops(graph):
        raise ValueError('Require simple undirected 50-node benchmark graphs')
    degree=np.sort(np.array([d for _,d in graph.degree()],dtype=float))
    gini=float(np.sum((2*np.arange(1,n+1)-n-1)*degree)/(n*degree.sum())) if degree.sum() else 0.
    result=dict(density=nx.density(graph),avg_clustering_coef=nx.average_clustering(graph),
        global_efficiency=nx.global_efficiency(graph),transitivity=nx.transitivity(graph),degree_gini=gini,
        prop_nodes_lcc=max(map(len,nx.connected_components(graph)))/n)
    if not all(np.isfinite(x) and 0<=x<=1 for x in result.values()):
        raise ValueError('Invalid bounded benchmark metric')
    return result


def sample_nodes(graph,draw):
    if len(graph)<50: raise ValueError('Cohort smaller than protocol size')
    nodes=random.Random(seed('node-sample',graph.graph['source'],graph.graph['cohort'],draw)).sample(sorted(graph),50)
    result=nx.Graph()
    result.add_nodes_from(sorted(nodes))
    result.add_edges_from(graph.subgraph(nodes).edges())
    return nx.convert_node_labels_to_integers(result,ordering='sorted')


def degree_control(graph,random_seed):
    clone=graph.copy()
    m=clone.number_of_edges()
    if m<2: return clone,'exact_degenerate'
    before=sorted(dict(clone.degree()).values())
    try:
        nx.double_edge_swap(clone,nswap=10*m,max_tries=100*m,seed=random_seed)
    except (nx.NetworkXAlgorithmError,nx.NetworkXError):
        return None,'swap_failed_missing'
    if (set(clone)!=set(graph) or clone.number_of_edges()!=m or nx.number_of_selfloops(clone)
            or sorted(dict(clone.degree()).values())!=before):
        raise ValueError('Degree control violated exact graph invariants')
    return clone,'finite_swaps_not_uniformity_proof'


def fit_baselines(development,protocol):
    frame=development.copy()
    frame['community']=frame.reference.replace({'Tailor 1':'Tailor','Tailor 2':'Tailor'})
    community=frame.groupby('community')[['density','avg_clustering_coef']].mean()
    if len(frame)!=8 or len(community)!=7: raise ValueError('Development panel membership differs')
    density=float(community.density.mean())
    clustering=float(community.avg_clustering_coef.mean())
    m=min(range(1,25),key=lambda value:(abs(value*(50-value)/1225-density),value))
    k=min(range(2,49,2),key=lambda value:(abs(value/49-density),value))
    grid=[]
    for p in protocol['transfer_baselines']['WS_p_grid']:
        values=[nx.average_clustering(nx.watts_strogatz_graph(50,k,p,
            seed=seed('WS-fit',p,draw))) for draw in range(protocol['transfer_baselines']['WS_fit_draws'])]
        mean=float(np.mean(values))
        grid.append(dict(p=p,mean_clustering=mean,squared_error=(mean-clustering)**2))
    best=min(grid,key=lambda row:(row['squared_error'],row['p']))
    return dict(development_records=8,development_communities=7,density_target=density,
        clustering_target=clustering,ER_p=density,BA_m=m,WS_k=k,WS_p=best['p'],WS_grid=grid,
        fitted_from_heldout=False,fit_scope='Legacy development panel only; Tailor waves averaged first, then seven equal community weights.')


def generate_baseline(name,fit,bank,draw):
    random_seed=seed('transfer-generator',name,bank,draw)
    if name=='ER': return nx.erdos_renyi_graph(50,fit['ER_p'],seed=random_seed)
    if name=='BA': return nx.barabasi_albert_graph(50,fit['BA_m'],seed=random_seed)
    if name=='WS': return nx.watts_strogatz_graph(50,fit['WS_k'],fit['WS_p'],seed=random_seed)
    raise ValueError('Unknown development-calibrated baseline')


def candidate_weights(frame,metric):
    valid=frame[frame.valid & frame[metric].notna()].copy()
    if valid.empty: raise ValueError('No defined candidate outcome')
    if frame.kind.iloc[0]=='LLM':
        conditions=set(frame.condition)
        counts=valid.condition.value_counts()
        if len(conditions)!=7 or set(counts.index)!=conditions:
            raise ValueError('Missing complete instruction condition; no zero fill or silent reweighting')
        weights=valid.condition.map(lambda value:1/(7*counts[value])).to_numpy(float)
    else:
        weights=np.full(len(valid),1/len(valid))
    return valid,weights


def transfer_distances(generated,targets):
    rows=[]
    for (source,cohort),target in targets.groupby(['source','cohort'],sort=True):
        for (candidate,kind,bank),group in generated.groupby(['candidate','kind','bank'],sort=True):
            for metric in METRICS:
                valid,weights=candidate_weights(group,metric)
                rows.append(dict(source=source,cohort=cohort,candidate=candidate,kind=kind,bank=int(bank),metric=metric,
                    w1=float(wasserstein_distance(valid[metric],target[metric],u_weights=weights)),
                    candidate_attempts=len(group),candidate_defined=len(valid),target_node_draws=len(target)))
    distances=pd.DataFrame(rows)
    bank_means=distances.groupby(['source','candidate','kind','metric','bank'],sort=True).w1.mean().reset_index()
    summary=bank_means.groupby(['source','candidate','kind','metric'],sort=True).agg(
        mean_w1=('w1','mean'),min_bank_w1=('w1','min'),max_bank_w1=('w1','max'),banks=('bank','size')).reset_index()
    return distances,summary


def conditional_residuals(targets,controls):
    rows=[]
    for (source,cohort,control),group in controls.groupby(['source','cohort','control'],sort=True):
        target=targets[(targets.source==source)&(targets.cohort==cohort)].set_index('draw')
        for metric in METRICS:
            differences=[]
            for draw,replicates in group.groupby('draw',sort=True):
                if len(replicates)==8 and replicates.success.all() and replicates[metric].notna().all():
                    differences.append(float(target.loc[draw,metric]-replicates[metric].mean()))
            complete=len(differences)==len(target)
            rows.append(dict(source=source,cohort=cohort,control=control,metric=metric,
                target_draws=len(target),complete_target_draws=len(differences),
                failed_control_draws=int((~group.success).sum()),
                mean_target_minus_control=float(np.mean(differences)) if complete else None,
                status='complete_descriptive' if complete else 'missing_incomplete_controls'))
    return pd.DataFrame(rows)


def validate_previous(root):
    base=root/'outputs/primary_analysis/20261105'
    seal=base/'evaluation_manifest.json'
    manifest=json.loads(seal.read_text())
    if manifest['status']!='COMPLETE_PRIMARY_ANALYSIS_WITH_CAVEATS': raise ValueError('Primary analysis not complete')
    protected={seal.relative_to(root).as_posix():sha(seal)}
    for name,info in manifest['artifacts'].items():
        path=base/name
        if sha(path)!=info['sha256']: raise ValueError('Original derived artifact differs: '+name)
        protected[path.relative_to(root).as_posix()]=info['sha256']
    for name,expected in {**manifest['code_sha256'],**manifest['preserved_original_artifacts']}.items():
        if sha(root/name)!=expected: raise ValueError('Original code/data differs: '+name)
        protected[name]=expected
    return protected


def run_benchmark(root,cache,output):
    protocol_path=root/'analysis_heldout/protocol.json'
    if sha(protocol_path)!=PROTOCOL_SHA256 or sha(root/'analysis_heldout/clarifications.json')!=CLARIFICATIONS_SHA256:
        raise ValueError('Protocol or review clarification changed; do not tune on evaluation outcomes')
    protocol=json.loads(protocol_path.read_text())
    protected=validate_previous(root)
    spec,primary,_=load_inputs(root)
    # Fit and save parameters before parsing new public graphs.
    fit=fit_baselines(pd.read_csv(root/'outputs/primary_analysis/20261105/empirical_reference_metrics.csv'),protocol)
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/'development_fit.json',fit)
    graphs=load_public(cache)
    quality=[]
    overlap=[]
    for graph in graphs:
        quality.append(dict(**graph.graph,observed_roster_nodes=len(graph),observed_edges=graph.number_of_edges(),
            observed_density=nx.density(graph),observed_clustering=nx.average_clustering(graph),
            observed_isolates=nx.number_of_isolates(graph),source_sha256=SOURCES[graph.graph['source']]['sha256']))
    for i,left in enumerate(graphs):
        for right in graphs[i+1:]:
            if left.graph['source']==right.graph['source']:
                overlap.append(dict(source=left.graph['source'],left=left.graph['cohort'],right=right.graph['cohort'],
                    shared_nodes=len(set(left)&set(right)),shared_edges=len(set(map(frozenset,left.edges()))&set(map(frozenset,right.edges())))))
    pd.DataFrame(quality).to_csv(output/'dataset_quality.csv',index=False)
    pd.DataFrame(overlap).to_csv(output/'cohort_overlap.csv',index=False)
    generated=[]
    for row in primary.itertuples(index=False):
        item=dict(candidate=f'LLM:{row.method}:{row.model}',kind='LLM',bank=0,draw=int(row.seed),
            run_id=row.run_id,condition=f'{row.culture}:{row.language}',valid=bool(row.valid_graph),failure=row.failure)
        item.update(metrics(_read_graph(root,row)) if row.valid_graph else dict.fromkeys(METRICS,None))
        generated.append(item)
    for name in ('ER','BA','WS'):
        for bank in range(protocol['transfer_baselines']['banks']):
            for draw in range(protocol['transfer_baselines']['graphs_per_bank']):
                generated.append(dict(candidate='Baseline:'+name,kind='baseline',bank=bank,draw=draw,
                    run_id='',condition='not_applicable',valid=True,failure='',**metrics(generate_baseline(name,fit,bank,draw))))
        print('TRANSFER_BASELINE_COMPLETE='+name,flush=True)
    generated=pd.DataFrame(generated)
    if int(generated.loc[generated.kind=='LLM','valid'].sum())!=890: raise ValueError('Primary valid count differs')
    generated.to_csv(output/'candidate_metrics.csv',index=False)
    targets=[]
    controls=[]
    for graph in graphs:
        source,cohort=graph.graph['source'],graph.graph['cohort']
        for draw in range(protocol['node_sampling']['draws_per_cohort']):
            target=sample_nodes(graph,draw)
            targets.append(dict(source=source,cohort=cohort,draw=draw,**metrics(target)))
            for replicate in range(protocol['conditional_controls']['draws_per_sample']):
                for control in ('Gnm','degree_swap'):
                    random_seed=seed('conditional-control',source,cohort,draw,replicate,control)
                    if control=='Gnm':
                        sample=nx.gnm_random_graph(50,target.number_of_edges(),seed=random_seed)
                        status='target_informed'
                    else:
                        sample,status=degree_control(target,random_seed)
                    controls.append(dict(source=source,cohort=cohort,draw=draw,replicate=replicate,control=control,
                        success=sample is not None,status=status,
                        **(metrics(sample) if sample is not None else dict.fromkeys(METRICS,None))))
        print('HELDOUT_COHORT_COMPLETE='+cohort,flush=True)
    targets,controls=pd.DataFrame(targets),pd.DataFrame(controls)
    targets.to_csv(output/'heldout_sample_metrics.csv',index=False)
    controls.to_csv(output/'conditional_control_metrics.csv',index=False)
    distances,summary=transfer_distances(generated,targets)
    distances.to_csv(output/'per_cohort_distances.csv',index=False)
    summary.to_csv(output/'source_distance_summary.csv',index=False)
    conditional_residuals(targets,controls).to_csv(output/'conditional_residuals.csv',index=False)
    for name,expected in protected.items():
        if sha(root/name)!=expected: raise ValueError('Previous analysis changed during benchmark')
    for name,expected in spec['sources'].items():
        if sha(root/name)!=expected: raise ValueError('Frozen source changed during benchmark')
    result=dict(primary_attempts=896,primary_valid=890,primary_failures=6,supplemental_graphs=0,
        public_studies=2,cohort_records=len(graphs),sampled_graphs=len(targets),conditional_graph_draws=len(controls),
        failed_conditional_draws=int((~controls.success).sum()),baseline_graphs=int((generated.kind=='baseline').sum()),
        protocol_sha256=PROTOCOL_SHA256,clarifications_sha256=CLARIFICATIONS_SHA256,
        fitted_from_heldout=False,paid_calls=0,ledger_queries=0,preserved_previous_artifacts=protected,
        frozen_source_pins=spec['sources'],sources=SOURCES,
        limitations=['Evaluation-held-out from this benchmark fitting only; not held out from LLM pretraining.',
            'Two public convenience studies, overlapping Facebook egonets, one incomplete school survey.',
            'Uniform induced 50-node draws are overlapping finite-network samples, not independent communities.',
            'Primary W1 values condition on valid outcomes and equally weight seven instructions; failures remain separate.',
            'Baseline-bank ranges measure Monte Carlo sensitivity, not CIs or human-population uncertainty.',
            'Degree swaps are finite-chain diagnostics, not proven uniform null samples.',
            'No named-person link prediction, cultural fidelity or real behavior validation; no model ranking based on a composite.'])
    write_json(output/'summary.json',result)
    return result
