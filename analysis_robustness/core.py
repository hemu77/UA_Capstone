"""Three offline checks on saved primary graphs and pinned public records.

1. Density-informed ER for ALL six statistics on the 11 original held-out records
   (the earlier release reported it for efficiency only).
2. Connected (snowball) sampling of the observed records, against uniform sampling.
3. Stratified label permutation for demographic mixing: is same-group tie share for
   one attribute above what the other two attributes and the roster's joint
   composition already produce?

Nothing here generates text, calls a provider, refits a baseline or touches a sealed file.
The six supplementary graphs are never read.
"""
import hashlib
import json
import random
from collections import deque

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from analysis_extension.sampling import copenhagen, graph_metrics
from analysis_figures.deep_review import v2_seed
from analysis_heldout.benchmark import METRICS, seed, transfer_distances
from analysis_heldout.data import load_public
from analysis_primary.graph_evaluation import _read_graph

ATTRIBUTES = ('political orientation', 'religion', 'gender')
PERMUTATIONS = 500
DRAWS = 32


def _sub_metrics(graph, nodes):
    sub = nx.Graph()
    sub.add_nodes_from(nodes)
    sub.add_edges_from(graph.subgraph(nodes).edges())
    return {k: v for k, v in graph_metrics(sub).items()}


def snowball_nodes(graph, draw, n=50):
    """Breadth-first sample from a random seed; restart from a new random node if a component is exhausted."""
    rng = random.Random(seed('snowball-v1', graph.graph['source'], graph.graph['cohort'], draw))
    order = sorted(graph)
    chosen, seen, queue = [], set(), deque()
    while len(chosen) < n:
        if not queue:
            start = rng.choice([v for v in order if v not in seen])
            seen.add(start)
            queue.append(start)
        node = queue.popleft()
        chosen.append(node)
        neighbours = sorted(set(graph[node]) - seen)
        rng.shuffle(neighbours)
        seen.update(neighbours)
        queue.extend(neighbours)
    return chosen


def snowball_targets(graphs):
    rows = []
    for g in graphs:
        for draw in range(DRAWS):
            nodes = snowball_nodes(g, draw)
            if len(set(nodes)) != 50:
                raise ValueError('Snowball sample must have 50 distinct nodes')
            rows.append(dict(source=g.graph['source'], cohort=g.graph['cohort'], draw=draw, n=50,
                             **_sub_metrics(g, nodes)))
    return pd.DataFrame(rows)


def procedure_distances(candidates, targets, scheme):
    distances, _ = transfer_distances(candidates, targets)
    distances['procedure'] = distances.candidate.str.split(':').str[1]
    out = distances.groupby(['source', 'cohort', 'procedure', 'kind', 'metric']).agg(
        mean_w1=('w1', 'mean'), min_model_or_bank=('w1', 'min'), max_model_or_bank=('w1', 'max'),
        models_or_banks=('w1', 'size')).reset_index()
    out.insert(0, 'sampling', scheme)
    return out


def target_er(quality, targets):
    """ER given each full record's density: same random streams as the released efficiency diagnostic."""
    triu = np.triu_indices(50, 1)
    rows, graphs = [], []
    for r in quality.itertuples():
        target = targets[(targets.source == r.source) & (targets.cohort == r.cohort)]
        if len(target) != DRAWS:
            raise ValueError('Missing observed draw')
        for bank in range(8):
            rng = np.random.default_rng(v2_seed(r.source, r.cohort, bank))
            values = []
            for draw in range(56):
                a = np.zeros((50, 50), bool)
                a[triu] = rng.random(len(triu[0])) < r.observed_density
                g = nx.from_numpy_array(a | a.T)
                m = graph_metrics(g)
                values.append(m)
                graphs.append(dict(source=r.source, cohort=r.cohort, bank=bank, draw=draw, **m))
            for metric in METRICS:
                rows.append(dict(source=r.source, cohort=r.cohort, bank=bank, metric=metric,
                                 w1=float(wasserstein_distance([v[metric] for v in values], target[metric]))))
    banks = pd.DataFrame(rows)
    summary = banks.groupby(['source', 'cohort', 'metric']).w1.agg(['mean', 'min', 'max', 'size']).reset_index()
    summary.columns = ['source', 'cohort', 'metric', 'mean_w1', 'min_bank', 'max_bank', 'banks']
    return banks, summary, pd.DataFrame(graphs)


def _same_share(u, v, labels):
    return float(np.mean(labels[u] == labels[v]))


def mixing_permutation(root, primary, roster):
    """Observed same-group tie share minus its mean under label permutation within strata of the other two attributes."""
    ids = [str(i) for i in range(50)]
    codes = {a: pd.Categorical([roster[i][a] for i in ids]).codes for a in ATTRIBUTES}
    strata = {}
    for a in ATTRIBUTES:
        others = [b for b in ATTRIBUTES if b != a]
        key = codes[others[0]] * 100 + codes[others[1]]
        strata[a] = [np.flatnonzero(key == k) for k in np.unique(key)]
    index = {v: i for i, v in enumerate(ids)}
    rows = []
    for r in primary.itertuples(index=False):
        if not r.valid_graph or bool(r.empty_graph):
            continue
        graph = _read_graph(root, r)
        edges = np.array([(index[str(a)], index[str(b)]) for a, b in graph.edges()])
        if len(edges) != r.edges:
            raise ValueError('Edge count differs from the primary record')
        u, v = edges[:, 0], edges[:, 1]
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(('mixperm-v1:' + r.run_id).encode()).digest()[:8], 'big'))
        for a in ATTRIBUTES:
            observed = _same_share(u, v, codes[a])
            null = np.empty(PERMUTATIONS)
            for k in range(PERMUTATIONS):
                labels = codes[a].copy()
                for members in strata[a]:
                    labels[members] = labels[rng.permutation(members)]
                null[k] = _same_share(u, v, labels)
            rows.append(dict(run_id=r.run_id, method=r.method, model=r.model, culture=r.culture, language=r.language,
                             seed=r.seed, attribute=a, edges=len(edges), same_share=observed,
                             stratified_null_mean=null.mean(), stratified_null_sd=null.std(ddof=1),
                             conditional_excess=observed - null.mean(),
                             upper_tail_p=(1 + np.sum(null >= observed - 1e-12)) / (PERMUTATIONS + 1)))
    return pd.DataFrame(rows)


def mixing_summary(scores):
    cell = ['method', 'model', 'culture', 'language']
    cells = scores.groupby(cell + ['attribute']).agg(conditional_excess=('conditional_excess', 'mean'),
                                                     graphs=('run_id', 'size'),
                                                     share_p_below_05=('upper_tail_p', lambda p: float(np.mean(p < .05)))).reset_index()
    wide = cells.pivot(index=cell, columns='attribute', values='conditional_excess')
    lead = wide.idxmax(axis=1)
    rows = []
    for method, part in cells.groupby('method'):
        for a in ATTRIBUTES:
            x = part[part.attribute == a]
            rows.append(dict(method=method, attribute=a, cells=len(x), mean_conditional_excess=x.conditional_excess.mean(),
                             min_cell=x.conditional_excess.min(), max_cell=x.conditional_excess.max(),
                             cells_positive=int((x.conditional_excess > 0).sum()),
                             cells_first=int((lead.xs(method, level='method') == a).sum()),
                             mean_share_graphs_p_below_05=x.share_p_below_05.mean()))
    return cells, pd.DataFrame(rows)


def run(root, output, protocol):
    old = root / 'outputs/heldout_evaluation/20261105_v1'
    primary = pd.read_csv(root / 'outputs/analysis_ready/060dcd3a86d5/primary_attempts.csv')
    if (len(primary), int(primary.valid_graph.sum()), set(primary.cohort)) != (896, 890, {'primary'}):
        raise ValueError('Not the primary cohort')
    candidates = pd.read_csv(old / 'candidate_metrics.csv')
    llm = candidates[candidates.kind.eq('LLM')]
    if set(llm.run_id) != set(primary.run_id):
        raise ValueError('Candidate table is not the primary cohort')
    graphs = load_public(root / 'data_external/heldout_20261105')
    cns, _ = copenhagen(root, protocol)
    quality = pd.read_csv(old / 'dataset_quality.csv')
    uniform = pd.read_csv(old / 'heldout_sample_metrics.csv')

    # 1. density-informed ER on all six statistics, uniform draws, 11 original records
    banks, er_summary, er_graphs = target_er(quality, uniform)
    released = pd.read_csv(root / 'outputs/publication_figures/20261111_v4/efficiency_records.csv')
    check = er_summary[er_summary.metric.eq('global_efficiency')].merge(released, on=['source', 'cohort'])
    np.testing.assert_allclose(check.mean_w1, check.target_ER, rtol=0, atol=1e-9)
    transfer = pd.read_csv(root / 'outputs/research_extension/20261114_v2/transfer_record_summary.csv')
    transfer = transfer[transfer.n.eq(50) & transfer.cohort.ne('Copenhagen')]
    wide = transfer.pivot(index=['source', 'cohort', 'metric'], columns='procedure', values='mean_w1').reset_index()
    density_table = wide.merge(er_summary[['source', 'cohort', 'metric', 'mean_w1']].rename(columns={'mean_w1': 'target_ER'}),
                               on=['source', 'cohort', 'metric'], validate='one_to_one')

    # 2. snowball sampling of all 12 records
    snow = snowball_targets(graphs + [cns])
    snow_dist = procedure_distances(candidates, snow[snow.cohort.ne('Copenhagen')], 'snowball')
    uni_dist = transfer.rename(columns={'n': 'sampling'}).assign(sampling='uniform')[snow_dist.columns]
    sampling = pd.concat([uni_dist, snow_dist], ignore_index=True)

    # 3. stratified label permutation
    roster = json.loads((root / 'text-files/revision224_adults.json').read_text(encoding='utf8'))
    perm = mixing_permutation(root, primary, roster)
    if perm.run_id.nunique() != 878:
        raise ValueError('Expected 878 non-empty primary graphs')
    cells, mix = mixing_summary(perm)

    for name, frame in [('target_er_all_metrics_banks', banks), ('target_er_all_metrics_records', density_table),
                        ('target_er_graph_metrics', er_graphs), ('snowball_sample_metrics', snow),
                        ('sampling_scheme_distances', sampling), ('mixing_stratified_permutation', perm),
                        ('mixing_stratified_cells', cells), ('mixing_stratified_summary', mix)]:
        frame.to_csv(output / (name + '.csv'), index=False, lineterminator='\n')
    return dict(density=density_table, sampling=sampling, snow=snow, uniform=uniform, mix=mix, cells=cells)
