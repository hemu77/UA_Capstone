"""Finite-network sampling diagnostics and a separate Copenhagen comparison."""
import csv
import json
import random
from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd

from analysis_heldout.benchmark import METRICS, generate_baseline, seed, transfer_distances
from analysis_heldout.data import load_public
from analysis_primary.core import sha
from analysis_primary.graph_evaluation import _read_graph

SIZES = (25, 40, 50)


def graph_metrics(graph):
    n = len(graph)
    if n < 2 or graph.is_directed() or graph.is_multigraph() or nx.number_of_selfloops(graph):
        raise ValueError('Require simple undirected graph with retained roster')
    degrees = np.sort([d for _, d in graph.degree()])
    gini = float(np.sum((2*np.arange(1, n+1)-n-1)*degrees)/(n*degrees.sum())) if degrees.sum() else 0.
    return dict(density=nx.density(graph), avg_clustering_coef=nx.average_clustering(graph),
        global_efficiency=nx.global_efficiency(graph), transitivity=nx.transitivity(graph),
        degree_gini=gini, prop_nodes_lcc=max(map(len, nx.connected_components(graph)))/n,
        edges=graph.number_of_edges(), isolates=nx.number_of_isolates(graph))


def strict_pairs(raw, header):
    if len(raw) > 60_000:
        raise ValueError('Public CSV exceeds bound')
    lines = raw.decode('ascii').splitlines()
    if not lines or lines[0] != header:
        raise ValueError('Unexpected public CSV header')
    pairs = []
    for row in csv.reader(lines[1:]):
        if len(row) != 2 or any(not value.isdecimal() for value in row):
            raise ValueError('Expected two nonnegative integer columns')
        pairs.append(tuple(map(int, row)))
    return pairs


def copenhagen(root, protocol):
    cache = root/'data_external/copenhagen_20261114'
    source = json.loads((cache/'source_manifest.json').read_text())
    if source['doi'] != protocol['sources']['doi'] or source['license']['name'] != 'MIT':
        raise ValueError('Copenhagen license/version evidence missing')
    for name in ('fb_friends.csv', 'genders.csv', 'fb_friends.README'):
        if sha(cache/name) != protocol['sources'][name]:
            raise ValueError('Copenhagen input changed: '+name)
    pairs = strict_pairs((cache/'fb_friends.csv').read_bytes(), '# user_a,user_b')
    graph = nx.Graph()
    graph.add_edges_from(pairs)
    if (len(graph) != 800 or graph.number_of_edges() != 6429 or len(pairs) != 6429
            or nx.number_of_selfloops(graph) != 11):
        raise ValueError('Published Copenhagen N/M or edge inventory mismatch')
    # Raw release includes self-links; remove only those, retaining every observed endpoint.
    graph.remove_edges_from(list(nx.selfloop_edges(graph)))
    if len(graph) != 800 or graph.number_of_edges() != 6418 or nx.number_of_isolates(graph):
        raise ValueError('Copenhagen simple-graph cleaning differs')
    gender_rows = strict_pairs((cache/'genders.csv').read_bytes(), '# user,female')
    genders = dict(gender_rows)
    if len(genders) != len(gender_rows) or len(genders) != 787 or not set(genders.values()) <= {0, 1}:
        raise ValueError('Unexpected gender release schema')
    graph.graph.update(source='copenhagen_friendship', cohort='Copenhagen',
        roster='friendship endpoint union; opt-in roster and full-network isolates unknown')
    expected = 1225*nx.density(graph)
    if expected < 1:
        raise ValueError('Insufficient expected edges for main benchmark')
    source.update(observed_roster_nodes=len(graph), observed_edges=graph.number_of_edges(),
        raw_edge_rows=6429, removed_self_links=11, duplicate_undirected_pairs=0,
        endpoint_roster=True, full_network_isolates_observable=False,
        expected_edges_in_50=expected, gender_rows=len(genders),
        friendship_nodes_with_gender=len(set(graph)&set(genders)),
        gender_only_ids=len(set(genders)-set(graph)),
        no_politics_or_religion_released=True,
        suitability='endpoint-conditioned structural comparison only; not representative friendship or RQ2 validation')
    return graph, source


def observed_nodes(graph, draw, n):
    if n not in SIZES or len(graph) < 50:
        raise ValueError('Unexpected sampling size')
    source, cohort = graph.graph['source'], graph.graph['cohort']
    selected = sorted(random.Random(seed('node-sample', source, cohort, draw)).sample(sorted(graph), 50))
    order = random.Random(seed('v3-induced-sensitivity', source, cohort, draw)).sample(range(50), 50)
    return set(selected if n == 50 else [selected[i] for i in order[:n]])


def candidate_samples(graph, key):
    if len(graph) != 50:
        raise ValueError('Candidate must have the original 50 nodes')
    order = random.Random(seed('v3-induced-sensitivity', *key)).sample(sorted(graph), 50)
    return {n: graph_metrics(graph if n == 50 else graph.subgraph(order[:n])) for n in SIZES}


def compare_sizes(candidates, targets):
    frames = []
    for n in SIZES:
        distances, _ = transfer_distances(candidates[candidates.n.eq(n)], targets[targets.n.eq(n)])
        distances['n'] = n
        distances['procedure'] = distances.candidate.str.split(':').str[1]
        frames.append(distances)
    distances = pd.concat(frames, ignore_index=True)
    records = distances.groupby(['n', 'source', 'cohort', 'procedure', 'kind', 'metric']).agg(
        mean_w1=('w1', 'mean'), min_model_or_bank=('w1', 'min'),
        max_model_or_bank=('w1', 'max'), models_or_banks=('w1', 'size')).reset_index()
    return distances, records


def build_sampling(root, output, primary, protocol):
    old = root/'outputs/heldout_evaluation/20261105_v1'
    archived = pd.read_csv(old/'candidate_metrics.csv')
    original = archived[archived.kind.eq('LLM')]
    if len(original) != 896 or set(original.run_id) != set(primary.run_id) or original.valid.sum() != 890:
        raise ValueError('Candidate table is not the full primary cohort')
    fit = json.loads((old/'development_fit.json').read_text())
    graphs = load_public(root/'data_external/heldout_20261105')
    cns, source = copenhagen(root, protocol)
    graphs.append(cns)
    (output/'copenhagen_source.json').write_text(json.dumps(source, indent=2, sort_keys=True)+'\n')
    full, sampled, overlaps = [], [], []
    for g in graphs:
        src, cohort = g.graph['source'], g.graph['cohort']
        full.append(dict(source=src, cohort=cohort, roster=g.graph['roster'], nodes=len(g), **graph_metrics(g)))
        sets = {n: [] for n in SIZES}
        for draw in range(32):
            for n in SIZES:
                nodes = observed_nodes(g, draw, n)
                sets[n].append(nodes)
                sampled.append(dict(source=src, cohort=cohort, draw=draw, n=n,
                    sample_fraction=n/len(g), **graph_metrics(g.subgraph(nodes))))
        for n, groups in sets.items():
            for a, b in combinations(range(32), 2):
                shared = len(groups[a]&groups[b])
                overlaps.append(dict(source=src, cohort=cohort, n=n, left_draw=a, right_draw=b,
                                     shared_nodes=shared, shared_fraction=shared/n))
        print('SAMPLING_RECORD='+cohort, flush=True)
    targets = pd.DataFrame(sampled)
    historical_targets = pd.read_csv(old/'heldout_sample_metrics.csv').sort_values(['source', 'cohort', 'draw'])
    recovered = targets[targets.n.eq(50) & targets.source.ne('copenhagen_friendship')].sort_values(['source', 'cohort', 'draw'])
    np.testing.assert_allclose(recovered[list(METRICS)], historical_targets[list(METRICS)], rtol=0, atol=1e-12)
    primary_index = {r.run_id: r for r in primary.itertuples(index=False)}
    candidates = []
    for r in archived.itertuples(index=False):
        info = dict(candidate=r.candidate, kind=r.kind, bank=r.bank, draw=r.draw,
                    run_id=r.run_id if r.kind == 'LLM' else '', condition=r.condition, valid=r.valid)
        if not r.valid:
            candidates.extend(dict(info, n=n, **dict.fromkeys(METRICS, np.nan)) for n in SIZES)
            continue
        if r.kind == 'LLM':
            graph, key = _read_graph(root, primary_index[r.run_id]), (r.run_id,)
        else:
            name = r.candidate.split(':')[1]
            graph, key = generate_baseline(name, fit, r.bank, r.draw), (name, r.bank, r.draw)
        values = candidate_samples(graph, key)
        np.testing.assert_allclose([values[50][k] for k in METRICS], [getattr(r, k) for k in METRICS], rtol=0, atol=1e-12)
        candidates.extend(dict(info, n=n, **m) for n, m in values.items())
    candidates = pd.DataFrame(candidates)
    distances, records = compare_sizes(candidates, targets)
    reference = pd.read_csv(old/'per_cohort_distances.csv')
    check = distances[distances.n.eq(50) & distances.source.ne('copenhagen_friendship')].merge(
        reference, on=['source', 'cohort', 'candidate', 'kind', 'bank', 'metric'], validate='one_to_one', suffixes=('', '_old'))
    if len(check) != len(reference):
        raise ValueError('Historical distance coverage differs')
    np.testing.assert_allclose(check.w1, check.w1_old, rtol=0, atol=1e-12)
    sensitivity = pd.read_csv(root/'outputs/publication_figures/20261105_v3/sensitivity_bank_distances.csv')
    sensitivity = sensitivity[sensitivity.scope.eq('original weighting')]
    check = distances.merge(sensitivity, on=['n', 'source', 'cohort', 'candidate', 'bank', 'metric'],
                            validate='one_to_one', suffixes=('', '_old'))
    if len(check) != len(sensitivity):
        raise ValueError('Historical 25/40/50-node sensitivity coverage differs')
    np.testing.assert_allclose(check.w1, check.w1_old, rtol=0, atol=1e-12)
    # Target-informed G(n,p) is a diagnostic tier, never included in transfer rankings.
    diagnostic = []
    for bank in range(8):
        for draw in range(56):
            g = nx.erdos_renyi_graph(50, nx.density(cns), seed=seed('copenhagen-target-ER', bank, draw))
            for n, m in candidate_samples(g, ('target_ER', bank, draw)).items():
                diagnostic.append(dict(candidate='Diagnostic:target_ER', kind='diagnostic', bank=bank,
                    draw=draw, run_id='', condition='not_applicable', valid=True, n=n, **m))
    diag = pd.DataFrame(diagnostic)
    diag_distances, diag_records = compare_sizes(diag, targets[targets.source.eq('copenhagen_friendship')])
    for name, frame in [('full_network_metrics', pd.DataFrame(full)), ('observed_sample_metrics', targets),
                        ('sample_overlap', pd.DataFrame(overlaps)), ('candidate_size_metrics', candidates),
                        ('transfer_distances', distances), ('transfer_record_summary', records),
                        ('target_er_metrics', diag), ('target_er_distances', diag_distances),
                        ('target_er_summary', diag_records)]:
        frame.to_csv(output/(name+'.csv'), index=False)
    return source
