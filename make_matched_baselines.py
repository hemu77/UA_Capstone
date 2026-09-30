"""Offline graph controls matched to each retained network's roster and edge count.

These are synthetic controls, not real people or new LLM experiments. The
degree control randomizes existing edges while preserving every node's degree.
"""
import itertools
import json
from pathlib import Path

import networkx as nx
import pandas as pd

from analyze_networks import compute_network_metrics, compute_coleman_homophily, compute_age_assortativity


def matched_baselines(graph, personas, seed=0, categorical=None):
    if graph.is_directed() or nx.number_of_selfloops(graph) or set(graph) != set(personas):
        raise ValueError('Require a simple undirected graph on the exact supplied roster.')
    nodes, edges = list(graph), graph.number_of_edges()
    random_graph = nx.gnm_random_graph(len(nodes), edges, seed=seed)
    random_graph = nx.relabel_nodes(random_graph, dict(enumerate(nodes)))
    categorical = ['gender', 'race/ethnicity', 'religion', 'political affiliation'] if categorical is None else list(categorical)
    if not categorical or len(set(categorical)) != len(categorical) or 'age' in categorical:
        raise ValueError('Provide distinct categorical attributes; age is handled separately.')
    age_range = max(float(p['age']) for p in personas.values()) - min(float(p['age']) for p in personas.values())
    def similarity(pair):
        left, right = (personas[node] for node in pair)
        age = 1 - abs(float(left['age']) - float(right['age'])) / max(1, age_range)
        return (sum(left[d] == right[d] for d in categorical) + age) / (len(categorical) + 1)
    ranked = sorted(itertools.combinations(sorted(nodes), 2), key=lambda pair: (-similarity(pair), pair))
    feature_graph = nx.empty_graph(nodes)
    feature_graph.add_edges_from(ranked[:edges])
    degree_graph = graph.copy()
    swaps_complete = False
    if edges >= 2 and len(graph) >= 4:
        try:
            nx.double_edge_swap(degree_graph, nswap=10*edges, max_tries=200*edges, seed=seed)
            swaps_complete = True
        except nx.NetworkXAlgorithmError:
            # A constrained degree sequence may not permit enough swaps. Keep
            # the actual partial result and flag it; never claim full mixing.
            pass
    return [('uniform_edge_count', random_graph, True),
            ('equal_weight_demographic_similarity', feature_graph, True),
            ('degree_preserving_rewire', degree_graph, swaps_complete)]


def main():
    root = Path(__file__).resolve().parent
    personas = json.loads((root / 'text-files/us_50_gpt4o_w_interests.json').read_text(encoding='utf-8'))
    runs = pd.read_csv(root / 'stats/revision_v1/network_metrics.csv')
    folder = root / 'stats/revision_v1/baselines'
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in runs.itertuples():
        graph = nx.read_adjlist(root / 'text-files' / (row.run_id + '.adj'))
        for name, baseline, mixed in matched_baselines(graph, personas):
            nx.write_adjlist(baseline, folder / (row.run_id + '_' + name + '.adj'))
            metrics = compute_network_metrics(baseline)
            scores = {d: compute_coleman_homophily(baseline, personas, d)[0]
                      for d in ['gender', 'race/ethnicity', 'religion', 'political affiliation']}
            rows.append(dict(run_id=row.run_id, baseline=name, baseline_seed=0,
                             rewiring_target_reached=mixed, evidence_type='offline_synthetic_control',
                             **metrics, **{'coleman_' + d: v for d,v in scores.items()},
                             age_assortativity=compute_age_assortativity(baseline, personas)))
    pd.DataFrame(rows).to_csv(folder / 'matched_metrics.csv', index=False)
    (folder / 'README.md').write_text(
        'Each control matches the retained source graph in node and edge count.\n'
        'Uniform controls sample G(n,m); feature controls rank all candidate pairs by equal-weight\n'
        'four categorical matches and age similarity normalized by the roster age range.\n'
        'Tie breaks use sorted IDs. Degree controls target 10 swaps per edge and report failure\n'
        'to reach that target. Rewiring completion is not proof of convergence. One baseline seed\n'
        'per retained graph is exploratory, not a confidence interval or realism benchmark.\n'
        'These controls cost no API tokens and do not establish closeness to real networks.\n', encoding='utf-8')
    print(f'{len(rows)} matched offline controls; no API calls. Source networks unchanged.')


if __name__ == '__main__':
    main()
