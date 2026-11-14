"""Primary-only categorical mixing and paired descriptive contrasts."""
from collections import Counter
from itertools import combinations
import hashlib
import json

import networkx as nx
import numpy as np
import pandas as pd

from analysis_primary.graph_evaluation import PRIMARY_CSV, _read_graph, _validate_primary

METHODS = ('global', 'local', 'sequential', 'iterative')
ATTRIBUTES = ('political orientation', 'religion', 'gender')
SCORES = ('excess', 'coleman', 'newman')
CELL = ['model', 'culture', 'language']
PAIR_KEY = CELL + ['seed', 'attribute']


def categorical_mixing(graph, labels):
    if (graph.is_directed() or graph.is_multigraph() or nx.number_of_selfloops(graph)
            or set(labels) != set(graph) or len(graph) < 2):
        raise ValueError('Require a simple undirected graph and its full labeled roster')
    categories = sorted(set(labels.values()))
    index = {v: i for i, v in enumerate(categories)}
    counts = np.zeros((len(categories), len(categories)), dtype=float)
    for u, v in graph.edges():
        i, j = index[labels[u]], index[labels[v]]
        counts[i, j] += 1
        counts[j, i] += 1
    n, m = len(graph), graph.number_of_edges()
    roster_counts = Counter(labels.values())
    sizes = np.array([roster_counts[c] for c in categories])
    chance = float(np.sum(sizes * (sizes - 1)) / (n * (n - 1)))
    matrix = counts / (2 * m) if m else np.full_like(counts, np.nan)
    a = matrix.sum(axis=1)
    same = float(np.trace(matrix))
    denominator = 1 - float(np.sum(a * a))
    newman = (same - np.sum(a * a)) / denominator if denominator > 1e-14 else np.nan
    p = sizes / n
    coleman = float(np.sum(p * (np.diag(matrix) / a - p) / (1 - p))) if np.all(a > 0) and np.all(p < 1) else np.nan
    return dict(categories=categories, counts=counts, matrix=matrix, marginals=a,
                same_share=same, roster_chance=chance, excess=same-chance,
                newman=float(newman), coleman=coleman)


def paired_cells(frame, score, common=False, omitted=None):
    if score not in SCORES or frame.duplicated(['method'] + PAIR_KEY).any():
        raise ValueError('Unknown score or duplicate paired key')
    data = frame if omitted is None else frame[frame.seed.ne(omitted)]
    rows = []
    for first, second in combinations(METHODS, 2):
        left, right = data[data.method.eq(first)], data[data.method.eq(second)]
        paired = left.merge(right, on=PAIR_KEY, suffixes=('_first', '_second'), validate='one_to_one')
        for keys, group in paired.groupby(CELL + ['attribute'], sort=True):
            columns = [s + suffix for s in (SCORES if common else (score,)) for suffix in ('_first', '_second')]
            ok = np.isfinite(group[columns].to_numpy(float)).all(axis=1)
            differences = group.loc[ok, score+'_second'] - group.loc[ok, score+'_first']
            rows.append(dict(zip(CELL + ['attribute'], keys), first=first, second=second,
                score=score, support='all_three_scores' if common else 'score_specific',
                attempted_pairs=len(group), defined_seeds=len(differences),
                difference=float(differences.mean()) if len(differences) else np.nan))
    return pd.DataFrame(rows)


def method_summary(cells, omitted):
    keys = ['first', 'second', 'attribute', 'score', 'support']
    result = cells.groupby(keys).agg(mean_difference=('difference', 'mean'),
        min_cell=('difference', 'min'), max_cell=('difference', 'max'),
        defined_cells=('difference', 'count'), observed_cells=('difference', 'size'),
        defined_pairs=('defined_seeds', 'sum')).reset_index()
    if cells.duplicated(keys + CELL).any():
        raise ValueError('Duplicate method contrast cell')
    result['planned_cells'] = 28
    # Incomplete cells cannot silently acquire greater weight in a pooled estimate.
    result.loc[result.defined_cells.ne(28) | result.observed_cells.ne(28), 'mean_difference'] = np.nan
    result['omitted_seed'] = omitted
    return result


def rank_cells(frame, omitted):
    data = frame if omitted == -1 else frame[frame.seed.ne(omitted)]
    rows = []
    for keys, group in data.groupby(['method'] + CELL):
        for support in ('score_specific', 'all_three_scores'):
            wide = group.set_index(['seed', 'attribute'])[list(SCORES)].unstack('attribute')
            for score in SCORES:
                subset = wide.dropna() if support == 'all_three_scores' else wide[score].dropna()
                values = subset[score] if support == 'all_three_scores' else subset
                if len(values):
                    means = values.mean().reindex(ATTRIBUTES)
                    leaders = means.index[means.eq(means.max())]
                    winner = leaders[0] if len(leaders) == 1 else 'tie'
                    margin = means.iloc[0] - means.iloc[1:].max()
                else:
                    winner, margin = 'undefined', np.nan
                rows.append(dict(zip(['method'] + CELL, keys), score=score, support=support,
                    omitted_seed=omitted, defined_seeds=len(values), winner=winner,
                    politics_minus_competitor=margin))
    return pd.DataFrame(rows)


def build_mixing(root, output):
    primary = pd.read_csv(root / PRIMARY_CSV)
    _validate_primary(root, primary)
    roster = json.loads((root / 'text-files/revision224_adults.json').read_text())
    digest = hashlib.sha256(json.dumps(roster, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    if set(roster) != set(map(str, range(50))) or not primary.roster_sha256.eq(digest).all():
        raise ValueError('Frozen fictional roster differs')
    rows, matrices = [], []
    for row in primary.itertuples(index=False):
        graph = _read_graph(root, row) if row.valid_graph else None
        for attribute in ATTRIBUTES:
            record = dict(run_id=row.run_id, method=row.method, model=row.model,
                culture=row.culture, language=row.language, seed=row.seed, attribute=attribute,
                valid_graph=row.valid_graph, empty_graph=bool(row.empty_graph) if row.valid_graph else False,
                status='failed' if graph is None else 'empty' if not graph.number_of_edges() else 'nonempty')
            if graph is None:
                rows.append(dict(record, **dict.fromkeys(SCORES, np.nan)))
                continue
            result = categorical_mixing(graph, {node: roster[node][attribute] for node in graph})
            archived = primary.loc[primary.run_id.eq(row.run_id), 'coleman_'+attribute].iloc[0]
            if not np.isclose(archived, result['coleman'], rtol=0, atol=1e-12, equal_nan=True):
                raise ValueError('Independent Coleman reconstruction differs: '+row.run_id)
            rows.append(dict(record, **{s: result[s] for s in SCORES},
                             same_share=result['same_share'], roster_chance=result['roster_chance']))
            if graph.number_of_edges():
                for i, a in enumerate(result['categories']):
                    for j, b in enumerate(result['categories']):
                        matrices.append(dict(method=row.method, model=row.model, culture=row.culture,
                            language=row.language, seed=row.seed, attribute=attribute, row_category=a,
                            column_category=b, fraction=result['matrix'][i, j],
                            row_edge_end_share=result['marginals'][i]))
    scores = pd.DataFrame(rows)
    if len(scores) != 896*3 or scores.run_id.nunique() != 896 or scores.status.eq('empty').sum() != 12*3:
        raise ValueError('Primary mixing denominator changed')
    scores.to_csv(output/'mixing_scores.csv', index=False)
    matrices = pd.DataFrame(matrices)
    keys = ['method'] + CELL + ['attribute', 'row_category', 'column_category']
    cells = matrices.groupby(keys).agg(fraction=('fraction', 'mean'),
        row_edge_end_share=('row_edge_end_share', 'mean'), defined_graphs=('fraction', 'size')).reset_index()
    cells.to_csv(output/'mixing_matrices.csv', index=False)
    coverage = scores.groupby(['method', 'attribute']).agg(attempted=('run_id', 'size'),
        valid=('valid_graph', 'sum'), empty=('empty_graph', 'sum'),
        excess_defined=('excess', 'count'), coleman_defined=('coleman', 'count'),
        newman_defined=('newman', 'count')).reset_index()
    coverage.to_csv(output/'mixing_coverage.csv', index=False)
    contrasts, summaries, ranks = [], [], []
    for omitted in [-1, *sorted(scores.seed.unique())]:
        parts = [paired_cells(scores, score, common, None if omitted == -1 else omitted)
                 for score in SCORES for common in (False, True)]
        combined = pd.concat(parts, ignore_index=True)
        if omitted == -1:
            contrasts.append(combined)
        summaries.append(method_summary(combined, omitted))
        ranks.append(rank_cells(scores, omitted))
    pd.concat(contrasts).to_csv(output/'mixing_paired_cells.csv', index=False)
    pd.concat(summaries).to_csv(output/'mixing_method_sensitivity.csv', index=False)
    pd.concat(ranks).to_csv(output/'mixing_rank_sensitivity.csv', index=False)
    return primary
