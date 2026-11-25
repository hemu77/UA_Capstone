"""Tie-level checks on the saved primary graphs (offline, exploratory).

1. Dyadic logistic regression per graph: all three shared attributes and the age gap enter together.
2. Tie overlap (Jaccard) within a setting and between settings, on different schedule blocks.
3. Smallest detectable difference for each planned contrast given the observed block variability.

No generation, no provider call, no supplemental graph, no sealed file is written.
"""
import itertools
import json

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression

from analysis_primary.graph_evaluation import _read_graph

ATTRIBUTES = ('political orientation', 'religion', 'gender')
TERMS = ATTRIBUTES + ('age gap (decades)',)
PERSON = ('local', 'sequential', 'iterative')
BOOT = 10000


def _design(roster):
    ids = [str(i) for i in range(50)]
    pairs = np.array(list(itertools.combinations(range(50), 2)))
    cols = [np.array([roster[ids[a]][t] == roster[ids[b]][t] for a, b in pairs], float) for t in ATTRIBUTES]
    cols.append(np.array([abs(roster[ids[a]]['age'] - roster[ids[b]]['age']) / 10 for a, b in pairs]))
    return ids, pairs, np.column_stack(cols)


def load_edges(root, primary, ids):
    index = {v: i for i, v in enumerate(ids)}
    out = {}
    for r in primary.itertuples(index=False):
        if not r.valid_graph or bool(r.empty_graph):
            continue
        g = _read_graph(root, r)
        e = frozenset((min(index[str(a)], index[str(b)]), max(index[str(a)], index[str(b)])) for a, b in g.edges())
        if len(e) != r.edges:
            raise ValueError('Edge count differs from the primary record')
        out[r.run_id] = e
    return out


def dyad_models(primary, edges, pairs, X):
    """Ridge-penalised logistic regression (C=10) of tie presence on the four dyad terms, one model per graph."""
    key = {tuple(p): i for i, p in enumerate(map(tuple, pairs))}
    rows = []
    for r in primary.itertuples(index=False):
        if r.run_id not in edges:
            continue
        y = np.zeros(len(pairs), int)
        y[[key[e] for e in edges[r.run_id]]] = 1
        if y.sum() < 5:
            continue
        fit = LogisticRegression(C=10, max_iter=2000).fit(X, y)
        rows.append(dict(run_id=r.run_id, method=r.method, model=r.model, culture=r.culture, language=r.language,
                         seed=r.seed, edges=int(y.sum()), **dict(zip(TERMS, fit.coef_[0]))))
    return pd.DataFrame(rows)


def _block_boot(wide, rng):
    """Percentile interval for the mean over blocks; rows are schedule blocks."""
    v = wide.to_numpy()
    draws = v[rng.integers(0, len(v), (BOOT, len(v)))].mean(1)
    return np.percentile(draws, [2.5, 97.5], axis=0)


def dyad_summary(models):
    cell = models.groupby(['method', 'model', 'culture', 'language'])[list(TERMS)].mean().reset_index()
    cell['first'] = cell[list(ATTRIBUTES)].idxmax(axis=1)
    block = models.groupby(['method', 'seed'])[list(TERMS)].mean()
    rng = np.random.default_rng(20261125)
    rows = []
    for method in sorted(models.method.unique()):
        b = block.loc[method]
        lo, hi = _block_boot(b, rng)
        c = cell[cell.method.eq(method)]
        for j, t in enumerate(TERMS):
            rows.append(dict(method=method, term=t, mean_log_odds=b[t].mean(), odds_ratio=np.exp(b[t].mean()),
                             ci_lower=lo[j], ci_upper=hi[j], blocks=len(b), cells=len(c),
                             cells_first=int(c['first'].eq(t).sum()) if t in ATTRIBUTES else np.nan,
                             cells_positive=int((c[t] > 0).sum())))
    # politics minus religion, per procedure, and person-level minus Global, paired on block
    gap = (block['political orientation'] - block['religion']).unstack('method')
    lo, hi = _block_boot(gap, rng)
    contrasts = [dict(contrast=f'{m}: politics minus religion', mean=gap[m].mean(), ci_lower=lo[j], ci_upper=hi[j],
                      blocks_same_sign=int((np.sign(gap[m]) == np.sign(gap[m].mean())).sum())) for j, m in enumerate(gap.columns)]
    for t in ATTRIBUTES:
        w = block[t].unstack('method')
        d = pd.DataFrame({m: w[m] - w['global'] for m in PERSON})
        lo, hi = _block_boot(d, rng)
        contrasts += [dict(contrast=f'{t}: {m} minus global', mean=d[m].mean(), ci_lower=lo[j], ci_upper=hi[j],
                           blocks_same_sign=int((np.sign(d[m]) == np.sign(d[m].mean())).sum())) for j, m in enumerate(PERSON)]
    return cell, pd.DataFrame(rows), pd.DataFrame(contrasts)


def _jaccard(a, b):
    u = len(a | b)
    return len(a & b) / u if u else np.nan


def tie_overlap(primary, edges):
    """Mean Jaccard overlap of tie sets between runs on different schedule blocks: same setting, and reference vs other setting."""
    p = primary[primary.run_id.isin(edges)].assign(setting=lambda d: d.culture + ':' + d.language)
    rows = []
    for (method, model), g in p.groupby(['method', 'model']):
        by = {s: dict(zip(h.seed, h.run_id)) for s, h in g.groupby('setting')}
        ref = by.get('us:english', {})
        for s, runs in by.items():
            within = [_jaccard(edges[runs[a]], edges[runs[b]]) for a, b in itertools.combinations(sorted(runs), 2)]
            rows.append(dict(method=method, model=model, setting=s, kind='within setting', pairs=len(within),
                             mean_jaccard=np.nanmean(within)))
            if s != 'us:english':
                between = [_jaccard(edges[ref[a]], edges[runs[b]]) for a in ref for b in runs if a != b]
                rows.append(dict(method=method, model=model, setting=s, kind='reference vs setting', pairs=len(between),
                                 mean_jaccard=np.nanmean(between)))
    table = pd.DataFrame(rows)
    w = table[table.kind.eq('within setting')].set_index(['method', 'model', 'setting']).mean_jaccard
    out = []
    for (method, model, s), b in table[table.kind.eq('reference vs setting')].set_index(['method', 'model', 'setting']).mean_jaccard.items():
        base = (w[(method, model, 'us:english')] + w[(method, model, s)]) / 2
        out.append(dict(method=method, model=model, setting=s, between=b, within_average=base, retained=b / base))
    cells = pd.DataFrame(out)
    summary = cells.groupby('method').agg(within_average=('within_average', 'mean'), between=('between', 'mean'),
                                          retained=('retained', 'mean'), min_retained=('retained', 'min'),
                                          cells=('retained', 'size')).reset_index()
    by_setting = cells.groupby(['method', 'setting']).retained.mean().unstack('setting').reset_index()
    return table, cells, summary, by_setting


def detectable(root):
    """Smallest true difference a two-sided paired t test would detect with 80% power at the Bonferroni level 0.05/24."""
    folder = root / 'outputs/primary_analysis/20261105'
    blocks = pd.read_csv(folder / 'paired_schedule_blocks.csv')
    contrasts = pd.read_csv(folder / 'primary_contrasts.csv')
    primary = pd.read_csv(root / 'outputs/analysis_ready/060dcd3a86d5/primary_attempts.csv')
    ref = primary[primary.valid_graph & primary.culture.eq('us') & primary.language.eq('english')]
    ref_mean = {('global', 'density'): ref[ref.method.eq('global')].density.mean()}
    for m in PERSON:
        ref_mean[(m, 'avg_clustering_coef')] = ref[ref.method.eq(m)].avg_clustering_coef.mean()
    rows = []
    for c in contrasts.itertuples(index=False):
        d = blocks[blocks.hypothesis_id.eq(c.hypothesis_id) & blocks.complete].difference
        n = len(d)
        if n < 6:
            continue
        se = d.std(ddof=1) / np.sqrt(n)
        alpha = 0.05 / 24
        mde = (stats.t.ppf(1 - alpha / 2, n - 1) + stats.t.ppf(0.8, n - 1)) * se
        base = [v for k, v in ref_mean.items() if k[0] == c.method][0]
        rows.append(dict(hypothesis_id=c.hypothesis_id, method=c.method, endpoint=c.endpoint, blocks=n,
                         estimate=d.mean(), block_sd=d.std(ddof=1), detectable_difference=mde, reference_mean=base,
                         detectable_relative=mde / base, ci_half_width=(c.ci_upper - c.ci_lower) / 2))
    table = pd.DataFrame(rows)
    summary = table.groupby(['method', 'endpoint']).agg(
        contrasts=('blocks', 'size'), reference_mean=('reference_mean', 'first'),
        median_detectable=('detectable_difference', 'median'), max_detectable=('detectable_difference', 'max'),
        median_relative=('detectable_relative', 'median'), max_ci_half_width=('ci_half_width', 'max'),
        max_abs_estimate=('estimate', lambda s: s.abs().max())).reset_index()
    return table, summary


def run(root, output):
    primary = pd.read_csv(root / 'outputs/analysis_ready/060dcd3a86d5/primary_attempts.csv')
    if (len(primary), int(primary.valid_graph.sum()), set(primary.cohort)) != (896, 890, {'primary'}):
        raise ValueError('Not the primary cohort')
    roster = json.loads((root / 'text-files/revision224_adults.json').read_text(encoding='utf8'))
    ids, pairs, X = _design(roster)
    edges = load_edges(root, primary, ids)
    if len(edges) != 878:
        raise ValueError('Expected 878 non-empty primary graphs')
    models = dyad_models(primary, edges, pairs, X)
    cell, terms, contrasts = dyad_summary(models)
    overlap, overlap_cells, overlap_summary, overlap_setting = tie_overlap(primary, edges)
    mde, mde_summary = detectable(root)
    frames = dict(dyad_models=models, dyad_cells=cell, dyad_terms=terms, dyad_contrasts=contrasts,
                  tie_overlap_pairs=overlap, tie_overlap_cells=overlap_cells, tie_overlap_summary=overlap_summary,
                  tie_overlap_by_setting=overlap_setting, detectable_differences=mde, detectable_summary=mde_summary)
    for name, frame in frames.items():
        frame.to_csv(output / (name + '.csv'), index=False, lineterminator='\n')
    return frames
