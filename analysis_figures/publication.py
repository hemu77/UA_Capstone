"""Publication v2: new presentation of sealed results, never new inference."""
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from analysis_figures.manuscript import _load_sealed
from analysis_heldout.benchmark import candidate_weights

PRIMARY = Path('outputs/primary_analysis/20261105')
HELDOUT = Path('outputs/heldout_evaluation/20261105_v1')
OUTPUT = Path('outputs/publication_figures/20261105_v2')
METHODS = ('global', 'local', 'sequential', 'iterative')
COLORS = ('#2274A5', '#C17817', '#24836B', '#A65474')
MODELS = ('gpt-4.1', 'gpt-5.6-luna', 'gpt-6-luna', 'gpt-6-sol')
MARKERS = ('o', 's', '^', 'D')
METRICS = ('density', 'avg_clustering_coef', 'global_efficiency', 'transitivity',
           'degree_gini', 'prop_nodes_lcc')
LABELS = ('Density', 'Clustering', 'Efficiency', 'Transitivity', 'Degree inequality', 'Largest component')
SOURCES = ('snap_facebook', 'sociopatterns_friendship')
SOURCE_LABELS = ('Facebook friendship networks', 'School friendship survey')
INK, GRAY = '#26333D', '#7B858D'
STYLE = {'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.labelsize': 9,
         'axes.titlesize': 10, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
         'legend.fontsize': 8, 'axes.spines.top': False, 'axes.spines.right': False,
         'axes.linewidth': .65, 'axes.edgecolor': '#859099', 'text.color': INK,
         'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK,
         'svg.fonttype': 'none', 'svg.hashsalt': 'publication-060dcd3a-v2',
         'pdf.fonttype': 42, 'savefig.facecolor': 'white'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protected_files(root):
    """Bind old data, rendering code, captions, and all frozen collection sources."""
    root = Path(root)
    protected = {}
    for folder, name, status in (
        (PRIMARY, 'evaluation_manifest.json', 'COMPLETE_PRIMARY_ANALYSIS_WITH_CAVEATS'),
        (HELDOUT, 'evaluation_manifest.json', 'COMPLETE_HELDOUT_STRUCTURAL_EVALUATION_WITH_CAVEATS'),
        (Path('outputs/manuscript_figures/20261105_v1'), 'figure_manifest.json',
         'COMPLETE_MANUSCRIPT_FIGURES_WITH_CAVEATS')):
        seal = root / folder / name
        manifest = json.loads(seal.read_text())
        if manifest['status'] != status:
            raise ValueError('Incomplete source: ' + str(seal))
        protected[seal] = sha(seal)
        protected.update({root / folder / p: r['sha256'] for p, r in manifest['artifacts'].items()})
        protected.update({root / p: h for p, h in manifest['code_sha256'].items()})
        protected.update({root / p: h for p, h in manifest.get('preserved_original_artifacts', {}).items()})
    summary = json.loads((root / HELDOUT / 'summary.json').read_text())
    protected.update({root / p: h for p, h in summary['preserved_previous_artifacts'].items()})
    protected.update({root / p: h for p, h in summary['frozen_source_pins'].items()})
    verify_hashes(protected)
    return protected


def verify_hashes(protected):
    for path, expected in protected.items():
        if sha(path) != expected:
            raise ValueError('Protected input differs: ' + str(path))


def load_tables(root):
    protected = protected_files(root)
    a, primary, tables = _load_sealed(root)
    for name in ('resources_by_method_model', 'synthetic_reference_per_run', 'empirical_reference_metrics'):
        tables[name] = pd.read_csv(root / PRIMARY / (name + '.csv'))
    held = {name: pd.read_csv(root / HELDOUT / (name + '.csv')) for name in (
        'source_distance_summary', 'per_cohort_distances', 'candidate_metrics',
        'heldout_sample_metrics', 'conditional_residuals', 'dataset_quality')}
    original = held['candidate_metrics'].query("kind == 'LLM'")
    if (len(original) != 896 or int(original.valid.sum()) != 890
            or set(original.run_id) != set(primary.original_run_id)):
        raise ValueError('Primary-only identities differ')
    return a, primary, tables, held, protected


def panel(ax, letter, title, axis='x'):
    ax.set_title(title, loc='left', pad=12, fontweight='medium')
    ax.text(-.035, 1.065, letter, transform=ax.transAxes, ha='right', va='bottom', fontweight='bold')
    ax.grid(axis=axis, color='#E6E9EC', linewidth=.55)
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', length=3, width=.65)
    ax.tick_params(axis='y' if axis == 'x' else 'x', length=0)
    (ax.xaxis if axis == 'x' else ax.yaxis).set_major_locator(MaxNLocator(4))


def method_handles():
    return [Line2D([], [], color=c, lw=1.8, label=m.title()) for m, c in zip(METHODS, COLORS)]


def model_handles():
    return [Line2D([], [], color=INK, marker=s, ls='', ms=4, label=m) for m, s in zip(MODELS, MARKERS)]


def condition_labels(a, baseline=False):
    country = {'us': 'US', 'india': 'India', 'japan': 'Japan', 'brazil': 'Brazil'}
    return [country[c] + ' / ' + l.title() for c, l in a['conditions'][0 if baseline else 1:]]


def primary_contrasts(a, primary, tables, held):
    fig, axes = plt.subplots(1, 4, figsize=(11.8, 3.7), sharey=True)
    frame, blocks = tables['primary_contrasts'], tables['paired_schedule_blocks']
    # One row grid exposes all methods without repeating long condition labels.
    y = np.array([0, 1, 2, 3.5, 4.5, 5.5])
    for i, (ax, method, color) in enumerate(zip(axes, METHODS, COLORS)):
        rows = frame[frame.method.eq(method)].set_index(['culture', 'language']).loc[
            [tuple(c) for c in a['conditions'][1:]]]
        panel(ax, chr(97 + i), method.title())
        ax.axvline(0, color=GRAY, ls='--', lw=.8)
        for yy, row in zip(y, rows.itertuples()):
            raw = blocks[blocks.hypothesis_id.eq(row.hypothesis_id) & blocks.complete].sort_values('seed')
            ax.scatter(raw.difference, yy + np.linspace(.12, .27, len(raw)), s=8,
                       color=color, alpha=.28, edgecolors='none', zorder=2)
            if pd.notna(row.ci_lower):
                ax.plot([row.ci_lower, row.ci_upper], [yy, yy], lw=1.4, color=color)
                ax.plot(row.complete_block_mean, yy, 'o', ms=4, color=color)
            else:
                ax.plot(row.complete_block_mean, yy, 'x', ms=5, color=INK)
                ax.annotate('no CI', (row.complete_block_mean, yy), (6, -1),
                            textcoords='offset points', va='center', fontsize=7)
            if row.complete_schedule_blocks < 8:
                ax.text(.98, yy + .28, f'n={row.complete_schedule_blocks}',
                        transform=ax.get_yaxis_transform(), ha='right', fontsize=7, color=GRAY)
        ax.axhline(2.75, color='#D8DDE1', lw=.6)
        ax.set_yticks(y, condition_labels(a))
        ax.set_ylim(6.1, -.6)
        relevant = frame[frame.endpoint.eq('density' if method == 'global' else 'avg_clustering_coef')]
        observed = blocks[blocks.complete & blocks.endpoint.eq(relevant.endpoint.iloc[0])].difference
        values = np.r_[relevant.ci_lower, relevant.ci_upper, observed, 0.]
        values = values[np.isfinite(values)]
        pad = (values.max() - values.min()) * .08
        ax.set_xlim(values.min() - pad, values.max() + pad)
        ax.set_xlabel('Density difference' if method == 'global' else 'Clustering difference')
    fig.subplots_adjust(left=.155, right=.99, bottom=.22, top=.88, wspace=.30)
    fig.text(.155, .035, 'Relative to US-English   |   Points: means; lines: 95% simultaneous intervals (24 contrasts); faint dots: paired blocks', fontsize=8, color=GRAY)
    return fig


def robustness(a, primary, tables, held):
    hid = 'iterative:us:hindi'
    blocks = tables['paired_schedule_blocks'].query('hypothesis_id == @hid').sort_values('seed')
    loo = tables['leave_one_seed_out'].query('hypothesis_id == @hid').sort_values('omitted_seed')
    full = tables['primary_contrasts'].set_index('hypothesis_id').loc[hid]
    fig, axes = plt.subplots(1, 2, figsize=(8.1, 3.8), sharey=True, sharex=True)
    for ax, letter, title in zip(axes, 'ab', ('Observed block differences', 'Leave-one-block-out means')):
        panel(ax, letter, title)
        ax.axvline(0, color=GRAY, ls=':', lw=.8)
        ax.axvline(full.complete_block_mean, color=COLORS[3], ls='--', lw=1)
        ax.set_ylim(7.6, -.6)
        ax.set_yticks(range(8), [f'Block {i + 1}' for i in range(8)])
        ax.set_xlabel('Clustering difference from US-English')
        ax.set_xlim(min(blocks.difference.min(), loo['mean'].min()) - .008, .008)
    axes[0].scatter(blocks.difference, range(8), c=COLORS[3], s=24, zorder=3)
    for y, row in enumerate(loo.itertuples()):
        mean = blocks.loc[blocks.seed.ne(row.omitted_seed), 'difference'].mean()
        if not np.isclose(mean, row.mean, atol=1e-12, rtol=0):
            raise ValueError('Leave-one-block-out mean differs')
        axes[1].plot(row.mean, y, 'o', ms=5, mec=COLORS[3],
                     mfc=COLORS[3] if row.holm_p < .05 else 'white')
    axes[1].set_ylabel('Omitted block')
    fig.legend(handles=[Line2D([], [], color=COLORS[3], ls='--', label='Full-data mean'),
        Line2D([], [], color=COLORS[3], marker='o', ls='', label='Adjusted p < .05'),
        Line2D([], [], color=COLORS[3], marker='o', mfc='white', ls='', label='Adjusted p >= .05')],
        loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(.53, .01))
    fig.subplots_adjust(left=.11, right=.98, top=.86, bottom=.24, wspace=.35)
    return fig


def model_profiles(a, primary, tables, held):
    fig, axes = plt.subplots(1, 4, figsize=(11.8, 3.7), sharey=True)
    data = tables['descriptive_metrics']
    for i, (ax, method, color) in enumerate(zip(axes, METHODS, COLORS)):
        panel(ax, chr(97 + i), method.title())
        metric = a['primary_endpoints'][method]
        for j, (model, marker) in enumerate(zip(MODELS, MARKERS)):
            rows = data[data.method.eq(method) & data.model.eq(model) & data.metric.eq(metric)]
            rows = rows.set_index(['culture', 'language']).loc[[tuple(c) for c in a['conditions']]]
            ax.scatter(rows['mean'], np.arange(7) + (j - 1.5) * .16,
                       marker=marker, s=18, color=color, linewidths=.5)
        ax.set(ylim=(6.6, -.6), xlim=(0, .16) if method == 'global' else (0, 1),
               xlabel='Mean density' if method == 'global' else 'Mean clustering')
        ax.set_yticks(range(7), condition_labels(a, baseline=True))
        ax.axhline(.5, color='#D8DDE1', lw=.6)
    fig.legend(handles=model_handles(), loc='lower center', ncol=4, frameon=False)
    fig.subplots_adjust(left=.155, right=.99, bottom=.23, top=.88, wspace=.30)
    return fig


def coverage_missingness(a, primary, tables, held):
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.7), gridspec_kw={'width_ratios': [1.1, 1]})
    ax = axes[0]
    panel(ax, 'a', 'Original attempt outcomes')
    for y, (method, color) in enumerate(zip(METHODS, COLORS)):
        group = primary[primary.method.eq(method)]
        valid = int(group.valid_graph.sum())
        ax.barh(y, valid, height=.45, color=color)
        ax.barh(y, len(group) - valid, left=valid, height=.45, color='white', edgecolor=INK, hatch='///', lw=.7)
        ax.text(232, y, f'{valid} valid / {len(group) - valid} failed', va='center', fontsize=8)
    ax.set_yticks(range(4), [m.title() for m in METHODS])
    ax.set(xlim=(0, 345), ylim=(3.6, -.6), xlabel='Attempts')
    ax.set_xticks([0, 56, 112, 168, 224])
    ax = axes[1]
    panel(ax, 'b', 'Contrasts with incomplete blocks')
    rows = tables['primary_contrasts'].query('complete_schedule_blocks < 8')
    for y, row in enumerate(rows.itertuples()):
        ax.plot([row.missingness_lower, row.missingness_upper], [y, y], color='#B57C2C', lw=2)
        ax.plot(row.complete_block_mean, y, 'o', color=INK, ms=4)
    ax.axvline(0, color=GRAY, ls='--', lw=.8)
    ax.set_yticks(range(len(rows)), [f'{r.method.title()} / {r.language.title()}\n{r.complete_schedule_blocks}/8 blocks' for r in rows.itertuples()])
    ax.set(ylim=(len(rows) - .4, -.6), xlabel='Endpoint difference from US-English')
    fig.subplots_adjust(left=.10, right=.98, top=.85, bottom=.22, wspace=.66)
    fig.text(.10, .04, '12 valid empty graphs remain included. Right: missing-value identification bounds, NOT confidence intervals.', fontsize=8, color=GRAY)
    return fig


def resource_use(a, primary, tables, held):
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.6), sharey=True)
    data = tables['resources_by_method_model']
    for ax, letter, column, label, divisor in zip(axes, 'abc',
        ('requests', 'received_conservative_usd', 'api_seconds'),
        ('Received requests (thousands)', 'Received cost estimate (USD)', 'Serial provider time (hours)'),
        (1000, 1, 3600)):
        panel(ax, letter, label)
        for y, (method, color) in enumerate(zip(METHODS, COLORS)):
            group = data[data.method.eq(method)].set_index('model').loc[list(MODELS)]
            for j, marker in enumerate(MARKERS):
                ax.plot(group.iloc[j][column] / divisor, y + (j - 1.5) * .15, marker=marker,
                        ls='', color=color, ms=4)
        ax.set(xlim=(0, None), ylim=(3.6, -.6))
        ax.set_yticks(range(4), [m.title() for m in METHODS])
    fig.legend(handles=model_handles(), loc='lower center', ncol=4, frameon=False)
    fig.subplots_adjust(left=.105, right=.98, top=.83, bottom=.22, wspace=.30)
    return fig


def gnm_context(a, primary, tables, held):
    fig, axes = plt.subplots(1, 4, figsize=(11.8, 3.4), sharex=True, sharey=True)
    reference = tables['synthetic_reference_per_run']
    reference = reference[reference.metric.eq('avg_clustering_coef') & reference.valid_graph].copy()
    if not reference.run_id.isin(primary.run_id).all():
        raise ValueError('Supplemental reference refused')
    reference['residual'] = reference.observed_value - reference.reference_mean
    extent = max(.01, float(reference.residual.abs().max()) * 1.06)
    for ax, letter, method, color in zip(axes, 'abcd', METHODS, COLORS):
        panel(ax, letter, method.title())
        ax.axvline(0, color=GRAY, ls='--', lw=.8)
        for y, model in enumerate(MODELS):
            values = reference[reference.method.eq(method) & reference.model.eq(model)].sort_values('run_id').residual.dropna()
            ax.scatter(values, y + np.linspace(-.22, .22, len(values)), c=color, s=8, alpha=.30, edgecolors='none')
            ax.plot(values.mean(), y, MARKERS[y], color=color, ms=5, mec=INK, mew=.6)
        ax.set(ylim=(3.6, -.6), xlim=(-extent, extent), xlabel='Clustering excess')
        ax.set_yticks(range(4), MODELS)
    fig.subplots_adjust(left=.15, right=.99, top=.85, bottom=.22, wspace=.28)
    return fig


def legacy_comparison(a, primary, tables, held):
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 4.5), sharey=True)
    empirical = tables['empirical_reference_metrics']
    for ax, letter, metric, label in zip(axes, 'abc', METRICS[:3], LABELS[:3]):
        panel(ax, letter, label)
        for y, (method, color) in enumerate(zip(METHODS, COLORS)):
            values = primary.loc[primary.valid_graph & primary.method.eq(method), metric].dropna().to_numpy()
            ax.boxplot([values], positions=[y], vert=False, widths=.42, manage_ticks=False,
                       showfliers=False, patch_artist=True, boxprops={'facecolor': color, 'alpha': .20},
                       medianprops={'color': color, 'lw': 1.7}, whiskerprops={'color': color}, capprops={'color': color})
            ax.scatter(values, y + np.linspace(-.17, .17, len(values)), c=color, s=6, alpha=.25, edgecolors='none')
        ax.axhline(4, color='#CCD3D9', lw=.8)
        ax.scatter(empirical[metric], np.arange(5, 13), c=INK, marker='D', s=17)
        ax.set(xlim=(0, 1), ylim=(12.7, -.7))
        ax.set_yticks([0, 1, 2, 3, *range(5, 13)],
                      [m.title() for m in METHODS] + [f'{r.reference} (n={r.nodes})' for r in empirical.itertuples()])
    fig.subplots_adjust(left=.21, right=.98, top=.89, bottom=.12, wspace=.24)
    return fig


def heldout_distances(a, primary, tables, held):
    frame = held['source_distance_summary']
    fig, axes = plt.subplots(2, 6, figsize=(13.4, 6.3), sharey=True)
    positions = [0, 1, 2, 3, 4.5, 5.5, 6.5]
    for row, source in enumerate(SOURCES):
        for col, metric in enumerate(METRICS):
            ax = axes[row, col]
            panel(ax, chr(97 + row * 6 + col), LABELS[col])
            part = frame[frame.source.eq(source) & frame.metric.eq(metric)].set_index('candidate')
            for y, (method, color) in enumerate(zip(METHODS, COLORS)):
                for j, (model, marker) in enumerate(zip(MODELS, MARKERS)):
                    value = part.loc[f'LLM:{method}:{model}', 'mean_w1']
                    ax.plot(value, y + (j - 1.5) * .16, marker=marker, ls='', ms=3.5, color=color)
            for y, baseline in zip(positions[4:], ('ER', 'BA', 'WS')):
                record = part.loc['Baseline:' + baseline]
                ax.plot([record.min_bank_w1, record.max_bank_w1], [y, y], color=INK, lw=1.4)
                ax.plot(record.mean_w1, y, 'D', mfc='white', mec=INK, ms=4)
            maximum = frame.loc[frame.metric.eq(metric), 'max_bank_w1'].max()
            maximum = max(maximum, frame.loc[frame.metric.eq(metric), 'mean_w1'].max())
            ax.set(xlim=(0, max(.05, maximum * 1.13)), ylim=(7.1, -.65))
            ax.axhline(3.75, color='#CCD3D9', lw=.7)
            ax.set_yticks(positions, ['Global', 'Local', 'Sequential', 'Iterative', 'ER', 'BA', 'WS'])
            if row == 1:
                ax.set_xlabel('Distance')
    fig.subplots_adjust(left=.12, right=.99, top=.91, bottom=.13, wspace=.30, hspace=.50)
    fig.text(.012, .70, 'Facebook', rotation=90, va='center', weight='bold', fontsize=10)
    fig.text(.012, .31, 'School', rotation=90, va='center', weight='bold', fontsize=10)
    handles = model_handles() + [Line2D([], [], marker='D', mfc='white', color=INK, lw=1.3,
                                       label='Baseline mean + bank range')]
    fig.legend(handles=handles, loc='lower center', ncol=5, frameon=False, bbox_to_anchor=(.53, .005))
    return fig


def weighted_ecdf(values, weights):
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    if (len(values) != len(weights) or not len(values) or not np.isfinite(values).all()
            or (values < 0).any() or (values > 1).any()
            or not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0):
        raise ValueError('Invalid weighted empirical distribution')
    order = np.argsort(values, kind='stable')
    return np.r_[0., values[order], 1.], np.r_[0., np.cumsum(weights[order]) / weights.sum(), 1.]


def heldout_distributions(a, primary, tables, held):
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 5.9), sharex=True, sharey=True)
    candidate = held['candidate_metrics']
    targets = held['heldout_sample_metrics']
    for row, source in enumerate(SOURCES):
        for col, metric in enumerate(METRICS[:3]):
            ax = axes[row, col]
            panel(ax, chr(97 + row * 3 + col), LABELS[col], axis='y')
            target = targets[targets.source.eq(source)]
            counts = target.cohort.value_counts()
            weights = target.cohort.map(lambda c: 1 / (len(counts) * counts[c])).to_numpy()
            x, y = weighted_ecdf(target[metric], weights)
            ax.step(x, y, where='post', color=INK, lw=1.9, zorder=5)
            for method, color in zip(METHODS, COLORS):
                values, weights = [], []
                for model in MODELS:
                    group = candidate[candidate.candidate.eq(f'LLM:{method}:{model}')]
                    valid, weight = candidate_weights(group, metric)
                    values.extend(valid[metric])
                    weights.extend(weight / len(MODELS))
                x, y = weighted_ecdf(values, weights)
                ax.step(x, y, where='post', color=color, lw=1.1)
            for baseline, line in zip(('ER', 'BA', 'WS'), (':', '--', '-.')):
                group = candidate[candidate.candidate.eq('Baseline:' + baseline)]
                x, y = weighted_ecdf(group[metric], np.ones(len(group)))
                ax.step(x, y, where='post', color='#9AA3A9', ls=line, lw=.9)
            ax.set(xlim=(0, 1), ylim=(0, 1.02))
            ax.set_yticks([0, .25, .5, .75, 1.])
            if col == 0:
                ax.set_ylabel(('Facebook' if row == 0 else 'School') + '\nCumulative fraction')
            if row == 1:
                ax.set_xlabel('Metric value')
    handles = [Line2D([], [], color=INK, lw=1.9, label='Observed 50-node draws')] + method_handles()
    handles += [Line2D([], [], color='#9AA3A9', ls=s, label=n) for n, s in zip(('ER', 'BA', 'WS'), (':', '--', '-.'))]
    fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(.54, .005))
    fig.subplots_adjust(left=.09, right=.98, top=.92, bottom=.18, hspace=.43, wspace=.24)
    return fig


def sampling(a, primary, tables, held):
    fig, axes = plt.subplots(1, 2, figsize=(8.7, 4.5), sharey=True)
    quality, target = held['dataset_quality'], held['heldout_sample_metrics']
    for ax, letter, metric, column, label in zip(axes, 'ab', METRICS[:2],
        ('observed_density', 'observed_clustering'), LABELS[:2]):
        panel(ax, letter, label)
        for y, row in enumerate(quality.itertuples()):
            values = target.loc[target.cohort.eq(row.cohort), metric]
            ax.plot([values.min(), values.max()], [y, y], color='#C6CED3', lw=1.5)
            ax.plot([getattr(row, column), values.mean()], [y, y], color='#768792', lw=.7)
            ax.plot(getattr(row, column), y, 'o', mfc='white', mec=INK, ms=4.5)
            ax.plot(values.mean(), y, 'D', color=COLORS[0], ms=4)
        ax.set(xlim=(0, .35) if metric == 'density' else (0, 1), ylim=(10.6, -.6), xlabel='Coefficient')
        ax.set_yticks(range(11), [f'{r.cohort}  (n={r.observed_roster_nodes})' for r in quality.itertuples()])
        ax.axhline(9.5, color='#CCD3D9', lw=.8)
    fig.legend(handles=[Line2D([], [], color=INK, marker='o', mfc='white', ls='', label='Full observed roster'),
        Line2D([], [], color=COLORS[0], marker='D', ls='', label='Mean of 50-node draws'),
        Line2D([], [], color='#C6CED3', lw=1.5, label='Node-draw range (not a CI)')],
        loc='lower center', ncol=3, frameon=False)
    fig.subplots_adjust(left=.19, right=.98, top=.89, bottom=.19, wspace=.30)
    return fig


def controls(a, primary, tables, held):
    fig, axes = plt.subplots(1, 2, figsize=(8.7, 4.4), sharey=True)
    cohorts = held['dataset_quality'].cohort.tolist()
    data = held['conditional_residuals']
    for ax, letter, metric, label in zip(axes, 'ab', ('avg_clustering_coef', 'transitivity'), ('Clustering', 'Transitivity')):
        panel(ax, letter, label)
        for control, marker, color, offset in (('Gnm', 's', GRAY, -.12), ('degree_swap', 'o', COLORS[0], .12)):
            rows = data[data.control.eq(control) & data.metric.eq(metric)].set_index('cohort').loc[cohorts]
            ax.scatter(rows.mean_target_minus_control, np.arange(11) + offset, marker=marker, c=color, s=20)
        ax.axvline(0, color=GRAY, ls='--', lw=.8)
        ax.set(ylim=(10.6, -.6), xlabel='Observed minus randomized control', xlim=(-.02, .65))
        ax.set_yticks(range(11), cohorts)
        ax.axhline(9.5, color='#CCD3D9', lw=.8)
    fig.legend(handles=[Line2D([], [], color=GRAY, marker='s', ls='', label='Same edge count'),
        Line2D([], [], color=COLORS[0], marker='o', ls='', label='Same degree sequence')],
        loc='lower center', ncol=2, frameon=False)
    fig.subplots_adjust(left=.13, right=.98, top=.88, bottom=.19, wspace=.28)
    return fig


def cohort_sensitivity(a, primary, tables, held):
    data = held['per_cohort_distances'].query("metric == 'avg_clustering_coef'")
    cohorts = held['dataset_quality'].cohort.tolist()
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.8), sharey=True)
    for ax, letter, title in zip(axes, 'ab', ('LLM procedure means (four fixed models)', 'Development-fitted baseline means (eight banks)')):
        panel(ax, letter, title)
        ax.set(ylim=(10.6, -.6), xlim=(0, .8), xlabel='Clustering distribution distance')
        ax.set_yticks(range(11), cohorts)
        ax.axhline(9.5, color='#CCD3D9', lw=.8)
    for j, (method, color) in enumerate(zip(METHODS, COLORS)):
        group = data[data.candidate.str.startswith('LLM:' + method + ':')].groupby('cohort').w1.mean().reindex(cohorts)
        axes[0].scatter(group, np.arange(11) + (j - 1.5) * .15, c=color, marker=MARKERS[j], s=18, label=method.title())
    for j, (baseline, marker, gray) in enumerate(zip(('ER', 'BA', 'WS'), ('o', 's', '^'), ('#26333D', '#667682', '#A2ADB5'))):
        group = data[data.candidate.eq('Baseline:' + baseline)].groupby('cohort').w1.mean().reindex(cohorts)
        axes[1].scatter(group, np.arange(11) + (j - 1) * .17, c=gray, marker=marker, s=18, label=baseline)
    for ax in axes:
        ax.legend(loc='upper center', bbox_to_anchor=(.5, -.15), ncol=4, frameon=False, handletextpad=.3, columnspacing=.9)
    fig.subplots_adjust(left=.11, right=.98, top=.88, bottom=.20, wspace=.27)
    return fig


DRAWERS = {
    '01_primary_contrasts': primary_contrasts,
    '02_primary_robustness': robustness,
    '03_model_profiles': model_profiles,
    '04_coverage_missingness': coverage_missingness,
    '05_resource_use': resource_use,
    '06_gnm_context': gnm_context,
    '07_legacy_comparison': legacy_comparison,
    '08_heldout_distances': heldout_distances,
    '09_heldout_distributions': heldout_distributions,
    '10_sampling_effect': sampling,
    '11_conditional_controls': controls,
    '12_cohort_sensitivity': cohort_sensitivity,
}


def render(root, output, names=None):
    root, output = Path(root).resolve(), Path(output).resolve()
    seal = output / 'figure_manifest.json'
    if seal.exists() and json.loads(seal.read_text()).get('status') == 'COMPLETE_PUBLICATION_FIGURES_V2':
        raise ValueError('Published revision is immutable; render to a new directory')
    allowed = list(DRAWERS) if names is None else list(names)
    if not allowed or len(set(allowed)) != len(allowed) or not set(allowed) <= set(DRAWERS):
        raise ValueError('Unknown or repeated figure')
    for folder in ('outputs/final_study', 'outputs/analysis_ready', str(PRIMARY), str(HELDOUT),
                   'outputs/manuscript_figures/20261105_v1'):
        if output.is_relative_to(root / folder):
            raise ValueError('Cannot overwrite protected evidence')
    a, primary, tables, held, protected = load_tables(root)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    existing = set(plt.get_fignums())
    with plt.rc_context(STYLE):
        try:
            for name in allowed:
                fig = DRAWERS[name](a, primary, tables, held)
                for suffix in ('png', 'svg', 'pdf'):
                    path = output / (name + '.' + suffix)
                    metadata = {'Date': None} if suffix == 'svg' else (
                        {'CreationDate': None, 'ModDate': None} if suffix == 'pdf' else None)
                    fig.savefig(path, dpi=220, metadata=metadata)
                    paths.append(path)
                plt.close(fig)
        finally:
            for number in set(plt.get_fignums()) - existing:
                plt.close(number)
    verify_hashes(protected)
    return paths


def show(root, names):
    """Portable notebook display: verify the published bytes, not temporary URLs."""
    from IPython.display import Image, Markdown, display
    root = Path(root).resolve()
    manifest = json.loads((root / OUTPUT / 'figure_manifest.json').read_text())
    if manifest['status'] != 'COMPLETE_PUBLICATION_FIGURES_V2':
        raise ValueError('Incomplete publication figure set')
    verify_hashes({root / p: h for p, h in manifest['protected_sha256'].items()})
    verify_hashes({root / p: h for p, h in manifest['code_sha256'].items()})
    for name in names:
        if name not in DRAWERS:
            raise ValueError('Unknown figure: ' + name)
        for suffix in ('png', 'svg', 'pdf'):
            filename = name + '.' + suffix
            if sha(root / OUTPUT / filename) != manifest['artifacts'][filename]['sha256']:
                raise ValueError('Figure differs: ' + filename)
        display(Image(filename=str(root / OUTPUT / (name + '.png')), width=1100))
        display(Markdown(manifest['captions'][name]))
