"""Compact manuscript figures; sealed primary evidence is read-only."""
import hashlib
import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from analysis_primary.core import load_inputs, validate_primary


SEALED = Path('outputs/primary_analysis/20261105')
STATUS = 'COMPLETE_PRIMARY_ANALYSIS_WITH_CAVEATS'
TABLES = ('primary_contrasts', 'paired_schedule_blocks', 'leave_one_seed_out',
          'descriptive_metrics', 'sensitivity')
TEAL, INK, GRAY = '#246C6C', '#282828', '#787878'
MODEL_COLORS = (TEAL, INK, '#707070', '#A0A0A0')
MARKERS = ('o', 's', '^', 'D')
STYLE = {'font.family': 'STIXGeneral', 'font.size': 9, 'axes.titlesize': 10,
         'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
         'legend.fontsize': 8, 'axes.spines.top': False, 'axes.spines.right': False,
         'axes.linewidth': .6, 'xtick.major.width': .6, 'ytick.major.width': .6,
         'svg.fonttype': 'none', 'svg.hashsalt': 'manuscript-060dcd3a86d5-v1',
         'pdf.fonttype': 42, 'savefig.facecolor': 'white'}


def _condition(culture, language):
    name = {'us': 'US', 'india': 'India', 'japan': 'Japan', 'brazil': 'Brazil'}[culture]
    return f'{name} / {language.title()}'


def _load_sealed(root):
    directory = root / SEALED
    manifest = json.loads((directory / 'evaluation_manifest.json').read_text(encoding='utf8'))
    if manifest.get('status') != STATUS:
        raise ValueError('Require evaluation status ' + STATUS)
    tables = {}
    for name in TABLES:
        filename = name + '.csv'
        data = (directory / filename).read_bytes()
        if hashlib.sha256(data).hexdigest() != manifest['artifacts'][filename]['sha256']:
            raise ValueError('Sealed input hash mismatch: ' + filename)
        tables[name] = pd.read_csv(io.BytesIO(data))
    spec, primary, _ = load_inputs(root)
    validate_primary(primary, spec)
    a = spec['analysis']
    expected = {f'{m}:{c}:{l}' for m in a['methods'] for c, l in a['conditions'][1:]}
    contrasts = tables['primary_contrasts']
    if len(contrasts) != 24 or set(contrasts.hypothesis_id) != expected:
        raise ValueError('Require the complete 24-contrast family')
    for name, seed in (('paired_schedule_blocks', 'seed'), ('leave_one_seed_out', 'omitted_seed')):
        frame = tables[name]
        actual = set(frame[['hypothesis_id', seed]].itertuples(index=False, name=None))
        if len(frame) != 192 or actual != {(hid, s) for hid in expected for s in a['schedule_seeds']}:
            raise ValueError('Require all contrast/schedule rows: ' + name)
    blocks = tables['paired_schedule_blocks']
    if (not blocks.complete.map(type).eq(bool).all()
            or blocks.loc[~blocks.complete, 'difference'].notna().any()):
        raise ValueError('Incomplete blocks cannot be plotted as observed differences')
    return a, primary, tables


def _axes_grid():
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 5.6))
    fig.subplots_adjust(left=.19, right=.97, bottom=.13, top=.95, wspace=.75, hspace=.50)
    return fig, axes.flat


def _axis(ax, title):
    ax.set_title(title, loc='left', pad=7)
    ax.axvline(0, color='#B0B0B0', linewidth=.7, linestyle=(0, (2, 2)), zorder=0)
    ax.grid(axis='x', color='#E8E8E8', linewidth=.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis='y', length=0)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))


def _limits(frame, observed=()):
    values = frame[['ci_lower', 'ci_upper', 'complete_block_mean']].to_numpy().ravel()
    values = np.r_[values, np.asarray(observed, dtype=float)]
    values = np.r_[0., values[np.isfinite(values)]]
    lower, upper = values.min(), values.max()
    pad = max(.005, (upper - lower) * .08)
    return lower - pad, upper + pad


def _forest(a, primary, tables):
    contrasts = tables['primary_contrasts']
    blocks = tables['paired_schedule_blocks']
    clustering = _limits(contrasts[contrasts.endpoint.eq('avg_clustering_coef')],
                         blocks.loc[blocks.complete & blocks.endpoint.eq('avg_clustering_coef'), 'difference'])
    fig, axes = _axes_grid()
    for letter, ax, method in zip('abcd', axes, a['methods']):
        rows = contrasts[contrasts.method.eq(method)].set_index(['culture', 'language']).loc[
            [tuple(c) for c in a['conditions'][1:]]]
        endpoint = 'density' if method == 'global' else 'clustering'
        _axis(ax, f'{letter}   {method.title()}: {endpoint}')
        labels = []
        for y, ((culture, language), row) in enumerate(rows.iterrows()):
            n = int(row.complete_schedule_blocks)
            observed = blocks.loc[blocks.hypothesis_id.eq(row.hypothesis_id) & blocks.complete].sort_values('seed')
            ax.scatter(observed.difference, y + np.linspace(.09, .23, len(observed)),
                       s=5, color=GRAY, alpha=.55, linewidths=0, zorder=2)
            ci = n >= a['fallback_min_complete_blocks'] and pd.notna(row.ci_lower) and pd.notna(row.ci_upper)
            if ci:
                ax.plot([row.ci_lower, row.ci_upper], [y, y], color=GRAY, linewidth=.9)
            if pd.notna(row.complete_block_mean):
                ax.plot(row.complete_block_mean, y, marker='o' if ci else 'x',
                        color=TEAL if ci else INK, markersize=4, markeredgewidth=.8)
            if not ci:
                ax.annotate('no CI', (row.complete_block_mean, y), xytext=(7, 0),
                            textcoords='offset points', va='center', fontsize=8)
            labels.append(f'{_condition(culture, language)} (n={n})')
        ax.set_yticks(range(6), labels)
        ax.set_ylim(5.5, -.5)
        ax.set_xlim(_limits(rows, blocks.loc[blocks.complete & blocks.method.eq(method), 'difference'])
                    if method == 'global' else clustering)
        ax.set_xlabel('Difference from US-English')
    fig.legend(handles=[Line2D([], [], color=GRAY, marker='o', markerfacecolor=TEAL,
                               markeredgecolor=TEAL, markersize=4, linewidth=.9,
                               label='95% simultaneous CI (24)'),
                        Line2D([], [], color=GRAY, marker='o', markersize=2.5, linestyle='none',
                               label='Paired blocks'),
                        Line2D([], [], color=INK, marker='x', linestyle='none', markersize=4,
                               label='Inference suppressed')],
               loc='lower center', bbox_to_anchor=(.5, .005), ncol=3, frameon=False)
    return fig


def _robustness(a, primary, tables):
    hid = 'iterative:us:hindi'
    blocks = tables['paired_schedule_blocks'].query('hypothesis_id == @hid').sort_values('seed')
    loo = tables['leave_one_seed_out'].query('hypothesis_id == @hid').sort_values('omitted_seed')
    full = tables['primary_contrasts'].set_index('hypothesis_id').loc[hid]
    if len(blocks) != 8 or not blocks.complete.all() or len(loo) != 8 or not loo.n.eq(7).all():
        raise ValueError('Iterative Hindi robustness requires eight complete blocks and eight seven-block refits')
    sample = blocks.difference.to_numpy()
    if not np.isclose(sample.mean(), full.complete_block_mean, atol=1e-12, rtol=0):
        raise ValueError('Robustness full-sample mean differs')
    for row in loo.itertuples():
        mean = blocks.loc[blocks.seed.ne(row.omitted_seed), 'difference'].mean()
        if not np.isclose(row.mean, mean, atol=1e-12, rtol=0):
            raise ValueError('Leave-one-out mean differs from its retained blocks')
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.9), sharex=True)
    fig.subplots_adjust(left=.12, right=.97, bottom=.26, top=.89, wspace=.48)
    for ax, title in zip(axes, ('a   Eight paired blocks', 'b   Seven-block refits')):
        _axis(ax, title)
        ax.axvline(full.complete_block_mean, color=GRAY, linewidth=.8, linestyle='--')
        ax.set_xlabel('Clustering difference from US-English')
        ax.set_ylim(7.5, -.5)
    for y, row in enumerate(blocks.itertuples()):
        axes[0].plot([0, row.difference], [y, y], color='#C0C0C0', linewidth=.8)
        axes[0].plot(row.difference, y, 'o', color=TEAL, markersize=4)
    for y, row in enumerate(loo.itertuples()):
        significant = pd.notna(row.holm_p) and row.holm_p < .05
        axes[1].plot(row.mean, y, marker='o', markersize=4.5, color=TEAL if significant else GRAY,
                     markerfacecolor=TEAL if significant else 'white', markeredgewidth=.9)
    axes[0].set_yticks(range(8), [str(s) for s in blocks.seed])
    axes[1].set_yticks(range(8), [str(s) for s in loo.omitted_seed])
    axes[0].set_ylabel('Schedule seed')
    axes[1].set_ylabel('Omitted schedule seed')
    lower = min(0, float(blocks.difference.min()), float(loo['mean'].min()))
    upper = max(0, float(blocks.difference.max()), float(loo['mean'].max()))
    pad = max(.005, (upper - lower) * .08)
    axes[0].set_xlim(lower - pad, upper + pad)
    fig.legend(handles=[Line2D([], [], color=GRAY, linestyle='--', linewidth=.8, label='Full-data mean'),
                        Line2D([], [], color=TEAL, marker='o', linestyle='none', markersize=4,
                               label='LOO: Holm p < 0.05'),
                        Line2D([], [], color=GRAY, marker='o', markerfacecolor='white', linestyle='none',
                               markersize=4, label='LOO: Holm p >= 0.05')],
               loc='lower center', bbox_to_anchor=(.5, .015), ncol=3, frameon=False)
    return fig


def _model_means(a, primary, tables):
    descriptive = tables['descriptive_metrics']
    conditions = [tuple(c) for c in a['conditions']]
    fig, axes = _axes_grid()
    for letter, ax, method in zip('abcd', axes, a['methods']):
        metric = a['primary_endpoints'][method]
        name = 'density' if method == 'global' else 'clustering'
        _axis(ax, f'{letter}   {method.title()}: {name}')
        for i, model in enumerate(a['models']):
            rows = descriptive[descriptive.method.eq(method) & descriptive.model.eq(model)
                               & descriptive.metric.eq(metric)].set_index(['culture', 'language'])
            if len(rows) != 7 or set(rows.index) != set(conditions):
                raise ValueError('Require all seven endpoint means per fixed model/method')
            rows = rows.loc[conditions]
            ax.plot(rows['mean'], np.arange(7) + (i - 1.5) * .13, linestyle='none',
                    marker=MARKERS[i], color=MODEL_COLORS[i], markersize=3.7, markeredgewidth=.6)
        ax.set_yticks(range(7), [_condition(*c) for c in conditions])
        ax.set_ylim(6.5, -.5)
        ax.set_xlim((0, .16) if method == 'global' else (0, 1))
        ax.set_xlabel('Mean density' if method == 'global' else 'Mean clustering coefficient')
    fig.legend(handles=[Line2D([], [], color=color, marker=marker, linestyle='none', markersize=4, label=model)
                        for color, marker, model in zip(MODEL_COLORS, MARKERS, a['models'])],
               loc='lower center', bbox_to_anchor=(.5, .005), ncol=4, frameon=False)
    return fig


def render_manuscript(root: Path, output: Path) -> list[Path]:
    """Read hash-sealed original tables; write three PNG/SVG/PDF figure sets.

    Use outputs/manuscript_figures/20261105_v1 for publication, or a temporary
    directory for notebook display. Existing primary evidence is never updated.
    """
    root, output = Path(root).resolve(), Path(output).resolve()
    protected = ('outputs/primary_analysis', 'outputs/final_study', 'outputs/analysis_ready')
    if any(output.is_relative_to(root / name) for name in protected):
        raise ValueError('Manuscript output cannot be inside a protected evidence directory')
    a, primary, tables = _load_sealed(root)
    paths = []
    existing = set(plt.get_fignums())
    with plt.rc_context(STYLE):
        try:
            for name, draw in (('01_primary_contrasts', _forest),
                               ('02_iterative_hindi_robustness', _robustness),
                               ('03_model_endpoint_means', _model_means)):
                fig = draw(a, primary, tables)
                output.mkdir(parents=True, exist_ok=True)
                for suffix in ('png', 'svg', 'pdf'):
                    metadata = {'Date': None} if suffix == 'svg' else (
                        {'CreationDate': None, 'ModDate': None} if suffix == 'pdf' else None)
                    path = output / f'{name}.{suffix}'
                    fig.savefig(path, dpi=300, metadata=metadata)
                    paths.append(path)
                plt.close(fig)
        finally:
            for number in set(plt.get_fignums()) - existing:
                plt.close(number)
    return paths
