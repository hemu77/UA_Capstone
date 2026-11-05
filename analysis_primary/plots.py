"""Primary-only, offline figures from the checked core analysis tables."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from .core import validate_primary


MODEL_COLORS = ('#0072B2', '#D55E00', '#009E73', '#CC79A7')
MODEL_MARKERS = ('o', 's', '^', 'D')
ENDPOINT_LABELS = {'density': 'Density', 'avg_clustering_coef': 'Mean clustering coefficient'}
SCOPE = 'Original primary cohort only: 896 attempts; 890 valid graphs; 6 terminal failures.'


def _condition(culture, language):
    countries = {'us': 'US', 'india': 'India', 'japan': 'Japan', 'brazil': 'Brazil'}
    return f'{countries.get(culture, culture)}-{language.title()}'


def render_figures(root: Path, primary: pd.DataFrame, output: Path) -> list[str]:
    """Write six PNG/SVG pairs, plus optional synthetic/empirical context pairs.

    Core CSVs must already exist. No collection, pipeline execution, or network calls.
    The caller should obtain primary through core.load_inputs for source-hash validation.
    """
    root, output = Path(root), Path(output)
    spec = json.loads((root / 'final_study/contract.json').read_text(encoding='utf8'))
    validate_primary(primary, spec)
    a = spec['analysis']
    methods, models = a['methods'], a['models']
    conditions = [tuple(c) for c in a['conditions']]
    contrasts = pd.read_csv(output / 'primary_contrasts.csv')
    blocks = pd.read_csv(output / 'paired_schedule_blocks.csv')
    descriptive = pd.read_csv(output / 'descriptive_metrics.csv')
    resources = pd.read_csv(output / 'resources_by_method_model.csv')
    expected = {f'{m}:{c}:{l}' for m in methods for c, l in conditions[1:]}
    if len(contrasts) != 24 or set(contrasts.hypothesis_id) != expected:
        raise ValueError('Require exactly the 24 primary contrasts')
    if len(blocks) != 192 or set(blocks.hypothesis_id) != expected:
        raise ValueError('Require all 192 paired schedule blocks')
    if not blocks.complete.map(type).eq(bool).all():
        raise ValueError('Complete block status must be boolean')
    if blocks.loc[~blocks.complete, 'difference'].notna().any():
        raise ValueError('Incomplete blocks cannot carry differences')
    reference_path = output / 'synthetic_reference_per_run.csv'
    reference = None
    if reference_path.is_file():
        reference = pd.read_csv(reference_path)
        reference = reference[reference.metric.eq('avg_clustering_coef')].copy()
        if reference.empty or reference.run_id.duplicated().any():
            raise ValueError('Require unique clustering reference rows')
        identity = primary.set_index('run_id')
        if not reference.run_id.isin(identity.index).all():
            raise ValueError('Supplemental reference rows refused')
        for column in ('valid_graph', 'method', 'model'):
            if not reference[column].reset_index(drop=True).equals(
                    identity.loc[reference.run_id, column].reset_index(drop=True)):
                raise ValueError('Reference primary identity differs: ' + column)
        reference = reference[reference.valid_graph].copy()
        comparable = reference[['observed_value', 'reference_mean']].notna().all(axis=1)
        reference = reference[comparable].copy()
        values = reference[['observed_value', 'reference_mean', 'reference_sd']]
        if (not np.isfinite(values.to_numpy()).all()
                or not reference.reference_replicates.eq(8).all()
                or not values[['observed_value', 'reference_mean']].ge(0).all().all()
                or not values[['observed_value', 'reference_mean']].le(1).all().all()
                or not reference.reference_sd.ge(0).all()):
            raise ValueError('Require bounded reference metrics and eight finite descriptive draws')
        reference['difference'] = reference.observed_value - reference.reference_mean
    empirical_path = output / 'empirical_reference_metrics.csv'
    empirical = None
    context_metrics = ('density', 'avg_clustering_coef', 'global_efficiency')
    if empirical_path.is_file():
        empirical = pd.read_csv(empirical_path)
        if len(empirical) != 8 or empirical.reference.isna().any() or not empirical.reference.is_unique:
            raise ValueError('Require eight uniquely labeled empirical references')
        values = empirical[list(context_metrics)]
        if (not np.isfinite(values.to_numpy()).all() or not values.ge(0).all().all()
                or not values.le(1).all().all() or not empirical.nodes.between(31, 70).all()):
            raise ValueError('Require bounded empirical metrics and reference node counts 31-70')
        for metric in context_metrics:
            if metric in primary:
                values = primary.loc[primary.valid_graph, metric].dropna()
                if not np.isfinite(values).all() or not values.between(0, 1).all():
                    raise ValueError('Primary descriptive metric out of bounds: ' + metric)
    destination = output.resolve() / 'figures'
    destination.mkdir(parents=True, exist_ok=True)
    paths = []

    def save(fig, name, title, caption):
        try:
            fig.suptitle(title, fontsize=15, fontweight='bold', y=.99)
            fig.text(.02, .015, SCOPE + '\n' + caption, ha='left', va='bottom', fontsize=9)
            bottom = .25 if name == '05_resource_use' else .22 if name == '03_model_endpoint_means' else .14
            fig.tight_layout(rect=(0, bottom, 1, .94), h_pad=2.2, w_pad=2)
            for suffix in ('png', 'svg'):
                path = destination / f'{name}.{suffix}'
                metadata = {'Date': None} if suffix == 'svg' else None
                fig.savefig(path, dpi=180, facecolor='white', bbox_inches='tight', metadata=metadata)
                paths.append(str(path))
        finally:
            plt.close(fig)

    def facets():
        return plt.subplots(2, 2, figsize=(14, 10), squeeze=False)

    def contrast_rows(method):
        indexed = contrasts[contrasts.method.eq(method)].set_index(['culture', 'language'])
        return indexed.loc[conditions[1:]]

    def contrast_axis(ax, method, rows):
        ax.set_title(f'{method.title()} | {ENDPOINT_LABELS[a["primary_endpoints"][method]]}')
        ax.set_yticks(range(6), [_condition(*c) for c in conditions[1:]])
        ax.invert_yaxis()
        ax.axvline(0, color='#555555', linewidth=.9, linestyle='--')
        ax.grid(axis='x', alpha=.2)
        ax.set_xlabel('Difference from US-English (endpoint units)')
        ax.set_ylabel('Instruction condition')

    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'svg.fonttype': 'none', 'svg.hashsalt': 'primary-analysis-060dcd3a86d5',
                         'axes.spines.top': False,
                         'axes.spines.right': False}):
        fig, axes = facets()
        for ax, method in zip(axes.flat, methods):
            rows = contrast_rows(method)
            contrast_axis(ax, method, rows)
            labels = []
            for y, ((culture, language), row) in enumerate(rows.iterrows()):
                n = int(row.complete_schedule_blocks)
                suppressed = n < a['fallback_min_complete_blocks'] or 'SUPPRESS' in str(row.status).upper()
                ci = not suppressed and pd.notna(row.ci_lower) and pd.notna(row.ci_upper)
                if ci:
                    ax.plot([row.ci_lower, row.ci_upper], [y, y], color='#0072B2', linewidth=2)
                if pd.notna(row.complete_block_mean):
                    ax.plot(row.complete_block_mean, y, marker='o' if ci else 'x', color='#222222')
                p = f'Holm p={row.holm_p:.3g}' if pd.notna(row.holm_p) and not suppressed else 'no primary test'
                labels.append(f'{_condition(culture, language)}\nn={n}/8; {p}' + ('; no CI' if not ci else ''))
            ax.set_yticks(range(6), labels)
            ax.margins(x=.15)
        save(fig, '01_primary_forest', 'Primary contrasts: 95% simultaneous confidence intervals',
             'Bonferroni family of 24; paired schedule-block t assumptions required. Points average four fixed models.\n'
             'Global US-Japanese: five complete blocks; inference suppressed, no CI. No equivalence or population claims.')

        fig, axes = facets()
        for ax, method in zip(axes.flat, methods):
            contrast_axis(ax, method, contrast_rows(method))
            labels = []
            for y, (culture, language) in enumerate(conditions[1:]):
                group = blocks[blocks.method.eq(method) & blocks.culture.eq(culture) & blocks.language.eq(language)].sort_values('seed')
                complete = group[group.complete]
                offsets = np.linspace(-.2, .2, len(group))
                for offset, row in zip(offsets, group.itertuples()):
                    if row.complete:
                        ax.plot(row.difference, y + offset, 'o', color='#0072B2', markersize=5, alpha=.8)
                labels.append(f'{_condition(culture, language)}\nn={len(complete)}/8 complete')
            ax.set_yticks(range(6), labels)
        save(fig, '02_paired_blocks', 'All 24 contrasts: individual paired schedule-block differences',
             'Each dot is one complete schedule block: four-model mean minus its same-seed US-English baseline.\n'
             'Vertical jitter follows seed order. Incomplete blocks omitted, never zero-filled. Dots show sign variability, not independent models.')

        fig, axes = facets()
        for ax, method in zip(axes.flat, methods):
            metric = a['primary_endpoints'][method]
            group = descriptive[descriptive.method.eq(method) & descriptive.metric.eq(metric)]
            ax.set_title(f'{method.title()} | {ENDPOINT_LABELS[metric]}')
            ax.set_xticks(range(7), [_condition(*c).replace('-', '\n') for c in conditions], rotation=30, ha='right')
            for i, model in enumerate(models):
                indexed = group[group.model.eq(model)].set_index(['culture', 'language']).reindex(conditions)
                x = np.arange(7) + (i - 1.5) * .16
                ax.plot(x, indexed['mean'], linestyle='none', marker=MODEL_MARKERS[i], color=MODEL_COLORS[i], label=model)
                for xx, (_, row) in zip(x, indexed.iterrows()):
                    if pd.notna(row['mean']):
                        ax.annotate(str(int(row.endpoint_defined)), (xx, row['mean']), xytext=(0, 6), textcoords='offset points', ha='center', fontsize=7)
            ax.set_ylim(0, 1.08)
            ax.set_ylabel(f'{ENDPOINT_LABELS[metric]} (0-1)')
            ax.set_xlabel('Instruction condition (first: US-English baseline)')
            ax.grid(axis='y', alpha=.2)
        handles = [Line2D([], [], color=c, marker=m, linestyle='none', label=n)
                   for c, m, n in zip(MODEL_COLORS, MODEL_MARKERS, models)]
        fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .095), ncol=4)
        save(fig, '03_model_endpoint_means', 'Descriptive endpoint means by fixed model and instruction',
             'Numbers beside points: defined endpoints, out of eight attempts per model-condition. Failures excluded, not zero.\n'
             'Global uses density; other methods use clustering. Separate endpoints are not a quality ranking; no inferential intervals here.')

        fig, axes = facets()
        for ax, method in zip(axes.flat, methods):
            matrix, labels = [], []
            metric = a['primary_endpoints'][method]
            for model in models:
                row, text = [], []
                for culture, language in conditions:
                    cell = primary[primary.method.eq(method) & primary.model.eq(model) & primary.culture.eq(culture) & primary.language.eq(language)]
                    valid = cell.valid_graph
                    defined = int(cell.loc[valid, metric].notna().sum())
                    failures = int((~valid).sum())
                    undefined = int(valid.sum()) - defined
                    row.append(defined / len(cell))
                    text.append(f'{defined}/{len(cell)}\nF{failures} U{undefined}')
                matrix.append(row)
                labels.append(text)
            ax.imshow(matrix, vmin=0, vmax=1, cmap='Blues', aspect='auto')
            for i, row in enumerate(labels):
                for j, text in enumerate(row):
                    ax.text(j, i, text, ha='center', va='center', fontsize=8,
                            color='white' if matrix[i][j] > .65 else 'black')
            ax.set_title(f'{method.title()} | defined primary endpoints')
            ax.set_yticks(range(4), models)
            ax.set_xticks(range(7), [_condition(*c).replace('-', '\n') for c in conditions], rotation=30, ha='right')
            ax.set_xlabel('Instruction condition (US-English baseline first)')
            ax.set_ylabel('Fixed model')
        save(fig, '04_endpoint_coverage', 'Coverage and missingness: every original model-condition cell',
             'Cell text: defined endpoints / attempts; F = terminal failures; U = valid graphs with undefined endpoint.\n'
             'Blue intensity is the defined fraction (0-1); exact counts carry the information. Valid empty graphs are not failures.')

        fig, axes = plt.subplots(1, 3, figsize=(15, 6))
        for ax, column, label in zip(axes, ('requests', 'received_conservative_usd', 'api_seconds'),
                ('Received request count (calls)', 'Received conservative cost (USD)', 'Serial provider time (seconds)')):
            for i, model in enumerate(models):
                indexed = resources[resources.model.eq(model)].set_index('method').reindex(methods)
                ax.bar(np.arange(4) + (i - 1.5) * .19, indexed[column], width=.18,
                       color=MODEL_COLORS[i], hatch=('', '//', '..', 'xx')[i], label=model)
            ax.set_xticks(range(4), [m.title() for m in methods], rotation=25)
            ax.set_ylim(bottom=0)
            ax.set_ylabel(label)
            ax.set_xlabel('Generation method')
            ax.grid(axis='y', alpha=.2)
        fig.legend(handles=axes[0].get_legend_handles_labels()[0], labels=models,
                   loc='lower center', bbox_to_anchor=(.5, .13), ncol=4)
        counts = ', '.join(f'{m}: n={int(resources.loc[resources.method.eq(m), "attempts"].sum())}' for m in methods)
        save(fig, '05_resource_use', 'Resource use: unequal request budgets, not an efficiency winner',
             counts + ' attempts. Includes failed attempts.\n'
             'Received-cost estimates exclude unallocated retained reservations; not an invoice. Serial API time is not wall time. No equal-compute claim.')

        fig, axes = facets()
        for ax, method in zip(axes.flat, methods):
            rows = contrast_rows(method)
            contrast_axis(ax, method, rows)
            for y, (_, row) in enumerate(rows.iterrows()):
                ax.plot([row.missingness_lower, row.missingness_upper], [y, y], color='#D55E00', linewidth=3, marker='|')
                if pd.notna(row.complete_block_mean):
                    ax.plot(row.complete_block_mean, y, 'o', color='#222222', markersize=4)
            ax.set_yticks(range(6), [f'{_condition(*c)}\nn={int(r.complete_schedule_blocks)}/8 complete'
                                   for c, (_, r) in zip(conditions[1:], rows.iterrows())])
        save(fig, '06_missingness_bounds', 'Missingness sensitivity: bounded extremes over all eight blocks',
             'Orange segments are identification bounds with missing endpoints allowed anywhere in [0, 1], NOT confidence intervals.\n'
             'Black dots: complete-block means. A collapsed bound is not certainty. All differences use the same-seed US-English baseline.')
        if reference is not None:
            fig, axes = facets()
            extent = max(.01, float(reference.difference.abs().max())) if len(reference) else .01
            for ax, method in zip(axes.flat, methods):
                labels = []
                for i, model in enumerate(models):
                    values = reference.loc[reference.method.eq(method) & reference.model.eq(model)].sort_values('run_id').difference
                    labels.append(f'{model}\nn={len(values)} defined graphs')
                    ax.scatter(values, i + np.linspace(-.2, .2, len(values)), s=18,
                               color=MODEL_COLORS[i], marker=MODEL_MARKERS[i], alpha=.5)
                    if len(values):
                        ax.scatter(values.mean(), i, s=95, color=MODEL_COLORS[i],
                                   marker=MODEL_MARKERS[i], edgecolors='black', linewidths=1.2, zorder=3)
                ax.set_yticks(range(4), labels)
                ax.set_ylim(3.5, -.5)
                ax.set_xlim(-extent * 1.1, extent * 1.1)
                ax.axvline(0, color='#555555', linewidth=.9, linestyle='--')
                ax.set_title(method.title())
                ax.set_xlabel('Observed clustering minus G(n,m) mean (endpoint units)')
                ax.set_ylabel('Fixed model')
                ax.grid(axis='x', alpha=.2)
            save(fig, '07_gnm_clustering_context', 'Clustering relative to equal-node, equal-edge G(n,m) context',
                 'Small marks: defined valid graphs, pooling seven instruction conditions; outlined marks: mean residual per model/method.\n'
                 'Eight synthetic draws per graph; no CI or new p-values. Descriptive context, not realism or empirical ground truth; not a US-English contrast.')
        if empirical is not None:
            fig, axes = plt.subplots(1, 3, figsize=(18, 9), sharey=True)
            labels = [m.title() for m in methods] + ['Published networks'] + [
                f'{row.reference} ({int(row.nodes)} nodes)' for row in empirical.itertuples()]
            for ax, metric in zip(axes, context_metrics):
                for y, method in enumerate(methods):
                    group = primary[primary.method.eq(method)]
                    values = group.loc[group.valid_graph, metric].dropna() if metric in group else pd.Series(dtype=float)
                    if len(values):
                        ax.boxplot([values], positions=[y], vert=False, widths=.45, manage_ticks=False,
                                   showfliers=False, patch_artist=True,
                                   boxprops=dict(facecolor='#DAE3E8', edgecolor='#526875'),
                                   medianprops=dict(color='black'),
                                   whiskerprops=dict(color='#526875'), capprops=dict(color='#526875'))
                        ax.scatter(values, y + np.linspace(-.15, .15, len(values)), s=6,
                                   color='#526875', alpha=.25, zorder=2)
                    ax.text(.98, y - .3, f'n={len(values)}/{len(group)}', ha='right',
                            va='center', fontsize=8, transform=ax.get_yaxis_transform())
                ax.scatter(empirical[metric], np.arange(5, 13), marker='D', color='#222222', s=35, zorder=3)
                ax.axhline(4, color='#888888', linestyle='--', linewidth=.8)
                ax.text(.98, 4.3, '8 published network records', ha='right', fontsize=8,
                        transform=ax.get_yaxis_transform())
                ax.set_title(ENDPOINT_LABELS.get(metric, 'Global efficiency'))
                ax.set_xlabel('Metric value (0-1)')
                ax.set_xlim(0, 1)
                ax.set_ylim(12.6, -.8)
                ax.set_yticks(range(13), labels)
                ax.grid(axis='x', alpha=.2)
            axes[0].set_ylabel('Primary method (pooled fixed models) / named empirical record')
            save(fig, '08_empirical_context', 'Descriptive structural context: primary distributions and eight published networks',
                 'Primary CSV only: valid defined graphs, 50 nodes; four fixed models and seven instruction conditions pooled by method.\n'
                 'Boxes: median/IQR; whiskers: 1.5 IQR; small dots: graphs; diamonds: named references, 31-70 nodes. Tailor waves are not independent.\n'
                 'Different tie definitions and sizes; missing metrics stay missing. No CIs or tests; not a realism benchmark or winner ranking.')
    return paths
