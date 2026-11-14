"""Figures from the versioned offline tables; no changes to prior figures."""
from itertools import combinations

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from analysis_extension.core import ATTRIBUTES, METHODS
from analysis_figures.publication import STYLE, COLORS

CAPTIONS = {
    '16_rq2_paired_mixing': 'Second method minus first, paired on model, instruction setting and seed. Excess same-group edge share is the fraction of same-attribute edges minus finite-roster chance. Points equally average 28 cell means; spans show their minimum and maximum, NOT confidence intervals. Only common defined pairs for this score enter each cell; failures and empty graphs remain reported separately. No causal attribute importance or new significance test.',
    '17_rq2_mixing_matrices': 'Mean oriented edge fractions: each undirected edge counts in both directions. Each matrix sums to one. Average nonempty graphs within each model/instruction cell, then weight all 28 cells equally. Color scale shared across panels. Diagonal concentration depends on category and edge-end shares; it is not independent attribute importance. All 878 nonempty primary graphs; supplements excluded.',
    '18_sampling_distortion': 'Black diamonds are full observed-record values. Colored points are means of 32 uniformly sampled, overlapping induced graphs at each size; spans are the complete draw ranges, NOT confidence intervals. N labels the observed roster, not a complete population census. Sampling-induced isolates are retained. Copenhagen uses friendship endpoints only; unobserved full-network isolates cannot be recovered.',
    '19_copenhagen_transfer': 'W1 is smaller when distributions match on that metric alone. Points average four fixed model distances or eight simulation-bank distances; spans show their range, NOT confidence intervals. Seven instruction settings retain equal weight within each model. Fixed ER/BA/WS were fitted only to the old development panel. The separate target-ER row uses Copenhagen density and is an informed diagnostic, not a transfer competitor. One endpoint-conditioned university friendship record, 32 overlapping n=50 samples; no population or demographic validation.'}
for name in ('16_rq2_paired_mixing', '17_rq2_mixing_matrices', '19_copenhagen_transfer'):
    CAPTIONS[name] += ' Exploratory extension using primary-source graphs: 896 attempts, 890 valid including 12 empty, six failures; six supplements excluded.'
CAPTIONS['18_sampling_distortion'] += ' This panel uses 12 observed records: ten overlapping Facebook records, one school record and one Copenhagen record. Each supplies 32 draws at each nested size; these are not independent population replications.'
CAPTIONS['19_copenhagen_transfer'] += ' W1 conditions on valid outputs. All six specified metrics are shown. Copenhagen: 800 endpoints, 6418 simple edges after excluding 11 self-links from 6429 raw records.'


def paired(folder):
    table = pd.read_csv(folder/'mixing_method_sensitivity.csv')
    table = table[(table.omitted_seed == -1) & table.score.eq('excess') & table.support.eq('score_specific')]
    pairs = list(combinations(METHODS, 2))
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 5.3), sharex=True, sharey=True)
    for ax, attr in zip(axes, ATTRIBUTES):
        for y, (first, second) in enumerate(pairs):
            row = table[(table.attribute == attr) & table['first'].eq(first) & table['second'].eq(second)].iloc[0]
            color = COLORS[METHODS.index(second)]
            ax.plot([row.min_cell, row.max_cell], [y, y], color=color, lw=2, alpha=.5)
            ax.scatter(row.mean_difference, y, s=40, color=color, zorder=3)
        ax.axvline(0, color='#8a9298', lw=.8, linestyle='--')
        ax.set_title(attr.capitalize(), loc='left', fontweight='bold')
        ax.set_xlabel('Difference in excess edge share')
        ax.grid(axis='x', alpha=.15)
    axes[0].set_yticks(range(6), [b.title()+' - '+a.title() for a, b in pairs])
    axes[0].invert_yaxis()
    fig.suptitle('RQ2 | Procedure differences, measured on the same saved cases', x=.03, ha='left', fontsize=14)
    fig.text(.03, .04, 'Exploratory; supplements excluded. Dots: equal-cell means. Spans: 28-cell heterogeneity, not CIs.', fontsize=9)
    fig.subplots_adjust(left=.19, right=.98, top=.82, bottom=.19, wspace=.16)
    return fig


def matrices(folder):
    table = pd.read_csv(folder/'mixing_matrices.csv')
    means = table.groupby(['method', 'attribute', 'row_category', 'column_category']).fraction.mean()
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    upper = means.max()
    for i, attr in enumerate(ATTRIBUTES):
        for j, method in enumerate(METHODS):
            matrix = means.loc[method, attr].unstack().sort_index().sort_index(axis=1)
            ax = axes[i, j]
            im = ax.imshow(matrix, vmin=0, vmax=upper, cmap='Blues', interpolation='nearest')
            ax.set_xticks(range(len(matrix)), matrix.columns, rotation=40, ha='right')
            ax.set_yticks(range(len(matrix)), matrix.index if j == 0 else ['']*len(matrix))
            ax.tick_params(length=0, labelsize=8)
            if i == 0:
                ax.set_title(method.title(), fontweight='bold')
            if j == 0:
                ax.set_ylabel(attr.capitalize(), labelpad=12)
            for r in range(len(matrix)):
                for c in range(len(matrix)):
                    value = matrix.iloc[r, c]
                    ax.text(c, r, f'{value:.2f}', ha='center', va='center', fontsize=7.5,
                            color='white' if value > upper*.6 else '#23323f')
            for spine in ax.spines.values():
                spine.set_visible(False)
    fig.suptitle('RQ2 | Where the edge ends meet', x=.03, ha='left', fontsize=14)
    fig.text(.03, .025, 'Oriented edge fractions; each matrix sums to 1. Equal model/setting weights, nonempty primary graphs only.', fontsize=9)
    fig.subplots_adjust(left=.15, right=.88, top=.91, bottom=.12, hspace=.8, wspace=.3)
    cax = fig.add_axes([.91, .30, .015, .42])
    fig.colorbar(im, cax=cax, label='Mean fraction of oriented edges')
    return fig


def sampling(folder):
    full = pd.read_csv(folder/'full_network_metrics.csv')
    samples = pd.read_csv(folder/'observed_sample_metrics.csv')
    metrics = ['density', 'avg_clustering_coef', 'global_efficiency']
    fig, axes = plt.subplots(1, 3, figsize=(12, 7.4), sharey=True)
    colors, symbols = ['#bd721e', '#38805d', '#2274a5'], ['o', 's', '^']
    for ax, metric, label in zip(axes, metrics, ['Density', 'Clustering', 'Global efficiency']):
        for y, row in enumerate(full.itertuples()):
            group = samples[(samples.source == row.source) & (samples.cohort == row.cohort)]
            for n, offset, color, marker in zip((25, 40, 50), (-.20, 0, .20), colors, symbols):
                values = group[group.n.eq(n)][metric]
                ax.plot([values.min(), values.max()], [y+offset]*2, color=color, alpha=.35, lw=1)
                ax.scatter(values.mean(), y+offset, s=18, color=color, marker=marker)
            ax.scatter(getattr(row, metric), y, color='#202a32', marker='D', s=23, zorder=4)
        ax.set_title(label, loc='left', fontweight='bold')
        ax.set_xlabel('Coefficient')
        ax.grid(axis='x', alpha=.15)
    axes[0].set_yticks(range(len(full)), [f'{r.cohort} (N={r.nodes})' for r in full.itertuples()])
    axes[0].invert_yaxis()
    handles = [Line2D([], [], marker='D', color='#202a32', linestyle='None', label='Full observed record')]
    handles += [Line2D([], [], marker=m, color=c, linestyle='-', label=f'n={n} mean / draw range')
                for n, c, m in zip((25, 40, 50), colors, symbols)]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.56, .94), ncol=2, frameon=False)
    fig.suptitle('Observation size changes what a network looks like', x=.025, ha='left', fontsize=14)
    fig.text(.025, .025, '32 overlapping draws per record. Ranges show sampling variation, not confidence intervals. Isolates retained.', fontsize=9)
    fig.subplots_adjust(left=.19, right=.98, top=.82, bottom=.11, wspace=.16)
    return fig


def copenhagen(folder):
    data = pd.concat([pd.read_csv(folder/'transfer_record_summary.csv'), pd.read_csv(folder/'target_er_summary.csv')])
    data = data[data.source.eq('copenhagen_friendship') & data.n.eq(50)]
    names = [*METHODS, 'ER', 'BA', 'WS', 'target_ER']
    labels = [m.title() for m in METHODS]+['ER (fixed)', 'BA (fixed)', 'WS (fixed)', 'ER (target density)']
    colors = [*COLORS, '#6d777d', '#6d777d', '#6d777d', '#ad7627']
    metrics = ['density', 'avg_clustering_coef', 'global_efficiency', 'transitivity', 'degree_gini', 'prop_nodes_lcc']
    fig, axes = plt.subplots(1, 6, figsize=(14, 5.8), sharey=True)
    for ax, metric, label in zip(axes, metrics, ['Density', 'Clustering', 'Efficiency', 'Transitivity', 'Degree inequality', 'Largest component']):
        for y, (name, color) in enumerate(zip(names, colors)):
            row = data[(data.procedure == name) & (data.metric == metric)].iloc[0]
            ax.plot([row.min_model_or_bank, row.max_model_or_bank], [y]*2, color=color, alpha=.45, lw=2)
            ax.scatter(row.mean_w1, y, color=color, marker='D' if y == 7 else 'o', s=32)
        ax.axhline(6.5, color='#9ca4a9', lw=.8, ls='--')
        ax.set_title(label, loc='left', fontsize=10, fontweight='bold')
        ax.set_xlabel('W1 distance\nLower is closer')
        ax.grid(axis='x', alpha=.15)
        ax.set_xlim(left=-.01)
    axes[0].set_yticks(range(8), labels)
    axes[0].invert_yaxis()
    fig.suptitle('Copenhagen | Transfer results and a separate density-informed diagnostic', x=.025, ha='left', fontsize=13)
    fig.text(.025, .045, 'Exploratory, valid-output conditional; one endpoint-conditioned record, n=50. Spans: model/bank ranges, not CIs.\nThe bottom row uses target information, NOT a fair transfer competitor. 896 attempts: 890 valid (12 empty), 6 failed; no supplements.', fontsize=9)
    fig.subplots_adjust(left=.17, right=.99, top=.84, bottom=.20, wspace=.24)
    return fig


def render(folder):
    drawers = [paired, matrices, sampling, copenhagen]
    with plt.rc_context(STYLE):
        for name, draw in zip(CAPTIONS, drawers):
            fig = draw(folder)
            for ext, metadata in [('png', None), ('svg', {'Date': None}), ('pdf', {'CreationDate': None, 'ModDate': None})]:
                fig.savefig(folder/(name+'.'+ext), dpi=200, metadata=metadata)
            plt.close(fig)
