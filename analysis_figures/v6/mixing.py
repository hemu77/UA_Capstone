"""Figures 14, 16 and 17: RQ2 demographic mixing in the primary cohort (878 non-empty graphs).

Descriptive only. One fixed roster, correlated attributes and a fixed prompt field order:
no score here measures independent attribute importance or human preference.
"""
import itertools

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

from analysis_figures.v6 import style as S

CELL = ['model', 'culture', 'language']
ATTRIBUTE_COLOR = {'political orientation': '#0072B2', 'religion': '#D55E00', 'gender': '#59636B'}
ATTRIBUTE_MARKER = {'political orientation': 'o', 'religion': 'D', 'gender': '^'}
SCORES = (('excess', 'Excess edge share'), ('coleman', 'Coleman-type index'), ('newman', 'Newman assortativity'))
PAIRS = tuple(itertools.combinations(S.METHODS, 2))


def _cell_means(D):
    scores = D['ext']['mixing_scores']
    defined = scores[scores.status.eq('nonempty')]
    if defined.run_id.nunique() != 878:
        raise ValueError('Expected 878 non-empty primary graphs')
    cells = defined.groupby(['method', 'attribute'] + CELL).excess.agg(['mean', 'size']).reset_index()
    if len(cells) != 4 * 3 * 28:
        raise ValueError('A model-setting cell has no defined graph')
    return cells


def fig14(D):
    """Same-group ties above roster chance, and how far politics leads its closest competitor."""
    cells = _cell_means(D)
    rank = D['ext']['mixing_rank_sensitivity']
    rank = rank[rank.omitted_seed.eq(-1) & rank.support.eq('all_three_scores')]
    fig, axes = plt.subplots(1, 2, figsize=(S.FULL, 4.35), gridspec_kw=dict(width_ratios=[1.12, 1]))
    ax = axes[0]
    S.panel(ax, 'a', 'Same-group tie share above roster chance')
    ax.axvline(0, color=S.INK, lw=.7)
    out, ticks = [], []
    for m, method in enumerate(S.METHODS):
        for a, attribute in enumerate(S.ATTRIBUTES):
            y = m * 3.7 + a
            values = cells[cells.method.eq(method) & cells.attribute.eq(attribute)]['mean'].to_numpy()
            color = ATTRIBUTE_COLOR[attribute]
            ax.plot(values, np.full(28, y), '|', ms=6, mew=.7, color=color, alpha=.5, zorder=2)
            ax.plot(values.mean(), y, ATTRIBUTE_MARKER[attribute], ms=4.8, color=color, mec='white', mew=.6, zorder=4)
            ticks.append((y, S.ATTRIBUTE_LABEL[attribute]))
            out.append(dict(panel='a', method=method, row=attribute, cells=28, equal_cell_mean=values.mean(),
                            min_cell=values.min(), max_cell=values.max()))
        ax.text(-.27, m * 3.7 + 1, method.title(), transform=ax.get_yaxis_transform(), rotation=90, ha='center',
                va='center', fontsize=7, fontweight='bold', color=S.METHOD_COLOR[method])
    ax.set_yticks([y for y, _ in ticks], [t for _, t in ticks])
    ax.set_ylim(3 * 3.7 + 2.6, -.7)
    ax.set_xlabel('Share of ties within a group minus the share\nexpected from roster group sizes (0 = chance)')
    bx = axes[1]
    S.panel(bx, 'b', 'Politics minus its closest competitor')
    bx.axvline(0, color=S.INK, lw=.7)
    ticks = []
    for m, method in enumerate(S.METHODS):
        for k, (score, label) in enumerate(SCORES):
            y = m * 3.7 + k
            part = rank[rank.method.eq(method) & rank.score.eq(score)]
            values = part.politics_minus_competitor.to_numpy()
            first = int(part.winner.eq('political orientation').sum())
            if len(values) != 28:
                raise ValueError('Expected 28 cells')
            color = S.METHOD_COLOR[method]
            bx.plot(values, np.full(28, y), '|', ms=6, mew=.7, color=color, alpha=.5, zorder=2)
            bx.plot(values.mean(), y, 'o', ms=4.4, mfc='white', mec=color, mew=1.1, zorder=4)
            bx.text(1.02, y, f'{first}/28', transform=bx.get_yaxis_transform(), va='center', fontsize=6.3,
                    fontweight='bold' if first in (0, 28) else 'normal', color=S.INK)
            ticks.append((y, label))
            out.append(dict(panel='b', method=method, row=score, cells=28, equal_cell_mean=values.mean(),
                            min_cell=values.min(), max_cell=values.max(), politics_first_cells=first))
    bx.text(1.02, -.95, 'Politics\nfirst', transform=bx.get_yaxis_transform(), va='center', fontsize=6, color=S.MUTED,
            linespacing=1.1)
    bx.set_yticks([y for y, _ in ticks], [t for _, t in ticks])
    bx.set_ylim(3 * 3.7 + 2.6, -1.5)
    ax.set_ylim(3 * 3.7 + 2.6, -1.5)
    bx.set_xlabel('Politics score minus the larger of religion and gender\n(score units; left of 0 = another attribute leads)')
    fig.legend(handles=[S.handle('One model × instruction cell (28 per row)', marker='|', ms=6, mew=.7, color=S.MUTED),
                        S.handle('Equal-weight mean of the 28 cells', marker='o', ms=4.6, color=S.MUTED)],
               loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(.5, .075))
    S.note(fig, '878 non-empty primary graphs; 12 empty graphs have no defined score and 6 failed attempts have no '
           'graph. Ticks show spread across fixed cells, not confidence intervals. One roster, correlated '
           'attributes and a fixed prompt order: ranks describe these graphs, not importance or human preference.')
    fig.subplots_adjust(left=.175, right=.94, top=.93, bottom=.275, wspace=.62)
    return fig, pd.DataFrame(out)


def fig16(D):
    """Paired procedure differences, seed by seed within a cell, then 28 cells weighted equally."""
    cells = D['ext']['mixing_paired_cells']
    cells = cells[cells.score.eq('excess') & cells.support.eq('score_specific')]
    summary = D['ext']['mixing_method_sensitivity']
    summary = summary[summary.score.eq('excess') & summary.support.eq('score_specific') & summary.omitted_seed.eq(-1)]
    fig, axes = plt.subplots(1, 3, figsize=(S.FULL, 3.75), sharey=True, sharex=True)
    out = []
    for ax, letter, attribute in zip(axes, 'abc', S.ATTRIBUTES):
        S.panel(ax, letter, S.ATTRIBUTE_LABEL[attribute])
        ax.axvline(0, color=S.INK, lw=.7)
        for y, (first, second) in enumerate(PAIRS):
            part = cells[cells.attribute.eq(attribute) & cells['first'].eq(first) & cells.second.eq(second)]
            row = summary[summary.attribute.eq(attribute) & summary['first'].eq(first) & summary.second.eq(second)].iloc[0]
            values = part.difference.to_numpy()
            if len(values) != 28 or not np.isclose(values.mean(), row.mean_difference, atol=1e-12, rtol=0):
                raise ValueError('Paired cell means differ from the published summary')
            color = S.METHOD_COLOR[second]
            ax.plot(values, np.full(28, y), '|', ms=6, mew=.7, color=color, alpha=.5, zorder=2)
            ax.plot(values.mean(), y, 'o', ms=4.6, color=color, mec='white', mew=.6, zorder=4)
            same = int(max((values > 0).sum(), (values < 0).sum()))
            ax.text(.985, y - .37, f'{same}/28 same sign', transform=ax.get_yaxis_transform(), ha='right', va='center',
                    fontsize=5.8, fontweight='bold' if same == 28 else 'normal',
                    color=S.INK if same == 28 else S.MUTED, zorder=5,
                    bbox=dict(facecolor='white', edgecolor='none', pad=.5, alpha=.85))
            out.append(dict(attribute=attribute, first=first, second=second, mean_difference=values.mean(),
                            min_cell=values.min(), max_cell=values.max(), cells_same_sign=same, cells=28,
                            defined_pairs=int(row.defined_pairs)))
        ax.set_xlabel('Difference in excess\nsame-group tie share')
    labels = []
    pairs_n = {(r['first'], r['second']): r['defined_pairs'] for r in out if r['attribute'] == S.ATTRIBUTES[0]}
    for first, second in PAIRS:
        labels.append(f'{second.title()} − {first.title()}\n{pairs_n[(first, second)]} paired seeds')
    axes[0].set_yticks(range(6), labels)
    axes[0].set_ylim(5.6, -.75)
    fig.legend(handles=[S.handle('One model × instruction cell (mean of its paired seeds)', marker='|', ms=6, mew=.7,
                                 color=S.MUTED),
                        S.handle('Equal-weight mean of 28 cells', marker='o', ms=4.6, color=S.MUTED)],
               loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(.5, .075))
    S.note(fig, 'Procedures are paired on model, instruction setting and seed before averaging. Pairs with an empty or '
           'failed Global graph are missing, not replaced. Spread across fixed cells is not a confidence interval '
           'and no test is implied. Supplementary graphs are excluded.')
    fig.subplots_adjust(left=.2, right=.985, top=.93, bottom=.3, wspace=.08)
    return fig, pd.DataFrame(out)


def fig17(D):
    """Who is tied to whom, relative to what the roster's group sizes alone would produce."""
    matrices = D['ext']['mixing_matrices']
    sizes = D['roster_sizes']
    mean = matrices.groupby(['method', 'attribute', 'row_category', 'column_category']).fraction.agg(['mean', 'size'])
    if mean['size'].ne(28).any():
        raise ValueError('Each matrix entry needs all 28 cells')
    fig = plt.figure(figsize=(S.FULL, 6.1))
    grid = fig.add_gridspec(3, 5, width_ratios=[1, 1, 1, 1, .06], height_ratios=[3, 5, 3], left=.15, right=.93,
                            top=.94, bottom=.115, wspace=.14, hspace=.36)
    norm = TwoSlopeNorm(vmin=-2.4, vcenter=0, vmax=2.4)
    out, image = [], None
    for r, attribute in enumerate(S.ATTRIBUTES):
        n = sizes[attribute]
        cats = list(n.index)
        total = n.sum()
        for c, method in enumerate(S.METHODS):
            ax = fig.add_subplot(grid[r, c])
            observed = np.array([[mean.loc[(method, attribute, a, b), 'mean'] for b in cats] for a in cats])
            expected = np.array([[n[a] * (n[b] - (a == b)) / (total * (total - 1)) for b in cats] for a in cats])
            if not np.isclose(observed.sum(), 1, atol=1e-9) or not np.isclose(expected.sum(), 1, atol=1e-12):
                raise ValueError('Mixing matrix does not sum to one')
            ratio = observed / expected
            image = ax.imshow(np.log2(ratio), cmap='PuOr_r', norm=norm, aspect='equal')
            for i, j in itertools.product(range(len(cats)), repeat=2):
                strong = abs(np.log2(ratio[i, j])) > 1.35
                ax.text(j, i, f'{ratio[i, j]:.1f}', ha='center', va='center', fontsize=5.6 if len(cats) > 3 else 6.3,
                        color='white' if strong else S.INK)
                out.append(dict(attribute=attribute, method=method, row_category=cats[i], column_category=cats[j],
                                mean_oriented_edge_fraction=observed[i, j], roster_chance_fraction=expected[i, j],
                                ratio_to_roster_chance=ratio[i, j], cells=28))
            short = [x.replace('-leaning', '').replace('Nonreligious', 'None').replace('Nonbinary', 'Nonbin.') for x in cats]
            ax.set_xticks(range(len(cats)), short, rotation=90 if len(cats) > 3 else 0, fontsize=5.8)
            ax.set_yticks(range(len(cats)), [f'{s} ({n[k]})' for s, k in zip(short, cats)] if c == 0 else [],
                          fontsize=5.8)
            ax.tick_params(length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
            if r == 0:
                ax.set_title(method.title(), fontsize=7.5, fontweight='bold', color=S.METHOD_COLOR[method], pad=6)
            if c == 0:
                ax.annotate(S.ATTRIBUTE_LABEL[attribute], (0, .5), xycoords='axes fraction', xytext=(-62, 0),
                            textcoords='offset points', rotation=90, ha='center', va='center', fontsize=7.5,
                            fontweight='bold', annotation_clip=False)
    bar = fig.colorbar(image, cax=fig.add_subplot(grid[:, 4]), ticks=[-2, -1, 0, 1, 2])
    bar.ax.set_yticklabels(['¼×', '½×', 'chance', '2×', '4×'], fontsize=6.3)
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0)
    S.note(fig, 'Number = mean share of tie ends in that pair of groups divided by the share expected from roster group '
           'sizes (group sizes in parentheses); 1.0 = chance. Color is anchored at chance within every matrix, so it '
           'compares pairs inside one attribute and one procedure; a darker attribute is not a more important one. '
           'Non-empty graphs averaged within each model × instruction cell, then 28 cells weighted equally.')
    return fig, pd.DataFrame(out)


FIGURES = {'14_demographic_mixing': fig14, '16_rq2_paired_mixing': fig16, '17_rq2_mixing_matrices': fig17}
