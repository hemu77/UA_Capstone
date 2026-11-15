"""Figures 10, 18 and 19: what a small induced sample keeps, and the Copenhagen record.

Draw-to-draw spread is sampling variation of overlapping node draws from ONE record. It is
never a confidence interval, and the draws are not independent communities.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis_figures.v6 import style as S
from analysis_figures.v6.heldout import PROCEDURES, ROW, _color, _label

THREE = ('density', 'avg_clustering_coef', 'global_efficiency')
SOURCE_COLOR = {'snap_facebook': S.FACEBOOK, 'sociopatterns_friendship': S.SCHOOL, 'copenhagen_friendship': S.COPENHAGEN}
SOURCE_MARKER = {'snap_facebook': 'o', 'sociopatterns_friendship': 'D', 'copenhagen_friendship': 's'}
SOURCE_LABEL = {'snap_facebook': 'Facebook record (10)', 'sociopatterns_friendship': 'School record',
                'copenhagen_friendship': 'Copenhagen record (endpoint-conditioned)'}


def _tables(D):
    full = D['ext']['full_network_metrics'].set_index('cohort')
    draws = D['ext']['observed_sample_metrics']
    if len(full) != 12 or draws.groupby(['cohort', 'n']).size().ne(32).any():
        raise ValueError('Expected 12 records with 32 draws per size')
    return full, draws


def _source_handles():
    return [S.handle(SOURCE_LABEL[s], marker=SOURCE_MARKER[s], ms=4.4, color=SOURCE_COLOR[s]) for s in SOURCE_COLOR]


def fig10(D):
    """How much of the full record's value survives in a 50-node induced draw."""
    full, draws = _tables(D)
    draws = draws[draws.n.eq(50)]
    fig, axes = plt.subplots(1, 3, figsize=(S.FULL, 3.35), sharey=True, sharex=True)
    out = []
    for ax, letter, metric in zip(axes, 'abc', THREE):
        S.panel(ax, letter, S.METRIC_LABEL[metric], grid='both')
        ax.axhline(1, color=S.INK, lw=.7)
        for cohort, row in full.iterrows():
            ratio = draws[draws.cohort.eq(cohort)][metric].to_numpy() / row[metric]
            x, color = 50 / row.nodes, SOURCE_COLOR[row.source]
            ax.plot([x, x], [ratio.min(), ratio.max()], color=color, lw=.8, alpha=.55, zorder=2)
            ax.plot(x, ratio.mean(), SOURCE_MARKER[row.source], ms=4.4, color=color, mec='white', mew=.5, zorder=3)
            out.append(dict(metric=metric, cohort=cohort, source=row.source, full_nodes=int(row.nodes),
                            sampling_fraction=x, full_value=row[metric], mean_draw_value=ratio.mean() * row[metric],
                            mean_ratio=ratio.mean(), min_ratio=ratio.min(), max_ratio=ratio.max(), draws=32))
        ax.set_xscale('log')
        ax.set_xticks([.05, .1, .2, .5, .85], ['5%', '10%', '20%', '50%', '85%'])
        ax.minorticks_off()
        ax.set_xlabel('Share of the record in the draw\n(50 ÷ record size, log scale)')
    axes[0].set_ylabel('50-node draw ÷ full record')
    axes[0].set_ylim(0, max(r['max_ratio'] for r in out) * 1.04)
    fig.legend(handles=_source_handles() + [S.handle('Range of 32 draws (not a CI)', ls='-', lw=.8, color=S.MUTED)],
               loc='lower center', ncol=4, frameon=False, columnspacing=1.3, bbox_to_anchor=(.5, .075))
    S.note(fig, 'Mark = mean of 32 overlapping 50-node induced draws divided by the value of the full observed record; '
           '1 = the draw keeps the full-record value. Nodes isolated by sampling stay in every metric.')
    fig.subplots_adjust(left=.085, right=.985, top=.92, bottom=.335, wspace=.08)
    return fig, pd.DataFrame(out)


def fig18(D):
    """Full-record structure and induced-sample structure are different objects."""
    full, draws = _tables(D)
    order = list(full.sort_values('nodes').index)
    metrics = THREE + ('isolate_share',)
    titles = dict(S.METRIC_LABEL, isolate_share='Isolated nodes (share)')
    draws = draws.assign(isolate_share=draws.isolates / draws.n)
    full = full.assign(isolate_share=full.isolates / full.nodes)
    fig, axes = plt.subplots(1, 4, figsize=(S.FULL, 4.3), sharey=True)
    sizes = ((25, -.27, '#B0BEC5', 3.2), (40, 0, '#607D8B', 3.6), (50, .27, '#1F2933', 4))
    out = []
    for ax, letter, metric in zip(axes, 'abcd', metrics):
        S.panel(ax, letter, titles[metric])
        for y, cohort in enumerate(order):
            for n, shift, color, ms in sizes:
                values = draws[draws.cohort.eq(cohort) & draws.n.eq(n)][metric].to_numpy()
                ax.plot([values.min(), values.max()], [y + shift] * 2, color=color, lw=.7, alpha=.7, zorder=2)
                ax.plot(values.mean(), y + shift, 'o', ms=ms, color=color, mec='white', mew=.4, zorder=3)
                out.append(dict(metric=metric, cohort=cohort, full_nodes=int(full.loc[cohort, 'nodes']), n=n,
                                full_value=full.loc[cohort, metric], mean_draw=values.mean(),
                                min_draw=values.min(), max_draw=values.max(), draws=32))
            ax.plot(full.loc[cohort, metric], y, 'D', ms=4.6, mfc='white', mec='#C8371B', mew=1.2, zorder=4)
        ax.set_xlim(0, None)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.set_xlabel(titles[metric] if metric == 'isolate_share' else f'{titles[metric]} (0–1)')
        for y in range(1, 12):
            ax.axhline(y - .5, color=S.GRID, lw=.4)
    axes[0].set_yticks(range(12), [f'{S.record_label(c)}  (N={int(full.loc[c, "nodes"])})' for c in order])
    axes[0].set_ylim(11.6, -.6)
    axes[0].set_ylabel('Observed record, smallest first')
    fig.legend(handles=[S.handle('Full observed record', marker='D', ms=4.6, mfc='white', mec='#C8371B', mew=1.2)]
               + [S.handle(f'{n}-node draws: mean and range', marker='o', ms=ms, color=color, ls='-', lw=.7)
                  for n, _, color, ms in sizes],
               loc='lower center', ncol=4, frameon=False, columnspacing=1.3, bbox_to_anchor=(.5, .06))
    S.note(fig, '32 overlapping induced draws per record and size; smaller draws are nested in the 50-node draw. Ranges '
           'are sampling variation, not confidence intervals. The Copenhagen record contains only people who appear '
           'in at least one friendship, so its zero full-record isolates is a property of the release.')
    fig.subplots_adjust(left=.19, right=.985, top=.93, bottom=.225, wspace=.09)
    return fig, pd.DataFrame(out)


def fig19(D):
    """Copenhagen at 50 nodes: six separate metric comparisons and one informed diagnostic."""
    records = D['ext']['transfer_record_summary']
    records = records[records.n.eq(50) & records.cohort.eq('Copenhagen')]
    bank = D['ext']['transfer_distances']
    bank = bank[bank.n.eq(50) & bank.cohort.eq('Copenhagen')]
    diag = D['ext']['target_er_summary']
    diag = diag[diag.n.eq(50)].set_index('metric')
    diag_bank = D['ext']['target_er_distances']
    diag_bank = diag_bank[diag_bank.n.eq(50)]
    metrics = tuple(S.METRIC_LABEL)
    fig, axes = plt.subplots(2, 3, figsize=(S.FULL, 5.25))
    out = []
    for i, (ax, letter, metric) in enumerate(zip(axes.flat, 'abcdef', metrics)):
        S.panel(ax, letter, S.METRIC_LABEL[metric])
        ax.axhspan(7.55, 9.1, color='#F1F3F5', lw=0, zorder=0)
        for procedure in PROCEDURES:
            y, color = ROW[procedure], _color(procedure)
            row = records[records.procedure.eq(procedure) & records.metric.eq(metric)].iloc[0]
            parts = bank[bank.procedure.eq(procedure) & bank.metric.eq(metric)].w1.to_numpy()
            if len(parts) != row.models_or_banks or not np.isclose(parts.mean(), row.mean_w1, atol=1e-12, rtol=0):
                raise ValueError('Copenhagen distances differ from the published summary')
            ax.plot(parts, np.full(len(parts), y), '|', ms=5.5, mew=.8, color=color, alpha=.75, zorder=2)
            ax.plot(row.mean_w1, y, 'o' if procedure in S.METHODS else S.BASELINE_MARKER[procedure], ms=4.6,
                    color=color, mec='white', mew=.6, zorder=4)
            out.append(dict(metric=metric, procedure=procedure, tier='transfer', mean_w1=row.mean_w1,
                            min_model_or_bank=parts.min(), max_model_or_bank=parts.max(), parts=len(parts)))
        parts = diag_bank[diag_bank.metric.eq(metric)].w1.to_numpy()
        ax.plot(parts, np.full(len(parts), 8.3), '|', ms=5.5, mew=.8, color=S.DIAGNOSTIC, alpha=.6, zorder=2)
        ax.plot(diag.loc[metric, 'mean_w1'], 8.3, 'D', ms=4.2, mfc='white', mec=S.DIAGNOSTIC, mew=1.2, zorder=4)
        out.append(dict(metric=metric, procedure='ER given target density', tier='diagnostic, target-informed',
                        mean_w1=diag.loc[metric, 'mean_w1'], min_model_or_bank=parts.min(),
                        max_model_or_bank=parts.max(), parts=len(parts)))
        ax.set_ylim(9.1, -.7)
        ax.axhline(3.9, color=S.GRID, lw=.6)
        ax.set_yticks(list(ROW.values()) + [8.3], [_label(p) for p in PROCEDURES] + ['Diagnostic: ER given\nCopenhagen density']
                      if i % 3 == 0 else [])
        for tick, procedure in zip(ax.get_yticklabels(), PROCEDURES):
            tick.set_color(_color(procedure))
        ax.set_xlim(0, None)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.set_xlabel(f'W1 distance in {S.METRIC_LABEL[metric].lower()}')
    fig.legend(handles=[S.handle('One model (LLM rows) or one simulation bank', marker='|', ms=5.5, mew=.8, color=S.MUTED),
                        S.handle('Mean of four models / eight banks', marker='o', ms=4.6, color=S.MUTED),
                        S.handle('Target-informed diagnostic, not a transfer competitor', marker='D', ms=4.2,
                                 mfc='white', mec=S.DIAGNOSTIC, mew=1.2)],
               loc='lower center', ncol=3, frameon=False, columnspacing=1.2, bbox_to_anchor=(.5, .055))
    S.note(fig, 'One endpoint-conditioned record (800 people who appear in at least one of 6,418 friendships), 32 '
           'overlapping 50-node draws. Smaller = closer on that metric only; panels have different units and are not '
           'combined. Ticks are model or bank spread, not confidence intervals.')
    fig.subplots_adjust(left=.165, right=.985, top=.94, bottom=.185, wspace=.14, hspace=.5)
    return fig, pd.DataFrame(out)


FIGURES = {'10_sampling_effect': fig10, '18_sampling_distortion': fig18, '19_copenhagen_transfer': fig19}
