"""Figures 08-09, 11-13 and 15: the 11 evaluation-held-out friendship records at 50 nodes.

Every distance is a one-dimensional Wasserstein-1 distance for ONE metric. Panels for
different metrics have different units and are never summed, ranked or averaged here.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis_figures.v6 import style as S

PROCEDURES = S.METHODS + tuple(S.BASELINES)
ROW = dict(zip(PROCEDURES, (0, 1, 2, 3, 4.5, 5.5, 6.5)))
FIVE = ('density', 'avg_clustering_coef', 'global_efficiency', 'transitivity', 'degree_gini')


def _label(procedure):
    return procedure.title() if procedure in S.METHODS else f'{procedure}, fixed'


def _color(procedure):
    return S.METHOD_COLOR.get(procedure, S.BASELINE_COLOR)


def _records(D, n=50):
    """Record-level mean W1 for the 11 historical held-out records (Copenhagen is figure 19)."""
    frame = D['ext']['transfer_record_summary']
    frame = frame[frame.n.eq(n) & frame.source.ne('copenhagen_friendship')]
    if frame.groupby(['metric', 'procedure']).cohort.nunique().ne(11).any() or len(frame) != 11 * 7 * 6:
        raise ValueError('Expected 11 records by seven procedures by six metrics')
    return frame


def _density_order(D):
    quality = D['held']['dataset_quality'].sort_values('observed_density')
    return list(quality.cohort), quality.set_index('cohort')


def _distance_rows(ax, part, out, metric):
    """Ten Facebook record distances, their mean, and the single school record, per procedure."""
    for procedure in PROCEDURES:
        y, rows = ROW[procedure], part[part.procedure.eq(procedure)]
        facebook = rows[rows.source.eq('snap_facebook')].mean_w1.to_numpy()
        school = rows[rows.source.eq('sociopatterns_friendship')].mean_w1.to_numpy()
        if len(facebook) != 10 or len(school) != 1:
            raise ValueError('Unexpected record coverage')
        color = _color(procedure)
        ax.plot([facebook.min(), facebook.max()], [y, y], color=color, lw=.7, alpha=.6, zorder=2)
        ax.plot(facebook, np.full(10, y), '|', ms=5.5, mew=.8, color=color, alpha=.75, zorder=3)
        ax.plot(facebook.mean(), y, 'o', ms=4.6, color=color, mec='white', mew=.6, zorder=4)
        ax.plot(school[0], y + .34, 'D', ms=3.6, mfc='white', mec=S.SCHOOL, mew=1, zorder=4)
        out.append(dict(metric=metric, procedure=procedure, facebook_mean_w1=facebook.mean(),
                        facebook_min_record=facebook.min(), facebook_max_record=facebook.max(),
                        school_w1=school[0], all_record_mean_w1=np.r_[facebook, school].mean(), records=11))


def _procedure_axis(ax, labels=True):
    ax.set_ylim(7.25, -.7)
    ax.set_yticks(list(ROW.values()), [_label(p) for p in PROCEDURES] if labels else [])
    ax.axhline(3.9, color=S.GRID, lw=.6)
    for tick, procedure in zip(ax.get_yticklabels(), PROCEDURES):
        tick.set_color(_color(procedure))


def _record_key():
    return [S.handle('One Facebook record (10)', marker='|', ms=5.5, mew=.8, color=S.MUTED),
            S.handle('Mean of the 10 Facebook records', marker='o', ms=4.6, color=S.MUTED),
            S.handle('School record (1)', marker='D', ms=3.6, mfc='white', mec=S.SCHOOL, mew=1)]


def fig08(D):
    """Five structural metrics, one panel each; no cross-metric score."""
    frame = _records(D)
    fig, axes = plt.subplots(2, 3, figsize=(S.FULL, 4.6))
    out = []
    for i, (ax, letter, metric) in enumerate(zip(axes.flat, 'abcde', FIVE)):
        S.panel(ax, letter, S.METRIC_LABEL[metric])
        _distance_rows(ax, frame[frame.metric.eq(metric)], out, metric)
        _procedure_axis(ax, labels=i % 3 == 0)
        ax.set_xlim(0, None)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.set_xlabel(f'W1 distance in {S.METRIC_LABEL[metric].lower()}')
    key = axes.flat[5]
    key.axis('off')
    key.legend(handles=_record_key(), loc='upper left', frameon=False, bbox_to_anchor=(-.05, 1.02),
               handletextpad=.5, labelspacing=.7)
    key.text(-.02, .40, 'Smaller = closer on that metric only.\nEach panel has its own units: distances\n'
             'are not comparable across panels and\nare not combined into one score.\n\n'
             'LLM rows: mean of four model-specific\ndistances. ER, BA, WS: mean of eight\n'
             'simulation banks, settings fixed on\nearlier development networks.',
             transform=key.transAxes, fontsize=6.3, color=S.MUTED, va='top', linespacing=1.4)
    fig.subplots_adjust(left=.115, right=.985, top=.94, bottom=.105, wspace=.14, hspace=.62)
    return fig, pd.DataFrame(out)


def _candidate_weights(candidates, procedure, metric):
    """Pool the models of one procedure: equal weight per model and instruction setting."""
    if procedure in S.METHODS:
        rows = candidates[candidates.kind.eq('LLM') & candidates.candidate.str.split(':').str[1].eq(procedure)]
        rows = rows[rows.valid & rows[metric].notna()]
        cell = rows.groupby(['candidate', 'condition'])[metric].transform('size')
        if rows.groupby('candidate').condition.nunique().ne(7).any() or rows.candidate.nunique() != 4:
            raise ValueError('A model-setting cell has no defined graph')
        return rows[metric].to_numpy(), (1 / (28 * cell)).to_numpy()
    rows = candidates[candidates.candidate.eq('Baseline:' + procedure)]
    return rows[metric].to_numpy(), np.full(len(rows), 1 / len(rows))


def _target_weights(targets, source, metric):
    rows = targets[targets.source.eq(source)]
    per = rows.groupby('cohort')[metric].transform('size')
    return rows[metric].to_numpy(), (1 / (rows.cohort.nunique() * per)).to_numpy()


DISTRIBUTION_ROWS = (('snap_facebook', 'Facebook draws'), ('sociopatterns_friendship', 'School draws'))


def _distribution_panel(ax, D, metric, out):
    targets, candidates = D['held']['heldout_sample_metrics'], D['held']['candidate_metrics']
    for y, (source, label), color in zip((0, 1), DISTRIBUTION_ROWS, (S.FACEBOOK, S.SCHOOL)):
        values, weights = _target_weights(targets, source, metric)
        stats = S.strip(ax, values, weights, y, color, 'D', alpha=.35)
        out.append(dict(metric=metric, row=label, kind='observed 50-node draws', graphs=len(values), **stats))
    for procedure in PROCEDURES:
        values, weights = _candidate_weights(candidates, procedure, metric)
        stats = S.strip(ax, values, weights, ROW[procedure] + 2.6, _color(procedure),
                        'o' if procedure in S.METHODS else S.BASELINE_MARKER[procedure],
                        alpha=.16 if procedure in S.METHODS else .1)
        out.append(dict(metric=metric, row=procedure, kind='LLM' if procedure in S.METHODS else 'fixed baseline',
                        graphs=len(values), **stats))


def _distribution_axis(ax, labels=True):
    ticks = [0, 1] + [ROW[p] + 2.6 for p in PROCEDURES]
    ax.set_ylim(9.75, -.7)
    ax.set_yticks(ticks, ['Facebook, 10 records', 'School, 1 record'] + [_label(p) for p in PROCEDURES]
                  if labels else [])
    for tick, color in zip(ax.get_yticklabels(), [S.FACEBOOK, S.SCHOOL] + [_color(p) for p in PROCEDURES]):
        tick.set_color(color)
    for y in (1.8, 6.5):
        ax.axhline(y, color=S.GRID, lw=.6)


def _strip_key(unit='graph'):
    return [S.handle(f'One {unit}', marker='|', ms=7, mew=.6, color=S.MUTED),
            S.handle('Median', marker='o', ms=4, mfc='white', mec=S.INK, mew=1),
            S.handle('Interquartile range', ls='-', lw=2.6),
            S.handle('5th–95th percentile', ls='-', lw=.8)]


def fig09(D):
    """The distributions behind the distances: observed 50-node draws against every generator."""
    metrics = ('density', 'avg_clustering_coef', 'global_efficiency')
    fig, axes = plt.subplots(1, 3, figsize=(S.FULL, 3.75))
    out = []
    for i, (ax, letter, metric) in enumerate(zip(axes, 'abc', metrics)):
        S.panel(ax, letter, S.METRIC_LABEL[metric])
        _distribution_panel(ax, D, metric, out)
        _distribution_axis(ax, labels=i == 0)
        ax.set_xlim(0, None)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.set_xlabel(f'{S.METRIC_LABEL[metric]} of a 50-node graph (0–1)')
    fig.legend(handles=_strip_key('50-node graph'), loc='lower center', ncol=4, frameon=False,
               columnspacing=1.6, bbox_to_anchor=(.5, .075))
    S.note(fig, 'Observed rows: 32 overlapping 50-node draws per record, records weighted equally. LLM rows: valid '
           'primary graphs, each model and instruction setting weighted equally; failed attempts have no graph. '
           'ER, BA, WS: 448 simulated graphs each with settings fixed on earlier development networks.')
    fig.subplots_adjust(left=.165, right=.985, top=.93, bottom=.27, wspace=.1)
    return fig, pd.DataFrame(out)


def fig11(D):
    """Closure beyond two target-informed controls, per record. Diagnostic, not a competitor."""
    held = D['held']
    order, quality = _density_order(D)
    controls = held['conditional_control_metrics']
    targets = held['heldout_sample_metrics']
    residual = held['conditional_residuals']
    metrics = ('avg_clustering_coef', 'transitivity')
    fig, axes = plt.subplots(1, 2, figsize=(S.FULL, 4.0), sharey=True)
    styles = {'Gnm': ('s', S.BASELINE_COLOR, -.17), 'degree_swap': ('o', S.METHOD_COLOR['global'], .17)}
    out = []
    for ax, letter, metric in zip(axes, 'ab', metrics):
        S.panel(ax, letter, f'{S.METRIC_LABEL[metric]}: observed draw minus its control')
        ax.axvline(0, color=S.INK, lw=.7)
        for y, cohort in enumerate(order):
            target = targets[targets.cohort.eq(cohort)].set_index('draw')[metric]
            for control, (marker, color, shift) in styles.items():
                rows = controls[controls.cohort.eq(cohort) & controls.control.eq(control) & controls.success]
                mean = rows.groupby('draw')[metric].mean()
                difference = (target - mean).dropna().to_numpy()
                archived = residual[residual.cohort.eq(cohort) & residual.control.eq(control)
                                    & residual.metric.eq(metric)].iloc[0]
                if (len(difference) != archived.complete_target_draws
                        or not np.isclose(difference.mean(), archived.mean_target_minus_control, atol=1e-12, rtol=0)):
                    raise ValueError('Conditional residual differs from the archived table')
                ax.plot(difference, np.full(len(difference), y + shift), '|', ms=4.5, mew=.6, color=color,
                        alpha=.45, zorder=2)
                ax.plot(difference.mean(), y + shift, marker, ms=4.4, color=color, mec='white', mew=.6, zorder=4)
                out.append(dict(metric=metric, cohort=cohort, control=control, draws=len(difference),
                                mean_difference=difference.mean(), min_draw=difference.min(),
                                max_draw=difference.max(), positive_draws=int((difference > 0).sum()),
                                full_record_density=quality.loc[cohort, 'observed_density']))
        ax.set_xlabel(f'Difference in {S.METRIC_LABEL[metric].lower()} (0 = control reproduces it)')
        ax.xaxis.set_major_locator(plt.MaxNLocator(6))
    axes[0].set_yticks(range(11), [f'{S.record_label(c)}  ({quality.loc[c, "observed_density"]:.3f})' for c in order])
    axes[0].set_ylim(10.6, -.6)
    axes[0].set_ylabel('Record (full-record density), sparsest first')
    fig.legend(handles=[
        S.handle('Same node and edge count, edges placed at random', marker='s', ms=4.4, color=S.BASELINE_COLOR),
        S.handle('Same degree of every node (edge swaps)', marker='o', ms=4.4, color=S.METHOD_COLOR['global']),
        S.handle('One of 32 overlapping 50-node draws', marker='|', ms=5, mew=.6, color=S.MUTED)],
        loc='lower center', ncol=3, frameon=False, columnspacing=1.5, bbox_to_anchor=(.5, .055))
    S.note(fig, 'Controls are rebuilt from each observed draw, so they hold information no LLM or fixed baseline had. '
           'Ticks are sampling variation of overlapping draws, not confidence intervals; no test is implied.')
    fig.subplots_adjust(left=.19, right=.985, top=.93, bottom=.235, wspace=.08)
    return fig, pd.DataFrame(out)


BASELINE_DASH = {'ER': (0, (1, 1.2)), 'BA': (0, (4, 1.5)), 'WS': '-'}


def fig12(D):
    """Clustering distance record by record: the ordering of procedures is not stable."""
    frame = _records(D)
    frame = frame[frame.metric.eq('avg_clustering_coef')]
    order, quality = _density_order(D)
    wide = frame.pivot(index='cohort', columns='procedure', values='mean_w1').loc[order, list(PROCEDURES)]
    fig = plt.figure(figsize=(S.FULL, 4.75))
    grid = fig.add_gridspec(2, 1, height_ratios=[2.5, 1], left=.085, right=.875, top=.94, bottom=.2, hspace=.16)
    ax = fig.add_subplot(grid[0])
    S.panel(ax, 'a', 'Clustering distance of each procedure, record by record', grid='y')
    x = np.arange(11)
    ends = []
    for procedure in PROCEDURES:
        llm = procedure in S.METHODS
        ax.plot(x, wide[procedure], color=_color(procedure), lw=1.3 if llm else 1,
                ls='-' if llm else BASELINE_DASH[procedure], marker='o' if llm else S.BASELINE_MARKER[procedure],
                ms=3.2, mfc=_color(procedure) if llm else 'white', mec=_color(procedure), mew=.8, zorder=3 if llm else 2)
        ends.append([wide[procedure].iloc[-1], procedure])
    ends.sort()
    for i in range(1, len(ends)):  # keep direct labels from overlapping
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + .028)
    for (y, procedure), value in zip(ends, [wide[p].iloc[-1] for _, p in ends]):
        ax.annotate(_label(procedure), (10, value), xytext=(10.55, y), color=_color(procedure), fontsize=6.5,
                    va='center', annotation_clip=False,
                    arrowprops=dict(arrowstyle='-', color=_color(procedure), lw=.4, shrinkA=0, shrinkB=2))
    ax.set_xlim(-.4, 10.4)
    ax.set_ylim(0, None)
    ax.set_xticks(x, [])
    ax.set_ylabel('W1 distance in clustering\n(smaller = closer)')
    bx = fig.add_subplot(grid[1], sharex=ax)
    S.panel(bx, 'b', 'Gap between the closest and the second-closest procedure', grid='y')
    ranked = np.sort(wide.to_numpy(), axis=1)
    margin = ranked[:, 1] - ranked[:, 0]
    closest = wide.idxmin(axis=1)
    out = []
    for i, cohort in enumerate(order):
        bx.plot([i, i], [0, margin[i]], color=_color(closest[cohort]), lw=1.1)
        bx.plot(i, margin[i], 'o' if closest[cohort] in S.METHODS else S.BASELINE_MARKER[closest[cohort]], ms=4.2,
                color=_color(closest[cohort]), mec='white', mew=.5)
        bx.text(i, margin[i] + .012, closest[cohort] if closest[cohort] in S.BASELINES else closest[cohort].title(),
                ha='center', va='bottom', fontsize=5.8, color=_color(closest[cohort]))
        row = dict(cohort=cohort, full_record_density=quality.loc[cohort, 'observed_density'],
                   closest=closest[cohort], second=wide.loc[cohort].sort_values().index[1], margin=margin[i])
        row.update({p: wide.loc[cohort, p] for p in PROCEDURES})
        out.append(row)
    bx.set_ylim(0, margin.max() * 1.55)
    bx.yaxis.set_major_locator(plt.MaxNLocator(3))
    bx.set_ylabel('W1 gap')
    bx.set_xticks(x, [f'{S.record_label(c)}\n{quality.loc[c, "observed_density"]:.3f}' for c in order])
    bx.set_xlabel('Held-out record and its full-record density (sparsest to densest)')
    plt.setp(ax.get_xticklabels(), visible=False)
    S.note(fig, 'Lines join separate records only to make changes of order visible; records are not a continuous '
           'scale and are not independent communities (the ten Facebook records overlap). b names the closest '
           'procedure for that record; a small gap is a near tie, not a win.')
    return fig, pd.DataFrame(out)


def fig13(D):
    """Largest-component share: why this metric cannot separate the connected baselines."""
    frame = _records(D)
    metric = 'prop_nodes_lcc'
    fig, axes = plt.subplots(1, 2, figsize=(S.FULL, 3.75), gridspec_kw=dict(width_ratios=[1.25, 1]))
    out = []
    ax = axes[0]
    S.panel(ax, 'a', 'Share of the 50 nodes in the largest component')
    _distribution_panel(ax, D, metric, out)
    _distribution_axis(ax)
    ax.set_xlim(0, 1.03)
    ax.set_xlabel('Largest-component share (0–1)')
    candidates = D['held']['candidate_metrics']
    for procedure in S.BASELINES:
        values, _ = _candidate_weights(candidates, procedure, metric)
        ax.text(.955, ROW[procedure] + 2.6, f'{(values == 1).mean():.0%} of graphs fully connected', ha='right',
                va='center', fontsize=5.8, color=S.MUTED)
    bx = axes[1]
    S.panel(bx, 'b', 'Resulting W1 distance')
    distances = []
    _distance_rows(bx, frame[frame.metric.eq(metric)], distances, metric)
    bx.set_ylim(9.75, -.7)
    offset = 2.6
    for line in list(bx.lines):
        line.set_ydata(np.asarray(line.get_ydata(), float) + offset)
    bx.set_yticks([ROW[p] + offset for p in PROCEDURES], [])
    bx.axhline(6.5, color=S.GRID, lw=.6)
    bx.set_xlim(0, None)
    bx.set_xlabel('W1 distance in largest-component share')
    bx.legend(handles=_record_key(), loc='upper right', frameon=False, bbox_to_anchor=(1.02, 1.0),
              handletextpad=.5, labelspacing=.5)
    fig.legend(handles=_strip_key('50-node graph'), loc='lower center', ncol=4, frameon=False,
               columnspacing=1.6, bbox_to_anchor=(.5, .075))
    S.note(fig, 'Nearly every ER, BA and WS graph is one connected component, so their three distances are almost '
           'the same number: this metric measures how fragmented the 50-node observed draws are, not which '
           'baseline is more realistic. Sampling creates much of that fragmentation (figures 10 and 18).')
    fig.subplots_adjust(left=.165, right=.985, top=.93, bottom=.27, wspace=.06)
    table = pd.DataFrame(out).merge(pd.DataFrame(distances).rename(columns={'procedure': 'row'}).drop(columns='metric'),
                                    on='row', how='left')
    return fig, table


def fig15(D):
    """Efficiency: the Global advantage depends on how dense the observed record is."""
    frame = D['v5']['efficiency_by_density'].sort_values('observed_density').reset_index(drop=True)
    if len(frame) != 11:
        raise ValueError('Expected 11 records')
    fig, axes = plt.subplots(1, 2, figsize=(S.FULL, 3.95), gridspec_kw=dict(width_ratios=[1.3, 1]))
    ax = axes[0]
    S.panel(ax, 'a', 'Efficiency distance for each record')
    blue = S.METHOD_COLOR['global']
    for y, row in frame.iterrows():
        ax.plot([min(row['global'], row.target_ER), max(row['global'], row.target_ER)], [y, y], color=S.RULE, lw=.9,
                zorder=1)
        ax.plot(row.WS, y, S.BASELINE_MARKER['WS'], ms=4.2, mfc='white', mec=S.BASELINE_COLOR, mew=.9)
        ax.plot(row['global'], y, 'o', ms=4.8, color=blue, mec='white', mew=.5, zorder=3)
        ax.plot(row.target_ER, y, 'D', ms=4, mfc='white', mec=S.DIAGNOSTIC, mew=1.1, zorder=3)
    ax.set_yticks(range(11), [f'{S.record_label(r.cohort)}  ({r.observed_density:.3f})' for r in frame.itertuples()])
    ax.set_ylim(10.6, -.6)
    ax.set_ylabel('Record (full-record density), sparsest first')
    ax.set_xlim(0, None)
    ax.set_xlabel('W1 distance in global efficiency (smaller = closer)')
    ax.legend(handles=[
        S.handle('Global LLM', marker='o', ms=4.8, color=blue),
        S.handle('WS, fixed on development networks', marker='v', ms=4.2, mfc='white', mec=S.BASELINE_COLOR, mew=.9),
        S.handle("Diagnostic: ER given that record's density", marker='D', ms=4, mfc='white', mec=S.DIAGNOSTIC,
                 mew=1.1)],
        loc='upper center', bbox_to_anchor=(.42, -.16), ncol=1, frameon=False, labelspacing=.45)
    bx = axes[1]
    S.panel(bx, 'b', 'Global minus the density-informed ER', grid='both')
    difference = frame['global'] - frame.target_ER
    bx.axhline(0, color=S.INK, lw=.7)
    for row, value in zip(frame.itertuples(), difference):
        school = row.source != 'snap_facebook'
        bx.plot(row.observed_density, value, 'D' if school else 'o', ms=4.4,
                color=S.SCHOOL if school else S.FACEBOOK, mec='white', mew=.5, zorder=3)
        bx.annotate(S.record_label(row.cohort).replace('FB', ''), (row.observed_density, value), xytext=(0, 5),
                    textcoords='offset points', ha='center', fontsize=5.6, color=S.MUTED)
    bx.set_xscale('log')
    bx.set_xticks([.03, .05, .1, .15], ['.03', '.05', '.10', '.15'])
    bx.minorticks_off()
    bx.set_xlabel('Full-record density (log scale)')
    bx.set_ylabel('Difference in efficiency W1')
    low, high = bx.get_ylim()
    bx.text(.97, .97, 'Above 0: Global farther\nthan the informed ER', transform=bx.transAxes, fontsize=6,
            color=S.MUTED, va='top', ha='right')
    bx.text(.03, .03, 'Below 0: Global closer\nthan the informed ER', transform=bx.transAxes, fontsize=6,
            color=S.MUTED, va='bottom', ha='left')
    bx.legend(handles=[S.handle('Facebook record (number)', marker='o', ms=4.4, color=S.FACEBOOK),
                       S.handle('School record', marker='D', ms=4.4, color=S.SCHOOL)],
              loc='upper center', bbox_to_anchor=(.5, -.2), ncol=1, frameon=False, labelspacing=.45)
    S.note(fig, 'The diamond uses the density of the record it is compared with, information no transfer competitor '
           'had: it is a diagnostic reference, not a ranked competitor. Eleven overlapping records are not eleven '
           'independent communities; b is a post hoc description, not a fitted trend or a test.')
    fig.subplots_adjust(left=.175, right=.985, top=.93, bottom=.345, wspace=.36)
    out = frame.assign(global_minus_target_ER=difference)
    return fig, out


FIGURES = {'08_heldout_distances': fig08, '09_heldout_distributions': fig09, '11_conditional_controls': fig11,
           '12_cohort_sensitivity': fig12, '13_appendix_lcc': fig13, '15_efficiency_density': fig15}
