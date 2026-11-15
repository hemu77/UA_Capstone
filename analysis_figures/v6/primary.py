"""Figures 01-07: locked instruction contrasts, robustness, coverage, resources, context."""
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from analysis_figures.v6 import style as S

ROWS = np.array([0, 1, 2, 3.6, 4.6, 5.6])  # country framing above, instruction language below
HINDI = 'iterative:us:hindi'
FAMILY = 24


def _contrast_rows(frame, method):
    rows = frame[frame.method.eq(method)].set_index(['culture', 'language'])
    return rows.loc[list(S.CONDITIONS[1:])]


def _group_labels(ax, x=-.46):
    for y, text in ((1, 'Country\nframing'), (4.6, 'Instruction\nlanguage')):
        ax.text(x, y, text, transform=ax.get_yaxis_transform(), rotation=90, ha='center', va='center',
                fontsize=6, color=S.MUTED, linespacing=1.1)


def _fmt(value, digits=3):
    text = f'{value:+.{digits}f}'
    return text[1:] if float(text) == 0 else text.replace('-', '−')


def _p(value):
    return 'n/a' if not np.isfinite(value) else '1' if value >= .9995 else f'{value:.3f}'.lstrip('0')


def _forest_axes(fig, left, right, top, bottom):
    """One shared difference axis for all 24 planned contrasts, grouped by procedure."""
    ax = fig.add_axes([left, bottom, right - left, top - bottom])
    y, rows, bands = 0., {}, []
    for method in S.METHODS:
        start = y
        for cond in S.CONDITIONS[1:]:
            rows[(method, *cond)] = y
            y += 1
        bands.append((method, start, y - 1))
        y += 1.15
    ax.set_ylim(y - 1.55, -1.5)
    ax.set_yticks(list(rows.values()), [S.condition_label(c, l) for (_, c, l) in rows])
    ax.axvline(0, color=S.RULE, lw=.7, zorder=1)
    ax.grid(axis='x', color=S.GRID, linewidth=.5)
    ax.set_axisbelow(True)
    for method, lo, hi in bands:
        endpoint = 'density' if method == 'global' else 'mean clustering'
        ax.annotate(method.title(), (0, lo - .95), xycoords=ax.get_yaxis_transform(), xytext=(-2, 0),
                    textcoords='offset points', ha='right', va='center', fontsize=7.5, fontweight='bold',
                    color=S.METHOD_COLOR[method], annotation_clip=False)
        ax.annotate(f'endpoint: {endpoint}', (0, lo - .95), xycoords=ax.get_yaxis_transform(), xytext=(4, 0),
                    textcoords='offset points', ha='left', va='center', fontsize=6.3, color=S.MUTED,
                    annotation_clip=False)
    return ax, rows, bands


def fig01(D):
    """All 24 planned contrasts with the archived simultaneous intervals and Holm p-values."""
    frame, blocks = D['tables']['primary_contrasts'], D['tables']['paired_schedule_blocks']
    fig = plt.figure(figsize=(S.FULL, 6.6))
    ax, rows, _ = _forest_axes(fig, .16, .60, .955, .135)
    cols = {'blocks': 1.045, 'mean': 1.20, 'ci': 1.375, 'p': 1.80}
    head = dict(transform=ax.get_yaxis_transform(), fontsize=6.3, color=S.MUTED, va='center')
    out = []
    for method in S.METHODS:
        color = S.METHOD_COLOR[method]
        part = frame[frame.method.eq(method)].set_index(['culture', 'language'])
        top = rows[(method, *S.CONDITIONS[1])] - .95
        ax.text(cols['blocks'], top, 'Blocks', ha='left', **head)
        ax.text(cols['mean'], top, 'Mean', ha='left', **head)
        ax.text(cols['ci'], top, '95% simultaneous CI', ha='left', **head)
        ax.text(cols['p'], top, 'Holm p', ha='right', **head)
        for cond in S.CONDITIONS[1:]:
            row, y = part.loc[cond], rows[(method, *cond)]
            raw = blocks[blocks.hypothesis_id.eq(row.hypothesis_id) & blocks.complete]
            ax.scatter(raw.difference, np.full(len(raw), y + .3), s=4.5, color=color, alpha=.5,
                       edgecolors='none', zorder=2)
            tested = bool(np.isfinite(row.ci_lower))
            significant = tested and row.holm_p < .05
            weight = 'bold' if significant else 'normal'
            ink = dict(transform=ax.get_yaxis_transform(), va='center', fontsize=6.3, fontweight=weight,
                       color=S.INK if tested else S.MUTED)
            if tested:
                ax.plot([row.ci_lower, row.ci_upper], [y, y], lw=1.3, color=color, zorder=3)
                ax.plot(row.complete_block_mean, y, 'o', ms=4.6, mec=color, mew=1.1,
                        mfc=color if significant else 'white', zorder=4)
                ax.text(cols['ci'], y, f'[{_fmt(row.ci_lower)}, {_fmt(row.ci_upper)}]', ha='left', **ink)
                ax.text(cols['p'], y, _p(row.holm_p), ha='right', **ink)
            else:
                ax.plot(row.complete_block_mean, y, 'x', ms=4.5, mew=1.1, color=S.MUTED, zorder=4)
                ax.text(cols['ci'], y, 'not tested: fewer than 6 blocks', ha='left', **ink)
            ax.text(cols['blocks'], y, f'{row.complete_schedule_blocks}/8', ha='left', **ink)
            ax.text(cols['mean'], y, _fmt(row.complete_block_mean), ha='left', **ink)
            out.append(dict(figure_row=row.hypothesis_id, method=method, endpoint=row.endpoint,
                            blocks=row.complete_schedule_blocks, mean=row.complete_block_mean,
                            ci_lower=row.ci_lower, ci_upper=row.ci_upper, holm_p=row.holm_p,
                            raw_p=row.raw_p, status=row.status))
    extent = max(np.nanmax(np.abs(np.r_[frame.ci_lower, frame.ci_upper])),
                 blocks[blocks.complete].difference.abs().max()) * 1.04
    ax.set_xlim(-extent, extent)
    ax.set_xlabel('Difference from US · English in the same seed block (endpoint units, 0–1 scale)')
    fig.legend(handles=[
        S.handle('Holm-adjusted p < .05', marker='o', ms=4.6, mfc=S.INK, mec=S.INK),
        S.handle('Holm-adjusted p ≥ .05', marker='o', ms=4.6, mfc='white', mec=S.INK, mew=1.1),
        S.handle('Not tested', marker='x', ms=4.5, color=S.MUTED, mew=1.1),
        S.handle('One paired seed block (four-model mean)', marker='o', ms=2.6, color=S.MUTED, alpha=.7),
        S.handle('Bonferroni simultaneous 95% interval (24 contrasts)', ls='-', lw=1.3)],
        loc='lower left', ncol=3, frameon=False, columnspacing=1.4, handletextpad=.5,
        bbox_to_anchor=(.02, .0))
    return fig, pd.DataFrame(out)


def fig02(D):
    """Why the one adjusted finding is fragile: blocks, omissions and test resolution."""
    t = D['tables']
    blocks = t['paired_schedule_blocks'].query('hypothesis_id == @HINDI').sort_values('seed')
    loo = t['leave_one_seed_out'].query('hypothesis_id == @HINDI').sort_values('omitted_seed')
    full = t['primary_contrasts'].set_index('hypothesis_id').loc[HINDI]
    sens = t['sensitivity'].set_index('hypothesis_id').loc[HINDI]
    if len(blocks) != 8 or len(loo) != 8 or not blocks.complete.all():
        raise ValueError('Expected eight complete Hindi blocks')
    color = S.METHOD_COLOR['iterative']
    fig = plt.figure(figsize=(S.FULL, 2.95))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.05, 1.25], left=.085, right=.985, top=.86,
                            bottom=.3, wspace=.34)
    ax = fig.add_subplot(grid[0])
    y = np.arange(8)
    S.panel(ax, 'a', 'Eight paired seed blocks')
    ax.axvspan(full.ci_lower, full.ci_upper, color=color, alpha=.12, lw=0)
    ax.axvline(full.complete_block_mean, color=color, lw=1)
    ax.axvline(0, color=S.RULE, lw=.7)
    ax.scatter(blocks.difference, y, s=20, color=color, zorder=3)
    ax.set_yticks(y, [str(s) for s in blocks.seed])
    ax.set_ylabel('Seed block')
    ax.set_ylim(7.6, -.9)
    ax.set_xlim(-.108, .012)
    ax.set_xlabel('Clustering difference,\nUS · Hindi minus US · English')
    ax.text(full.complete_block_mean, -.62, f'mean {_fmt(full.complete_block_mean, 4)}', color=color,
            fontsize=6, ha='center', va='center',
            bbox=dict(facecolor='white', edgecolor='none', pad=.6))
    ax = fig.add_subplot(grid[1])
    S.panel(ax, 'b', 'Adjusted p after omitting one block')
    ax.axvline(.05, color=S.INK, lw=.8, ls=(0, (4, 2)))
    for yy, row in zip(y, loo.itertuples()):
        mean = blocks.loc[blocks.seed.ne(row.omitted_seed), 'difference'].mean()
        if not np.isclose(mean, row.mean, atol=1e-12, rtol=0):
            raise ValueError('Leave-one-block-out mean differs')
        ax.plot(row.holm_p, yy, 'o', ms=4.6, mec=color, mew=1.1, mfc=color if row.holm_p < .05 else 'white')
    ax.set_xscale('log')
    ax.set_xlim(.0008, .2)
    ax.set_xticks([.001, .01, .05, .1], ['.001', '.01', '.05', '.1'])
    ax.minorticks_off()
    ax.set_yticks(y, [str(s) for s in loo.omitted_seed])
    ax.set_ylabel('Omitted seed block')
    ax.set_ylim(7.6, -.9)
    ax.set_xlabel('Holm-adjusted p of the seven-block refit\n(log scale, 24-contrast family)')
    passed = int((loo.holm_p < .05).sum())
    ax.text(.05, -.62, f'{passed} of 8 refits below .05', fontsize=6, ha='center', va='center',
            bbox=dict(facecolor='white', edgecolor='none', pad=.6))
    ax = fig.add_subplot(grid[2])
    S.panel(ax, 'c', 'What each test can resolve')
    threshold = .05 / FAMILY
    attainable = np.arange(1, 129) * 2 / 256
    ax.axvline(threshold, color=S.INK, lw=.8, ls=(0, (4, 2)))
    ax.vlines(attainable[attainable <= .25], .82, 1.18, color=S.MUTED, lw=.5)
    ax.plot(full.raw_p, 0, 'o', ms=4.6, color=color)
    ax.plot(sens.sign_flip_p, 1, 'o', ms=4.6, mfc='white', mec=color, mew=1.1)
    ax.text(full.raw_p * 1.35, 0, f'raw p = {full.raw_p:.5f}'.replace('0.', '.'), fontsize=6, ha='left', va='center')
    ax.text(sens.sign_flip_p * 1.25, 1.42, f'raw p = 2/256 = {sens.sign_flip_p:.4f}'.replace('0.', '.')
            + '\nis the smallest possible', fontsize=6, ha='left', va='center', linespacing=1.3)
    ax.text(threshold, 1.98, f'first Holm threshold\n.05/24 = {threshold:.4f}'.replace('0.', '.'), fontsize=6,
            ha='center', va='center', linespacing=1.3,
            bbox=dict(facecolor='white', edgecolor='none', pad=.6))
    ax.set_xscale('log')
    ax.set_xlim(.0005, .3)
    ax.set_xticks([.001, .01, .1], ['.001', '.01', '.1'])
    ax.minorticks_off()
    ax.set_yticks([0, 1], ['Paired t test\n(locked analysis)', 'Exact sign-flip\n(8 blocks, post hoc)'])
    ax.set_ylim(2.35, -.75)
    ax.set_xlabel('Unadjusted two-sided p (log scale)')
    S.note(fig, 'a: band = archived Bonferroni simultaneous 95% interval. b: filled = below .05; overlapping refits are '
           'not independent replications.\nc: grey ticks = every p-value an eight-block sign-flip test can produce; '
           f'none reaches the threshold, so its adjusted p of {sens.sign_flip_holm_p:.4f} is not a refutation.'.replace('0.', '.'))
    out = blocks[['seed', 'difference']].merge(
        loo.rename(columns={'omitted_seed': 'seed', 'mean': 'loo_mean', 'holm_p': 'loo_holm_p',
                            'raw_p': 'loo_raw_p'})[['seed', 'loo_mean', 'loo_raw_p', 'loo_holm_p']], on='seed')
    out['full_mean'], out['full_raw_p'], out['full_holm_p'] = full.complete_block_mean, full.raw_p, full.holm_p
    out['sign_flip_raw_p'], out['sign_flip_holm_p'] = sens.sign_flip_p, sens.sign_flip_holm_p
    out['first_holm_threshold'] = threshold
    return fig, out


def fig03(D):
    """RQ3: does each fixed model move in the same direction? Descriptive, no intervals."""
    frame = D['tables']['model_contrasts_descriptive']
    fig = plt.figure(figsize=(S.FULL, 6.45))
    ax, rows, _ = _forest_axes(fig, .16, .70, .955, .115)
    head = dict(transform=ax.get_yaxis_transform(), fontsize=6.3, color=S.MUTED, va='center')
    out = []
    for method in S.METHODS:
        color = S.METHOD_COLOR[method]
        part = frame[frame.method.eq(method)].set_index(['culture', 'language', 'model']).sort_index()
        top = rows[(method, *S.CONDITIONS[1])] - .95
        ax.text(1.04, top, 'Models with the same sign', ha='left', **head)
        ax.text(1.40, top, 'Range', ha='left', **head)
        for cond in S.CONDITIONS[1:]:
            y = rows[(method, *cond)]
            values = part.loc[cond].loc[list(S.MODELS)]
            x = values.common_panel_mean.to_numpy()
            ax.plot([x.min(), x.max()], [y, y], color=color, lw=.7, alpha=.55, zorder=2)
            for model, value in zip(S.MODELS, x):
                ax.plot(value, y, S.MODEL_MARKER[model], ms=4.2, mfc='white', mec=color, mew=1, zorder=3)
            ax.plot(x.mean(), y, '|', ms=9, mew=1.6, color=S.INK, zorder=4)
            same = int(max((x > 0).sum(), (x < 0).sum()))
            blocks = int(values.common_panel_blocks.iloc[0])
            ink = dict(transform=ax.get_yaxis_transform(), va='center', fontsize=6.3,
                       fontweight='bold' if same == 4 else 'normal', color=S.INK if same == 4 else S.MUTED)
            ax.text(1.04, y, f'{same} of 4' + (f'   ({blocks}/8 blocks)' if blocks < 8 else ''), ha='left', **ink)
            ax.text(1.40, y, f'{x.max() - x.min():.3f}', ha='left', **ink)
            for model, value in zip(S.MODELS, x):
                out.append(dict(method=method, culture=cond[0], language=cond[1], model=model,
                                common_panel_blocks=blocks, model_mean_difference=value, models_same_sign=same,
                                model_range=x.max() - x.min()))
    extent = frame.common_panel_mean.abs().max() * 1.06
    ax.set_xlim(-extent, extent)
    ax.set_xlabel('Model-specific mean difference from US · English over the common seed blocks (endpoint units)')
    fig.legend(handles=S.model_handles() + [S.handle('Four-model mean', marker='|', ms=9, mew=1.6)],
               loc='lower left', ncol=5, frameon=False, columnspacing=1.6, handletextpad=.4,
               bbox_to_anchor=(.02, .0))
    return fig, pd.DataFrame(out)


def fig04(D):
    """Every one of the 896 attempts as one square, then what the six failures cost."""
    primary, frame = D['primary'], D['tables']['primary_contrasts']
    fig = plt.figure(figsize=(S.FULL, 3.9))
    grid = fig.add_gridspec(2, 2, width_ratios=[2.5, 1], height_ratios=[2, 1.15], left=.155, right=.985,
                            top=.86, bottom=.2, wspace=.42, hspace=.95)
    ax = fig.add_subplot(grid[:, 0])
    S.panel(ax, 'a', 'All 896 primary attempts: one square per attempt, seeds 44000–44007 left to right', grid=None)
    seeds = sorted(primary.seed.unique())
    index = primary.set_index(['method', 'model', 'culture', 'language', 'seed'])
    out, y, yticks = [], 0, []
    for method in S.METHODS:
        for model in S.MODELS:
            for c, cond in enumerate(S.CONDITIONS):
                for k, seed in enumerate(seeds):
                    row = index.loc[(method, model, *cond, seed)]
                    state = 'failed' if not row.valid_graph else 'empty' if bool(row.empty_graph) else 'nonempty'
                    x = c * 9.2 + k
                    ax.add_patch(Rectangle((x + .08, y + .1), .84, .8, lw=.5,
                                           facecolor={'failed': S.FAIL, 'empty': 'white', 'nonempty': S.ADD}[state],
                                           edgecolor={'failed': S.FAIL, 'empty': S.INK, 'nonempty': S.ADD}[state]))
                    out.append(dict(method=method, model=model, culture=cond[0], language=cond[1], seed=seed,
                                    outcome=state))
            yticks.append((y + .5, model))
            y += 1
        y += .7
    ax.set_xlim(-.3, 6 * 9.2 + 8.3)
    ax.set_ylim(y - .5, -.3)
    ax.set_yticks([p for p, _ in yticks], [m for _, m in yticks])
    ax.set_xticks([c * 9.2 + 4 for c in range(7)], [S.condition_label(*c).replace(' · ', '\n') for c in S.CONDITIONS])
    ax.tick_params(axis='x', length=0, pad=3)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for m, method in enumerate(S.METHODS):
        ax.text(-.215, m * 4.7 + 2, method.title(), transform=ax.get_yaxis_transform(), rotation=90,
                ha='center', va='center', fontsize=7, color=S.METHOD_COLOR[method], fontweight='bold')
    table = pd.DataFrame(out)
    counts = table.outcome.value_counts()
    if (len(table), counts['nonempty'], counts['empty'], counts['failed']) != (896, 878, 12, 6):
        raise ValueError('Attempt outcomes differ from the primary cohort')
    ax.legend(handles=[
        S.handle(f"Valid, non-empty graph ({counts['nonempty']})", marker='s', ms=5, mfc=S.ADD, mec=S.ADD),
        S.handle(f"Valid graph with no edges ({counts['empty']})", marker='s', ms=5, mfc='white', mec=S.INK, mew=.6),
        S.handle(f"Failed attempt, no graph ({counts['failed']})", marker='s', ms=5, mfc=S.FAIL, mec=S.FAIL)],
        loc='upper center', bbox_to_anchor=(.44, -.105), ncol=3, frameon=False, columnspacing=1.2,
        handletextpad=.2)
    affected = frame[frame.complete_schedule_blocks < 8]
    for position, endpoint, letter, title in ((0, 'density', 'b', 'Affected contrasts: density'),
                                              (1, 'avg_clustering_coef', 'c', 'Affected contrast: clustering')):
        bx = fig.add_subplot(grid[position, 1])
        rows = affected[affected.endpoint.eq(endpoint)]
        S.panel(bx, letter, title)
        bx.axvline(0, color=S.RULE, lw=.7)
        for yy, row in enumerate(rows.itertuples()):
            bx.plot([row.missingness_lower, row.missingness_upper], [yy, yy], color=S.FAIL, lw=3, alpha=.35,
                    solid_capstyle='butt')
            bx.plot(row.complete_block_mean, yy, 'o', ms=4.4, color=S.INK)
            status = 'no test' if not np.isfinite(row.raw_p) else f'tested on {row.complete_schedule_blocks}'
            bx.text(1.0, yy - .34, f'{row.complete_schedule_blocks}/8 blocks, {status}',
                    transform=bx.get_yaxis_transform(), ha='right', va='center', fontsize=5.8, color=S.MUTED)
        bx.set_yticks(range(len(rows)), [f'{r.method.title()}\nUS · {r.language.title()}' for r in rows.itertuples()])
        bx.set_ylim(len(rows) - .45, -.75)
        lo, hi = rows.missingness_lower.min(), rows.missingness_upper.max()
        pad = (max(hi, 0) - min(lo, 0)) * .08
        bx.set_xlim(min(lo, 0) - pad, max(hi, 0) + pad)
        bx.xaxis.set_major_locator(plt.MaxNLocator(4))
        bx.set_xlabel('Difference from US · English')
    S.note(fig, 'b, c: dot = mean of complete blocks; bar = range the block mean could take if each missing endpoint lay '
           'anywhere in [0, 1].\nThese are identification bounds, not confidence intervals. Failures are never zero-filled '
           'or replaced by the six later supplements.')
    return fig, table


def fig05(D):
    """Compute per attempt on a log scale: procedures were not given equal resources."""
    data = D['tables']['resources_by_method_model'].set_index(['method', 'model'])
    fig, axes = plt.subplots(1, 3, figsize=(S.FULL, 2.55), sharey=True)
    out = []
    specs = (('requests', 'Requests per attempt', 1), ('received_conservative_usd', 'Estimated cost per attempt (USD)', 1),
             ('api_seconds', 'Provider seconds per attempt', 1))
    for ax, letter, (column, label, scale) in zip(axes, 'abc', specs):
        S.panel(ax, letter, label)
        base = np.array([data.loc[('global', m), column] / data.loc[('global', m), 'attempts'] for m in S.MODELS])
        for y, method in enumerate(S.METHODS):
            values = np.array([data.loc[(method, m), column] / data.loc[(method, m), 'attempts'] for m in S.MODELS])
            ax.plot([values.min(), values.max()], [y, y], color=S.METHOD_COLOR[method], lw=.7, alpha=.55)
            for model, value in zip(S.MODELS, values):
                ax.plot(value, y, S.MODEL_MARKER[model], ms=4.2, mfc='white', mec=S.METHOD_COLOR[method], mew=1)
                out.append(dict(quantity=column, method=method, model=model, per_attempt=value,
                                attempts=int(data.loc[(method, model), 'attempts'])))
            if method != 'global':
                ratio = np.median(values / base)
                ax.text(.985, y - .33, f'×{ratio:,.0f} Global' if ratio >= 10 else f'×{ratio:.1f} Global',
                        transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=5.8, color=S.MUTED)
        ax.set_xscale('log')
        ax.set_ylim(3.55, -.65)
        ax.set_yticks(range(4), [m.title() for m in S.METHODS])
        ax.set_xlabel('Log scale')
        ax.grid(axis='x', which='major', color=S.GRID, lw=.5)
        ax.minorticks_off()
    fig.legend(handles=S.model_handles(), loc='lower center', ncol=4, frameon=False, columnspacing=1.6,
               handletextpad=.4, bbox_to_anchor=(.5, .085))
    S.note(fig, 'Each mark: one method-model total over 56 attempts (failed attempts included), divided by 56. '
           '× = median ratio to Global across the four models. Cost is the received-usage estimate.')
    fig.subplots_adjust(left=.095, right=.985, top=.9, bottom=.34, wspace=.12)
    return fig, pd.DataFrame(out)


def fig06(D):
    """Clustering beyond what the same number of edges placed at random would give."""
    ref = D['tables']['synthetic_reference_per_run']
    ref = ref[ref.metric.eq('avg_clustering_coef') & ref.valid_graph].copy()
    ref['excess'] = ref.observed_value - ref.reference_mean
    ref = ref.dropna(subset=['excess'])
    fig, ax = plt.subplots(figsize=(S.FULL, 3.85))
    S.panel(ax, '', 'Observed clustering minus the mean of eight random graphs with the same nodes and edge count')
    ax.axvline(0, color=S.INK, lw=.7)
    out, y, ticks = [], 0, []
    for method in S.METHODS:
        color = S.METHOD_COLOR[method]
        for model in S.MODELS:
            values = ref[ref.method.eq(method) & ref.model.eq(model)].excess.to_numpy()
            S.barcode(ax, values, y, color, height=.3, alpha=.4)
            stats = S.summary_mark(ax, values, np.ones(len(values)), y, color, S.MODEL_MARKER[model])
            ax.text(1.012, y, f'n = {len(values)}', transform=ax.get_yaxis_transform(), va='center',
                    fontsize=5.8, color=S.MUTED)
            out.append(dict(method=method, model=model, valid_graphs=len(values), **stats))
            ticks.append((y, model))
            y += 1
        y += .6
    ax.set_yticks([p for p, _ in ticks], [m for _, m in ticks])
    ax.set_ylim(y - .9, -.7)
    for m, method in enumerate(S.METHODS):
        ax.text(-.135, m * 4.6 + 1.5, method.title(), transform=ax.get_yaxis_transform(), rotation=90,
                ha='center', va='center', fontsize=7, color=S.METHOD_COLOR[method], fontweight='bold')
    ax.set_xlabel('Clustering excess over same-size, same-edge-count random graphs (0 = no excess)')
    ax.legend(handles=[S.handle('One valid primary graph', marker='|', ms=7, mew=.6, color=S.MUTED),
                       S.handle('Median', marker='o', ms=4, mfc='white', mec=S.INK, mew=1),
                       S.handle('Interquartile range', ls='-', lw=2.6),
                       S.handle('5th–95th percentile', ls='-', lw=.8)],
              loc='upper center', bbox_to_anchor=(.45, -.115), ncol=4, frameon=False, columnspacing=1.6)
    fig.subplots_adjust(left=.16, right=.93, top=.93, bottom=.17)
    return fig, pd.DataFrame(out)


def fig07(D):
    """Where the generated graphs sit relative to eight archived small friendship networks."""
    primary, records = D['primary'], D['tables']['empirical_reference_metrics'].reset_index(drop=True)
    metrics = ('density', 'avg_clustering_coef', 'global_efficiency')
    fig, axes = plt.subplots(1, 3, figsize=(S.FULL, 2.95), sharey=True)
    out = []
    for ax, letter, metric in zip(axes, 'abc', metrics):
        S.panel(ax, letter, S.METRIC_LABEL[metric])
        ax.axvspan(records[metric].min(), records[metric].max(), color='#F1F3F5', lw=0, zorder=0)
        for y, method in enumerate(S.METHODS):
            values = primary.loc[primary.valid_graph & primary.method.eq(method), metric].dropna().to_numpy()
            S.barcode(ax, values, y, S.METHOD_COLOR[method], alpha=.22)
            stats = S.summary_mark(ax, values, np.ones(len(values)), y, S.METHOD_COLOR[method])
            out.append(dict(metric=metric, row=method, n=len(values), **stats))
        order = records[metric].rank(method='first').astype(int) - 1
        for i, record in records.iterrows():
            lane = 4.72 + .4 * (order[i] % 3)
            ax.plot(record[metric], 4.3, 'D', ms=3.6, color=S.OBSERVED, zorder=4)
            ax.text(record[metric], lane + .22, str(i + 1), ha='center', va='center', fontsize=5.6, color=S.INK)
            out.append(dict(metric=metric, row=f'{i + 1} {record.reference}', n=int(record.nodes),
                            median=record[metric]))
        ax.set_ylim(6.1, -.65)
        ax.set_xlabel(S.METRIC_LABEL[metric] + ' (0–1)')
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    axes[0].set_yticks([0, 1, 2, 3, 4.3], [m.title() for m in S.METHODS] + ['8 archived\nnetworks'])
    key = '   '.join(f'{i + 1} {r.reference} (n={r.nodes})' for i, r in records.iterrows())
    S.note(fig, 'Generated rows: all valid primary graphs (50 nodes); tick = one graph, circle = median, thick = interquartile, '
           'thin = 5th–95th percentile.\nShaded band: range of the eight archived networks. ' + key)
    fig.subplots_adjust(left=.11, right=.985, top=.91, bottom=.3, wspace=.1)
    return fig, pd.DataFrame(out)


FIGURES = {'01_primary_contrasts': fig01, '02_primary_robustness': fig02, '03_model_profiles': fig03,
           '04_coverage_missingness': fig04, '05_resource_use': fig05, '06_gnm_context': fig06,
           '07_legacy_comparison': fig07}
