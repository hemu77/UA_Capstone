"""Figure 20: how the closest generator depends on the sampling scheme. Same visual grammar as release v6."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis_figures.v6 import style as S

PROCEDURES = S.METHODS + tuple(S.BASELINES)
DASH = {'ER': (0, (1, 1.2)), 'BA': (0, (4, 1.5)), 'WS': '-'}
PANELS = (('density', 'Density'), ('global_efficiency', 'Efficiency'), ('avg_clustering_coef', 'Clustering'),
          ('transitivity', 'Transitivity'))


def render(folder):
    table = pd.read_csv(folder / 'sampling_scheme_distances.csv')
    means = table.groupby(['sampling', 'metric', 'procedure']).mean_w1.agg(['mean', 'size']).reset_index()
    if means['size'].ne(11).any():
        raise ValueError('Expected 11 records per sampling scheme, metric and procedure')
    with plt.rc_context(S.STYLE):
        fig, axes = plt.subplots(1, 4, figsize=(S.FULL, 2.75))
        for ax, letter, (metric, title) in zip(axes, 'abcd', PANELS):
            S.panel(ax, letter, title, grid='y')
            ends = []
            for p in PROCEDURES:
                rows = means[(means.metric == metric) & (means.procedure == p)].set_index('sampling')['mean']
                y = [rows['uniform'], rows['snowball']]
                llm = p in S.METHODS
                color = S.METHOD_COLOR.get(p, S.BASELINE_COLOR)
                ax.plot([0, 1], y, color=color, lw=1.3 if llm else 1, ls='-' if llm else DASH[p],
                        marker='o' if llm else S.BASELINE_MARKER[p], ms=3.4, mfc=color if llm else 'white', mec=color,
                        mew=.8, zorder=3 if llm else 2)
            ax.set_xlim(-.25, 1.25)
            ax.set_ylim(0, None)
            ax.set_xticks([0, 1], ['Uniform\ndraws', 'Snowball\ndraws'])
            ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        axes[0].set_ylabel('Mean $W_1$ over 11 records\n(smaller = closer)')
        handles = [S.handle(p.title(), marker='o', ms=3.4, color=S.METHOD_COLOR[p], ls='-', lw=1.3) for p in S.METHODS]
        handles += [S.handle(p + ', fixed', marker=S.BASELINE_MARKER[p], ms=3.4, mfc='white', mec=S.BASELINE_COLOR,
                             color=S.BASELINE_COLOR, ls=DASH[p], lw=1) for p in S.BASELINES]
        fig.legend(handles=handles, loc='lower center', ncol=7, frameon=False, columnspacing=1.1, handletextpad=.4,
                   bbox_to_anchor=(.5, .0))
        fig.subplots_adjust(left=.1, right=.99, top=.9, bottom=.3, wspace=.34)
        for ext, meta in (('png', {'Software': None}), ('svg', {'Date': None}),
                          ('pdf', {'CreationDate': None, 'ModDate': None, 'Producer': None, 'Creator': None})):
            fig.savefig(folder / f'20_sampling_scheme.{ext}', dpi=300, metadata=meta)
        plt.close(fig)
