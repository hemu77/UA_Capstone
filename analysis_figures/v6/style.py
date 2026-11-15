"""Shared visual grammar for release v6. One role per color and marker."""
import textwrap

from matplotlib.lines import Line2D
import numpy as np

# Sizes are final print sizes in inches: text stays legible without zooming.
FULL = 7.2
INK, MUTED, RULE, GRID = '#1F2933', '#6B7780', '#9AA5AE', '#E4E8EB'
METHODS = ('global', 'local', 'sequential', 'iterative')
# Okabe-Ito hues: distinguishable under common color-vision deficiencies.
METHOD_COLOR = dict(zip(METHODS, ('#0072B2', '#D55E00', '#009E73', '#AA4499')))
MODELS = ('gpt-4.1', 'gpt-5.6-luna', 'gpt-6-luna', 'gpt-6-sol')
MODEL_MARKER = dict(zip(MODELS, ('o', 's', '^', 'D')))
BASELINES = ('ER', 'BA', 'WS')
BASELINE_MARKER = dict(zip(BASELINES, ('s', '^', 'v')))
BASELINE_COLOR = '#59636B'
OBSERVED = '#111111'
FACEBOOK, SCHOOL, COPENHAGEN = '#4F6D7A', '#B8500F', '#111111'
DIAGNOSTIC = '#111111'
ADD, MISS, FAIL = '#CBD3D9', '#FFFFFF', '#C8371B'
CONDITIONS = (('us', 'english'), ('india', 'english'), ('japan', 'english'), ('brazil', 'english'),
              ('us', 'hindi'), ('us', 'japanese'), ('us', 'portuguese'))
COUNTRY = {'us': 'US', 'india': 'India', 'japan': 'Japan', 'brazil': 'Brazil'}
METRIC_LABEL = {'density': 'Density', 'avg_clustering_coef': 'Clustering', 'global_efficiency': 'Efficiency',
                'transitivity': 'Transitivity', 'degree_gini': 'Degree inequality',
                'prop_nodes_lcc': 'Largest component'}
ATTRIBUTES = ('political orientation', 'religion', 'gender')
ATTRIBUTE_LABEL = {'political orientation': 'Politics', 'religion': 'Religion', 'gender': 'Gender'}

STYLE = {
    'font.family': 'DejaVu Sans', 'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 7.5,
    'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 6.5,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .6,
    'axes.edgecolor': RULE, 'text.color': INK, 'axes.labelcolor': INK,
    'xtick.color': INK, 'ytick.color': INK, 'xtick.major.width': .6, 'ytick.major.width': .6,
    'xtick.major.size': 2.5, 'ytick.major.size': 0, 'lines.solid_capstyle': 'butt',
    'svg.fonttype': 'none', 'svg.hashsalt': 'figure-release-060dcd3a-v6', 'pdf.fonttype': 42,
    'savefig.facecolor': 'white', 'figure.facecolor': 'white', 'hatch.linewidth': .5,
}


def condition_label(culture, language):
    return f'{COUNTRY[culture]} · {language.title()}'


def panel(ax, letter, title, grid='x'):
    """Left-aligned sentence-case title with a bold panel letter."""
    ax.set_title(title, loc='left', pad=5, fontsize=7.5, color=INK)
    if letter:
        ax.annotate(letter, (0, 1), xytext=(-3, 5), xycoords='axes fraction', textcoords='offset points',
                    ha='right', va='bottom', fontsize=8, fontweight='bold', color=INK, annotation_clip=False)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=.5)
    ax.set_axisbelow(True)


def note(fig, text, y=.012, x=.01, width=None):
    """Small reading key under the panels, wrapped to the figure width."""
    width = width or int(fig.get_figwidth() * 20.5)
    lines = [w for part in text.split('\n') for w in (textwrap.wrap(part, width) or [''])]
    fig.text(x, y, '\n'.join(lines), fontsize=6, color=MUTED, ha='left', va='bottom', linespacing=1.35)


def weighted_quantile(values, weights, q):
    """Left-continuous weighted quantile; weights need not sum to one."""
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    if (len(values) != len(weights) or not len(values) or not np.isfinite(values).all()
            or not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0):
        raise ValueError('Invalid weighted sample')
    order = np.argsort(values, kind='stable')
    values, cumulative = values[order], np.cumsum(weights[order]) / weights.sum()
    return np.array([values[min(np.searchsorted(cumulative, p - 1e-12, side='left'), len(values) - 1)]
                     for p in np.atleast_1d(q)])


def barcode(ax, values, y, color, height=.32, alpha=.28, lw=.5):
    ax.vlines(values, y - height, y + height, color=color, alpha=alpha, lw=lw, zorder=1)


def summary_mark(ax, values, weights, y, color, marker='o'):
    """Median, interquartile range and 5-95% span of a weighted sample."""
    lo, q1, med, q3, hi = weighted_quantile(values, weights, [.05, .25, .5, .75, .95])
    ax.plot([lo, hi], [y, y], color=color, lw=.8, zorder=3)
    ax.plot([q1, q3], [y, y], color=color, lw=2.6, zorder=3)
    ax.plot(med, y, marker, mfc='white', mec=color, mew=1, ms=4, zorder=4)
    return dict(p05=lo, p25=q1, median=med, p75=q3, p95=hi)


def handle(label, **kw):
    kw.setdefault('color', INK)
    kw.setdefault('ls', '')
    return Line2D([], [], label=label, **kw)


def model_handles():
    return [handle(m, marker=MODEL_MARKER[m], ms=3.5, mfc='white', mec=INK, mew=.8) for m in MODELS]


def strip(ax, values, weights, y, color, marker='o', alpha=.3, height=.3):
    """Every observation as a tick, with its weighted median, quartiles and 5-95% span."""
    values = np.asarray(values, float)
    barcode(ax, values, y, color, height=height, alpha=alpha)
    return summary_mark(ax, values, weights, y, color, marker)


def record_label(cohort):
    return {'School2013': 'School', 'Copenhagen': 'Copenhagen'}.get(cohort, cohort)
