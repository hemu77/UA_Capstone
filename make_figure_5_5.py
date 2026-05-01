"""
Figure 5.5 — Religion and political affiliation homophily by prompt language
and generation method (language study only).
"""

import os, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from collections import Counter

# ── Config ──────────────────────────────────────────────────────────────────
TEXT_FILES   = './text-files'
PERSONA_FILE = 'text-files/us_50_gpt4o_w_interests.json'
OUT_PATH     = 'plots/figure_5_5_language_homophily.png'
N_NODES      = 50

METHODS  = ['sequential', 'global', 'local', 'iterative']
MODELS   = ['gpt-4.1-nano', 'gpt-4.1-mini', 'gpt-4.1']
LANGS    = ['english', 'spanish', 'hindi', 'japanese']
SEEDS    = [0, 1]
DIMS     = ['religion', 'political affiliation']

LANG_LABELS = {
    'english':  'English',
    'spanish':  'Spanish',
    'hindi':    'Hindi',
    'japanese': 'Japanese',
}
METHOD_LABELS = {
    'sequential': 'Sequential',
    'global':     'Global',
    'local':      'Local',
    'iterative':  'Iterative',
}
METHOD_COLORS = {
    'Sequential': '#4C72B0',
    'Global':     '#DD8452',
    'Local':      '#55A868',
    'Iterative':  '#C44E52',
}
DIM_TITLES = {
    'religion':             'Religion',
    'political affiliation':'Political Affiliation',
}

# ── Load personas ────────────────────────────────────────────────────────────
with open(PERSONA_FILE) as f:
    raw = json.load(f)
personas = {int(k): v for k, v in raw.items()}

def get_label(node, dim):
    return personas[node][dim]

def random_baseline(dim):
    labels = [get_label(i, dim) for i in range(N_NODES)]
    counts = Counter(labels)
    N = len(labels)
    return sum((c / N) ** 2 for c in counts.values())

baselines = {d: random_baseline(d) for d in DIMS}
print('Random baselines:')
for d, b in baselines.items():
    print(f'  {d:25s}: {b:.4f}')

# ── Graph loader ─────────────────────────────────────────────────────────────
def load_graph(path):
    G = nx.Graph()
    G.add_nodes_from(range(N_NODES))
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = list(map(int, line.split()))
            src = parts[0]
            for dst in parts[1:]:
                if src != dst:
                    G.add_edge(src, dst)
    return G

def same_group_ratio(G, dim):
    edges = list(G.edges())
    if not edges:
        return np.nan
    labels  = {n: get_label(n, dim) for n in range(N_NODES)}
    same    = sum(1 for u, v in edges if labels[u] == labels[v])
    total   = len(edges)
    counts  = Counter(labels.values())
    N = N_NODES
    expected_frac = sum(c * (c - 1) for c in counts.values()) / (N * (N - 1))
    observed_frac = same / total
    return observed_frac / expected_frac if expected_frac > 0 else np.nan

def adj_path(method, model, lang, seed):
    suffix = '' if method == 'global' else '_n5'
    return os.path.join(TEXT_FILES,
        f'{method}_{model}{suffix}_culture_us_lang_{lang}_{seed}.adj')

# ── Aggregate: results[dim][method][lang] = list of ratios ───────────────────
results = {
    dim: {m: {l: [] for l in LANGS} for m in METHODS}
    for dim in DIMS
}

for method in METHODS:
    for model in MODELS:
        for lang in LANGS:
            for seed in SEEDS:
                p = adj_path(method, model, lang, seed)
                if not os.path.exists(p):
                    print(f'  MISSING: {p}')
                    continue
                G = load_graph(p)
                for dim in DIMS:
                    r = same_group_ratio(G, dim)
                    if not np.isnan(r):
                        results[dim][method][lang].append(r)

# Print summary
for dim in DIMS:
    print(f'\n{dim}:')
    for method in METHODS:
        row = '  '.join(
            f'{LANG_LABELS[l]}: {np.mean(results[dim][method][l]):.3f}'
            for l in LANGS if results[dim][method][l]
        )
        print(f'  {METHOD_LABELS[method]:12s} | {row}')

# ── Plot ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':    'sans-serif',
    'font.size':      11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize':10,
    'ytick.labelsize':10,
    'legend.fontsize':10,
})

method_order = METHODS
lang_order   = LANGS
n_langs      = len(lang_order)
n_methods    = len(method_order)
x            = np.arange(n_langs)
width        = 0.18
offsets      = np.linspace(-(n_methods - 1) / 2, (n_methods - 1) / 2, n_methods) * width

fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
fig.subplots_adjust(wspace=0.32)

for ax, dim in zip(axes, DIMS):
    baseline = baselines[dim]

    for i, method in enumerate(method_order):
        mlabel = METHOD_LABELS[method]
        means  = [np.mean(results[dim][method][l]) if results[dim][method][l] else 0
                  for l in lang_order]
        bars = ax.bar(
            x + offsets[i],
            means,
            width=width * 0.9,
            label=mlabel,
            color=METHOD_COLORS[mlabel],
            edgecolor='white',
            linewidth=0.6,
            zorder=3,
        )
        for bar, val in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f'{val:.2f}',
                ha='center', va='bottom',
                fontsize=7.5, color='#222222', fontweight='bold',
            )

    # Random baseline
    ax.axhline(
        baseline, color='#333333', linestyle='--', linewidth=1.3,
        zorder=2, label=f'Random baseline ({baseline:.3f})'
    )

    ax.set_xticks(x)
    ax.set_xticklabels([LANG_LABELS[l] for l in lang_order], fontsize=10)
    ax.set_xlabel('Prompt Language', fontsize=11, labelpad=6)
    ax.set_ylabel('Average Same-Group Ratio', fontsize=11, labelpad=6)
    ax.set_title(DIM_TITLES[dim], fontsize=12, fontweight='bold', pad=10)

    ymax = max(
        np.mean(results[dim][m][l])
        for m in method_order for l in lang_order
        if results[dim][m][l]
    )
    ax.set_ylim(0, ymax + 0.22)
    ax.yaxis.grid(True, alpha=0.4, linewidth=0.7, zorder=0)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

# Shared legend — collect from left subplot, place below figure
handles, labels = axes[0].get_legend_handles_labels()
# Separate method bars from baseline line
bar_h  = [h for h, l in zip(handles, labels) if 'baseline' not in l]
bar_l  = [l for l in labels if 'baseline' not in l]
# Each subplot has its own baseline line with different value — show both
base_h = [h for h, l in zip(handles, labels) if 'baseline' in l]
base_l = [l for l in labels if 'baseline' in l]
# Add right subplot baseline handle too
rh, rl = axes[1].get_legend_handles_labels()
base_h += [h for h, l in zip(rh, rl) if 'baseline' in l]
base_l += [l for l in rl if 'baseline' in l]

fig.legend(
    bar_h + base_h,
    bar_l + base_l,
    title='Generation Method',
    title_fontsize=10,
    fontsize=9.5,
    loc='lower center',
    bbox_to_anchor=(0.5, -0.12),
    ncol=6,
    frameon=True,
    framealpha=0.92,
    edgecolor='#cccccc',
)

fig.suptitle(
    'Religion and Political Affiliation Homophily\nby Prompt Language and Generation Method',
    fontsize=13, fontweight='bold', y=1.02,
)

plt.tight_layout()

os.makedirs('plots', exist_ok=True)
plt.savefig(OUT_PATH, dpi=300, bbox_inches='tight')
print(f'\nSaved → {OUT_PATH}')
