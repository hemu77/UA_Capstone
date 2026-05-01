"""
Figure 5.4 — Pairwise edge distance between model pairs across three studies.

Edge distance(G1, G2) = |E1 △ E2| / (N*(N-1))
where N=50, so denominator = 2450. Both graphs are treated as directed
(each undirected edge counts as two directed edges) so the symmetric
difference is also counted in directed form.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

# ── Config ──────────────────────────────────────────────────────────────────
TEXT_FILES = './text-files'
OUT_PATH   = 'plots/figure_5_4_model_edge_distance.png'
N_NODES    = 50
DENOM      = N_NODES * (N_NODES - 1)   # 2450 directed edges

MODELS   = ['gpt-4.1', 'gpt-4.1-mini', 'gpt-4.1-nano']
CULTURES = ['us', 'india', 'japan', 'brazil']
SEEDS    = [0, 1]
LANGS    = ['english', 'hindi', 'japanese', 'spanish']

MODEL_PAIRS = [
    ('gpt-4.1',      'gpt-4.1-mini'),
    ('gpt-4.1',      'gpt-4.1-nano'),
    ('gpt-4.1-mini', 'gpt-4.1-nano'),
]
PAIR_LABELS = {
    ('gpt-4.1',      'gpt-4.1-mini'): '4.1 vs 4.1-mini',
    ('gpt-4.1',      'gpt-4.1-nano'): '4.1 vs 4.1-nano',
    ('gpt-4.1-mini', 'gpt-4.1-nano'): '4.1-mini vs 4.1-nano',
}
PAIR_COLORS = {
    ('gpt-4.1',      'gpt-4.1-mini'): '#4C72B0',
    ('gpt-4.1',      'gpt-4.1-nano'): '#DD8452',
    ('gpt-4.1-mini', 'gpt-4.1-nano'): '#55A868',
}

# ── Helpers ──────────────────────────────────────────────────────────────────
def load_graph(path):
    G = nx.DiGraph()
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
                    G.add_edge(dst, src)   # undirected → both directions
    return G

def edge_distance(G1, G2):
    e1 = set(G1.edges())
    e2 = set(G2.edges())
    return len(e1.symmetric_difference(e2)) / DENOM

def adj_path(method, model, culture, seed, lang=None):
    suffix = '' if method == 'global' else '_n5'
    if lang:
        return os.path.join(TEXT_FILES,
            f'{method}_{model}{suffix}_culture_{culture}_lang_{lang}_{seed}.adj')
    return os.path.join(TEXT_FILES,
        f'{method}_{model}{suffix}_culture_{culture}_{seed}.adj')

# ── Study definitions ────────────────────────────────────────────────────────
def get_distances(conditions):
    """
    conditions: list of (method, culture, seed, lang=None)
    Returns dict {pair: [list of edge distances]}
    """
    distances = {p: [] for p in MODEL_PAIRS}
    for (method, culture, seed, lang) in conditions:
        graphs = {}
        for model in MODELS:
            p = adj_path(method, model, culture, seed, lang)
            if not os.path.exists(p):
                print(f'  MISSING: {p}')
                continue
            graphs[model] = load_graph(p)
        for pair in MODEL_PAIRS:
            m1, m2 = pair
            if m1 in graphs and m2 in graphs:
                distances[pair].append(edge_distance(graphs[m1], graphs[m2]))
    return distances

# Cultural study: sequential, all 4 cultures, 2 seeds
cultural_conditions = [
    ('sequential', culture, seed, None)
    for culture in CULTURES for seed in SEEDS
]

# Method study: global + local + iterative, all 4 cultures, 2 seeds
method_conditions = [
    (method, culture, seed, None)
    for method in ['global', 'local', 'iterative']
    for culture in CULTURES for seed in SEEDS
]

# Language study: all 4 methods, culture=us, all 4 langs, 2 seeds
language_conditions = [
    (method, 'us', seed, lang)
    for method in ['sequential', 'global', 'local', 'iterative']
    for lang in LANGS for seed in SEEDS
]

studies = [
    ('Cultural',  cultural_conditions),
    ('Method',    method_conditions),
    ('Language',  language_conditions),
]

results = {}
for study_name, conditions in studies:
    dists = get_distances(conditions)
    results[study_name] = {pair: np.mean(vals) for pair, vals in dists.items()}
    print(f'\n{study_name} study:')
    for pair, val in results[study_name].items():
        n = len(dists[pair])
        print(f'  {PAIR_LABELS[pair]:25s}: {val:.4f}  (n={n})')

# ── Plot ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':    'sans-serif',
    'font.size':      12,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize':11,
    'ytick.labelsize':11,
    'legend.fontsize':11,
})

study_names  = [s[0] for s in studies]
n_studies    = len(study_names)
n_pairs      = len(MODEL_PAIRS)
x            = np.arange(n_studies)
width        = 0.22
offsets      = np.linspace(-(n_pairs - 1) / 2, (n_pairs - 1) / 2, n_pairs) * width

fig, ax = plt.subplots(figsize=(11, 6))

for i, pair in enumerate(MODEL_PAIRS):
    means = [results[s][pair] for s in study_names]
    bars  = ax.bar(
        x + offsets[i],
        means,
        width=width * 0.9,
        label=PAIR_LABELS[pair],
        color=PAIR_COLORS[pair],
        edgecolor='white',
        linewidth=0.7,
        zorder=3,
    )
    # Value labels above each bar
    for bar, val in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f'{val:.3f}',
            ha='center', va='bottom',
            fontsize=9, color='#222222', fontweight='bold',
        )

ax.set_xticks(x)
ax.set_xticklabels(study_names, fontsize=11)
ax.set_xlabel('Study', fontsize=12, labelpad=8)
ax.set_ylabel('Average Pairwise Edge Distance', fontsize=12, labelpad=8)
ax.set_title('Pairwise Edge Distance Between Model Pairs Across Studies',
             fontsize=13, fontweight='bold', pad=12)

ymax = max(results[s][p] for s in study_names for p in MODEL_PAIRS)
ax.set_ylim(0, ymax + 0.06)

# Grid y-only
ax.yaxis.grid(True, alpha=0.4, linewidth=0.7, zorder=0)
ax.xaxis.grid(False)
ax.set_axisbelow(True)
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

# Note
ax.text(0.99, 0.97, 'Lower distance = more similar outputs',
        transform=ax.transAxes, ha='right', va='top',
        fontsize=9, style='italic', color='#555555')

# Legend outside right
ax.legend(
    title='Model Pair',
    title_fontsize=11,
    fontsize=10,
    loc='upper left',
    bbox_to_anchor=(1.01, 1.0),
    borderaxespad=0,
    frameon=True,
    framealpha=0.92,
    edgecolor='#cccccc',
)

plt.tight_layout(rect=[0, 0, 0.82, 1])

os.makedirs('plots', exist_ok=True)
plt.savefig(OUT_PATH, dpi=300, bbox_inches='tight')
print(f'\nSaved → {OUT_PATH}')
