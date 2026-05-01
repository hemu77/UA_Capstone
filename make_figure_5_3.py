"""
Figure 5.3 — Average same-group ratio heatmap by demographic dimension × generation method.

Computes same-group ratio directly from adjacency list files and the persona roster.
Age is binned into 10-year intervals. Interests are matched by Jaccard similarity
on the comma-split token set (same-group = Jaccard >= 0.25).
Averages across all cultures, models, and seeds within each method × dimension cell.
"""

import os, re, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import networkx as nx
from collections import defaultdict, Counter
from itertools import combinations

# ── Config ─────────────────────────────────────────────────────────────────
TEXT_FILES   = './text-files'
PERSONA_FILE = 'text-files/us_50_gpt4o_w_interests.json'
OUT_PATH     = 'plots/figure_5_3_homophily_heatmap.png'
N_NODES      = 50

CULTURES = ['us', 'india', 'japan', 'brazil']
METHODS  = ['sequential', 'global', 'local', 'iterative']
MODELS   = ['gpt-4.1-nano', 'gpt-4.1-mini', 'gpt-4.1']
SEEDS    = [0, 1]

METHOD_LABELS = {
    'sequential': 'Sequential',
    'global':     'Global',
    'local':      'Local',
    'iterative':  'Iterative',
}
DIMS = ['gender', 'age', 'race/ethnicity', 'religion', 'political affiliation', 'interests']
DIM_LABELS = {
    'gender':               'Gender',
    'age':                  'Age (10-yr bins)',
    'race/ethnicity':       'Race / Ethnicity',
    'religion':             'Religion',
    'political affiliation':'Political Affiliation',
    'interests':            'Interests',
}

# ── Load personas ───────────────────────────────────────────────────────────
with open(PERSONA_FILE) as f:
    raw = json.load(f)
personas = {int(k): v for k, v in raw.items()}

def age_bin(age):
    return (age // 10) * 10   # 0-9 → 0, 10-19 → 10, etc.

def interest_tokens(s):
    return set(t.strip().lower() for t in s.split(',') if t.strip())

def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

# Precompute per-node group labels for each dimension
def get_label(node, dim):
    p = personas[node]
    if dim == 'age':
        return age_bin(p['age'])
    if dim == 'interests':
        return interest_tokens(p['interests'])   # set; handled separately
    return p[dim]

# ── Random baseline per dimension ──────────────────────────────────────────
# Expected same-group ratio under random friendship:
# sum_g (n_g/N)^2  ← probability that a random pair share the same group
def random_baseline(dim):
    if dim == 'interests':
        # mean Jaccard across all pairs as baseline
        tokens = [interest_tokens(personas[i]['interests']) for i in range(N_NODES)]
        vals = [jaccard(tokens[a], tokens[b])
                for a, b in combinations(range(N_NODES), 2)]
        return float(np.mean(vals))
    labels = [get_label(i, dim) for i in range(N_NODES)]
    counts = Counter(labels)
    total  = sum(counts.values())
    return sum((c / total) ** 2 for c in counts.values())

baselines = {d: random_baseline(d) for d in DIMS}
print('Random baselines:')
for d, b in baselines.items():
    print(f'  {d:25s}: {b:.4f}')

# ── Load adjacency list ─────────────────────────────────────────────────────
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

# ── Compute same-group ratio per graph × dimension ─────────────────────────
def same_group_ratio(G, dim):
    edges = list(G.edges())
    if not edges:
        return np.nan

    if dim == 'interests':
        tokens = [interest_tokens(personas[n]['interests']) for n in range(N_NODES)]
        observed = np.mean([jaccard(tokens[u], tokens[v]) for u, v in edges])
        baseline  = baselines['interests']
        return observed / baseline if baseline > 0 else np.nan

    labels = {n: get_label(n, dim) for n in range(N_NODES)}
    same   = sum(1 for u, v in edges if labels[u] == labels[v])
    total  = len(edges)

    # Expected fraction of same-group edges under random
    counts = Counter(labels.values())
    N = N_NODES
    # E[same edges] / total edges = sum_g n_g*(n_g-1) / (N*(N-1))
    expected_frac = sum(c * (c - 1) for c in counts.values()) / (N * (N - 1))
    observed_frac = same / total

    return observed_frac / expected_frac if expected_frac > 0 else np.nan

def adj_filename(method, model, culture, seed):
    if method == 'global':
        return f'{method}_{model}_culture_{culture}_{seed}.adj'
    return f'{method}_{model}_n5_culture_{culture}_{seed}.adj'

# ── Aggregate ───────────────────────────────────────────────────────────────
# data[method][dim] = list of ratio values
data = {m: {d: [] for d in DIMS} for m in METHODS}

for method in METHODS:
    for culture in CULTURES:
        for model in MODELS:
            for seed in SEEDS:
                fname = adj_filename(method, model, culture, seed)
                fpath = os.path.join(TEXT_FILES, fname)
                if not os.path.exists(fpath):
                    print(f'  MISSING: {fname}')
                    continue
                G = load_graph(fpath)
                for dim in DIMS:
                    r = same_group_ratio(G, dim)
                    if not np.isnan(r):
                        data[method][dim].append(r)

# Build matrix: rows=dims, cols=methods
method_order = ['sequential', 'global', 'local', 'iterative']
matrix = np.array([
    [np.mean(data[m][d]) if data[m][d] else np.nan for m in method_order]
    for d in DIMS
])

print('\nMean same-group ratios:')
print(f"{'Dimension':25s}", '  '.join(f'{METHOD_LABELS[m]:12s}' for m in method_order))
for i, d in enumerate(DIMS):
    vals = '  '.join(f'{matrix[i,j]:12.3f}' for j in range(len(method_order)))
    print(f'{d:25s} {vals}')

# ── Plot ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':    'sans-serif',
    'font.size':      12,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
})

fig, ax = plt.subplots(figsize=(9, 6))

# Diverging colormap centered at 1.0 (the same-group ratio baseline)
# Rescale matrix so that 1.0 maps to 0 for the diverging norm
vmin = min(0.7, np.nanmin(matrix) - 0.05)
vmax = max(1.3, np.nanmax(matrix) + 0.05)
norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)
cmap = plt.cm.RdYlBu_r   # warm above 1.0, cool below

im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect='auto')

# Colorbar
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Same-Group Ratio\n(1.0 = random baseline)', fontsize=10)
cbar.ax.tick_params(labelsize=9)

# Cell annotations
for i in range(len(DIMS)):
    for j in range(len(method_order)):
        val = matrix[i, j]
        if np.isnan(val):
            continue
        # Choose text color for contrast
        normed = norm(val)
        bg_rgb = plt.cm.RdYlBu_r(normed)
        luminance = 0.299 * bg_rgb[0] + 0.587 * bg_rgb[1] + 0.114 * bg_rgb[2]
        txt_color = 'black' if luminance > 0.45 else 'white'
        ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                fontsize=10.5, color=txt_color, fontweight='bold')

# Axes labels
ax.set_xticks(range(len(method_order)))
ax.set_xticklabels([METHOD_LABELS[m] for m in method_order], fontsize=11)
ax.set_yticks(range(len(DIMS)))
ax.set_yticklabels([DIM_LABELS[d] for d in DIMS], fontsize=11)
ax.set_xlabel('Generation Method', fontsize=12, labelpad=8)
ax.set_ylabel('Demographic Dimension', fontsize=12, labelpad=8)
ax.set_title('Average Same-Group Ratio by Demographic Dimension\nand Generation Method',
             fontsize=13, fontweight='bold', pad=12)

# Baseline annotation strip on the right — show expected ratio per dimension
ax2 = ax.twinx()
ax2.set_ylim(ax.get_ylim())
ax2.set_yticks(range(len(DIMS)))
ax2.set_yticklabels(
    [f'baseline={baselines[d]:.3f}' for d in DIMS],
    fontsize=8.5, color='#555555'
)
ax2.tick_params(axis='y', length=0)
for spine in ax2.spines.values():
    spine.set_visible(False)

# Subtle row separators
for y in np.arange(0.5, len(DIMS) - 0.5, 1):
    ax.axhline(y, color='white', linewidth=1.5, zorder=5)

plt.tight_layout()

os.makedirs('plots', exist_ok=True)
plt.savefig(OUT_PATH, dpi=300, bbox_inches='tight')
print(f'\nSaved → {OUT_PATH}')
