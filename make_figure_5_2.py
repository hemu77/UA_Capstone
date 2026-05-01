"""
Figure 5.2 — Proportion of nodes in the largest connected component (prop_nodes_lcc)
per culture × generation method, averaged across all three models and both seeds.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

# ── Config ─────────────────────────────────────────────────────────────────
TEXT_FILES = './text-files'
OUT_DIR    = './figures'
os.makedirs(OUT_DIR, exist_ok=True)

CULTURES = ['brazil', 'india', 'japan', 'us']
METHODS  = ['sequential', 'global', 'local', 'iterative']
MODELS   = ['gpt-4.1-nano', 'gpt-4.1-mini', 'gpt-4.1']
SEEDS    = [0, 1]
N_NODES  = 50

CULTURE_LABELS = {'us': 'US', 'india': 'India', 'japan': 'Japan', 'brazil': 'Brazil'}
METHOD_LABELS  = {
    'sequential': 'Sequential',
    'global':     'Global',
    'local':      'Local',
    'iterative':  'Iterative',
}
METHOD_COLORS  = {
    'Sequential': '#4C72B0',
    'Global':     '#DD8452',
    'Local':      '#55A868',
    'Iterative':  '#C44E52',
}

# ── Helpers ─────────────────────────────────────────────────────────────────
def adj_filename(method, model, culture, seed):
    if method == 'global':
        return f'{method}_{model}_culture_{culture}_{seed}.adj'
    return f'{method}_{model}_n5_culture_{culture}_{seed}.adj'

def prop_lcc(path):
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
                G.add_edge(src, dst)
    lcc = max(nx.connected_components(G), key=len)
    return len(lcc) / G.number_of_nodes()

# ── Compute ─────────────────────────────────────────────────────────────────
# results[method_label][culture_label] = list of prop_lcc values
results = {METHOD_LABELS[m]: {CULTURE_LABELS[c]: [] for c in CULTURES} for m in METHODS}

for method in METHODS:
    for culture in CULTURES:
        for model in MODELS:
            for seed in SEEDS:
                fname = adj_filename(method, model, culture, seed)
                fpath = os.path.join(TEXT_FILES, fname)
                if not os.path.exists(fpath):
                    print(f'  MISSING: {fname}')
                    continue
                val = prop_lcc(fpath)
                results[METHOD_LABELS[method]][CULTURE_LABELS[culture]].append(val)

# Summary
for method in METHOD_LABELS.values():
    for culture in CULTURE_LABELS.values():
        vals = results[method][culture]
        if vals:
            print(f'{method:12s} {culture:8s}  n={len(vals)}  mean={np.mean(vals):.3f}  sd={np.std(vals):.3f}')

# ── Plot ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':    'sans-serif',
    'font.size':      12,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize':11,
    'ytick.labelsize':11,
    'legend.fontsize':11,
})

culture_order = [CULTURE_LABELS[c] for c in CULTURES]
method_order  = [METHOD_LABELS[m]  for m in METHODS]
n_cultures    = len(culture_order)
n_methods     = len(method_order)

x       = np.arange(n_cultures)
width   = 0.17
offsets = np.linspace(-(n_methods - 1) / 2, (n_methods - 1) / 2, n_methods) * width

fig, ax = plt.subplots(figsize=(13, 6))

for i, method in enumerate(method_order):
    means = [np.mean(results[method][c]) if results[method][c] else 0 for c in culture_order]
    stds  = [np.std(results[method][c])  if results[method][c] else 0 for c in culture_order]

    ax.bar(
        x + offsets[i],
        means,
        width=width * 0.92,
        yerr=stds,
        label=method,
        color=METHOD_COLORS[method],
        edgecolor='white',
        linewidth=0.6,
        error_kw=dict(elinewidth=1.2, capsize=3, capthick=1.2, ecolor='#333333'),
        zorder=3,
    )

# Reference line
ax.axhline(1.0, color='#555555', linestyle='--', linewidth=1.3, zorder=2, label='Full connectivity (1.0)')

ax.set_xticks(x)
ax.set_xticklabels(culture_order, fontsize=11)
ax.set_xlabel('Culture', fontsize=12, labelpad=8)
ax.set_ylabel('Proportion of Nodes in Largest Connected Component', fontsize=12, labelpad=8)
ax.set_title('Proportion of Nodes in Largest Connected Component\nby Culture and Generation Method',
             fontsize=13, fontweight='bold', pad=12)
ax.set_ylim(0, 1.12)

# Gridlines on y only
ax.yaxis.grid(True, alpha=0.4, linewidth=0.7, zorder=0)
ax.xaxis.grid(False)
ax.set_axisbelow(True)
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

# Legend outside right
handles, lbls = ax.get_legend_handles_labels()
bar_h  = [h for h, l in zip(handles, lbls) if l != 'Full connectivity (1.0)']
bar_l  = [l for l in lbls if l != 'Full connectivity (1.0)']
ref_h  = [h for h, l in zip(handles, lbls) if l == 'Full connectivity (1.0)']
ref_l  = ['Full connectivity (1.0)']
ax.legend(
    bar_h + ref_h, bar_l + ref_l,
    title='Generation Method',
    title_fontsize=11,
    fontsize=10,
    loc='upper left',
    bbox_to_anchor=(1.01, 1.0),
    borderaxespad=0,
    frameon=True,
    framealpha=0.92,
    edgecolor='#cccccc',
)

plt.tight_layout(rect=[0, 0, 0.83, 1])

out = os.path.join(OUT_DIR, 'figure_5_2_prop_nodes_lcc.png')
plt.savefig(out, dpi=300, bbox_inches='tight')
print(f'\nSaved → {out}')
