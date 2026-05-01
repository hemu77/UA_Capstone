"""
Figure 5.6 — Structural comparison: LLM-generated vs classical graph models.

Real and Sequential GPT-4.1-mini metrics come from stats/ CSVs.
ER, BA, WS graphs are generated synthetically, calibrated to the real networks'
mean node count (N=36) and mean density, then measured with the same metrics.
Each synthetic model is sampled 50 times to get stable mean ± SE estimates.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
import networkx.algorithms.community as nx_comm

# ── Config ───────────────────────────────────────────────────────────────────
OUT_PATH  = 'plots/figure_5_6_structural_comparison.png'
METRICS   = ['density', 'avg_clustering_coef', 'prop_nodes_lcc', 'modularity']
METRIC_LABELS = {
    'density':           'Density',
    'avg_clustering_coef':'Avg. Clustering\nCoefficient',
    'prop_nodes_lcc':    'Prop. Nodes in\nLargest CC',
    'modularity':        'Modularity',
}
N_SYNTH   = 50    # synthetic graph samples per model
SEED      = 42

# ── Load real metrics ─────────────────────────────────────────────────────────
real_df = pd.read_csv('stats/real/network_metrics.csv')
real_df = real_df[real_df['node'].isna()]   # graph-level rows only

# Calibration params from real networks
mean_density = float(real_df[real_df['metric_name'] == 'density']['_metric_value'].mean())
N_REAL       = 36
mean_edges   = int(round(mean_density * N_REAL * (N_REAL - 1) / 2))
p_er         = mean_density          # ER edge probability = density
k_ws         = max(2, round(mean_density * (N_REAL - 1)))  # WS avg degree
m_ba         = max(1, round(mean_edges / N_REAL))          # BA edges per step

print(f'Real: N={N_REAL}, mean density={mean_density:.4f}, '
      f'mean edges≈{mean_edges}, k_ws={k_ws}, m_ba={m_ba}')

# ── Load sequential GPT-4.1-mini metrics ─────────────────────────────────────
seq_df = pd.read_csv('stats/sequential_gpt-4.1-mini_n5/network_metrics.csv')
seq_df = seq_df[seq_df['node'].isna()]

# ── Compute metrics for a graph ───────────────────────────────────────────────
def compute_metrics(G):
    G_un = G.to_undirected() if G.is_directed() else G
    N    = G_un.number_of_nodes()
    if N < 2:
        return {m: np.nan for m in METRICS}
    density          = nx.density(G_un)
    avg_cc           = nx.average_clustering(G_un)
    comps            = sorted(nx.connected_components(G_un), key=len, reverse=True)
    prop_lcc         = len(comps[0]) / N if comps else 0.0
    try:
        communities  = nx_comm.greedy_modularity_communities(G_un)
        modularity   = nx_comm.modularity(G_un, communities)
    except Exception:
        modularity   = np.nan
    return {
        'density':           density,
        'avg_clustering_coef': avg_cc,
        'prop_nodes_lcc':    prop_lcc,
        'modularity':        modularity,
    }

# ── Generate synthetic graphs ─────────────────────────────────────────────────
rng = np.random.default_rng(SEED)

def sample_synthetic(gen_fn, n):
    rows = []
    for _ in range(n):
        G = gen_fn()
        rows.append(compute_metrics(G))
    return pd.DataFrame(rows)

er_df  = sample_synthetic(lambda: nx.erdos_renyi_graph(N_REAL, p_er,
             seed=int(rng.integers(1e6))), N_SYNTH)
ba_df  = sample_synthetic(lambda: nx.barabasi_albert_graph(N_REAL, m_ba,
             seed=int(rng.integers(1e6))), N_SYNTH)
ws_df  = sample_synthetic(lambda: nx.watts_strogatz_graph(N_REAL, k_ws, 0.3,
             seed=int(rng.integers(1e6))), N_SYNTH)

# ── Assemble results ──────────────────────────────────────────────────────────
def stats_from_csv(df):
    """Return {metric: (mean, se)} from a network_metrics.csv dataframe."""
    out = {}
    for m in METRICS:
        vals = df[df['metric_name'] == m]['_metric_value'].dropna().values
        out[m] = (float(np.mean(vals)), float(np.std(vals) / np.sqrt(len(vals))))
    return out

def stats_from_df(df):
    out = {}
    for m in METRICS:
        vals = df[m].dropna().values
        out[m] = (float(np.mean(vals)), float(np.std(vals) / np.sqrt(len(vals))))
    return out

conditions = {
    'Real':              stats_from_csv(real_df),
    'Erdős–Rényi':       stats_from_df(er_df),
    'Barabási–Albert':   stats_from_df(ba_df),
    'Watts–Strogatz':    stats_from_df(ws_df),
    'Sequential\nGPT-4.1-mini': stats_from_csv(seq_df),
}

print('\nMeans:')
header = f"{'Condition':30s}" + ''.join(f'{m:22s}' for m in METRICS)
print(header)
for cond, d in conditions.items():
    row = f"{cond.replace(chr(10),' '):30s}" + \
          ''.join(f'{d[m][0]:10.4f} ±{d[m][1]:.4f}    ' for m in METRICS)
    print(row)

# ── Plot ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':    'sans-serif',
    'font.size':      11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize':10,
    'ytick.labelsize':10,
    'legend.fontsize':10,
})

cond_names = list(conditions.keys())
n_conds    = len(cond_names)
n_metrics  = len(METRICS)

COLORS = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#9467BD']
cond_colors = dict(zip(cond_names, COLORS))

x       = np.arange(n_metrics)
width   = 0.14
offsets = np.linspace(-(n_conds - 1) / 2, (n_conds - 1) / 2, n_conds) * width

fig, ax = plt.subplots(figsize=(14, 6))

for i, cond in enumerate(cond_names):
    means = [conditions[cond][m][0] for m in METRICS]
    sems  = [conditions[cond][m][1] for m in METRICS]
    bars  = ax.bar(
        x + offsets[i],
        means,
        width=width * 0.88,
        yerr=sems,
        label=cond.replace('\n', ' '),
        color=cond_colors[cond],
        edgecolor='white',
        linewidth=0.6,
        error_kw=dict(elinewidth=1.2, capsize=3, capthick=1.2, ecolor='#333333'),
        zorder=3,
    )

ax.set_xticks(x)
ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS], fontsize=10)
ax.set_xlabel('Network Metric', fontsize=11, labelpad=8)
ax.set_ylabel('Mean Value', fontsize=11, labelpad=8)
ax.set_title('Structural Comparison: LLM-Generated vs Classical Graph Models',
             fontsize=12, fontweight='bold', pad=12)
ax.set_ylim(0, 1.05)

ax.yaxis.grid(True, alpha=0.4, linewidth=0.7, zorder=0)
ax.xaxis.grid(False)
ax.set_axisbelow(True)
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

ax.legend(
    title='Condition',
    title_fontsize=10,
    fontsize=9.5,
    loc='upper left',
    bbox_to_anchor=(1.01, 1.0),
    borderaxespad=0,
    frameon=True,
    framealpha=0.92,
    edgecolor='#cccccc',
)

ax.text(0.99, 0.97,
        f'ER/BA/WS calibrated to real networks (N={N_REAL}, mean density={mean_density:.3f})\n'
        f'Synthetic: {N_SYNTH} samples each. Error bars = ±1 SE.',
        transform=ax.transAxes, ha='right', va='top',
        fontsize=8, style='italic', color='#555555')

plt.tight_layout(rect=[0, 0, 0.82, 1])

os.makedirs('plots', exist_ok=True)
plt.savefig(OUT_PATH, dpi=300, bbox_inches='tight')
print(f'\nSaved → {OUT_PATH}')
