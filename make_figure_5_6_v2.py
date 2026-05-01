"""
Figure 5.6 v2 — Structural comparison: LLM-generated vs classical graph models.

Improvements: sentence-case title, subtitle with methodology notes, colorblind
palette, value labels on bars, prop_nodes_lcc annotation, wider figure,
consistent condition order (Real first, GPT last), no in-figure annotation clutter.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import networkx as nx
import networkx.algorithms.community as nx_comm

# ── Config ───────────────────────────────────────────────────────────────────
OUT_PATH  = 'plots/figure_5_6_structural_comparison_v2.png'
# Metric display order (x-axis groups)
METRICS   = ['avg_clustering_coef', 'modularity', 'density', 'prop_nodes_lcc']
METRIC_LABELS = {
    'avg_clustering_coef': 'Avg. Clustering\nCoefficient',
    'modularity':          'Modularity',
    'density':             'Density',
    'prop_nodes_lcc':      'Prop. Nodes in\nLargest CC',
}
N_SYNTH   = 50
SEED      = 42

# Condition order: Real always first, GPT always last
COND_ORDER = ['Real', 'Erdős–Rényi', 'Barabási–Albert', 'Watts–Strogatz', 'Sequential GPT-4.1-mini']

# Colorblind-friendly palette (seaborn colorblind)
CB_COLORS = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#56B4E9']
COND_COLORS = dict(zip(COND_ORDER, CB_COLORS))

# ── Load real metrics ─────────────────────────────────────────────────────────
real_df = pd.read_csv('stats/real/network_metrics.csv')
real_df = real_df[real_df['node'].isna()]

mean_density = float(real_df[real_df['metric_name'] == 'density']['_metric_value'].mean())
N_REAL       = 36
mean_edges   = int(round(mean_density * N_REAL * (N_REAL - 1) / 2))
p_er         = mean_density
k_ws         = max(2, round(mean_density * (N_REAL - 1)))
m_ba         = max(1, round(mean_edges / N_REAL))
print(f'Calibration: N={N_REAL}, density={mean_density:.4f}, k_ws={k_ws}, m_ba={m_ba}')

# ── Load sequential GPT-4.1-mini metrics ─────────────────────────────────────
seq_df = pd.read_csv('stats/sequential_gpt-4.1-mini_n5/network_metrics.csv')
seq_df = seq_df[seq_df['node'].isna()]

# ── Generate & measure synthetic graphs ──────────────────────────────────────
def compute_metrics(G):
    G_un = G.to_undirected() if G.is_directed() else G
    N    = G_un.number_of_nodes()
    if N < 2:
        return {m: np.nan for m in METRICS}
    comps = sorted(nx.connected_components(G_un), key=len, reverse=True)
    try:
        communities = nx_comm.greedy_modularity_communities(G_un)
        mod = nx_comm.modularity(G_un, communities)
    except Exception:
        mod = np.nan
    return {
        'avg_clustering_coef': nx.average_clustering(G_un),
        'modularity':          mod,
        'density':             nx.density(G_un),
        'prop_nodes_lcc':      len(comps[0]) / N if comps else 0.0,
    }

rng = np.random.default_rng(SEED)

def sample_synthetic(gen_fn, n):
    return pd.DataFrame([compute_metrics(gen_fn()) for _ in range(n)])

er_df = sample_synthetic(
    lambda: nx.erdos_renyi_graph(N_REAL, p_er, seed=int(rng.integers(1e6))), N_SYNTH)
ba_df = sample_synthetic(
    lambda: nx.barabasi_albert_graph(N_REAL, m_ba, seed=int(rng.integers(1e6))), N_SYNTH)
ws_df = sample_synthetic(
    lambda: nx.watts_strogatz_graph(N_REAL, k_ws, 0.3, seed=int(rng.integers(1e6))), N_SYNTH)

# ── Assemble stats ────────────────────────────────────────────────────────────
def stats_from_csv(df):
    out = {}
    for m in METRICS:
        vals = df[df['metric_name'] == m]['_metric_value'].dropna().values
        out[m] = (float(np.mean(vals)), float(np.std(vals) / np.sqrt(max(len(vals), 1))))
    return out

def stats_from_df(df):
    out = {}
    for m in METRICS:
        vals = df[m].dropna().values
        out[m] = (float(np.mean(vals)), float(np.std(vals) / np.sqrt(max(len(vals), 1))))
    return out

conditions = {
    'Real':                   stats_from_csv(real_df),
    'Erdős–Rényi':            stats_from_df(er_df),
    'Barabási–Albert':        stats_from_df(ba_df),
    'Watts–Strogatz':         stats_from_df(ws_df),
    'Sequential GPT-4.1-mini': stats_from_csv(seq_df),
}

# ── Plot ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':    'sans-serif',
    'font.size':      11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize':11,
    'ytick.labelsize':10,
    'legend.fontsize':10,
})

n_conds   = len(COND_ORDER)
n_metrics = len(METRICS)
x         = np.arange(n_metrics)
width     = 0.14
offsets   = np.linspace(-(n_conds - 1) / 2, (n_conds - 1) / 2, n_conds) * width

fig, ax = plt.subplots(figsize=(16, 6))

for i, cond in enumerate(COND_ORDER):
    means = [conditions[cond][m][0] for m in METRICS]
    sems  = [conditions[cond][m][1] for m in METRICS]
    bars  = ax.bar(
        x + offsets[i],
        means,
        width=width * 0.88,
        yerr=sems,
        label=cond,
        color=COND_COLORS[cond],
        edgecolor='white',
        linewidth=0.6,
        error_kw=dict(elinewidth=1.1, capsize=2.5, capthick=1.1, ecolor='#333333'),
        zorder=3,
    )
    # Value labels above each bar
    for bar, val in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + sems[list(means).index(val)] + 0.012,
            f'{val:.2f}',
            ha='center', va='bottom',
            fontsize=7, color='#222222',
        )

# Annotation for prop_nodes_lcc group (last metric at index 3)
lcc_x = x[METRICS.index('prop_nodes_lcc')]
ax.text(
    lcc_x, 0.04,
    'All conditions near 1.0\nat this density',
    ha='center', va='bottom',
    fontsize=7.5, style='italic', color='#555555',
    bbox=dict(boxstyle='round,pad=0.3', facecolor='#f5f5f5', edgecolor='#cccccc', linewidth=0.7),
    zorder=4,
)

ax.set_xticks(x)
ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS], fontsize=11)
ax.set_xlabel('Network metric', fontsize=11, labelpad=8)
ax.set_ylabel('Mean value', fontsize=11, labelpad=8)

# Sentence-case title + subtitle
ax.set_title(
    'Structural comparison: LLM-generated vs classical graph models',
    fontsize=13, fontweight='bold', pad=18,
)
fig.text(
    0.5, 0.97,
    'ER, BA, and WS calibrated to match real network density.  '
    'Error bars = ±1 SE.  N=36 real networks, 50 samples each synthetic.',
    ha='center', va='top', fontsize=9, style='italic', color='#555555',
)

ax.set_ylim(0, 1.18)
ax.yaxis.grid(True, alpha=0.35, linewidth=0.7, zorder=0)
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

plt.tight_layout(rect=[0, 0, 0.83, 0.95])

os.makedirs('plots', exist_ok=True)
plt.savefig(OUT_PATH, dpi=300, bbox_inches='tight')
print(f'Saved → {OUT_PATH}')
