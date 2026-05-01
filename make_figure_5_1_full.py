"""
Figure 5.1 — Dominant Homophily Dimension per Culture by Generation Method.

Aggregates homophily.csv files across all three models (gpt-4.1-nano,
gpt-4.1-mini, gpt-4.1) and both seeds (graph_nr 0 and 1) for each
method × culture condition, then plots the dominant dimension
(highest mean same_ratio) as a grouped bar chart.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
STATS   = './stats'
PLOTS   = './plots'
CULTURES = ['us', 'india', 'japan', 'brazil']
METHODS  = ['sequential', 'global', 'local', 'iterative']
MODELS   = ['gpt-4.1-nano', 'gpt-4.1-mini', 'gpt-4.1']

CULTURE_LABELS = {'us': 'US', 'india': 'India', 'japan': 'Japan', 'brazil': 'Brazil'}
METHOD_LABELS  = {'sequential': 'Sequential', 'global': 'Global',
                  'local': 'Local',           'iterative': 'Iterative'}

DEMO_SHORT = {
    'political affiliation': 'Pol. Affil.',
    'race/ethnicity':        'Race/Eth.',
    'age':                   'Age',
    'gender':                'Gender',
    'religion':              'Religion',
}

METHOD_COLORS = {
    'Sequential': '#4C72B0',
    'Global':     '#DD8452',
    'Local':      '#55A868',
    'Iterative':  '#C44E52',
}

# ── Load & aggregate ──────────────────────────────────────────────────────────
records = []
for method in METHODS:
    for culture in CULTURES:
        for model in MODELS:
            folder = (f'{method}_{model}_culture_{culture}' if method == 'global'
                      else f'{method}_{model}_n5_culture_{culture}')
            path = os.path.join(STATS, folder, 'homophily.csv')
            if not os.path.exists(path):
                print(f'  MISSING: {path}')
                continue
            df = pd.read_csv(path)
            df['method']  = METHOD_LABELS[method]
            df['culture'] = CULTURE_LABELS[culture]
            df['model']   = model
            records.append(df)

all_df = pd.concat(records, ignore_index=True)
same_df = all_df[all_df['metric_name'] == 'same_ratio'].copy()

# Mean across models and seeds per (method, culture, demo)
agg = (same_df
       .groupby(['method', 'culture', 'demo'], as_index=False)['_metric_value']
       .mean()
       .rename(columns={'_metric_value': 'mean_ratio'}))

# Dominant dimension per (method, culture)
dominant = (agg
            .loc[agg.groupby(['method', 'culture'])['mean_ratio'].idxmax()]
            .reset_index(drop=True))

print(dominant[['method', 'culture', 'demo', 'mean_ratio']].to_string(index=False))

# ── Plot ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':    'sans-serif',
    'font.size':      12,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize':11,
    'ytick.labelsize':11,
    'legend.fontsize':11,
    'figure.dpi':     150,
})

culture_order = [CULTURE_LABELS[c] for c in CULTURES]
method_order  = [METHOD_LABELS[m]  for m in METHODS]
n_cultures    = len(culture_order)
n_methods     = len(method_order)

x        = np.arange(n_cultures)
width     = 0.17
offsets   = np.linspace(-(n_methods - 1) / 2, (n_methods - 1) / 2, n_methods) * width

fig, ax = plt.subplots(figsize=(15, 7))

for i, method in enumerate(method_order):
    mdf    = dominant[dominant['method'] == method].set_index('culture')
    values = [mdf.loc[c, 'mean_ratio'] if c in mdf.index else 0.0 for c in culture_order]
    labels = [DEMO_SHORT.get(mdf.loc[c, 'demo'], mdf.loc[c, 'demo'])
               if c in mdf.index else '' for c in culture_order]

    bars = ax.bar(
        x + offsets[i],
        values,
        width=width * 0.92,
        label=method,
        color=METHOD_COLORS[method],
        edgecolor='white',
        linewidth=0.6,
        zorder=3,
    )

    for bar, val, lbl in zip(bars, values, labels):
        if val == 0 or not lbl:
            continue
        y_min = ax.get_ylim()[0]
        bar_height_in_data = val - y_min
        # place label 88 % up the visible bar height
        y_pos = y_min + bar_height_in_data * 0.88
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_pos,
            lbl,
            ha='center', va='top',
            fontsize=8, color='white', fontweight='bold',
            rotation=90,
        )

# Baseline
ax.axhline(1.0, color='#555555', linestyle='--', linewidth=1.3, zorder=2, label='Random baseline (1.0)')

ax.set_xticks(x)
ax.set_xticklabels(culture_order, fontsize=11)
ax.set_xlabel('Culture', fontsize=12, labelpad=8)
ax.set_ylabel('Observed / Expected Same-Group Ties\n(Dominant Homophily Dimension)', fontsize=12, labelpad=8)
ax.set_title('Dominant Homophily Dimension per Culture by Generation Method',
             fontsize=13, fontweight='bold', pad=12)

ymax = dominant['mean_ratio'].max()
ax.set_ylim(1.3, ymax + 0.15)   # zoom in — all values are well above 1.0
ax.yaxis.grid(True, alpha=0.4, linewidth=0.7, zorder=0)
ax.set_axisbelow(True)
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

# Legend outside — right of plot; baseline entry last
handles, lbls = ax.get_legend_handles_labels()
# separate baseline from method bars
bar_h   = [h for h, l in zip(handles, lbls) if l != 'Random baseline (1.0)']
bar_l   = [l for l in lbls if l != 'Random baseline (1.0)']
base_h  = [h for h, l in zip(handles, lbls) if l == 'Random baseline (1.0)']
base_l  = ['Random baseline (1.0)']

ax.legend(
    bar_h + base_h,
    bar_l + base_l,
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

plt.tight_layout(rect=[0, 0, 0.82, 1])

out = os.path.join(PLOTS, 'figure_5_1_dominant_homophily_by_culture_method.png')
plt.savefig(out, dpi=300, bbox_inches='tight')
print(f'\nSaved → {out}')
