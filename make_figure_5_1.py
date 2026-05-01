"""
Generate Figure 5.1: dominant homophily dimension per culture × generation method.

Reads homophily.csv files for global and sequential gpt-4.1-mini across the four
core cultures (Brazil, India, Japan, US), finds the dominant dimension (highest
mean same_ratio), and draws a grouped bar chart of that dominant ratio value.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

STATS = './stats'
PLOTS = './plots'
CULTURES = ['brazil', 'india', 'japan', 'us']
METHODS = ['global', 'sequential']
MODEL = 'gpt-4.1-mini'

sns.set_theme(context='paper', style='whitegrid', font='sans-serif', font_scale=1.3)

CULTURE_LABELS = {'brazil': 'Brazil', 'india': 'India', 'japan': 'Japan', 'us': 'US'}
METHOD_LABELS  = {'global': 'Global', 'sequential': 'Sequential'}
DEMO_SHORT = {
    'political affiliation': 'Pol. Affil.',
    'race/ethnicity':        'Race/Eth.',
    'age':                   'Age',
    'gender':                'Gender',
    'religion':              'Religion',
}

rows = []
for method in METHODS:
    for culture in CULTURES:
        folder = (f'{method}_{MODEL}_culture_{culture}' if method == 'global'
                  else f'{method}_{MODEL}_n5_culture_{culture}')
        path = os.path.join(STATS, folder, 'homophily.csv')
        if not os.path.exists(path):
            print(f'Missing: {path}')
            continue
        df = pd.read_csv(path)
        same = df[df['metric_name'] == 'same_ratio'].copy()
        mean_by_demo = same.groupby('demo')['_metric_value'].mean()
        dominant_demo = mean_by_demo.idxmax()
        dominant_val  = mean_by_demo.max()
        rows.append({
            'culture':       CULTURE_LABELS[culture],
            'method':        METHOD_LABELS[method],
            'dominant_demo': dominant_demo,
            'same_ratio':    dominant_val,
        })

fig_df = pd.DataFrame(rows)
print(fig_df.to_string(index=False))

PALETTE = {'Global': '#5B9BD5', 'Sequential': '#ED7D31'}

fig, ax = plt.subplots(figsize=(11, 6))
bar_plot = sns.barplot(
    data=fig_df,
    x='culture',
    y='same_ratio',
    hue='method',
    palette=PALETTE,
    edgecolor='white',
    linewidth=0.8,
    width=0.55,
    ax=ax,
)

# annotate bars — label just above each bar
for container, method in zip(bar_plot.containers, METHODS):
    mkey = METHOD_LABELS[method]
    method_df = fig_df[fig_df['method'] == mkey].set_index('culture')
    for bar, culture in zip(container, [CULTURE_LABELS[c] for c in CULTURES]):
        if culture not in method_df.index:
            continue
        demo  = method_df.loc[culture, 'dominant_demo']
        label = DEMO_SHORT.get(demo, demo)
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.04,
            label,
            ha='center', va='bottom',
            fontsize=9, color='#222222', fontweight='semibold',
        )

ax.axhline(1.0, color='#555555', linestyle='--', linewidth=1.2, label='Random baseline (1.0)')

ax.set_ylim(0, 2.45)
ax.set_xlabel('Culture', fontsize=13, labelpad=8)
ax.set_ylabel('Observed / Expected Same-Group Ties\n(Dominant Homophily Dimension)', fontsize=12, labelpad=8)
ax.set_title('Dominant Homophily Dimension per Culture by Generation Method\n(GPT-4.1-mini)',
             fontsize=14, fontweight='bold', pad=14)

# legend outside plot area — top-right corner, no overlap
handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles, labels,
    title='Generation Method',
    title_fontsize=11,
    fontsize=10,
    loc='upper left',
    bbox_to_anchor=(1.01, 1),
    borderaxespad=0,
    frameon=True,
    framealpha=0.9,
    edgecolor='#cccccc',
)

ax.tick_params(axis='both', labelsize=11)
ax.grid(axis='y', alpha=0.35, linewidth=0.7)
ax.set_axisbelow(True)
sns.despine(left=False, bottom=False)

plt.tight_layout(rect=[0, 0, 0.85, 1])   # leave room for legend on the right

out = os.path.join(PLOTS, 'figure_5_1_dominant_homophily.png')
plt.savefig(out, dpi=180, bbox_inches='tight')
print(f'Saved → {out}')
plt.show()
