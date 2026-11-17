"""Proposed replacement figures (review v3). Reads sealed held-out tables only; writes only into this directory.
No provider calls, no network, no change to frozen data or v1/v2 figures.  Run: python build_review_figures.py <repo_root>"""
import csv, sys, collections, statistics, os
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
root = sys.argv[1] if len(sys.argv) > 1 else '.'
D = f'{root}/outputs/heldout_evaluation/20261105_v1/'; OUT = os.path.dirname(os.path.abspath(__file__))
pc = list(csv.DictReader(open(D+'per_cohort_distances.csv')))
cm = list(csv.DictReader(open(D+'candidate_metrics.csv')))
hs = list(csv.DictReader(open(D+'heldout_sample_metrics.csv')))
PROC = ['Global','Local','Sequential','Iterative']; BASE = ['ER','BA','WS']
COL = {'Global':'#2171a5','Local':'#c07818','Sequential':'#23866b','Iterative':'#a05070','ER':'#222','BA':'#555','WS':'#999'}
MK = {'ER':'o','BA':'s','WS':'^'}
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#6b7785','figure.dpi':150})
FB = sorted({r['cohort'] for r in pc if r['source']=='snap_facebook'}); SC = ['School2013'] if any(r['cohort']=='School2013' for r in pc) else sorted({r['cohort'] for r in pc if r['source']!='snap_facebook'})
# per-cohort distance: LLM procedure = mean of 4 model W1s; baseline = mean of 8 bank W1s
acc = collections.defaultdict(list)
for r in pc:
    c = r['candidate']
    if r['kind']=='LLM': key = c.split(':')[1].capitalize()
    else: key = c.split(':')[1]
    acc[(key, r['cohort'], r['metric'])].append(float(r['w1']))
dist = {k: statistics.mean(v) for k, v in acc.items()}
SRC = [('Facebook (10 records)', FB), ('School 2013 (1 record)', SC)]
def cand_src(cand, coh, metric): 
    v = [dist[(cand, c, metric)] for c in coh]; return statistics.mean(v), min(v), max(v)

# ---- R08: one row of metrics; both sources stacked inside each panel, shared zero-based axis ----
mets = [('density','Density'),('avg_clustering_coef','Clustering'),('global_efficiency','Efficiency'),('transitivity','Transitivity'),('degree_gini','Degree inequality')]
fig, axes = plt.subplots(1, 5, figsize=(17, 5.6), sharey=True)
rows = []; y = 0
for sname, coh in SRC:
    for cand in PROC+BASE: rows.append((sname, cand, y)); y += 1
    y += 0.8
for ax, (m, label) in zip(axes, mets):
    xmax = 0
    for sname, cand, y in rows:
        coh = dict(SRC)[sname]; mean, lo, hi = cand_src(cand, coh, m)
        base = cand in BASE
        if len(coh) > 1: ax.plot([lo, hi], [y, y], color=COL[cand], lw=1.4, alpha=.45, zorder=1)
        ax.scatter([mean], [y], s=70, color='white' if base else COL[cand], edgecolor=COL[cand], marker=MK.get(cand, 'o'), lw=1.8, zorder=3)
        xmax = max(xmax, hi)
    ax.set_xlim(0, xmax*1.08); ax.set_title(label, loc='left', fontsize=12); ax.set_xlabel('Wasserstein-1 distance'); ax.grid(axis='x', color='#e6e9ee')
axes[0].set_yticks([r[2] for r in rows]); axes[0].set_yticklabels([r[1] if r[1] in PROC else 'Dev. '+r[1] for r in rows]); axes[0].invert_yaxis()
for sname, coh in SRC:
    ys = [r[2] for r in rows if r[0]==sname]; axes[0].text(-0.62, np.mean(ys), sname.split(' (')[0], transform=axes[0].get_yaxis_transform(), rotation=90, va='center', ha='center', fontweight='bold')
fig.text(.5, .005, 'Dots = source mean over records (LLM rows: mean of 4 fixed models; baselines: mean of 8 banks). Faint bars (Facebook only) = min-max across the 10 cohort records: heterogeneity, NOT a CI. '
         'School is one record, so no spread. Largest-component fraction omitted here (see R_appendix_lcc). No composite score.', ha='center', fontsize=9, color='#555', wrap=True)
fig.tight_layout(rect=(0.03, .04, 1, 1)); [fig.savefig(f'{OUT}/R08_heldout_distances.{e}', bbox_inches='tight') for e in ('png','svg','pdf')]; plt.close(fig)

# ---- R09: ECDFs: clustering + zoomed density, both sources, observed + procedures + baselines ----
def ecdf(ax, vals, w, **kw):
    o = np.argsort(vals); v = np.asarray(vals)[o]; c = np.cumsum(np.asarray(w, float)[o]); c /= c[-1]; ax.step(np.r_[0, v], np.r_[0, c], where='post', **kw)
def llm_mix(proc, col):
    by = collections.defaultdict(list)
    for r in cm:
        if r['candidate'].startswith(f'LLM:{proc.lower()}:') and r['valid']=='True': by[(r['candidate'], r['condition'])].append(float(r[col]))
    xs, ws = [], []
    for (cand, cond), v in by.items():
        for x in v: xs.append(x); ws.append(1/7/len(v))
    return xs, ws
def base_vals(name, col): v = [float(r[col]) for r in cm if r['candidate']=='Baseline:'+name]; return v, [1]*len(v)
fig, axes = plt.subplots(2, 2, figsize=(11, 7.6))
for i, (col, label, xlim) in enumerate([('avg_clustering_coef','Clustering',(0,1)), ('density','Density',(0,.35))]):
    for j, (sname, coh) in enumerate(SRC):
        ax = axes[i, j]
        obs = [float(r[col]) for r in hs if r['cohort'] in coh]
        for p in PROC: ecdf(ax, *llm_mix(p, col), color=COL[p], lw=1.8, label=p)
        for b, ls in zip(BASE, [':', '--', '-.']): ecdf(ax, *base_vals(b, col), color='#8a8f98', lw=1.5, ls=ls, label='Dev. '+b)
        ecdf(ax, obs, [1]*len(obs), color='black', lw=3, label='Observed 50-node draws', zorder=5)
        ax.set_xlim(*xlim); ax.set_ylim(0, 1.01); ax.grid(color='#eceef2'); ax.set_title(f'{label} · {sname}', loc='left', fontsize=11)
        if j == 0: ax.set_ylabel('Cumulative fraction')
        if i == 1: ax.set_xlabel('Metric value (x-range zoomed to data)')
h, l = axes[0, 0].get_legend_handles_labels(); fig.legend(h, l, ncol=4, loc='lower center', frameon=False, bbox_to_anchor=(.5, -.01))
fig.text(.5, -.075, 'Observed = pooled draws of the listed records (descriptive mixture; 32 overlapping draws per record). Generated = pooled valid graphs, equal weight across 7 instruction conditions. No CIs.', ha='center', fontsize=9, color='#555')
fig.tight_layout(rect=(0, .04, 1, 1)); [fig.savefig(f'{OUT}/R09_heldout_distributions.{e}', bbox_inches='tight') for e in ('png','svg','pdf')]; plt.close(fig)

# ---- R12: per-record clustering distance, all 7 candidates on one shared axis ----
recs = FB + SC; fig, ax = plt.subplots(figsize=(8.5, 6.2))
for i, c in enumerate(recs):
    for k, cand in enumerate(PROC+BASE):
        v = dist[(cand, c, 'avg_clustering_coef')]; b = cand in BASE
        ax.scatter(v, i + (k-3)*0.09, s=42, marker=MK.get(cand, 'o'), color='white' if b else COL[cand], edgecolor=COL[cand], lw=1.4, label=('Dev. '+cand if b else cand) if i == 0 else None)
ax.axhline(len(FB)-.5, color='#cbd3da'); ax.set_yticks(range(len(recs))); ax.set_yticklabels(recs); ax.invert_yaxis(); ax.set_xlim(0, .5); ax.grid(axis='x', color='#eceef2')
ax.set_xlabel('Clustering Wasserstein-1 distance (lower = closer)'); ax.legend(ncol=4, loc='lower center', bbox_to_anchor=(.5, -.2), frameon=False, fontsize=9)
ax.set_title('Per-record clustering distance. Records FB01-FB10 share people and are not independent.', loc='left', fontsize=10.5)
fig.tight_layout(); [fig.savefig(f'{OUT}/R12_cohort_sensitivity.{e}', bbox_inches='tight') for e in ('png','svg','pdf')]; plt.close(fig)

# ---- Appendix: largest-component fraction (why the metric cannot discriminate here) ----
fig, ax = plt.subplots(figsize=(6.4, 4.2))
obs = [float(r['prop_nodes_lcc']) for r in hs]
for p in PROC: ecdf(ax, *llm_mix(p, 'prop_nodes_lcc'), color=COL[p], lw=1.8, label=p)
for b, ls in zip(BASE, [':', '--', '-.']): ecdf(ax, *base_vals(b, 'prop_nodes_lcc'), color='#8a8f98', lw=1.5, ls=ls, label='Dev. '+b)
ecdf(ax, obs, [1]*len(obs), color='black', lw=3, label='Observed 50-node draws (all 352)')
ax.set_xlim(0, 1.02); ax.set_xlabel('Largest-component fraction'); ax.set_ylabel('Cumulative fraction'); ax.grid(color='#eceef2'); ax.legend(frameon=False, fontsize=8.5, loc='upper left')
ax.set_title('Baselines and Local/Sequential are ~always connected (mass at 1.0)', loc='left', fontsize=10)
fig.tight_layout(); [fig.savefig(f'{OUT}/R_appendix_lcc.{e}', bbox_inches='tight') for e in ('png','svg','pdf')]; plt.close(fig)
print('done')
