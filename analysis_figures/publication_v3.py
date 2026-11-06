"""Review revision. Reuse v2 where unchanged; never write into its release."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from analysis_figures import publication as v2

OUTPUT = Path('outputs/publication_figures/20261105_v3')
KEEP = {'02_primary_robustness', '04_coverage_missingness', '10_sampling_effect'}
BASE = ('ER', 'BA', 'WS')
PROCEDURES = (*v2.METHODS, *BASE)
ROWS = np.array([0, 1, 2, 3, 4.5, 5.5, 6.5])
M = 'avg_clustering_coef'

def load_tables(root):
    a,primary,tables,held,protected=v2.load_tables(root)
    held['conditional_control_metrics']=pd.read_csv(root/v2.HELDOUT/'conditional_control_metrics.csv')
    return a,primary,tables,held,protected


def primary_contrasts(a, primary, tables, held):
    fig = v2.primary_contrasts(a, primary, tables, held)
    loo = tables['leave_one_seed_out'].query("hypothesis_id == 'iterative:us:hindi'")
    fig.axes[-1].text(.03, 2.96, f'Hindi: {sum(loo.holm_p < .05)}/{len(loo)} omission refits pass Holm',
                      transform=fig.axes[-1].get_yaxis_transform(), fontsize=7, va='center',
                      bbox=dict(facecolor='white',edgecolor='none',pad=.4), color=v2.INK)
    return fig


def _limits(values, include_zero=False):
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    if include_zero:
        x = np.r_[x, 0.]
    lo, hi = x.min(), x.max()
    pad = max((hi-lo)*.06, .001)
    return lo-pad, hi+pad


def profiles(*args):
    fig = v2.model_profiles(*args)
    for axes in (fig.axes[:1], fig.axes[1:]):
        x = np.concatenate([c.get_offsets()[:, 0] for ax in axes for c in ax.collections])
        for ax in axes:
            ax.set_xlim(*_limits(x))
    return fig


def resources(*args):
    fig = v2.resource_use(*args)
    for ax in fig.axes:
        hi = ax.get_xlim()[1]
        ax.set_xlim(-hi*.04, hi)
        ax.set_xticks([t for t in ax.get_xticks() if 0 <= t <= hi])
    return fig


def gnm(*args):
    fig = v2.gnm_context(*args)
    x = np.concatenate([c.get_offsets()[:, 0] for ax in fig.axes for c in ax.collections])
    fig.axes[0].set_xlim(*_limits(x, True))
    return fig


def legacy(a, primary, tables, held):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.6), sharey=True)
    records = tables['empirical_reference_metrics']
    for ax, letter, metric, label in zip(axes, 'abc', v2.METRICS, v2.LABELS):
        v2.panel(ax, letter, label)
        for y, (method, color) in enumerate(zip(v2.METHODS, v2.COLORS)):
            q = primary.loc[primary.valid_graph & primary.method.eq(method), metric].quantile([.25,.5,.75])
            ax.plot([q.iloc[0], q.iloc[2]], [y,y], color=color, lw=2)
            ax.plot(q.iloc[1], y, 'o', color=color, ms=4)
        ax.scatter(records[metric], np.arange(5, 5+len(records)), marker='D', s=20, color=v2.INK)
        ax.axhline(4, lw=.6, color=v2.GRAY)
        ax.set(xlim=(-.025,1.025), ylim=(12.7,-.7), xlabel='Coefficient')
        ax.set_yticks([*range(4), *range(5,13)], [m.title() for m in v2.METHODS] +
                      [f'{r.reference} (n={r.nodes})' for r in records.itertuples()])
    fig.subplots_adjust(left=.20, right=.98, top=.90, bottom=.14, wspace=.25)
    return fig


def cohort_scores(frame):
    """Average distances (not graphs) within one record before comparing records."""
    keys = ['source', 'cohort', 'candidate', 'metric']
    if not np.isfinite(frame.w1).all() or not frame.w1.between(0,1).all():
        raise ValueError('Undefined or out-of-domain distance')
    if frame.duplicated(keys + ['bank']).any():
        raise ValueError('Duplicate candidate bank')
    groups = frame.groupby(keys, sort=True)
    n = groups.size()
    baseline = n.index.get_level_values('candidate').str.startswith('Baseline:')
    if not np.all(n == np.where(baseline, 8, 1)):
        raise ValueError('Incomplete candidate banks')
    fixed = groups.w1.mean().rename('w1').reset_index()
    fixed['procedure'] = fixed.candidate.str.split(':').str[1]
    groups = fixed.groupby(['source', 'cohort', 'procedure', 'metric'], sort=True)
    n = groups.size()
    if not np.all(n == np.where(n.index.get_level_values('procedure').isin(BASE), 1, 4)):
        raise ValueError('Incomplete fixed-model groups')
    return fixed, groups.w1.mean().rename('w1').reset_index()


def distances(a, primary, tables, held, appendix=False):
    _, scores = cohort_scores(held['per_cohort_distances'])
    indexes = [5] if appendix else list(range(5))
    fig, axes = plt.subplots(1, len(indexes), figsize=(5.7 if appendix else 12.8, 4.3), sharey=True, squeeze=False)
    for ax, j in zip(axes[0], indexes):
        v2.panel(ax, chr(97+j) if not appendix else '', v2.LABELS[j])
        frame = scores[scores.metric.eq(v2.METRICS[j])]
        for y, proc in zip(ROWS, PROCEDURES):
            fb, school = [frame.loc[frame.procedure.eq(proc) & frame.source.eq(s), 'w1'] for s in v2.SOURCES]
            if len(fb) != 10 or len(school) != 1:
                raise ValueError('Incomplete source records')
            ax.plot([fb.min(),fb.max()], [y-.15]*2, color='#6595AA', lw=1.3)
            ax.plot(fb.mean(), y-.15, 'o', color='#176785', ms=4)
            ax.plot(school.iloc[0], y+.15, 'D', color='#AE6018', ms=4)
        ax.set(xlim=(-.03*frame.w1.max(), 1.05*frame.w1.max()), ylim=(7.1,-.65), xlabel='W1 distance')
        ax.axhline(3.75, lw=.6, color='#D2D8DD')
        ax.set_yticks(ROWS, [m.title() if m not in BASE else m for m in PROCEDURES])
    fig.legend(handles=[Line2D([],[],marker='o',color='#176785',label='Facebook: mean and range across 10 records'),
                        Line2D([],[],marker='D',ls='',color='#AE6018',label='School: one record')],
               loc='lower center', ncol=1 if appendix else 2, frameon=False)
    fig.subplots_adjust(left=.16 if appendix else .085, right=.98, top=.88, bottom=.24, wspace=.24)
    return fig


def distributions(a, primary, tables, held):
    fig, axes = plt.subplots(2,2, figsize=(8.6,6.0), sharex='col', sharey=True)
    candidates, targets = held['candidate_metrics'], held['heldout_sample_metrics']
    for j, metric in enumerate(('density', M)):
        largest = max(candidates[metric].max(), targets[metric].max())
        upper = min(1., np.ceil(largest*10)/10)
        for i, ax in enumerate(axes[:,j]):
            v2.panel(ax, chr(97+i*2+j), ('Density' if j==0 else 'Clustering') + (' / LLMs' if i==0 else ' / baselines'), axis='y')
            def curve(values, weights, color, ls, name, bold=False):
                x,y = v2.weighted_ecdf(values,weights)
                ax.step(x,y,where='post',color=color,ls=ls,lw=1.65 if bold else 1.1,
                        zorder=4 if bold else 2,label=name)
            for s, ls, name in zip(v2.SOURCES, ('-', '--'), ('Facebook','School')):
                g = targets[targets.source.eq(s)]
                curve(g[metric], 1/g.cohort.map(g.cohort.value_counts()), v2.INK, ls, name, True)
            if i == 0:
                for proc,color in zip(v2.METHODS,v2.COLORS):
                    vals, ws = [], []
                    for model in v2.MODELS:
                        g,w = v2.candidate_weights(candidates[candidates.candidate.eq(f'LLM:{proc}:{model}')],metric)
                        vals.extend(g[metric]); ws.extend(w/4)
                    curve(vals,ws,color,'-',proc.title())
            else:
                for proc,color,ls in zip(BASE, ('#007D80','#BE7215','#8966A6'),(':','-.','-')):
                    g = candidates[candidates.candidate.eq('Baseline:'+proc)]
                    curve(g[metric],np.ones(len(g)),color,ls,proc)
            ax.set(xlim=(-.008*upper,upper), ylim=(-.01,1.025), yticks=[0,.25,.5,.75,1])
            if j==0: ax.set_ylabel('Cumulative fraction')
            if i==1: ax.set_xlabel('Metric value')
    handles, labels = axes[0,0].get_legend_handles_labels()
    h,l = axes[1,0].get_legend_handles_labels()
    fig.legend(handles+h[2:],labels+l[2:], loc='lower center',ncol=5,frameon=False)
    fig.subplots_adjust(left=.10,right=.98,top=.93,bottom=.18,hspace=.47,wspace=.18)
    return fig


def residual_draws(held):
    keys = ['source','cohort','draw']
    controls = held['conditional_control_metrics']
    targets = held['heldout_sample_metrics']
    join = controls.merge(targets, on=keys, validate='many_to_one', suffixes=('_control','_target'))
    columns=[m+s for m in (M,'transitivity') for s in ('_target','_control')]
    if (len(join)!=5632 or not np.isfinite(join[columns]).all().all()
            or join.duplicated(keys+['control','replicate']).any() or not join.success.all()
            or not (join.groupby(keys+['control']).size()==8).all()):
        raise ValueError('Incomplete conditional controls')
    for metric in (M,'transitivity'):
        join[metric] = join[metric+'_target']-join[metric+'_control']
    draws = join.groupby(keys+['control'])[[M,'transitivity']].mean().reset_index()
    if not draws.groupby(['source','cohort','control']).size().eq(32).all():
        raise ValueError('Incomplete target draws')
    for r in held['conditional_residuals'].itertuples():
        if r.metric in (M,'transitivity'):
            values = draws.loc[draws.cohort.eq(r.cohort) & draws.control.eq(r.control),r.metric]
            if not np.isclose(values.mean(),r.mean_target_minus_control,rtol=0,atol=1e-12):
                raise ValueError('Residual differs from sealed result')
    return draws


def controls(a, primary, tables, held):
    fig,axes = plt.subplots(1,2,figsize=(9,4.8),sharey=True)
    draws = residual_draws(held)
    cohorts = held['dataset_quality'].cohort
    for ax,letter,metric,label in zip(axes,'ab',(M,'transitivity'),('Clustering','Transitivity')):
        v2.panel(ax,letter,label)
        for ctrl,marker,c,offset in [('Gnm','s',v2.GRAY,-.16),('degree_swap','o',v2.COLORS[0],.16)]:
            g = draws[draws.control.eq(ctrl)].groupby('cohort')[metric].agg(['min','mean','max']).loc[cohorts]
            y = np.arange(len(cohorts))+offset
            ax.hlines(y,g['min'],g['max'],color=c,alpha=.68,lw=1)
            ax.scatter(g['mean'],y,s=18,marker=marker,color=c,label='Same edge count' if ctrl=='Gnm' else 'Same degrees')
        ax.axvline(0,color=v2.GRAY,ls='--',lw=.7)
        ax.set(ylim=(10.6,-.65),xlabel='Observed minus control',yticks=range(11),yticklabels=cohorts)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False)
    fig.subplots_adjust(left=.13,right=.98,top=.90,bottom=.18,wspace=.22)
    return fig


def records(a, primary, tables, held):
    fixed,_ = cohort_scores(held['per_cohort_distances'])
    fixed = fixed[fixed.metric.eq(M)]
    cohorts = held['dataset_quality'].cohort
    fig,axes=plt.subplots(1,5,figsize=(12.2,5),sharex=True,sharey=True)
    for i,(ax,proc) in enumerate(zip(axes,(*v2.METHODS,'Baselines'))):
        v2.panel(ax,chr(97+i),proc.title())
        names = [f'LLM:{proc}:{m}' for m in v2.MODELS] if i<4 else ['Baseline:'+p for p in BASE]
        for j,name in enumerate(names):
            x = fixed[fixed.candidate.eq(name)].set_index('cohort').loc[cohorts].w1
            ax.scatter(x,np.arange(11)+(j-(len(names)-1)/2)*.17,marker=v2.MARKERS[j],s=17,
                       color=v2.COLORS[i] if i<4 else ('#007D80','#BE7215','#8966A6')[j],
                       label=name.split(':')[-1])
        ax.axhline(9.5,color='#CFD6DA',lw=.7)
        ax.set(ylim=(10.6,-.65),yticks=range(11),yticklabels=cohorts,xlabel='Clustering W1')
    hb,lb=axes[-1].get_legend_handles_labels()
    fig.legend(handles=v2.model_handles()+hb,loc='lower center',ncol=7,frameon=False)
    fig.subplots_adjust(left=.087,right=.985,top=.90,bottom=.18,wspace=.24)
    return fig


DRAWERS = dict(v2.DRAWERS)
DRAWERS.update(zip(list(DRAWERS), (primary_contrasts,v2.robustness,profiles,v2.coverage_missingness,
                    resources,gnm,legacy,distances,distributions,v2.sampling,controls,records)))
DRAWERS['13_appendix_lcc'] = lambda *args: distances(*args, appendix=True)
EXPECTED = {'01': list(DRAWERS)[:2]+[list(DRAWERS)[6]], '02': list(DRAWERS)[:7],
            '03': list(DRAWERS)[7:], '04': list(DRAWERS)}


def show(root, names):
    from IPython.display import Image, Markdown, display
    folder = Path(root)/OUTPUT
    manifest = json.loads((folder/'figure_manifest.json').read_text(encoding='utf8'))
    if manifest['status']!='COMPLETE_REVIEW_V3':
        raise ValueError('Incomplete v3 release')
    v2.verify_hashes({Path(root)/p:h for p,h in manifest['inputs'].items()})
    for name in names:
        if name not in DRAWERS:
            raise ValueError('Unknown figure')
        for ext in ('png','svg','pdf'):
            path = folder/(name+'.'+ext)
            if v2.sha(path)!=manifest['artifacts'][path.name]:
                raise ValueError('Changed v3 figure: '+path.name)
        display(Image(filename=str(folder/(name+'.png')),width=1100))
        display(Markdown(manifest['captions'][name]))
