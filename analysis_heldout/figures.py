"""Compact manuscript-style figures; caveats live in the accompanying captions."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METRIC_LABELS={'density':'Density','avg_clustering_coef':'Clustering','global_efficiency':'Efficiency',
               'transitivity':'Transitivity','degree_gini':'Degree Gini','prop_nodes_lcc':'LCC fraction'}
SOURCES=('snap_facebook','sociopatterns_friendship')
SOURCE_LABELS=('Facebook alter networks','School friendship nominations')


def candidates(frame):
    methods=('global','local','sequential','iterative')
    models=('gpt-4.1','gpt-5.6-luna','gpt-6-luna','gpt-6-sol')
    order=[f'LLM:{method}:{model}' for method in methods for model in models]+['Baseline:'+x for x in ('ER','BA','WS')]
    if set(frame.candidate)!=set(order): raise ValueError('Candidate set differs from fixed benchmark')
    return order


def labels(order):
    return [x.replace('LLM:','').replace('Baseline:','Dev. baseline / ').replace(':',' / ') for x in order]


def render(output):
    summary=pd.read_csv(output/'source_distance_summary.csv')
    per_cohort=pd.read_csv(output/'per_cohort_distances.csv')
    quality=pd.read_csv(output/'dataset_quality.csv')
    target=pd.read_csv(output/'heldout_sample_metrics.csv')
    residual=pd.read_csv(output/'conditional_residuals.csv')
    order=candidates(summary)
    metrics=list(METRIC_LABELS)
    folder=output/'figures'
    folder.mkdir(parents=True,exist_ok=True)
    paths=[]
    style={'font.family':'STIXGeneral','font.size':9,'axes.labelsize':9,'axes.titlesize':10,
        'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'axes.spines.top':False,
        'axes.spines.right':False,'svg.hashsalt':'heldout-manuscript-20261105'}

    def save(fig,name):
        try:
            for suffix in ('png','svg','pdf'):
                path=folder/(name+'.'+suffix)
                metadata={'Date':None} if suffix=='svg' else {'CreationDate':None,'ModDate':None} if suffix=='pdf' else None
                fig.savefig(path,dpi=250,bbox_inches='tight',pad_inches=.08,metadata=metadata)
                paths.append(path)
        finally: plt.close(fig)

    with plt.rc_context(style):
        fig,axes=plt.subplots(1,2,figsize=(9,5.6),sharey=True)
        for ax,source,title in zip(axes,SOURCES,SOURCE_LABELS):
            values=summary[summary.source.eq(source)].pivot(index='candidate',columns='metric',values='mean_w1').loc[order,metrics].to_numpy()
            image=ax.imshow(values,vmin=0,vmax=1,cmap='YlGnBu',aspect='auto')
            ax.set_title(title,loc='left')
            ax.set_xticks(range(6),list(METRIC_LABELS.values()),rotation=45,ha='right')
            ax.set_yticks(range(len(order)),labels(order))
            for boundary in (3.5,7.5,11.5,15.5): ax.axhline(boundary,color='white',lw=1.5)
            for y in range(len(order)):
                for x in range(6):
                    ax.text(x,y,f'{values[y,x]:.2f}',ha='center',va='center',fontsize=7,color='white' if values[y,x]>.55 else '#15212b')
        fig.subplots_adjust(left=.25,right=.91,bottom=.18,top=.94,wspace=.12)
        color=fig.add_axes([.935,.21,.014,.67])
        fig.colorbar(image,cax=color,label='Wasserstein-1 distance')
        save(fig,'01_heldout_transfer')

        cohorts=quality.cohort.tolist()
        y=np.arange(len(cohorts))
        fig,axes=plt.subplots(1,2,figsize=(7.1,4.3),sharey=True)
        for ax,metric,column in zip(axes,('density','avg_clustering_coef'),('observed_density','observed_clustering')):
            for position,cohort in enumerate(cohorts):
                values=target.loc[target.cohort.eq(cohort),metric]
                ax.plot([values.min(),values.max()],[position,position],color='#adbac1',lw=1)
                ax.scatter(values.mean(),position,color='#007f83',marker='D',s=20,zorder=3)
                ax.scatter(float(quality.loc[quality.cohort.eq(cohort),column].iloc[0]),position,
                    facecolors='none',edgecolors='#29343b',s=24,zorder=3)
            ax.set(xlim=(0,1),xlabel=METRIC_LABELS[metric],ylim=(len(cohorts)-.5,-.5))
            ax.set_yticks(y,[f'{r.cohort} (n={r.observed_roster_nodes})' for r in quality.itertuples()])
            ax.grid(axis='x',lw=.4,alpha=.25)
        from matplotlib.lines import Line2D
        axes[0].legend(handles=[Line2D([],[],marker='o',mfc='none',color='#29343b',ls='',label='Observed full roster'),
            Line2D([],[],marker='D',color='#007f83',ls='',label='Mean of 50-node draws')],loc='upper center',bbox_to_anchor=(1.03,1.16),ncol=2,frameon=False)
        fig.subplots_adjust(left=.23,right=.98,bottom=.12,top=.86,wspace=.12)
        save(fig,'02_sampling_shift')

        fig,axes=plt.subplots(1,2,figsize=(7.1,4.3),sharey=True)
        for ax,metric in zip(axes,('avg_clustering_coef','transitivity')):
            for control,offset,color,marker in (('Gnm',-.12,'#707d86','D'),('degree_swap',.12,'#007f83','o')):
                selected=residual[(residual.control==control)&(residual.metric==metric)].set_index('cohort').loc[cohorts]
                ax.scatter(selected.mean_target_minus_control,y+offset,color=color,marker=marker,s=22,label=control)
                for position,row in enumerate(selected.itertuples()):
                    if pd.isna(row.mean_target_minus_control):
                        ax.text(.98,position+offset,f'{control}: incomplete',transform=ax.get_yaxis_transform(),ha='right',fontsize=6)
            ax.axvline(0,color='#69777e',lw=.7,ls='--')
            ax.set(xlabel='Observed minus control '+METRIC_LABELS[metric].lower(),ylim=(len(cohorts)-.5,-.5))
            ax.set_yticks(y,cohorts)
            ax.grid(axis='x',lw=.4,alpha=.25)
        axes[0].legend(['G(n,m), target edge count','Degree-preserving swaps'],loc='upper center',bbox_to_anchor=(1.03,1.16),ncol=2,frameon=False)
        fig.subplots_adjust(left=.15,right=.98,bottom=.13,top=.86,wspace=.12)
        save(fig,'03_conditional_structure')

        selected=per_cohort[per_cohort.metric.eq('avg_clustering_coef')].groupby(['candidate','cohort']).w1.mean().unstack().loc[order,cohorts]
        fig,ax=plt.subplots(figsize=(8,5.6))
        image=ax.imshow(selected.to_numpy(),vmin=0,vmax=1,cmap='YlGnBu',aspect='auto')
        ax.set_xticks(range(len(cohorts)),cohorts,rotation=45,ha='right')
        ax.set_yticks(range(len(order)),labels(order))
        ax.axvline(9.5,color='white',lw=2)
        ax.axhline(15.5,color='white',lw=2)
        fig.colorbar(image,ax=ax,label='Clustering Wasserstein-1 distance',fraction=.035,pad=.025)
        fig.subplots_adjust(left=.29,right=.94,bottom=.15,top=.98)
        save(fig,'04_cohort_sensitivity')
    return paths
