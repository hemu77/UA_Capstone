"""Post-review display corrections; frozen numerical analyses are not refitted."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from analysis_figures import deep_review as old, publication as style, publication_v3 as v3

OUT=Path('outputs/publication_figures/20261112_v5')
SOURCE=Path('analysis_figures/review_v5.py')
PREVIOUS='e8b88b8a454afd4f2d5d3b0cc9c1f5f877a00d5069f93c0ab421dca331f4913c'
CAPTIONS={
'08_heldout_distances': old.CAPTIONS['08_heldout_distances']+' The extra target-ER row in panel c uses each full target record density; it is NOT a held-out transfer candidate. Dashes elsewhere mean these metrics were not simulated for that diagnostic.',
'14_demographic_mixing': 'Both panels show all 28 model/instruction means per method and attribute. Large symbols are equal-combination means, not confidence intervals. Left: marginal Coleman. Right: excess same-attribute edge share above finite-roster chance. Axes have different units; neither is a causal importance scale. Outlined low-religion points are GPT-4.1 / US-Hindi, eight available graphs each: sequential Coleman 0.0256 and iterative 0.0310; no points were removed. Political-first counts (Coleman / excess): Global 6/28 / 1/28; each other method 28/28 / 28/28. Religion-politics association (V=0.316) describes this ONE designed roster, not independent evidence.',
'15_efficiency_density': 'Left: paired record-specific W1; symbols have small vertical offsets to reveal nearly coincident values, not uncertainty. School ER 0.1215 vs Global 0.1383 (difference 0.0168), not a significance or equivalence result. Right: weighted ECDFs illustrate shape, not the bank-averaged W1. Target-informed ER uses full-record density; fixed WS was development-fitted. All original weights are retained. No aggregate percentage is a bounded or causal measure.'}

def bands(root):
    records=old.table(root,'efficiency_records')
    held=v3.load_tables(root)[3]
    quality=held['dataset_quality'][['source','cohort','observed_density']]
    records=records.merge(quality,on=['source','cohort'],validate='one_to_one')
    records['density_band']=np.where(records.observed_density<.06,'below 0.06','at least 0.06')
    rows=[]
    for band,g in records.groupby('density_band'):
        delta=g.target_ER-g['global']
        rows.append(dict(density_band=band,records=len(g),facebook_records=int(g.source.eq('snap_facebook').sum()),
                         global_w1=g['global'].mean(),target_er_w1=g.target_ER.mean(),fixed_ws_w1=g.WS.mean(),
                         target_er_strictly_closer=int((delta<0).sum()),global_strictly_closer=int((delta>0).sum()),
                         within_0015=int((abs(delta)<=.015).sum())))
    return records,pd.DataFrame(rows)

def figure08(args,records):
    fig=v3.distances(*args)
    fig.set_size_inches(12.8,5.1)
    for ax in fig.axes:
        ax.set_ylim(8.7,-.65)
        ax.set_yticks([*v3.ROWS,8], [p.title() if p not in v3.BASE else p for p in v3.PROCEDURES]+['Target ER*'])
        ax.axhline(7.25,color='#BDC6CB',lw=.7,ls='--')
        if ax!=fig.axes[2]: ax.text(.5,8,'not estimated',transform=ax.get_yaxis_transform(),ha='center',va='center',size=7,color='#6D777F')
    ax=fig.axes[2]
    fb=records.loc[records.source.eq('snap_facebook'),'target_ER'];school=records.loc[~records.source.eq('snap_facebook'),'target_ER']
    ax.plot([fb.min(),fb.max()],[7.85]*2,color='#6595AA',lw=1.3)
    ax.plot(fb.mean(),7.85,'o',color='#176785',ms=4)
    ax.plot(school.iloc[0],8.15,'D',color='#AE6018',ms=4)
    ax.set_title('Efficiency\n+ target ER reference',loc='left',fontsize=10)
    fig.subplots_adjust(left=.115,bottom=.29,top=.86)
    fig.text(.53,.115,'* Diagnostic with target information, not a transfer competitor. Record ranges are not confidence intervals.',ha='center',size=9)
    return fig

def figure14(cells):
    fig,axes=plt.subplots(1,2,figsize=(11,5.4),sharey=True)
    ys=np.arange(4)*4
    for ax,score,title in zip(axes,['coleman','excess'],['A   Marginal Coleman','B   Excess same-group edge share']):
        for k,(attr,color,symbol) in enumerate(zip(old.ATTR,old.COLORS,['o','D','^'])):
            for j,method in enumerate(style.METHODS):
                g=cells[(cells.method==method)&(cells.attribute==attr)]
                values=np.sort(g[score])
                ax.scatter(values,ys[j]+k+np.linspace(-.15,.15,len(values)),s=12,c=color,alpha=.45,lw=0)
                ax.scatter(values.mean(),ys[j]+k,c=color,marker=symbol,s=45,edgecolor='white',lw=.6,zorder=3)
                marked=g[(g.model=='gpt-4.1')&(g.culture=='us')&(g.language=='hindi')]
                if attr=='religion' and method in ('sequential','iterative'):
                    ax.scatter(marked[score],[ys[j]+k]*len(marked),facecolors='none',edgecolors='#182D39',s=66,lw=1)
        ax.axvline(0,color='#889297',lw=.7)
        ax.set(yticks=ys+1,yticklabels=[m.title() for m in style.METHODS],ylim=(15,-1),xlabel='Coleman units' if score=='coleman' else 'Edge-share difference')
        ax.set_title(title,loc='left');ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
    fig.legend(handles=[Line2D([],[],color=c,marker=m,ls='',label=l.title()) for c,m,l in zip(old.COLORS,['o','D','^'],old.ATTR)],loc='lower center',bbox_to_anchor=(.5,.09),ncol=3,frameon=False)
    fig.text(.5,.035,'Outlined points: GPT-4.1 / US-Hindi religion, eight graphs per condition. Both panels retain all points.',ha='center',size=9)
    fig.subplots_adjust(left=.105,right=.98,bottom=.25,top=.9,wspace=.15)
    return fig

def figure15(held,records,sim):
    fig=old.fig15(held,records,sim)
    ax=fig.axes[0]
    for col,offset in zip(ax.collections,[-.18,0,.18]):
        xy=col.get_offsets();xy[:,1]+=offset;col.set_offsets(xy)
    for text in fig.texts: text.set_visible(False)
    ymax=max(held['heldout_sample_metrics'][old.EFF].max(),sim[old.EFF].max(),
             held['candidate_metrics'].loc[held['candidate_metrics'].candidate.str.startswith('LLM:global:'),old.EFF].max())
    upper=np.ceil((ymax+.01)*10)/10
    handles=[Line2D([],[],color='#242D33',label='Observed'),Line2D([],[],color='#176785',label='Global LLM'),
             Line2D([],[],color='#A95A18',ls='--',label='ER at target density')]
    for bx in fig.axes[1:]:
        bx.set_xlim(-.01,upper)
        bx.legend(handles=handles,frameon=False,fontsize=7,loc='lower right')
    return fig

def build(root,dest):
    if dest.exists():raise FileExistsError(dest)
    style.verify_hashes({root/old.OUT/'manifest.json':PREVIOUS})
    previous=old.verify(root)
    style.verify_hashes({root/old.OUT/p:h for p,h in previous['artifacts'].items()})
    args=v3.load_tables(root)[:4]
    records,summary=bands(root)
    cells=old.table(root,'attribute_cells');sim=old.table(root,'efficiency_simulations')
    extrema=cells.sort_values('coleman').groupby(['method','attribute']).head(1)
    dest.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=dest.parent) as tmp:
        folder=Path(tmp)/'release';folder.mkdir()
        for name,table in [('density_band_summary',summary),('efficiency_by_density',records),('attribute_lowest_cells',extrema)]:
            table.to_csv(folder/(name+'.csv'),index=False)
        with plt.rc_context(style.STYLE):
            figs=[figure08(args,records),figure14(cells),figure15(args[3],records,sim)]
            for name,fig in zip(CAPTIONS,figs):
                for ext,meta in [('png',None),('svg',{'Date':None}),('pdf',{'CreationDate':None,'ModDate':None})]:
                    fig.savefig(folder/(name+'.'+ext),dpi=220,metadata=meta)
                plt.close(fig)
        manifest=dict(previous_manifest_sha256=PREVIOUS,code_sha256=style.sha(root/SOURCE),
                      paid_calls=0,primary_tests_changed=False,density_cutoff=.06,cutoff_status='post-hoc descriptive review diagnostic',
                      artifacts={p.name:style.sha(p) for p in sorted(folder.iterdir())})
        (folder/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        folder.rename(dest)

def verify(root):
    old.verify(root)
    style.verify_hashes({root/old.OUT/'manifest.json':PREVIOUS})
    m=json.loads((root/OUT/'manifest.json').read_bytes())
    if m['previous_manifest_sha256']!=PREVIOUS:raise ValueError('Unexpected provenance')
    style.verify_hashes({root/SOURCE:m['code_sha256'],**{root/OUT/f:h for f,h in m['artifacts'].items()}})
    return m

def table(root,name):
    verify(root)
    if (root/OUT/(name+'.csv')).is_file():return pd.read_csv(root/OUT/(name+'.csv'))
    return old.table(root,name)

def show(root,names):
    from IPython.display import Image,Markdown,display
    verify(root)
    for name in names:
        if name not in CAPTIONS:old.show(root,[name]);continue
        display(Image(filename=str(root/OUT/(name+'.png')),width=1100));display(Markdown(CAPTIONS[name]))

if __name__=='__main__':
    import sys
    root=Path(__file__).resolve().parents[1]
    build(root,Path(sys.argv[1]).resolve() if len(sys.argv)>1 else root/OUT)
