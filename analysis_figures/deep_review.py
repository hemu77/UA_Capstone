"""Post-hoc target-density and roster diagnostics. No collection or baseline refitting."""
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import shortest_path
from scipy.stats import wasserstein_distance
from matplotlib.lines import Line2D

from analysis_figures import publication as v2, publication_v3 as v3
from analysis_primary.graph_evaluation import _read_graph

OUT = Path('outputs/publication_figures/20261111_v4')
PRIMARY = Path('outputs/analysis_ready/060dcd3a86d5/primary_valid_graphs.csv')
ROSTER = Path('text-files/revision224_adults.json')
SOURCE = Path('analysis_figures/deep_review.py')
PINS = {v3.OUTPUT/'figure_manifest.json': '9c5a7575d6fb718ac36d3bc1f995092dc1920a718b2c3153590416df09f2d94a',
        PRIMARY: 'f62db1d6ae788193af39005fd92a969d5418bd2d7fa1acef437577aa5edf5591'}
ATTR = ['political orientation','religion','gender']
COLORS = ['#205E78','#98601E','#535769']
KEY = ['method','model','culture','language']
EFF = 'global_efficiency'
SEED = 20261111
CAPTIONS = {
    '08_heldout_distances': 'Lower W1 is closer on one metric. Blue spans are record ranges, not confidence intervals. Substantial overlap and small rank margins preclude an overall winner. All three baseline settings were fixed on development data. See figure 15 for the separate target-informed efficiency diagnostic.',
    '11_conditional_controls': 'Means and full ranges of 32 paired target-minus-control values per record, not confidence intervals. The two controls preserve different properties of the observed graphs.',
    '14_demographic_mixing': 'Small points: all 28 model/instruction means per method and attribute. Large symbols: equally weighted mean of those 28. The 878 nonempty primary graphs have all three scores. Categorical scores adjust roster shares but are not calibrated attribute-importance measures. No causal ranking or significance test.',
    '15_efficiency_density': 'At 50 nodes the post-hoc ER diagnostic uses the FULL observed density of each record, unavailable to transfer competitors. Left: W1 per record (mean of bank or model distances). Right: illustrative distributions, not the left-panel W1. Facebook records receive equal weight within source. Unreachable pairs contribute zero to efficiency.'}


def efficiency(adj):
    distances = shortest_path(adj, directed=False, unweighted=True, method='FW')
    off = ~np.eye(len(adj), dtype=bool)
    return np.mean(1 / distances[off])


def diagnostic(held):
    quality, target = held['dataset_quality'], held['heldout_sample_metrics']
    _, scores = v3.cohort_scores(held['per_cohort_distances'])
    scores = scores[scores.metric.eq(EFF)].pivot(index=['source','cohort'], columns='procedure',values='w1')
    if quality.duplicated(['source','cohort']).any() or len(quality)!=11 or len(target)!=352:
        raise ValueError('Incomplete held-out records')
    triu = np.triu_indices(50,1)
    banks, samples = [], []
    for r in quality.itertuples():
        p = r.observed_density
        if not 0 < p < 1 or not np.isclose(p,2*r.observed_edges/(r.observed_roster_nodes*(r.observed_roster_nodes-1)),rtol=0,atol=1e-12):
            raise ValueError('Invalid full-record density')
        t = target[(target.source==r.source)&(target.cohort==r.cohort)]
        if set(t.draw)!=set(range(32)) or len(t)!=32 or not np.isfinite(t[EFF]).all():
            raise ValueError('Missing or duplicate observed draw')
        for b in range(8):
            rng = np.random.default_rng(v2_seed(r.source,r.cohort,b))
            vals=[]
            for _ in range(56):
                a=np.zeros((50,50),bool)
                a[triu]=rng.random(len(triu[0])) < p
                vals.append(efficiency(a|a.T))
            samples.extend((r.source,r.cohort,b,i,e) for i,e in enumerate(vals))
            banks.append((r.source,r.cohort,p,b,wasserstein_distance(vals,t[EFF])))
    banks=pd.DataFrame(banks,columns=['source','cohort','p_full','bank','target_ER'])
    per=scores[['global','ER','WS']].join(banks.groupby(['source','cohort']).target_ER.mean(),validate='one_to_one').reset_index()
    if len(per)!=11 or not np.isfinite(per.iloc[:,2:]).all().all():
        raise ValueError('Incomplete efficiency comparison')
    return banks,per,pd.DataFrame(samples,columns=['source','cohort','bank','draw',EFF])


def summarize(per):
    rows=[]
    for label,g in [*per.groupby('source'),('all_equal_record',per)]:
        means=g[['global','ER','WS','target_ER']].mean()
        gap=means.WS-means['global']
        delta=g.target_ER-g['global']
        rows.append(dict(source=label,records=len(g),**means.to_dict(),
                         ws_gap_closed=(means.WS-means.target_ER)/gap if gap else np.nan,
                         target_strictly_closer=int((delta<0).sum()),
                         target_closer_or_within_015=int((delta<=.015).sum())))
    return pd.DataFrame(rows)


def v2_seed(source,cohort,bank):
    return int.from_bytes(hashlib.sha256(f'{SEED}:{source}:{cohort}:{bank}'.encode()).digest()[:8],'big')


def mixing(root):
    v2.verify_hashes({root/PRIMARY:PINS[PRIMARY]})
    primary=pd.read_csv(root/PRIMARY)
    roster=json.loads((root/ROSTER).read_text())
    roster_hash=hashlib.sha256(json.dumps(roster,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
    if (len(primary)!=890 or primary.run_id.duplicated().any() or not primary.cohort.eq('primary').all()
            or not primary.roster_sha256.eq(roster_hash).all() or set(roster)!=set(map(str,range(50)))):
        raise ValueError('Primary roster/catalog mismatch')
    labels=pd.DataFrame(roster).T
    cross=pd.crosstab(labels.religion,labels['political orientation'])
    counts=cross.to_numpy(); expected=np.outer(counts.sum(1),counts.sum(0))/counts.sum()
    v=np.sqrt((((counts-expected)**2/expected).sum())/(counts.sum()*min(np.array(counts.shape)-1)))
    chance={a:sum(k*(k-1) for k in labels[a].value_counts())/(50*49) for a in ATTR}
    rows=[]
    coleman=primary[['coleman_'+a for a in ATTR]].to_numpy()
    if not np.array_equal(~np.isfinite(coleman).all(axis=1),primary.empty_graph.to_numpy()) or primary.empty_graph.sum()!=12:
        raise ValueError('Unexpected score missingness')
    for _,r in primary[~primary.empty_graph].iterrows():
        g=_read_graph(root,r)
        for a in ATTR:
            same=sum(roster[i][a]==roster[j][a] for i,j in g.edges)/g.number_of_edges()
            rows.append((*r[KEY],a,r['coleman_'+a],same-chance[a]))
    frame=pd.DataFrame(rows,columns=KEY+['attribute','coleman','excess'])
    cells=frame.groupby(KEY+['attribute']).agg(coleman=('coleman','mean'),excess=('excess','mean'),graphs=('coleman','size')).reset_index()
    if len(cells)!=336 or not cells.graphs.between(1,8).all() or cells.graphs.sum()!=878*3:
        raise ValueError('Incomplete RQ2 analysis')
    ranks=[]
    for metric in ['coleman','excess']:
        t=cells.pivot(index=KEY,columns='attribute',values=metric)
        # Do not silently assign exact ties to the first column.
        ranks.append(t.idxmax(axis=1).where(t.eq(t.max(axis=1),axis=0).sum(axis=1)==1,'tie').rename(metric))
    return cells,pd.concat(ranks,axis=1).reset_index(),cross,v


def margins(summary):
    q=summary[(summary.metric==v3.M)&(summary.scope=='original weighting')]
    rows=[]
    for (n,s),g in q.groupby(['n','source']):
        if set(g.procedure)!=set(v3.PROCEDURES) or len(g)!=7 or not np.isfinite(g['mean']).all():
            raise ValueError('Incomplete rank comparison')
        first,second=g.sort_values(['mean','procedure']).iloc[:2].to_dict('records')
        rows.append((n,s,first['procedure'],first['mean'],second['procedure'],second['mean'],second['mean']-first['mean']))
    if len(rows)!=6: raise ValueError('Incomplete sampling sizes')
    return pd.DataFrame(rows,columns=['n','source','closest','w1','runner_up','runner_up_w1','margin'])


def fig14(cells,ranks):
    fig,(a,b)=plt.subplots(1,2,figsize=(9.6,4.7),gridspec_kw={'width_ratios':[1.7,1]})
    y=np.arange(4)*4
    for i,(attr,color,symbol) in enumerate(zip(ATTR,COLORS,['o','D','^'])):
        for j,m in enumerate(v2.METHODS):
            x=np.sort(cells.loc[(cells.method==m)&(cells.attribute==attr),'coleman'])
            a.scatter(x,y[j]+i+np.linspace(-.15,.15,len(x)),s=10,c=color,alpha=.42,lw=0)
            a.scatter(x.mean(),y[j]+i,c=color,marker=symbol,s=44,edgecolor='white',lw=.6,zorder=3)
    a.axvline(0,color='#889297',lw=.7)
    a.set(yticks=y+1,yticklabels=[m.title() for m in v2.METHODS],ylim=(15,-1),xlabel='Marginal Coleman score')
    a.set_title('A   Spread across 28 combinations',loc='left')
    x=np.arange(4)
    for i,(metric,marker) in enumerate([('coleman','o'),('excess','x')]):
        n=ranks.coleman if metric=='coleman' else ranks.excess
        counts=n.eq(ATTR[0]).groupby(ranks.method).sum().reindex(v2.METHODS)
        b.scatter(counts,x+(i-.5)*.19,color=COLORS[0] if i==0 else '#834515',marker=marker,s=38,label='Coleman' if i==0 else 'Excess edge share')
    b.set(yticks=x,yticklabels=[],ylim=(3.5,-.5),xticks=[0,7,14,21,28],xlim=(-1,29),xlabel='Combinations (out of 28)')
    b.set_title('B   Political orientation ranked first',loc='left')
    b.legend(loc='lower center',bbox_to_anchor=(.5,-.45),frameon=False)
    a.legend(handles=[Line2D([],[],color=c,marker=m,ls='',label=l.title()) for c,m,l in zip(COLORS,['o','D','^'],ATTR)],loc='lower center',bbox_to_anchor=(.5,-.35),ncol=3,frameon=False,columnspacing=1)
    for ax in (a,b): ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
    fig.subplots_adjust(left=.095,right=.98,bottom=.29,top=.91,wspace=.24)
    return fig


def fig15(held,per,sim):
    fig=plt.figure(figsize=(9.5,5.3))
    grid=fig.add_gridspec(2,2,left=.10,right=.98,top=.91,bottom=.18,wspace=.32,hspace=.58,width_ratios=[1.2,1])
    ax=fig.add_subplot(grid[:,0]); ys=np.arange(len(per))
    for key,marker,color,label in [('WS','s','#777D82','Fixed WS'),('global','o','#176785','Global LLM'),('target_ER','D','#A95A18','ER (target density)')]:
        ax.scatter(per[key],ys,marker=marker,s=22,c=color,label=label)
    ax.set(yticks=ys,yticklabels=per.cohort,ylim=(len(per)-.4,-.6),xlabel='Efficiency W1 (smaller is closer)')
    ax.set_title('A   Paired comparisons by record',loc='left');ax.grid(axis='x',alpha=.18)
    ax.legend(frameon=False,loc='lower center',bbox_to_anchor=(.5,-.21),ncol=3,columnspacing=.65)
    cand=held['candidate_metrics']
    for j,(src,label) in enumerate(zip(v2.SOURCES,['Facebook','School'])):
        bx=fig.add_subplot(grid[j,1]);t=held['heldout_sample_metrics'];t=t[t.source==src];sim_s=sim[sim.source==src]
        vals=[];weights=[]
        for model in v2.MODELS:
            g,w=v2.candidate_weights(cand[cand.candidate.eq('LLM:global:'+model)],EFF)
            vals.extend(g[EFF]);weights.extend(w/4)
        for values,ws,c,ls in [(t[EFF],1/t.cohort.map(t.cohort.value_counts()),'#242D33','-'),(sim_s[EFF],np.ones(len(sim_s)),'#A95A18','--'),(vals,weights,'#176785','-')]:
            xx,yy=v2.weighted_ecdf(values,ws);bx.step(xx,yy,where='post',color=c,ls=ls,lw=1.4)
        bx.set(xlim=(-.02,1),ylim=(-.02,1.02),yticks=[0,.5,1],ylabel='Fraction at or below')
        bx.set_title('BC'[j]+'   '+label+' efficiency',loc='left')
        if j: bx.set_xlabel('Global efficiency')
    fig.text(.58,.035,'Black: observed   /   Blue: Global   /   Dashed: target ER',size=8)
    return fig


def build(root,dest):
    if dest.exists(): raise FileExistsError(dest)
    v2.verify_hashes({root/p:h for p,h in PINS.items()})
    args=v3.load_tables(root)[:4];held=args[-1]
    old=json.loads((root/v3.OUTPUT/'figure_manifest.json').read_text())
    summary=v3.OUTPUT/'sensitivity_source_summary.csv'
    v2.verify_hashes({root/summary:old['artifacts'][summary.name]})
    v2.verify_hashes({root/p:h for p,h in old['inputs'].items()})
    banks,per,sim=diagnostic(held);cells,ranks,cross,v=mixing(root)
    mg=margins(pd.read_csv(root/summary))
    dest.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=dest.parent) as tmp:
        folder=Path(tmp)/'release';folder.mkdir()
        for name,data in [('efficiency_banks',banks),('efficiency_records',per),('efficiency_simulations',sim),('efficiency_summary',summarize(per)),('clustering_margins',mg),('attribute_cells',cells),('attribute_ranks',ranks),('roster_crosstab',cross.reset_index())]:
            data.to_csv(folder/(name+'.csv'),index=False)
        with plt.rc_context(v2.STYLE):
            figs=[v3.distances(*args),v3.controls(*args),fig14(cells,ranks),fig15(held,per,sim)]
            fig=figs[0];fig.subplots_adjust(bottom=.32)
            fig.text(.50,.16,'Efficiency: fixed baselines are denser than the targets. See Fig. 15 for target-density ER.  |  Record ranges are not CIs.',ha='center',size=9)
            fig=figs[1];fig.subplots_adjust(bottom=.24)
            fig.text(.50,.115,'Lines span 32 overlapping draws (min to max), not confidence intervals.',ha='center',size=9)
            for (name,fig) in zip(CAPTIONS,figs):
                for ext,meta in [('png',None),('svg',{'Date':None}),('pdf',{'CreationDate':None,'ModDate':None})]:
                    fig.savefig(folder/(name+'.'+ext),dpi=220,metadata=meta)
                plt.close(fig)
        manifest={'seed':SEED,'banks':8,'graphs_per_bank':56,'n':50,'roster_cramers_v':v,
                  'roster_sha256_raw':v2.sha(root/ROSTER),'inputs':{str(p.as_posix()):h for p,h in PINS.items()},
                  'code_sha256':v2.sha(root/SOURCE),'artifacts':{p.name:v2.sha(p) for p in sorted(folder.iterdir())}}
        (folder/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
        folder.rename(dest)


def verify(root):
    folder=root/OUT
    manifest=json.loads((folder/'manifest.json').read_text())
    if manifest['inputs']!={p.as_posix():h for p,h in PINS.items()}:
        raise ValueError('Unexpected provenance')
    v2.verify_hashes({root/SOURCE:manifest['code_sha256'],root/ROSTER:manifest['roster_sha256_raw'],
                      **{root/p:h for p,h in PINS.items()}})
    old=json.loads((root/v3.OUTPUT/'figure_manifest.json').read_text())
    v2.verify_hashes({root/p:h for p,h in old['inputs'].items()})
    return manifest


def table(root,name):
    p=root/OUT/(name+'.csv')
    manifest=verify(root)
    v2.verify_hashes({p:manifest['artifacts'][p.name]})
    return pd.read_csv(p)


def show(root,names):
    from IPython.display import Image,Markdown,display
    folder=root/OUT
    manifest=verify(root)
    for name in names:
        if name not in CAPTIONS:
            v3.show(root,[name]);continue
        v2.verify_hashes({folder/(name+'.'+ext):manifest['artifacts'][name+'.'+ext] for ext in ('png','svg','pdf')})
        display(Image(filename=str(folder/(name+'.png')),width=1100))
        display(Markdown(CAPTIONS[name]))


if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    build(root,Path(sys.argv[1]).resolve() if len(sys.argv)>1 else root/OUT)
