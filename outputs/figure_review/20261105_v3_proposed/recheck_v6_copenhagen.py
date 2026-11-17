import csv,collections,statistics,bisect
E='outputs/research_extension/20261114_v2/'
obs=[r for r in csv.DictReader(open(E+'observed_sample_metrics.csv')) if r['cohort']=='Copenhagen' and r.get('n','50')=='50']
cm=list(csv.DictReader(open('outputs/heldout_evaluation/20261105_v1/candidate_metrics.csv')))
def w1(xs,ws,ys,wy):
    a=collections.defaultdict(float);b=collections.defaultdict(float)
    for v,w in zip(xs,ws):a[v]+=w
    for v,w in zip(ys,wy):b[v]+=w
    pts=sorted(set(a)|set(b));ca=cb=t=0
    for i,p in enumerate(pts[:-1]):
        ca+=a.get(p,0);cb+=b.get(p,0);t+=abs(ca-cb)*(pts[i+1]-p)
    return t
print(len(obs),list(obs[0].keys()))
def llm(proc,col):
    per=[]
    for g in sorted({r['candidate'] for r in cm if r['candidate'].startswith('LLM:%s:'%proc)}):
        by=collections.defaultdict(list)
        for r in cm:
            if r['candidate']==g and r['valid']=='True' and r[col]!='': by[r['condition']].append(float(r[col]))
        xs=[];ws=[]
        for c,v in by.items():
            for x in v: xs.append(x);ws.append(1/7/len(v))
        y=[float(r[col]) for r in obs]; per.append(w1(xs,ws,y,[1/len(y)]*len(y)))
    return statistics.mean(per)
ts={(r['procedure'],r['metric']):float(r['mean_w1']) for r in csv.DictReader(open(E+'transfer_record_summary.csv')) if r['cohort']=='Copenhagen' and r['n']=='50'}
for p in ['global','local','sequential','iterative']:
    for m in ['density','avg_clustering_coef','global_efficiency','transitivity','degree_gini','prop_nodes_lcc']:
        mine=llm(p,m); pub=ts.get((p,m))
        print(p,m,round(mine,4),pub and round(pub,4),'OK' if pub and abs(mine-pub)<1e-9 else 'DIFF')
