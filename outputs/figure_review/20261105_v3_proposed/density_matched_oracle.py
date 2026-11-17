import csv,collections,statistics,random,sys
import networkx as nx
D='outputs/heldout_evaluation/20261105_v1/'
hs=list(csv.DictReader(open(D+'heldout_sample_metrics.csv')))
pc=list(csv.DictReader(open(D+'per_cohort_distances.csv')))
def w1(x,y):
    xs=sorted(x);ys=sorted(y);pts=sorted(set(xs)|set(ys));t=0
    import bisect
    for i,p in enumerate(pts[:-1]):
        t+=abs(bisect.bisect_right(xs,p)/len(xs)-bisect.bisect_right(ys,p)/len(ys))*(pts[i+1]-p)
    return t
llm=collections.defaultdict(list)
for r in pc:
    if r['kind']=='LLM' and r['candidate'].split(':')[1]=='global': llm[(r['cohort'],r['metric'])].append(float(r['w1']))
base=collections.defaultdict(list)
for r in pc:
    if r['kind']=='baseline': base[(r['cohort'],r['metric'],r['candidate'])].append(float(r['w1']))
cohorts=sorted({r['cohort'] for r in hs}); out=[]
random.seed(7)
for c in cohorts:
    t=[r for r in hs if r['cohort']==c]
    p=statistics.mean(float(r['density']) for r in t)
    gs=[nx.gnp_random_graph(50,p,seed=random.randrange(10**9)) for _ in range(56)]
    ge=[nx.global_efficiency(g) for g in gs]; gc=[nx.average_clustering(g) for g in gs]
    te=[float(r['global_efficiency']) for r in t]; tc=[float(r['avg_clustering_coef']) for r in t]
    row=dict(c=c,p=round(p,3),
        eff_oracleER=w1(ge,te),eff_globalLLM=statistics.mean(llm[(c,'global_efficiency')]),eff_WS=statistics.mean(base[(c,'global_efficiency','Baseline:WS')]),
        clu_oracleER=w1(gc,tc),clu_globalLLM=statistics.mean(llm[(c,'avg_clustering_coef')]),clu_WS=statistics.mean(base[(c,'avg_clustering_coef','Baseline:WS')]))
    out.append(row)
print('%-11s %6s | eff: oracleER globalLLM WS | clus: oracleER globalLLM WS'%('cohort','dens'))
for r in out: print('%-11s %6.3f |      %.3f     %.3f   %.3f |       %.3f     %.3f   %.3f'%(r['c'],r['p'],r['eff_oracleER'],r['eff_globalLLM'],r['eff_WS'],r['clu_oracleER'],r['clu_globalLLM'],r['clu_WS']))
for k in ['eff_oracleER','eff_globalLLM','eff_WS','clu_oracleER','clu_globalLLM','clu_WS']:
    print(k,'mean',round(statistics.mean(r[k] for r in out),3))
