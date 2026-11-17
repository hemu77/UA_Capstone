import csv,json,collections,statistics,math,bisect,os
V='outputs/publication_figures/20261111_v4/'
# 1. target_ER W1 recompute from simulations vs heldout draws
hs=list(csv.DictReader(open('outputs/heldout_evaluation/20261105_v1/heldout_sample_metrics.csv')))
sim=collections.defaultdict(list)
for r in csv.DictReader(open(V+'efficiency_simulations.csv')): sim[(r['cohort'],r['bank'])].append(float(r['global_efficiency']))
def w1(x,y):
    xs=sorted(x);ys=sorted(y);pts=sorted(set(xs)|set(ys));t=0
    for i,p in enumerate(pts[:-1]): t+=abs(bisect.bisect_right(xs,p)/len(xs)-bisect.bisect_right(ys,p)/len(ys))*(pts[i+1]-p)
    return t
rec={r['cohort']:float(r['target_ER']) for r in csv.DictReader(open(V+'efficiency_records.csv'))}
mx=0
for c in rec:
    tgt=[float(r['global_efficiency']) for r in hs if r['cohort']==c]
    d=[w1(sim[(c,str(b))],tgt) for b in range(8)]; mx=max(mx,abs(statistics.mean(d)-rec[c]))
print('target_ER recompute max abs diff over 11 records:',mx, '| sims per bank',len(sim[('FB01','0')]))
# density used vs full-roster density
bk=list(csv.DictReader(open(V+'efficiency_banks.csv'))); print('p_full range',min(float(r['p_full']) for r in bk),max(float(r['p_full']) for r in bk))
# 2. Cramer's V
tab=[[2,3,5],[2,4,4],[3,3,4],[6,1,3],[3,6,1]]; n=sum(map(sum,tab)); rs=[sum(r) for r in tab]; cs=[sum(c) for c in zip(*tab)]
chi=sum((tab[i][j]-rs[i]*cs[j]/n)**2/(rs[i]*cs[j]/n) for i in range(5) for j in range(3)); print('Cramers V',math.sqrt(chi/(n*2)),'chi2',chi,'n',n)
# 3. rank counts
rk=list(csv.DictReader(open(V+'attribute_ranks.csv')))
for k in ['coleman','excess']: print(k,collections.Counter(r[k] for r in rk), collections.Counter((r['method'],r[k]) for r in rk if r[k]!='political orientation'))
# 4. explorer edges vs metrics
st=json.load(open('viewer/public/data/primary-study.json'))
m={r['run_id']:r for r in csv.DictReader(open('outputs/final_study/060dcd3a86d5/main/network_metrics.csv'))}
bad=0;miss=0;ordb=0;ext=0
for r in st:
    if r['edges'] is None: continue
    for c in r['edges']:
        a,b=divmod(c,50)
        if not a<b: ordb+=1
    if len(set(r['edges']))!=len(r['edges']): bad+=1
    d=m[r['id']]['density']
    if d!='' and abs(len(r['edges'])/1225-float(d))>1e-9: miss+=1
    if not os.path.exists(f"outputs/final_study/060dcd3a86d5/main/runs/{r['id']}.adj"): ext+=1
print('explorer: records',len(st),'dup-edge graphs',bad,'density mismatches',miss,'bad order',ordb,'missing adj files',ext, 'valid',sum(r['edges'] is not None for r in st))
