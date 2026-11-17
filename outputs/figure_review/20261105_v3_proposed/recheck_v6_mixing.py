import csv,json,collections,statistics,glob,os
R=json.load(open('text-files/revision224_adults.json'))
attrs=['gender','political orientation','religion']
m={r['run_id']:r for r in csv.DictReader(open('outputs/final_study/060dcd3a86d5/main/network_metrics.csv'))}
def edges(path):
    E=set()
    for line in open(path):
        if line.startswith('#') or not line.strip(): continue
        t=line.split(); a=int(t[0])
        for b in t[1:]:
            b=int(b)
            if a!=b: E.add((min(a,b),max(a,b)))
    return E
chance={a:sum(R[str(i)][a]==R[str(j)][a] for i in range(50) for j in range(i+1,50))/1225 for a in attrs}
print('chance',{a:round(v,4) for a,v in chance.items()})
cells=collections.defaultdict(lambda: collections.defaultdict(list)); bad=0;n=0
for rid,row in m.items():
    if row['valid_graph']!='True': continue
    E=edges(f'outputs/final_study/060dcd3a86d5/main/runs/{rid}.adj')
    if not E: continue
    if row['density']!='' and abs(len(E)/1225-float(row['density']))>1e-9: bad+=1
    key=(row['method'],row['model'],row['culture'],row['language'])
    for a in attrs:
        same=sum(R[str(i)][a]==R[str(j)][a] for i,j in E)/len(E)
        cells[key][a].append(same-chance[a])
    n+=1
print('nonempty graphs',n,'density mismatches',bad,'cells',len(cells))
pub=collections.defaultdict(dict)
for r in csv.DictReader(open('outputs/publication_figures/20261111_v4/attribute_cells.csv')):
    pub[(r['method'],r['model'],r['culture'],r['language'])][r['attribute']]=float(r['excess'])
mx=0
for k,v in cells.items():
    for a in attrs: mx=max(mx,abs(statistics.mean(v[a])-pub[k][a]))
print('max abs diff vs published excess (all 112 cells x 3 attrs):',mx)
wins=collections.Counter(); byM=collections.defaultdict(collections.Counter)
for k,v in cells.items():
    best=max(attrs,key=lambda a:statistics.mean(v[a])); wins[best]+=1; byM[k[0]][best]+=1
print(dict(wins)); print({k:dict(v) for k,v in byM.items()})
