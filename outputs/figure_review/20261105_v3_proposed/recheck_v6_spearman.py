import csv,collections,statistics,bisect,math
from scipy.stats import spearmanr, shapiro
E='outputs/research_extension/20261114_v2/'
ts=list(csv.DictReader(open(E+'transfer_record_summary.csv')))
# A. Spearman efficiency vs density W1 across record x procedure at n=50 (LLM procedures + 3 baselines)
d={(r['cohort'],r['procedure'],r['metric']):float(r['mean_w1']) for r in ts if r['n']=='50' and r['kind']!='diagnostic' and 'target' not in r['procedure'].lower()}
keys=sorted({(c,p) for c,p,m in d})
x=[d[(c,p,'density')] for c,p in keys if (c,p,'density') in d and (c,p,'global_efficiency') in d]
y=[d[(c,p,'global_efficiency')] for c,p in keys if (c,p,'density') in d and (c,p,'global_efficiency') in d]
print('procedures',sorted({p for c,p in keys}),'pairs',len(x),'Spearman eff~dens',round(spearmanr(x,y)[0],3))
x2=[d[(c,p,'density')] for c,p in keys if (c,p,'prop_nodes_lcc') in d and (c,p,'density') in d]
y2=[d[(c,p,'prop_nodes_lcc')] for c,p in keys if (c,p,'prop_nodes_lcc') in d and (c,p,'density') in d]
print('Spearman LCC~dens',round(spearmanr(x2,y2)[0],3))
# also within-Facebook only, excluding Copenhagen/School, and LLM-only
for lab,filt in [('FB only',lambda c:c.startswith('FB')),('LLM procs only',None)]:
    pass
xx=[d[(c,p,'density')] for c,p in keys if c.startswith('FB')]; yy=[d[(c,p,'global_efficiency')] for c,p in keys if c.startswith('FB')]
print('FB-only Spearman',round(spearmanr(xx,yy)[0],3),len(xx))
# B. Hindi block diffs
pb=list(csv.DictReader(open('outputs/primary_analysis/20261105/paired_schedule_blocks.csv'))); print(pb[0].keys())
