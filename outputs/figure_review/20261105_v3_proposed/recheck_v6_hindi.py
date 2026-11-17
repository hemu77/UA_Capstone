import csv,statistics,math,itertools,collections
from scipy import stats
pb=list(csv.DictReader(open('outputs/primary_analysis/20261105/paired_schedule_blocks.csv')))
h=[r for r in pb if r['hypothesis_id']=='iterative:us:hindi']
h=sorted(h,key=lambda r:r['seed']); v=[float(r['difference']) for r in h if r['complete'] in ('True','true','1')]
print('n blocks',len(v),[round(x,4) for x in v])
t=stats.ttest_1samp(v,0); print('mean',round(statistics.mean(v),5),'t p',t.pvalue,'(csv raw_p 0.000993)')
print('Shapiro',stats.shapiro(v))
# exact sign-flip two-sided
n=len(v); obs=abs(sum(v)); cnt=0
for s in itertools.product([1,-1],repeat=n):
    if abs(sum(a*b for a,b in zip(s,v)))>=obs-1e-15: cnt+=1
p=cnt/2**n; print('sign-flip p',p,'floor 2/256 =',2/256,'x24 =',min(1,p*24),'| min attainable adj',24*2/256)
# t critical for simultaneous 24 Bonferroni
se=statistics.stdev(v)/math.sqrt(n); crit=stats.t.ppf(1-0.05/48,n-1); print('CI',statistics.mean(v)-crit*se,statistics.mean(v)+crit*se)
# LOO
loo=[r for r in csv.DictReader(open('outputs/primary_analysis/20261105/leave_one_seed_out.csv')) if r['hypothesis_id']=='iterative:us:hindi']
print('LOO holm_p',sorted(round(float(r['holm_p']),4) for r in loo),'pass',sum(float(r['holm_p'])<.05 for r in loo))
# LOO raw t p recomputed, using fixed Holm multiplier = rank-1 -> 24
for i in range(n):
    w=v[:i]+v[i+1:]; pp=stats.ttest_1samp(w,0).pvalue; print(' omit',i+1,'raw',round(pp,5),'x24',round(min(1,pp*24),4))
# Fig 16 sign counts
pc=list(csv.DictReader(open('outputs/research_extension/20261114_v2/mixing_paired_cells.csv')))
sel=[r for r in pc if r['attribute']=='religion' and r['first']=='local' and r['second']=='sequential' and r['score']=='excess' and r['support']=='score_specific']
d=[float(r['difference']) for r in sel]; print('religion local->sequential cells',len(d),'mean',round(statistics.mean(d),6),'neg',sum(x<0 for x in d),'pos',sum(x>0 for x in d))
