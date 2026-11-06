"""Build a new release once, or check the current notebooks with the frozen kernel."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat
from nbclient import NotebookClient
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from analysis_figures import publication as old, publication_v3 as new
from analysis_figures.review_sensitivity import evaluate

V2_PIN = '25b5419d9857c31d1d69fa41d508976060351accd315d74a12f829c5d8a0b5fd'


def inputs(root):
    seal = root/old.OUTPUT/'figure_manifest.json'
    old.verify_hashes({seal:V2_PIN})
    previous = json.loads(seal.read_text())
    hashes = dict(previous['protected_sha256'], **previous['code_sha256'])
    hashes.update({(old.OUTPUT/f).as_posix():r['sha256'] for f,r in previous['artifacts'].items()})
    hashes[(old.OUTPUT/seal.name).as_posix()] = V2_PIN
    for f in ('analysis_figures/publication_v3.py','analysis_figures/review_sensitivity.py','scripts/review_publication.py'):
        hashes[f] = old.sha(root/f)
    old.verify_hashes({root/f:h for f,h in hashes.items()})
    return hashes,previous['captions']


def captions(original):
    result = dict(original)
    result['01_primary_contrasts'] += ' Iterative US-Hindi passes the adjusted threshold in only 3 of 8 omission refits; these are overlapping sensitivity checks.'
    result['03_model_profiles'] += ' Axes show the range of observed means, not the full theoretical domain. All three clustering panels use the same limits.'
    result['07_legacy_comparison'] = ('Circles show the median and lines the middle half of valid, defined primary outcomes within each procedure. '
        'These are distribution summaries, not confidence intervals. Diamonds are individual development records; size and tie definitions differ, and the Tailor waves share workers.')
    result['08_heldout_distances'] = ('Lower W1 means closer on that metric alone. First average the four fixed-model distances, or the eight classical simulation-bank distances, '
        'within each record. Blue points average the ten Facebook records equally; bars show the smallest and largest record values, NOT confidence intervals. '
        'Orange diamonds are one school record. The same people can appear in more than one Facebook record. LLM scores weight all seven instructions equally among valid outcomes; failures are not zeros. '
        'These are not country-matched populations. Efficiency reflects fragmentation as well as path length. Largest-component results are in appendix 13.')
    result['09_heldout_distributions'] = ('Height is the fraction of networks at or below the value. Compare each generated curve with both observed sources: Facebook (solid black, equal record weights) and school (dashed black). '
        'Candidates appear only once per metric: top panels mix the four fixed models equally within a procedure, retaining equal instruction weights; lower panels pool eight equal-size banks per baseline. '
        'The density axis covers all observed values. Pooled distributions illustrate shape; distance between these curves is NOT the cohort-wise score in figure 08.')
    result['11_conditional_controls'] = ('Each of the 32 target graphs contributes one residual: observed coefficient minus the mean of its eight matched controls. Dots average those 32 differences; '
        'lines span their actual minimum to maximum, not confidence intervals. These are target-informed controls, not competitors. Edge count fixes density, and the degree sequence fixes Gini: zero residuals there are identities. '
        'Finite swaps do not establish uniform mixing; a positive mean is not a significance test.')
    result['12_cohort_sensitivity'] = ('Each row is one observed record. Panels share one clustering W1 scale. Four fixed models per LLM procedure remain separate, and the last panel shows each classical baseline averaged over its eight banks. '
        'These 19 candidates show variability hidden by procedure means. Rows are not independent population replicates.')
    result['13_appendix_lcc'] = ('Appendix: largest-component fraction, with the same record-level W1 averaging and descriptive ranges as figure 08. '
        '447/448 ER graphs and 448/448 BA and WS graphs are connected. Distances can nearly coincide without being exactly identical; this metric is retained, not deleted.')
    return result


def build(root, dest, cache):
    if dest.exists():
        raise FileExistsError('Release or preview exists; choose an unused destination: '+str(dest))
    hashes,previous=inputs(root)
    a,primary,tables,held,_=new.load_tables(root)
    dest.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=dest.parent) as tmp:
        folder=Path(tmp)/'release'
        folder.mkdir()
        with plt.rc_context(old.STYLE):
            for name,drawer in new.DRAWERS.items():
                if name in new.KEEP:
                    for ext in ('png','svg','pdf'):
                        shutil.copyfile(root/old.OUTPUT/(name+'.'+ext),folder/(name+'.'+ext))
                    continue
                fig=drawer(a,primary,tables,held)
                try:
                    for ext,meta in [('png',None),('svg',{'Date':None}),('pdf',{'CreationDate':None,'ModDate':None})]:
                        fig.savefig(folder/(name+'.'+ext),dpi=220,metadata=meta)
                finally:
                    plt.close(fig)
        raw=evaluate(root,cache,primary,held)
        raw.to_csv(folder/'sensitivity_bank_distances.csv',index=False)
        groups=[]
        for (n,scope,metric),frame in raw.groupby(['n','scope','metric']):
            if scope=='US-English only':
                baseline=raw[(raw.n==n)&(raw.scope=='original weighting')&(raw.metric==metric)&raw.candidate.str.startswith('Baseline:')]
                frame=pd.concat([frame,baseline])
            _,records=new.cohort_scores(frame)
            summary=records.groupby(['source','procedure']).w1.agg(['mean','min','max','count']).reset_index()
            groups.append(summary.assign(n=n,scope=scope,metric=metric))
        pd.concat(groups).to_csv(folder/'sensitivity_source_summary.csv',index=False)
        old.verify_hashes({root/f:h for f,h in hashes.items()})
        manifest=dict(status='COMPLETE_REVIEW_V3',inputs=hashes,captions=captions(previous),
            primary_attempts=896,primary_valid=890,supplemental_graphs=0,paid_calls=0,ledger_queries=0,
            artifacts={p.name:old.sha(p) for p in sorted(folder.iterdir())})
        (folder/'figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
        folder.rename(dest)
    print(f'Built {len(new.DRAWERS)} figures and two diagnostic tables: {dest}',flush=True)


def check(root):
    folder=root/new.OUTPUT
    seal=folder/'figure_manifest.json'
    manifest=json.loads(seal.read_text())
    hashes,_=inputs(root)
    expected_files={k+'.'+ext for k in new.DRAWERS for ext in ('png','svg','pdf')} | {'sensitivity_source_summary.csv','sensitivity_bank_distances.csv'}
    if (manifest['status']!='COMPLETE_REVIEW_V3' or manifest['inputs']!=hashes or
            set(manifest['artifacts'])!=expected_files or set(manifest['captions'])!=set(new.DRAWERS)):
        raise ValueError('Release inputs or inventory differ')
    protected={root/f:h for f,h in hashes.items()}
    protected.update({folder/f:h for f,h in manifest['artifacts'].items()})
    protected[seal]=old.sha(seal)
    old.verify_hashes(protected)
    for number,names in new.EXPECTED.items():
        path=next((root/'analysis').glob(number+'_*.ipynb'))
        nb=nbformat.read(path,as_version=4)
        NotebookClient(nb,timeout=300,kernel_name='ua-capstone-final060d',resources={'metadata':{'path':str(root)}},
                       allow_errors=False).execute()
        nbformat.validate(nb)
        outputs=[o.get('data',{}) for c in nb.cells if c.cell_type=='code' for o in c.outputs]
        images=[hashlib.sha256(base64.b64decode(o['image/png'])).hexdigest() for o in outputs if 'image/png' in o]
        if images!=[manifest['artifacts'][k+'.png'] for k in names]:
            raise ValueError('Embedded figure order or bytes differ: '+path.name)
        md=[o['text/markdown'] for o in outputs if 'text/markdown' in o]
        if [s for s in md if s in manifest['captions'].values()]!=[manifest['captions'][k] for k in names]:
            raise ValueError('Embedded captions differ: '+path.name)
        old.verify_hashes(protected)
        nbformat.write(nb,path)
        print(path.name, len(images),'embedded v3 images verified',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='action',required=True)
    b=sub.add_parser('build')
    b.add_argument('--cache',type=Path,required=True)
    b.add_argument('--dest',type=Path,default=ROOT/new.OUTPUT)
    sub.add_parser('check')
    args=p.parse_args()
    if args.action=='build':
        build(ROOT,args.dest.resolve(),args.cache.resolve())
    else:
        check(ROOT)
