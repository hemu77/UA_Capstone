"""Verify shipped v4 notebook outputs and execute all five against the frozen kernel."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from analysis_figures import deep_review as v4, publication_v3 as v3, publication as v2

PIN='e8b88b8a454afd4f2d5d3b0cc9c1f5f877a00d5069f93c0ab421dca331f4913c'


def check(write=False):
    folder=ROOT/v4.OUT
    v2.verify_hashes({folder/'manifest.json':PIN})
    seal=v4.verify(ROOT)
    v2.verify_hashes({folder/p:h for p,h in seal['artifacts'].items()})
    old=json.loads((ROOT/v3.OUTPUT/'figure_manifest.json').read_bytes())
    v2.verify_hashes({ROOT/v3.OUTPUT/p:h for p,h in old['artifacts'].items()})
    names={n:list(k) for n,k in v3.EXPECTED.items()}
    names['03']+=['15_efficiency_density']
    names['04']+=['14_demographic_mixing','15_efficiency_density']
    names['05']=['14_demographic_mixing']
    captions={**old['captions'],**v4.CAPTIONS}
    def verify(nb,figs):
        nbformat.validate(nb)
        out=[o for c in nb.cells if c.cell_type=='code' for o in c.outputs]
        if any(o.output_type=='error' for o in out) or any(c.get('attachments') for c in nb.cells):
            raise ValueError('Unexpected error or alternate image attachment')
        hashes=[hashlib.sha256(base64.b64decode(o.data['image/png'])).hexdigest() for o in out if 'image/png' in o.get('data',{})]
        expected=[(seal if k in v4.CAPTIONS else old)['artifacts'][k+'.png'] for k in figs]
        md=[o.data['text/markdown'] for o in out if 'text/markdown' in o.get('data',{})]
        if hashes!=expected or [s for s in md if s in captions.values()]!=[captions[k] for k in figs]:
            raise ValueError('Figure or caption bytes/order differ: '+str(figs))
    for number,figs in names.items():
        path=next((ROOT/'analysis').glob(number+'_*.ipynb'))
        nb=nbformat.read(path,as_version=4)
        if write and number=='04':
            for c in nb.cells:
                if c.id=='rq2-primary-addendum': c.pop('attachments',None)
        if not write or number in ('01','02'): verify(nb,figs)
        NotebookClient(nb,timeout=300,kernel_name='ua-capstone-final060d',resources={'metadata':{'path':str(ROOT)}},
                       allow_errors=False).execute()
        verify(nb,figs)
        if write and number not in ('01','02'):
            nbformat.write(nb,path)
            verify(nbformat.read(path,as_version=4),figs)
        print(f'{path.name}: {len(figs)} images and captions verified',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write',action='store_true',help='Refresh generated outputs in 03, 04 and 05 only.')
    check(parser.parse_args().write)
