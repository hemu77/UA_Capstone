"""Execute notebook 04 with sealed figure/code and original-evidence checks."""
import argparse
import hashlib
import json
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    root=parser.parse_args().root.resolve()
    folder=root/'outputs/manuscript_figures/20261105_v1'
    seal=folder/'figure_manifest.json'
    manifest=json.loads(seal.read_text())
    if manifest['status']!='COMPLETE_MANUSCRIPT_FIGURES_WITH_CAVEATS': raise ValueError('Incomplete manuscript figures')
    protected={seal:digest(seal),**{folder/name:info['sha256'] for name,info in manifest['artifacts'].items()},
        **{root/name:expected for name,expected in manifest['code_sha256'].items()}}
    old=root/'outputs/primary_analysis/20261105/evaluation_manifest.json'
    previous=json.loads(old.read_text())
    protected.update({old:digest(old),**{old.parent/name:info['sha256'] for name,info in previous['artifacts'].items()}})
    protected.update({root/name:expected for name,expected in previous['code_sha256'].items()})
    protected.update({root/name:expected for name,expected in previous['preserved_original_artifacts'].items()})
    source_pins=json.loads((root/'final_study/contract.json').read_text())['sources']
    protected.update({root/name:expected for name,expected in source_pins.items()})
    if digest(old)!=manifest['previous_primary_seal_sha256']: raise ValueError('Previous primary seal changed')
    for path,expected in protected.items():
        if digest(path)!=expected: raise ValueError('Protected artifact differs: '+str(path))
    path=root/'analysis/04_manuscript_figures.ipynb'
    notebook=nbformat.read(path,as_version=4)
    nbformat.validate(notebook)
    NotebookClient(notebook,timeout=300,kernel_name='ua-capstone-final060d',
        resources={'metadata':{'path':str(root)}},allow_errors=False).execute()
    nbformat.validate(notebook)
    nbformat.write(notebook,path)
    for path,expected in protected.items():
        if digest(path)!=expected: raise ValueError('Notebook changed protected artifact: '+str(path))
    print('Notebook 04 executed/schema-valid; sealed manuscript and primary figures unchanged.',flush=True)


if __name__=='__main__': main()
