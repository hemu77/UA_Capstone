"""Execute the public-data notebook without changing any sealed output."""
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
    output=root/'outputs/heldout_evaluation/20261105_v1'
    seal=output/'evaluation_manifest.json'
    manifest=json.loads(seal.read_text())
    if manifest['status']!='COMPLETE_HELDOUT_STRUCTURAL_EVALUATION_WITH_CAVEATS': raise ValueError('Incomplete held-out evaluation')
    protected={seal:digest(seal),**{output/name:info['sha256'] for name,info in manifest['artifacts'].items()}}
    summary=json.loads((output/'summary.json').read_text())
    previous=summary['preserved_previous_artifacts']
    protected.update({root/name:expected for name,expected in previous.items()})
    protected.update({root/name:expected for name,expected in manifest['code_sha256'].items()})
    protected.update({root/name:expected for name,expected in summary['frozen_source_pins'].items()})
    for path,expected in protected.items():
        if digest(path)!=expected: raise ValueError('Protected artifact differs: '+str(path))
    path=root/'analysis/03_heldout_public_evaluation.ipynb'
    notebook=nbformat.read(path,as_version=4)
    nbformat.validate(notebook)
    NotebookClient(notebook,timeout=300,kernel_name='ua-capstone-final060d',
        resources={'metadata':{'path':str(root)}},allow_errors=False).execute()
    nbformat.validate(notebook)
    nbformat.write(notebook,path)
    for path,expected in protected.items():
        if digest(path)!=expected: raise ValueError('Notebook changed protected artifact: '+str(path))
    print('Notebook 03 executed/schema-valid; held-out and previous primary artifacts unchanged.',flush=True)


if __name__=='__main__': main()
