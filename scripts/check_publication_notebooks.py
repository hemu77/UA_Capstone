"""Run notebooks 01-04; verify embedded replacement images and immutable evidence."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis_figures.publication import OUTPUT, DRAWERS, protected_files, sha, verify_hashes

EXPECTED = {'01': ['01_primary_contrasts', '02_primary_robustness', '07_legacy_comparison'],
            '02': list(DRAWERS)[:7], '03': list(DRAWERS)[7:], '04': list(DRAWERS)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    seal = root / OUTPUT / 'figure_manifest.json'
    manifest = json.loads(seal.read_text())
    if manifest['status'] != 'COMPLETE_PUBLICATION_FIGURES_V2':
        raise ValueError('Incomplete figure revision')
    protected = protected_files(root)
    protected.update({root / p: h for p, h in manifest['protected_sha256'].items()})
    protected.update({root / p: h for p, h in manifest['code_sha256'].items()})
    protected.update({root / OUTPUT / p: r['sha256'] for p, r in manifest['artifacts'].items()})
    protected[seal] = sha(seal)
    verify_hashes(protected)
    for number, expected in EXPECTED.items():
        path = next((root / 'analysis').glob(number + '_*.ipynb'))
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        NotebookClient(notebook, timeout=300, kernel_name='ua-capstone-final060d',
                       resources={'metadata': {'path': str(root)}}, allow_errors=False).execute()
        images = [hashlib.sha256(base64.b64decode(o['data']['image/png'])).hexdigest()
                  for c in notebook.cells if c.cell_type == 'code' for o in c.outputs
                  if 'image/png' in o.get('data', {})]
        expected_hashes = [manifest['artifacts'][name + '.png']['sha256'] for name in expected]
        if sorted(images) != sorted(expected_hashes):
            raise ValueError('Notebook does not embed exactly the expected v2 figures: ' + path.name)
        verify_hashes(protected)
        nbformat.validate(notebook)
        nbformat.write(notebook, path)
        print(path.name + ': executed; ' + str(len(images)) + ' exact v2 PNGs embedded', flush=True)
    print('ALL_4_NOTEBOOKS_VERIFIED; OLD_RESULTS_UNCHANGED; PAID_CALLS=0', flush=True)


if __name__ == '__main__':
    main()
