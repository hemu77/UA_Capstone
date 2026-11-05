"""Execute both primary notebooks offline; sealed analysis outputs must not change."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    root = parser.parse_args().root.resolve()
    output = root/'outputs/primary_analysis/20261105'
    manifest_path = output/'evaluation_manifest.json'
    manifest_hash = digest(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest['status'] != 'COMPLETE_PRIMARY_ANALYSIS_WITH_CAVEATS':
        raise ValueError('A completed evaluation is required')
    for relative, record in manifest['artifacts'].items():
        path = output/relative
        if digest(path) != record['sha256'] or path.stat().st_size != record['bytes']:
            raise ValueError('Sealed artifact differs: '+relative)
    with (output/'empirical_reference_metrics.csv').open(newline='') as stream:
        for record in csv.DictReader(stream):
            data = (root/record['source_file']).read_bytes()
            if hashlib.sha256(data.replace(b'\r\n', b'\n')).hexdigest() != record['source_lf_sha256']:
                raise ValueError('Empirical source content differs beyond newlines: '+record['source_file'])
    for name in ('01_primary_statistical_analysis.ipynb', '02_primary_comparison_plots.ipynb'):
        path = root/'analysis'/name
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        NotebookClient(notebook, timeout=300, kernel_name='ua-capstone-final060d',
                       resources={'metadata':{'path':str(root)}}, allow_errors=False).execute()
        nbformat.validate(notebook)
        nbformat.write(notebook, path)
        print(name+': executed and schema-valid', flush=True)
    if digest(manifest_path) != manifest_hash:
        raise ValueError('Notebook execution changed the analysis seal')
    for relative, record in manifest['artifacts'].items():
        if digest(output/relative) != record['sha256']:
            raise ValueError('Notebook execution changed sealed artifact: '+relative)
    print('All sealed artifacts unchanged; no collection, provider calls or ledger queries.', flush=True)


if __name__ == '__main__':
    main()
