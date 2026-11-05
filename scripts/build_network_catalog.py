"""Publish verified graph PNGs and a browsable index, without mixing cohorts."""
import argparse
import csv
import hashlib
import json
from pathlib import Path, PureWindowsPath
import shutil


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def asset(root, relative):
    path = Path(relative)
    if path.is_absolute() or PureWindowsPath(relative).drive or '..' in path.parts or '..' in PureWindowsPath(relative).parts:
        raise ValueError('Require project-relative asset')
    resolved = (root/path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError('Asset escapes root')
    return resolved


def catalog(root, destination):
    ready = root/'outputs/analysis_ready/060dcd3a86d5'
    receipt = json.loads((ready/'readiness.json').read_text())
    name = 'all_valid_graphs_descriptive_only.csv'
    if digest(ready/name) != receipt['generated_files'][name]['sha256']:
        raise ValueError('Catalog hash differs')
    with (ready/name).open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    primary = [r for r in rows if r['cohort'] == 'primary']
    supplement = [r for r in rows if r['cohort'] != 'primary']
    if (len(rows),len(primary),len(supplement),len({r['run_id'] for r in rows})) != (896,890,6,896):
        raise ValueError('Require 890 primary plus six separate supplements')
    for row in rows:
        if row['valid_graph'] != 'True':
            raise ValueError('Invalid graph in valid-only catalog')
        for suffix in ('adj','png'):
            path = asset(root,row[suffix+'_path'])
            if digest(path) != row[suffix+'_sha256']:
                raise ValueError('Graph asset hash differs: '+row['run_id'])
            target = asset(destination,row[suffix+'_path'])
            if target.exists() and digest(target) != row[suffix+'_sha256']:
                raise ValueError('Refuse differing destination: '+str(target))
    for row in rows:
        for suffix in ('adj','png'):
            target = asset(destination,row[suffix+'_path'])
            if not target.exists():
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(asset(root,row[suffix+'_path']),target)
    folder = destination/'docs/network_catalog'
    folder.mkdir(parents=True,exist_ok=True)
    index = ['# Browse All 896 Saved Networks','',
        '**890 original primary networks + six separately labeled supplements.**',
        'The primary study remains 890 valid graphs out of 896 attempts, including 12 valid empty graphs.',
        'All original six failures remain recorded. Supplements never enter primary statistical tests.','',
        'Every row below links to the actual adjacency file and original PNG in this repository.',
        'The large response JSON records remain in the [verified public release](https://github.com/hemu77/UA_Capstone/releases/tag/main-study-20261105).','',
        '| Primary Method | Saved Networks | Browse Files |','| --- | ---: | --- |']
    groups = [(m,[r for r in primary if r['method']==m]) for m in ('global','local','sequential','iterative')]
    groups.append(('supplemental',supplement))
    for group, selected in groups:
        if group != 'supplemental':
            index.append(f'| {group.title()} | {len(selected)} | [All graphs and PNGs](docs/network_catalog/{group}.md) |')
        lines = [f'# {group.title()} Networks','',
            '**Separate, selectively completed supplements: excluded from primary inference.**' if group=='supplemental'
            else '**Original primary valid graphs only. Failed attempts are not successful zero graphs.**',
            '', '[Back to all networks](../../NETWORKS.md)','',
            '| Model | Framing | Language | Seed | Edges | Graph Data | Original Plot |',
            '| --- | --- | --- | ---: | ---: | --- | --- |']
        for row in selected:
            lines.append(f"| {row['model']} | {row['culture']} | {row['language']} | {row['seed']} | {row['edges']} | "
                         f"[ADJ](../../{row['adj_path']}) | [PNG](../../{row['png_path']}) |")
        (folder/(group+'.md')).write_text('\n'.join(lines)+'\n',encoding='utf8')
    index += ['', '[Browse the six separate supplemental graphs](docs/network_catalog/supplemental.md).',
        '[All nine retained failed attempts](outputs/analysis_ready/060dcd3a86d5/failures.csv).',
        '[Machine-readable 896-network catalog and hashes](outputs/analysis_ready/060dcd3a86d5/all_valid_graphs_descriptive_only.csv).',
        '', '## Example Original Plots', '']
    for method,selected in groups[:4]:
        index += [f'### {method.title()}',f"![Example {method} network]({selected[0]['png_path']})",'']
    (destination/'NETWORKS.md').write_text('\n'.join(index)+'\n',encoding='utf8')
    print(json.dumps(dict(valid_graphs=896,primary=890,supplemental=6,adjacencies_verified=896,pngs_verified=896,paid_calls=0)))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--destination',type=Path)
    args=parser.parse_args()
    root=args.root.resolve()
    catalog(root,args.destination.resolve() if args.destination else root)
