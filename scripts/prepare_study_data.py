"""Offline release installation and separated analysis inputs. Never generates data."""
import argparse
from collections import Counter
import csv
import hashlib
import importlib.metadata
import io
import itertools
import json
import math
import os
from pathlib import Path
import platform
import stat
import tempfile
import zipfile

CONTRACT = '060dcd3a86d58d0c288dfbebd3f958dfea4587f1e0225f98d24692ba9e93b52f'
PRIMARY = 'outputs/final_study/060dcd3a86d5/main/'
GROUPS = {PRIMARY:'primary', 'outputs/supplement_20261105_v1/':'supplement_initial',
          'outputs/supplement_completion_20261105_v1/':'supplement_completion'}
PINS = {'original-manifest.json':'342a3afe73f5fc00d5e50ddf925e54351b9d7b64927d84822f1328e777d7c3d5',
        'supplement-manifest.json':'6aa2bd5fc6f52962b5aca600ed9ac520e6d88717720113bc6c9490c3f0151118',
        'completion-manifest.json':'5cddbb4a70dae8d0ace1f2d739ddc1a72727c23803959964d85cc444eefbca52'}
DIMENSIONS = ('model','method','culture','language','seed','wording','roster_sha256')

def require(ok, message):
    if not ok:
        raise ValueError(message)

def digest(data):
    return hashlib.sha256(data).hexdigest()

def file_digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + '\n').encode('utf8')

def checked_name(name):
    parts = name.split('/')
    require(name and '\\' not in name and ':' not in name and not any(ord(c)<32 for c in name), 'Unsafe ZIP name')
    reserved = {'CON','PRN','AUX','NUL'} | {f'{p}{i}' for p in ('COM','LPT') for i in range(1,10)}
    require(all(p not in ('','.','..') and p.rstrip(' .') == p and p.split('.')[0].upper() not in reserved for p in parts), 'Unsafe ZIP component')
    require(name == 'final_study/contract.json' or name.startswith((*GROUPS, 'study_amendments/20261105/')), 'Unexpected archive location')
    require(not set(parts) & {'requests','transport','pretransport_rollbacks'} and
            parts[-1] not in {'authorization.json','private_snapshot.json'} and
            Path(name).suffix in {'.json','.csv','.adj','.png','.py','.ps1','.md'}, 'Private or unsupported archive member')
    return name

def destination(root, name):
    target = root / checked_name(name)
    require(target.resolve().is_relative_to(root), 'Destination escapes workspace')
    for part in [target, *target.parents]:
        if part == root.parent:
            break
        require(not part.is_symlink() and not (part.exists() and getattr(part.lstat(),'st_file_attributes',0) & 0x400), 'Redirected destination')
    return target

def verified_package(archive, manifest):
    require(archive.stat().st_size == manifest['archive_bytes'] and file_digest(archive) == manifest['archive_sha256'], 'Archive hash differs')
    entries = manifest['entries']
    require(sum(v['bytes'] for v in entries.values()) <= 3*1024**3, 'Archive too large')
    with zipfile.ZipFile(archive) as z:
        infos = z.infolist()
        names = [checked_name(v.filename) for v in infos]
        require(len(names) == len(set(n.casefold() for n in names)) and set(names) == set(entries), 'Duplicate or mismatched ZIP inventory')
        for info in infos:
            expected = entries[info.filename]
            kind = stat.S_IFMT(info.external_attr >> 16)
            require(not info.is_dir() and kind in (0,stat.S_IFREG) and not info.flag_bits & 1, 'Link/special/encrypted archive member')
            require(type(expected['bytes']) is int and 0 <= expected['bytes'] <= 128*1024**2 and info.file_size == expected['bytes'], 'Member size differs')
            with z.open(info) as stream:
                require(hashlib.file_digest(stream,'sha256').hexdigest() == expected['sha256'], 'Member hash differs')

def create_once(target, stream, expected):
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix='.study-install-', delete=False) as output:
        temporary = Path(output.name)
        try:
            h, size = hashlib.sha256(), 0
            for chunk in iter(lambda: stream.read(1024*1024), b''):
                size += len(chunk)
                require(size <= expected['bytes'], 'Unexpected extraction expansion')
                h.update(chunk); output.write(chunk)
            output.close()
            require(size == expected['bytes'] and h.hexdigest() == expected['sha256'], 'Content changed during installation')
            # Atomic, no-overwrite publication. The temporary file is on the same filesystem.
            os.link(temporary, target)
        finally:
            output.close()
            temporary.unlink(missing_ok=True)

def install_packages(root, packages):
    root = root.resolve()
    require(root.is_dir(), 'Workspace must exist')
    entries, names, missing = {}, set(), set()
    for archive, manifest in packages:
        verified_package(archive, manifest)
        for name, expected in manifest['entries'].items():
            require(name.casefold() not in names, 'Cross-archive duplicate')
            names.add(name.casefold()); entries[name] = expected
            target = destination(root,name)
            if target.exists():
                require(target.is_file() and target.stat().st_size == expected['bytes'] and file_digest(target) == expected['sha256'], 'Existing evidence differs: '+name)
            else:
                missing.add(name)
    # No writes until all three archives and all existing destinations have passed.
    for archive, manifest in packages:
        with zipfile.ZipFile(archive) as z:
            for name in manifest['entries']:
                if name in missing:
                    target = destination(root,name)
                    with z.open(name) as stream:
                        create_once(target, stream, entries[name])
    return dict(installed=len(missing), verified=len(entries), entries=entries)

def analysis_views(rows):
    primary = [r for r in rows if r['cohort']=='primary']
    supplement = [r for r in rows if r['cohort']!='primary']
    return {'all_attempts.csv':rows, 'primary_attempts.csv':primary,
            'primary_valid_graphs.csv':[r for r in primary if r['valid_graph']],
            'supplemental_attempts.csv':supplement,
            'supplemental_valid_graphs.csv':[r for r in supplement if r['valid_graph']],
            'all_valid_graphs_descriptive_only.csv':[r for r in rows if r['valid_graph'] and r['selected_for_896_view']],
            'failures.csv':[r for r in rows if not r['valid_graph']]}

def trusted_read(root, name, entries):
    data = destination(root,name).read_bytes()
    require(len(data) == entries[name]['bytes'] and digest(data) == entries[name]['sha256'], 'Installed source changed: '+name)
    return data

def graph_from_bytes(data):
    import networkx as nx
    return nx.read_adjlist(io.BytesIO(data))

def catalog(root, entries):
    import networkx as nx
    from PIL import Image
    spec = json.loads(trusted_read(root,'final_study/contract.json',entries))
    require(digest(json.dumps(spec,sort_keys=True,ensure_ascii=False).encode()) == CONTRACT, 'Wrong contract')
    for name, expected in spec['sources'].items():
        require(file_digest(root/name) == expected, 'Frozen source differs: '+name)
    runtime = {'python':platform.python_version(), **{k:importlib.metadata.version(k) for k in spec['runtime'] if k!='python'}}
    require(runtime == spec['runtime'], 'Use the frozen project virtual environment')
    selected_path = 'outputs/supplement_completion_20261105_v1/summary.json'
    selection = json.loads(trusted_read(root,selected_path,entries))['selected']
    rows, records, original_cells = [], {}, {}
    metrics = set()
    for prefix, cohort in GROUPS.items():
        expected = {name for name in entries if name.startswith(prefix+'runs/')}
        actual = {p.relative_to(root).as_posix() for p in (root/prefix/'runs').iterdir() if p.suffix in {'.json','.adj','.png'}}
        require(expected == actual, 'Unexpected or missing run files: '+prefix)
        for name in sorted(n for n in expected if n.endswith('.json')):
            record = json.loads(trusted_read(root,name,entries)); cell = record['cell']; rid=cell['run_id']
            require(rid == Path(name).stem and rid not in records and cell['final_contract_sha256'] == CONTRACT
                    and cell['frozen_source_sha256'] == spec['sources'] and cell['roster_sha256'] == spec['roster_sha256'], 'Receipt identity differs')
            require(record['evidence_type'] == ('received_final_study' if cohort=='primary' else 'received_supplemental_study'), 'Wrong evidence cohort')
            valid = record['failure'] is None
            require((valid and isinstance(record['metrics'],dict) and set(record['artifacts'])=={'.adj','.png'})
                    or (not valid and record['metrics'] is None and record['artifacts']=={}), 'Failure/graph contradiction')
            oid = rid if cohort=='primary' else cell['original_run_id']
            if cohort=='primary': original_cells[rid]=cell
            else:
                require(oid in original_cells and all(cell[k]==original_cells[oid][k] for k in DIMENSIONS), 'Supplement not linked to its original cell')
            row = dict(cohort=cohort,run_id=rid,original_run_id=oid, **{k:cell[k] for k in DIMENSIONS},
                       primary_member=cohort=='primary', eligible_primary_graph=cohort=='primary' and valid,
                       valid_graph=valid,failure=record['failure'],terminal_recorded=True,
                       selected_for_896_view=valid and (cohort=='primary' or selection.get(oid,{}).get('run_id')==rid),
                       selection_basis='fixed_primary' if cohort=='primary' else 'post_collection_supplement_not_primary',
                       request_receipts=len(record['requests']), record_path=name, record_sha256=entries[name]['sha256'],
                       adj_path=None,png_path=None,adj_sha256=None,png_sha256=None, nodes=None,edges=None,empty_graph=None,
                       **(record['metrics'] or {}))
            if valid:
                for extension, expected_hash in record['artifacts'].items():
                    artifact = str(Path(name).with_suffix(extension)).replace('\\','/')
                    require(artifact in entries and entries[artifact]['sha256']==expected_hash, 'Artifact binding differs')
                    data = trusted_read(root,artifact,entries)
                    row[extension[1:]+'_path']=artifact; row[extension[1:]+'_sha256']=expected_hash
                    if extension=='.adj':
                        graph=graph_from_bytes(data)
                        require(set(graph)=={str(i) for i in range(50)} and nx.number_of_selfloops(graph)==0, 'Invalid roster or loops')
                        require(math.isclose(nx.density(graph),record['metrics']['density'],abs_tol=1e-12,rel_tol=0), 'Density differs')
                        row.update(nodes=len(graph),edges=graph.number_of_edges(),empty_graph=graph.number_of_edges()==0)
                    else:
                        with Image.open(io.BytesIO(data)) as picture: picture.verify()
                metrics.update(record['metrics'])
            rows.append(row); records[rid]=dict(cell=cell,failure=record['failure'],metrics=record['metrics'])
    primary=[r for r in rows if r['primary_member']]
    analysis=spec['analysis']
    expected={(m,t,c,l,seed) for m,t,(c,l),seed in itertools.product(analysis['models'],analysis['methods'],analysis['conditions'],analysis['schedule_seeds'])}
    require(len(primary)==896 and {(r['model'],r['method'],r['culture'],r['language'],r['seed']) for r in primary}==expected, 'Primary schedule coverage differs')
    counts=Counter((r['cohort'],r['valid_graph']) for r in rows)
    require(counts == {('primary',True):890,('primary',False):6,('supplement_initial',True):4,
                      ('supplement_initial',False):2,('supplement_completion',True):2,('supplement_completion',False):1}, 'Dataset counts differ')
    failed_primary={r['run_id'] for r in primary if not r['valid_graph']}
    require(set(selection)==failed_primary and len({v['run_id'] for v in selection.values()})==6, 'Selection does not cover exactly the six original failures')
    for oid, item in selection.items():
        chosen=next(r for r in rows if r['run_id']==item['run_id'])
        require(chosen['valid_graph'] and not chosen['primary_member'] and chosen['original_run_id']==oid
                and chosen['record_path'].startswith('outputs/'+item['source']+'/'), 'Invalid supplemental selection')
    original_rows=json.loads(trusted_read(root,PRIMARY+'analysis_rows.json',entries))
    require(len(original_rows)==896 and {r['run_id'] for r in original_rows}==set(original_cells), 'Precomputed primary rows differ')
    for old in original_rows:
        source=records[old['run_id']]
        require(old['valid_graph']==(source['failure'] is None) and old['failure']==source['failure']
                and all(old.get(k)==v for k,v in (source['metrics'] or {}).items()), 'Precomputed metric differs')
    return rows, dict(contract_sha256=CONTRACT,frozen_source_files=len(spec['sources']),runtime=runtime,
                     primary_schedule_groups=len(expected)//8, primary_empty_graphs=sum(r['empty_graph'] is True for r in primary),
                     metric_columns=sorted(metrics))

def csv_bytes(rows, columns=None):
    columns=columns or list(dict.fromkeys(k for r in rows for k in r))
    stream=io.StringIO(newline=''); writer=csv.DictWriter(stream,fieldnames=columns,lineterminator='\n')
    writer.writeheader(); writer.writerows(rows)
    return stream.getvalue().encode('utf8')

def prepare(root, archives):
    packages=[]
    for name, expected in PINS.items():
        path=archives/name
        require(file_digest(path)==expected, 'Release manifest differs: '+name)
        manifest=json.loads(path.read_text(encoding='utf8'))
        require(Path(manifest['asset']).name==manifest['asset'], 'Unsafe asset path')
        packages.append((archives/manifest['asset'],manifest))
    installation=install_packages(root,packages)
    entries=installation['entries']
    for name in PINS:
        target=destination(root,'study_amendments/20261105/'+name)
        data=(archives/name).read_bytes()
        if target.exists(): require(target.read_bytes()==data,'Existing manifest differs')
        else: create_once(target,io.BytesIO(data),dict(bytes=len(data),sha256=digest(data)))
    rows, checks=catalog(root,entries)
    views=analysis_views(rows)
    selected=[r for r in views['supplemental_valid_graphs.csv'] if r['selected_for_896_view']]
    files={name:csv_bytes(table) for name,table in views.items()}
    files['selected_supplement_map.csv']=csv_bytes(selected)
    files['failures.json']=json_bytes(views['failures.csv'])
    files['DATA_DICTIONARY.md']=(
        '# Analysis Input Dictionary\n\nOne row is one recorded graph-generation attempt, not an API request.\n'
        'All paths are relative to the project root. Raw evidence stays in its original runs folder.\n\n'
        '- primary_member: original fixed study, including its six failures.\n'
        '- eligible_primary_graph: a valid original graph; use primary_attempts.csv to retain missingness.\n'
        '- valid_graph: frozen validator accepted the observation; empty graphs can be valid.\n'
        '- selected_for_896_view: descriptive inventory only, never a pooled primary sample.\n'
        '- original_run_id: links every supplemental draw to the unchanged primary failed record.\n'
        '- failure: recorded validation failure, not an unattempted or zero-valued graph.\n'
        '- *_sha256: byte identities for receipts and artifacts.\n'
        '- request_receipts: received provider replies used by the observation, not graph count.\n'
        '- Metrics are copied from verified terminal records, not recalculated statistically here.\n'
        '  Empty CSV cells mean missing/undefined, never zero-fill. density and clustering are [0,1].\n'
        '  Primary endpoints: density for global; avg_clustering_coef for the other three methods.\n'
        '- Supplemental observations are selectively completed and excluded from primary inference.\n'
    ).encode()
    files['README.md']=(
        '# Analysis-Ready Handoff\n\nPrepared offline; no generation, credentials, ledger queries or new inference.\n\n'
        '| Table | Rows | Use |\n| --- | ---: | --- |\n'
        '| primary_attempts.csv | 896 | Primary input; retain six failures as missing |\n'
        '| primary_valid_graphs.csv | 890 | Original valid graph inventory, not a missingness replacement |\n'
        '| supplemental_attempts.csv | 9 | All extra draws, including three failures |\n'
        '| supplemental_valid_graphs.csv | 6 | Separate post-collection graphs |\n'
        '| all_valid_graphs_descriptive_only.csv | 896 | Descriptive/visualization inventory, NOT primary inference |\n'
        '| all_attempts.csv | 905 | Full provenance catalog |\n| failures.csv / failures.json | 9 | All retained failures |\n'
        '| selected_supplement_map.csv | 6 | Original failure to selected extra graph |\n\n'
        'Graph/PNG/JSON files are unpacked under outputs/final_study/060dcd3a86d5/main/runs/, '
        'outputs/supplement_20261105_v1/runs/, and outputs/supplement_completion_20261105_v1/runs/.\n'
        'Do not move those evidence files into legacy text-files/ or plots/ directories.\n\n'
        'Start with analysis/00_data_readiness.ipynb using the project .venv kernel.\n'
        'Existing primary analysis_rows.json, network_metrics.csv, contrasts and missingness reports '
        'remain in the original main directory, byte-for-byte unchanged.\n'
        'Their 21 complete, two partial and one missing contrasts still need interpretation and assumption review.\n'
        'Do not run the frozen collector CLI as a setup command: its original replay does not recognize '
        'the separately documented billing exception. No new statistical results were calculated here.\n\n'
        'readiness.json seals these generated tables by hash. Re-run scripts/prepare_study_data.py '
        '--archives <verified-release-folder> for a complete installation and integrity check; '
        'different existing evidence is refused, never overwritten.\n'
    ).encode()
    report=dict(status='READY_FOR_ANALYSIS_INPUT_REVIEW',statistical_interpretation_performed=False,
                inference_recomputed=False,paid_calls=0,ledger_access=False, **checks,
                counts={name:len(table) for name,table in views.items()},
                archive_members_verified=installation['verified'],
                source_manifests={name:pin for name,pin in PINS.items()},
                generated_files={name:dict(bytes=len(data),sha256=digest(data)) for name,data in files.items()})
    files['readiness.json']=json_bytes(report)
    output=root/'outputs/analysis_ready/060dcd3a86d5'
    for name,data in files.items():
        target=destination_generated(root,output/name)
        require(not target.exists() or target.read_bytes()==data, 'Prepared output differs; preserve it and use a reviewed new version: '+str(target))
    for name,data in files.items():
        target=destination_generated(root,output/name)
        if not target.exists(): create_once(target,io.BytesIO(data),dict(bytes=len(data),sha256=digest(data)))
    print(json.dumps(dict(status=report['status'],installed_members=installation['installed'],verified_members=installation['verified'],
                          primary_attempts=896,valid_graphs=896,failed_attempts=9,output=str(output))),flush=True)

def destination_generated(root,target):
    require(target.resolve().is_relative_to(root), 'Generated output escapes workspace')
    for part in [target,*target.parents]:
        if part==root.parent: break
        require(not part.is_symlink() and not (part.exists() and getattr(part.lstat(),'st_file_attributes',0)&0x400), 'Redirected output')
    return target

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archives',type=Path,required=True)
    args=parser.parse_args()
    prepare(Path(__file__).resolve().parents[1],args.archives.resolve())
