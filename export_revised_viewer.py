"""Publish only verified V6 graph evidence; never call a model or expose paid logs."""
from collections import Counter
import json
import math
import shutil
from pathlib import Path

import networkx as nx
import calibration_v6 as calibration


def checked_inspection(target, spec):
    report = json.loads((target / 'inspection.json').read_text(encoding='utf-8'))
    expected = {cell['run_id'] + '.json' for cell in calibration.cells(spec)}
    if (report['status'] != 'COMPLETE_CALIBRATION' or report['verified_networks'] != len(expected)
            or report['contract_sha256'] != calibration.frozen.digest(spec)
            or report['inspector_sha256'] != calibration.sha(Path(__file__).with_name('inspect_calibration_v6.py'))
            or set(report['receipt_sha256']) != expected
            or {p.name for p in (target / 'runs').glob('*.json')} != expected):
        raise ValueError('Revised inspection is incomplete, stale or from another contract.')
    for name, digest in report['receipt_sha256'].items():
        if calibration.sha(target / 'runs' / name) != digest:
            raise ValueError('Revised receipt differs from inspected evidence.')
    for name, digest in report['tables_sha256'].items():
        if Path(name).name != name or calibration.sha(target / name) != digest:
            raise ValueError('Revised inspection table changed.')
    return report


def public_run(path, cell, record, graph, spec, roster):
    """Allowlist graph facts, not raw prompts, replies, request IDs or accounting."""
    run_id = cell['run_id']
    return dict(
        **{key: cell[key] for key in ['run_id', 'model', 'method', 'culture', 'language', 'seed', 'repetition']},
        study='revised_calibration', status='CALIBRATION_NOT_MAIN_STUDY',
        roster_id=calibration.frozen.digest(roster), protocol_sha256=spec['protocol_sha256'],
        prompt_variant=calibration.frozen.digest(spec), language_confound=False,
        source=f'outputs/calibration_v6/{calibration.frozen.digest(spec)[:12]}/runs/{run_id}.adj',
        source_sha256=record['artifacts']['.adj'], receipt_sha256=calibration.sha(path),
        png_sha256=record['artifacts']['.png'],
        png_url=f'./data/revised-calibration/{run_id}.png',
        adjacency_url=f'./data/revised-calibration/{run_id}.adj',
        png_provenance='Original V6 PNG/adjacency hashes verified against the replayed receipt. Browser layout differs.',
        analysis_version='revised-v6-replayed-export', metrics=record['metrics'],
        homophily={key.removeprefix('coleman_'): value for key, value in record['metrics'].items() if key.startswith('coleman_')},
        age_assortativity=record['metrics']['age_assortativity'],
        edges=sorted(sorted(edge) for edge in graph.edges()),
        events=[{key: event[key] for key in ['step', 'method', 'persona', 'added', 'removed', 'attempts']}
                for event in record['events']], replay_status='actual_recorded_graph_changes')


def export():
    # Serialize with collection: a partial or changing study is never a fallback.
    with calibration.frozen.workflow_lock():
        spec = calibration.contract()
        target = calibration.folder(spec)
        report = checked_inspection(target, spec)
        calibration.verify_probe()
        roster = calibration.revised.previous.adult_roster()
        manifest = calibration.cells(spec)
        runs, copies = [], []
        for cell in manifest:
            path = target / 'runs' / (cell['run_id'] + '.json')
            record = calibration.verify_receipt(path, cell)
            graph = nx.read_adjlist(path.with_suffix('.adj'))
            runs.append(public_run(path, cell, record, graph, spec, roster))
            copies.extend(path.with_suffix(suffix) for suffix in ['.adj', '.png'])
        cfg = calibration.revised.config()
        counts = Counter((c['model'], c['method'], c['culture'], c['language']) for c in manifest)
        personas = [dict(id=pid, attributes=attributes,
                        position=[75*math.cos(i*2*math.pi/len(roster)), 0, 75*math.sin(i*2*math.pi/len(roster))])
                    for i, (pid, attributes) in enumerate(sorted(roster.items(), key=lambda pair: int(pair[0])))]
        empty = sum(not run['edges'] for run in runs if run['method'] == 'global')
        global_count = sum(run['method'] == 'global' for run in runs)
        data = dict(schema_version=1, dataset='revised_calibration', roster_id=calibration.frozen.digest(roster),
            analysis_version='revised-v6-replayed-export', personas=personas, runs=runs,
            models=cfg['models'], methods=cfg['methods'], settings=cfg['settings'], planned_networks=len(manifest),
            coverage_plan=[dict(model=model, method=method, culture=culture, language=language,
                                repetitions=counts[(model, method, culture, language)])
                           for model in cfg['models'] for method in cfg['methods'] for culture, language in cfg['settings']],
            inspection_sha256=calibration.sha(target/'inspection.json'), main_authorized=False,
            evidence_notice=f'{empty}/{global_count} global graphs contain no ties: valid recorded NONE responses, not missing data. Main-study collection remains on hold.',
            calibration_note='GPT-6-Luna: two repetitions across seven settings. Other models: one per US language/method. No final main-study allocation approved.',
            compliance=dict(initial_decisions=report['first_decisions'], first_failures=report['first_failures']),
            limitations='One designed fictional adult roster, not national populations. Country framing and instruction language are separate; English persona values stay fixed. No human bilingual validation. Empty-graph homophily is undefined, not zero.',
            verification='All 104 receipts, original artifact hashes, exact prompt/parser/event replay and recomputed metrics checked offline. This public derivative excludes raw API replies and does not independently audit provider billing.')
        # Validate all source records before replacing the last usable public manifest.
        public = calibration.ROOT / 'viewer/public/data'
        destination = public / 'revised-calibration'
        destination.mkdir(parents=True, exist_ok=True)
        for path in copies:
            shutil.copyfile(path, destination / path.name)
        temporary = public / 'revised-calibration.json.tmp'
        temporary.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(',', ':')), encoding='utf-8')
        temporary.replace(public / 'revised-calibration.json')
        return dict(networks=len(runs), personas=len(personas), empty_global=empty, paid_calls=0)


if __name__ == '__main__':
    print(json.dumps(export()))
