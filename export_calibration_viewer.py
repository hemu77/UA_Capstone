"""Export already-analyzed calibration evidence. Never call a model or alter a receipt.

This is a separate dataset: persona 3 in the old roster is not persona 3 here.
The public export contains graph deltas, not raw prompts, replies or billing logs.
"""
import hashlib
import json
import math
import shutil
from pathlib import Path

import networkx as nx
import revision224 as study


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reviewed_report_hash(completion, relative_path):
    # Review receipts written on Windows must also verify on a POSIX checkout.
    target = str(relative_path).replace('\\', '/')
    matches = [value for key, value in completion['reports_sha256'].items()
               if key.replace('\\', '/') == target]
    if len(matches) != 1:
        raise ValueError('Reviewed report path is missing or ambiguous.')
    return matches[0]


def checked_record(path, planned, roster, receipt_hash, source_contract):
    if sha256(path) != receipt_hash:
        raise ValueError('Receipt differs from the analyzed evidence.')
    record = json.loads(path.read_text(encoding='utf-8'))
    cell = record['cell']
    if (any(cell.get(k) != v for k, v in planned.items() if k != 'evidence_type')
            or cell.get('evidence_type') != 'fresh_exploratory_llm_run'
            or cell.get('roster_sha256') != study.digest(roster)
            or cell.get('frozen_source_sha256') != source_contract
            or cell.get('runtime_versions') != study.runtime_versions()):
        raise ValueError('Receipt does not match the frozen calibration contract.')
    for suffix in ['.adj', '.png']:
        if sha256(path.with_suffix(suffix)) != record['artifact_sha256'][suffix]:
            raise ValueError('Calibration artifact hash mismatch.')
    graph = nx.read_adjlist(path.with_suffix('.adj'))
    metrics, homophily = study.verify_graph(graph, record['events'], roster)
    if metrics != record['metrics'] or homophily != record['homophily']:
        raise ValueError('Recomputed measurements differ from the receipt.')
    return record, graph


def load_calibration(root=study.ROOT):
    """Verify the complete reviewed V5 set without credentials or a billing ledger."""
    root = Path(root)
    config = study.load_config()
    roster = json.loads((root / config['persona_file']).read_text(encoding='utf-8'))
    report_path = root / study.STATS / 'analysis_report.json'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    completion = json.loads((root / study.DESTINATION / 'completion_review.json').read_text(encoding='utf-8'))
    # Bind this derivative to the completed review, not just a mutable folder glob.
    review_key = str(Path(study.STATS) / 'analysis_report.json')
    if (report['protocol_sha256'] != study.digest(config)
            or reviewed_report_hash(completion, review_key) != sha256(report_path)):
        raise ValueError('Analysis report differs from the completed calibration review.')
    expected = {cell['run_id']: cell for cell in study.calibration_cells(config)}
    hashes = report['source_receipts_sha256']
    if set(hashes) != {run_id + '.json' for run_id in expected}:
        raise ValueError('Analyzed calibration coverage is incomplete or unexpected.')
    source_contract = study.generation_source_hashes(config)
    verified = []
    for run_id, planned in expected.items():
        path = root / study.RESULTS / (run_id + '.json')
        record, graph = checked_record(path, planned, roster, hashes[path.name], source_contract)
        verified.append((path, planned, record, graph))
    return config, roster, verified


def export(root=study.ROOT):
    root = Path(root)
    config, roster, verified = load_calibration(root)
    source_contract = study.generation_source_hashes(config)
    runs, copies = [], []
    for path, planned, record, graph in verified:
        run_id = planned['run_id']
        scores = record['homophily']
        runs.append(dict(
            **{k: planned[k] for k in ['run_id', 'model', 'method', 'culture', 'language', 'seed', 'repetition']},
            study='calibration', status='CALIBRATION_NOT_MAIN_STUDY',
            roster_id=study.digest(roster), protocol_sha256=study.digest(config),
            prompt_variant=study.digest(source_contract), language_confound=False,
            source=path.with_suffix('.adj').relative_to(root).as_posix(),
            source_sha256=record['artifact_sha256']['.adj'],
            receipt_sha256=sha256(path),
            png_sha256=record['artifact_sha256']['.png'],
            png_url=f'./data/calibration/{run_id}.png',
            adjacency_url=f'./data/calibration/{run_id}.adj',
            png_provenance='Original calibration PNG and adjacency hashes match the analyzed receipt. Browser layout differs.',
            analysis_version='frozen-v5-recomputed-export', metrics=record['metrics'],
            homophily={key: value[0] for key, value in scores.items() if key != 'age_assortativity'},
            age_assortativity=scores['age_assortativity'],
            edges=sorted(sorted(edge) for edge in graph.edges()),
            events=[{key: event[key] for key in ['step', 'method', 'persona', 'added', 'removed', 'attempts']}
                    for event in record['events']],
            replay_status='actual_recorded_graph_changes'))
        copies.extend(path.with_suffix(suffix) for suffix in ['.adj', '.png'])
    personas = []
    for i, (pid, attributes) in enumerate(sorted(roster.items(), key=lambda pair: int(pair[0]))):
        angle = i * 2 * math.pi / len(roster)
        personas.append(dict(id=pid, attributes=attributes, position=[75*math.cos(angle), 0, 75*math.sin(angle)]))
    data = dict(schema_version=1, dataset='calibration', roster_id=study.digest(roster),
                analysis_version='frozen-v5-recomputed-export', personas=personas, runs=runs,
                models=config['models'], methods=config['methods'], settings=config['main_conditions'],
                planned_repetitions=config['confirmatory_repetitions'], planned_networks=config['main_networks_target'],
                limitations='V5 exploratory calibration: design under revision, not the revised main study. The 896-network target is historical. One designed fictional adult roster; not national populations. Instructions translated; persona values remain English. No human bilingual signoff.',
                verification='Receipt/report/artifact hashes, frozen source contract, replay and recomputed metrics checked offline. Export does not independently reconcile provider billing.')
    public = root / 'viewer/public/data'
    (public / 'calibration').mkdir(parents=True, exist_ok=True)
    # Validate everything above before replacing the usable public dataset.
    for path in copies:
        shutil.copyfile(path, public / 'calibration' / path.name)
    temporary = public / 'calibration.json.tmp'
    temporary.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(',', ':')), encoding='utf-8')
    temporary.replace(public / 'calibration.json')
    return {'networks': len(runs), 'personas': len(personas), 'paid_calls': 0}


if __name__ == '__main__':
    print(json.dumps(export()))
