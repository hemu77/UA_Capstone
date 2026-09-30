"""Offline calibration inspection and an assumption-labelled serial runtime forecast."""
import hashlib
import json
import sqlite3
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean

import networkx as nx
from PIL import Image

import revision224 as study


def runtime_forecast(config, rows, completed_ids):
    """Use measured Luna treatment ratios, never claim other-model timing was measured."""
    expected = {c['run_id'] for c in study.calibration_cells(config)}
    if {r['run_id'] for r in rows} != expected or len(rows) != len(expected):
        return None
    grouped = defaultdict(list)
    for row in rows:
        duration = row['request_roundtrip_seconds']
        if duration is None or duration <= 0:
            return None
        grouped[tuple(row[k] for k in ['model', 'method', 'culture', 'language'])].append(duration)
    durations = {key: mean(values) for key, values in grouped.items()}
    luna = config['calibration']['model']
    total, remaining = 0, 0
    for cell in study.cells(config):
        model, method, country, language = (cell[k] for k in ['model', 'method', 'culture', 'language'])
        seconds = durations[(luna, method, country, language)]
        if model != luna:
            seconds *= durations[(model, method, 'us', 'english')] / durations[(luna, method, 'us', 'english')]
        total += seconds
        if cell['run_id'] not in completed_ids:
            remaining += seconds
    return {'full_study_api_hours': total / 3600,
            'remaining_api_hours': remaining / 3600,
            'remaining_api_hours_with_50_percent_contingency': remaining * 1.5 / 3600}


def request_timing(timings):
    """Do not charge overnight pauses or restart downtime to API runtime estimates."""
    if not timings or any(end < start for start, end in timings):
        raise ValueError('Missing or backward request timestamps; timing cannot be verified.')
    return {'request_roundtrip_seconds': sum((end - start).total_seconds() for start, end in timings),
            'elapsed_span_seconds': (max(end for _, end in timings) - min(start for start, _ in timings)).total_seconds()}


def inspect():
    # These routines audit receipts/ledger and compute controls, without API clients.
    config = study.load_config()
    with study.workflow_lock():
        # Audit and read one locked snapshot, not a mix of pre- and post-resume files.
        folder = study.ROOT / study.STATS
        target = folder / 'calibration_inspection.json'
        study.write_json(target, {'status': 'RUNNING', 'main_generation_authorized': False})
        calibration = study._calibration_report(config)
        analysis = study._analyze(config, folder)
        study.write_json(study.DESTINATION / 'calibration_report.json', calibration)
        study.write_json(folder / 'analysis_report.json', {
            'protocol_sha256': study.digest(config), 'paid_calls': 0, **analysis})
        ledger = study.ROOT / 'outputs/revision_budget_v1/budget.sqlite'
        db = sqlite3.connect(ledger.resolve().as_uri() + '?mode=ro', uri=True)
        try:
            requests = {rid: (json.loads(req), json.loads(resp) if resp else None, status)
                        for rid, req, resp, status in db.execute('SELECT id,request,response,status FROM attempts')}
        finally:
            db.close()
        rows = []
        receipt_hashes = analysis.get('source_receipts_sha256', {})
        completed_ids = {Path(name).stem for name in receipt_hashes}
        for run in calibration['networks']:
            path = study.ROOT / study.RESULTS / (run['run_id'] + '.json')
            record = json.loads(path.read_text(encoding='utf-8'))
            graph = nx.read_adjlist(path.with_suffix('.adj'))
            with Image.open(path.with_suffix('.png')) as png:
                png.verify()
            if len(graph) != 50 or graph.number_of_edges() == 0 or nx.number_of_selfloops(graph):
                raise ValueError('Calibration graph violates full-roster nonempty simple-graph contract.')
            timings = []
            for receipt in record['requests']:
                request, response, status = requests[receipt['request_id']]
                if status != 'received':
                    raise ValueError('A completed network contains an unreceived request.')
                start = datetime.fromisoformat(request['date_utc'])
                end = datetime.fromisoformat(response['date_utc'])
                timings.append((start, end))
            rows.append({**run, 'nodes': len(graph), 'edges': graph.number_of_edges(),
                         'density': record['metrics']['density'],
                         **request_timing(timings), 'png_file_integrity_verified': True})
        report = {
            'status': calibration['status'], 'protocol_sha256': study.digest(config),
            'generation_source_sha256': study.generation_source_hashes(config),
            'execution_source_sha256': study.source_hashes(),
            'inspection_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'receipt_sha256': receipt_hashes, 'verified_networks': len(rows), 'networks': rows,
            'analysis_status': analysis['status'], 'cost_report': 'calibration_report.json in the current preflight directory',
            'runtime_forecast': runtime_forecast(config, rows, completed_ids) if calibration['status'] == 'COMPLETE' else None,
            'assumptions': [
                'Serial API-only forecast sums request/response intervals, excluding pauses between requests, rendering, local processing and post-run analysis. Wall-clock completion can take longer.',
                'Other-model treatment timings use Luna ratios; those other-model multilingual timings have not been measured.',
                'A 50 percent runtime contingency is a planning allowance, not a confidence interval or deadline.',
                'Calibration reuse requires unchanged reviewed protocol and source. Full-study approval remains separate.',
                'Two Luna repetitions and one base run on other models do not establish statistical power or human translation equivalence.',
                'PNG verification checks file integrity, not visual readability or that the drawing represents the saved adjacency list.'
            ],
            'api_calls_by_inspection': 0, 'main_generation_authorized': False,
        }
        study.write_json(study.ROOT / study.STATS / 'calibration_inspection.json', report)
        return report


if __name__ == '__main__':
    result = inspect()
    print(json.dumps({k: result[k] for k in ['status', 'verified_networks', 'runtime_forecast', 'main_generation_authorized']}, indent=2))
