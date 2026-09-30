"""Offline handoff gate: prove the browser export still matches saved artifacts.

No API client, network requests or source mutation. PNG hashes establish file
integrity, not a pixel-level proof of the depicted graph. The pilot writer saves
its PNG and adjacency from one graph; historical figure provenance is unknown.
"""
import json
import math
from pathlib import Path

import networkx as nx

from export_research_viewer import ROOT, DEMOS, sha256, pilot_records, finite_or_none
from analyze_networks import compute_network_metrics, compute_coleman_homophily, compute_age_assortativity


def validate_run_ids(runs, historical_ids, pilot_ids):
    ids = [run['run_id'] for run in runs]
    actual_pilots = {run['run_id'] for run in runs if run['study'] == 'engineering_pilot'}
    if len(ids) != len(set(ids)) or set(ids) != historical_ids | pilot_ids or actual_pilots != pilot_ids:
        raise ValueError('Export has missing, duplicate, misclassified or unexpected run identities.')


def verify(root=ROOT):
    root = Path(root)
    public = root / 'viewer/public/data/networks.json'
    built = root / 'viewer/dist/data/networks.json'
    if sha256(public) != sha256(built):
        raise ValueError('Browser build is stale: rebuild viewer before handoff.')
    data = json.loads(public.read_text(encoding='utf-8'))
    manifest = json.loads((root / 'stats/revision_v1/manifest.json').read_text(encoding='utf-8'))
    if sha256(public) != manifest['viewer_sha256']:
        raise ValueError('Export no longer matches its manifest.')
    for name, expected in manifest['source_sha256'].items():
        if sha256(root / name) != expected:
            raise ValueError(f'Historical source changed: {name}')
    pilots = {run['run_id']: run for run in pilot_records(root)}
    personas = json.loads((root / 'text-files/us_50_gpt4o_w_interests.json').read_text(encoding='utf-8'))
    exported_people = {person['id']: person['attributes'] for person in data['personas']}
    expected_people = {pid: {key: person[key] for key in DEMOS + ['age']} for pid, person in personas.items()}
    if len(data['personas']) != len(personas) or exported_people != expected_people:
        raise ValueError('Displayed persona attributes differ from the hashed source roster.')
    historical_ids = {Path(name).stem for name in manifest['source_sha256'] if name.endswith('.adj')}
    excluded = set(data['quarantined_runs'])
    if len(excluded) != 16 or len(data['quarantined_runs']) != 16 or not excluded <= historical_ids:
        raise ValueError('Unexpected historical exclusion identities.')
    for run_id in excluded:
        graph = nx.read_adjlist(root / 'text-files' / (run_id + '.adj'))
        if set(graph) == set(personas) and not nx.number_of_selfloops(graph) and graph.number_of_edges():
            raise ValueError('A valid historical graph was unexpectedly excluded.')
    validate_run_ids(data['runs'], historical_ids - excluded, set(pilots))
    checks = []
    for run in data['runs']:
        source = (root / run['source']).resolve()
        if not source.is_relative_to(root.resolve()):
            raise ValueError('Source escapes project directory.')
        if sha256(source) != run['source_sha256']:
            raise ValueError(f'Adjacency hash mismatch: {run["run_id"]}')
        graph = nx.read_adjlist(source)
        canonical = lambda edges: {tuple(sorted(edge)) for edge in edges}
        if set(graph) != {p['id'] for p in data['personas']} or canonical(graph.edges()) != canonical(run['edges']):
            raise ValueError(f'Web/source graph mismatch: {run["run_id"]}')
        for key, value in compute_network_metrics(graph).items():
            actual = run['metrics'][key]
            if (not math.isfinite(value) and actual is not None) or (math.isfinite(value) and (actual is None or not math.isclose(value, actual, abs_tol=1e-12))):
                raise ValueError(f'Metric mismatch: {run["run_id"]}/{key}')
        expected_homophily = {demo: finite_or_none(compute_coleman_homophily(graph, personas, demo)[0]) for demo in DEMOS}
        if run['homophily'] != expected_homophily or run['age_assortativity'] != finite_or_none(compute_age_assortativity(graph, personas)):
            raise ValueError(f'Homophily mismatch: {run["run_id"]}')
        pilot = pilots.get(run['run_id'])
        if pilot:
            for key in ['events', 'homophily', 'age_assortativity', 'metrics', 'png_sha256', 'prompt_variant', 'roster_id']:
                if pilot[key] != run[key]:
                    raise ValueError(f'Pilot receipt/export mismatch: {run["run_id"]}/{key}')
            for folder in ['viewer/public/data/artifacts', 'viewer/dist/data/artifacts']:
                if sha256(root / folder / (run['run_id'] + '.png')) != run['png_sha256']:
                    raise ValueError('Bundled PNG differs from verified receipt.')
        checks.append(dict(run_id=run['run_id'], adjacency_matches=True, metrics_match=True, homophily_matches=True,
                           replay_verified=bool(pilot), png_receipt_hash_verified=bool(pilot),
                           historical_png_graph_parity='not_applicable' if pilot else 'unknown'))
    if len(checks) != 176 + len(pilots) or not 24 <= len(pilots) <= 28:
        raise ValueError('Expected 176 historical + 24 original pilots and up to four completed Luna 5.6 pilots.')
    presentation_path = root / 'viewer/public/data/presentation.json'
    figures_verified = 0
    if not presentation_path.exists():
        raise ValueError('Presentation manifest missing: run render_pilot_figures.py and rebuild.')
    if presentation_path.exists():
        from export_research_viewer import inspect_png
        presentation = json.loads(presentation_path.read_text(encoding='utf-8'))['runs']
        if set(presentation) != set(pilots):
            raise ValueError('Presentation figures do not cover the exact pilot identities.')
        if sha256(presentation_path) != sha256(root / 'viewer/dist/data/presentation.json'):
            raise ValueError('Built presentation manifest is stale.')
        for run_id, figure in presentation.items():
            if figure['png_url'] != f'./data/presentation/{run_id}.png' or figure['source'] != pilots[run_id]['source']:
                raise ValueError('Presentation URL/source does not identify the verified run.')
            if figure['svg_url'] != f'./data/presentation/{run_id}.svg':
                raise ValueError('SVG URL does not identify the verified run.')
            if figure['source_sha256'] != pilots[run_id]['source_sha256'] or figure['edges'] != pilots[run_id]['edges'] or set(figure['nodes']) != set(personas):
                raise ValueError('Presentation manifest differs from source graph.')
            for folder in ['outputs/presentation', 'viewer/public/data/presentation', 'viewer/dist/data/presentation']:
                path = root / folder / (run_id + '.png')
                if sha256(path) != figure['png_sha256'] or not inspect_png(path):
                    raise ValueError('Presentation figure missing, changed or corrupt.')
                if sha256(path.with_suffix('.svg')) != figure['svg_sha256']:
                    raise ValueError('Vector figure missing or changed.')
            figures_verified += 1
    report = dict(status='PASS_ARTIFACT_PARITY_NOT_SCIENTIFIC_VALIDATION', api_calls=0,
                  browser_graphs_verified=len(checks), pilots_verified=sum(check['replay_verified'] for check in checks),
                  persona_attributes_verified=len(personas),
                  presentation_figures_verified=figures_verified,
                  preserved_historical_sources=len(manifest['source_sha256']),
                  historical_quarantined=len(data['quarantined_runs']),
                  api_response_reconciliation='not_performed_by_this_check', checks=checks)
    destination = root / 'outputs/qa/viewer_source_verification.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'checks'}, indent=2))
    return report


if __name__ == '__main__':
    verify()
