"""Prepare the 896-run study; the filename remains a compatible entry point.

Use --prepare to export the adult roster, exact cells and prompts for review.
Use --preflight to exercise the real engine/parser with labelled test replies.
Live execution requires a matching review receipt and explicit --execute.
"""
import argparse
import contextlib
import csv
import hashlib
import io
import itertools
import json
import math
import os
import random
import platform
import tempfile
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

import networkx as nx
import numpy as np
import pandas as pd

import constants_and_utils as shared
import generate_networks as generation
import revision224_prompts as prompts
from analyze_networks import compute_network_metrics, compute_coleman_homophily, compute_age_assortativity
from make_matched_baselines import matched_baselines
from paid_study import clean_numbers, Budget, PaidCaller, credential, OpenAI

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / 'study_protocol_896.json'
DESTINATION = ROOT / 'outputs/revision896_retry_v5_preflight'
RESULTS = Path('outputs/revision896_retry_v5')
STATS = Path('stats/revision896_retry_v5')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def source_hashes():
    files = ['revision224.py', 'revision224_prompts.py', 'generate_networks.py',
             'constants_and_utils.py', 'analyze_networks.py', 'make_matched_baselines.py', 'paid_study.py']
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files}


def runtime_versions():
    return {'python': platform.python_version(), **{name: version(name) for name in
            ['networkx', 'numpy', 'pandas', 'openai', 'matplotlib']}}


def generation_source_hashes(config):
    """Keep the original contract only for an explicitly audited wrapper change.

    This is not a general stale-source override. Prompts, engine, metrics, roster,
    runtime and protocol must remain identical; actual executing hashes are saved
    separately on new paid requests and receipts.
    """
    current = source_hashes()
    path = DESTINATION / 'source_compatibility.json'
    if not path.exists():
        return current
    review = json.loads(path.read_text(encoding='utf-8'))
    expected = dict(source_sha256=current, protocol_sha256=digest(config),
                    roster_sha256=digest(prompts.adult_roster()), runtime_versions=runtime_versions())
    if review.get('reviewed') is not True or any(review.get(k) != v for k, v in expected.items()):
        raise ValueError('Source compatibility review is stale or incomplete.')
    original = review['generation_source_sha256']
    if set(original) != set(current):
        raise ValueError('Source compatibility generation contract is incomplete.')
    for name in current:
        if original[name] == current[name]:
            continue
        if name not in {'paid_study.py', 'revision224.py'}:
            raise ValueError('Source compatibility cannot change prompts or generation/analysis modules.')
        snapshot = DESTINATION / 'source_snapshot' / name
        if hashlib.sha256(snapshot.read_bytes()).hexdigest() != original[name]:
            raise ValueError('Source compatibility snapshot differs from original generation contract.')
    return original


IMPORTED_SOURCE_SHA256 = source_hashes()
REQUEST_FIELDS = ['response_id', 'resolved_model', 'fingerprint', 'usage',
                  'finish_reason', 'parse', 'conservative_charge_usd']


@contextlib.contextmanager
def workflow_lock():
    """One fresh-study writer; the OS releases the lock even after a crash."""
    DESTINATION.mkdir(parents=True, exist_ok=True)
    with (DESTINATION / 'workflow.lock').open('a+b') as handle:
        if handle.tell() == 0:
            handle.write(b'0')
            handle.flush()
        handle.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError('Fresh-study workflow already running; no requests dispatched.') from error
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == 'nt':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def load_config():
    config = json.loads(CONFIG.read_text(encoding='utf-8'))
    expected = {
        'models': ['gpt-4.1', 'gpt-5.6-luna', 'gpt-6-luna', 'gpt-6-sol'],
        'methods': ['global', 'local', 'sequential', 'iterative'],
        'cultures': ['us', 'india', 'japan', 'brazil'],
        'languages': ['english', 'hindi', 'japanese', 'portuguese'],
        'main_conditions': [['us', 'english'], ['india', 'english'], ['japan', 'english'],
                            ['brazil', 'english'], ['us', 'hindi'], ['us', 'japanese'], ['us', 'portuguese']],
        'confirmatory_rosters': 1, 'confirmatory_repetitions': 8, 'expected_personas': 50,
        'iterative_rounds': 3, 'max_attempts_per_request': 3, 'total_budget_usd': None,
        'main_networks_target': 896,
        'calibration': {'model': 'gpt-6-luna', 'repetitions': 2, 'networks': 68},
        'visible_attributes': prompts.DEMOS, 'persona_file': 'text-files/revision224_adults.json',
    }
    for name, value in expected.items():
        if config.get(name) != value:
            raise ValueError(f'Fresh protocol differs at {name}; review before changing the matrix.')
    return config


def cells(config):
    fingerprint = digest(config)
    rows = []
    for model, method, (country, language), rep in itertools.product(
            config['models'], config['methods'], config['main_conditions'], range(config['confirmatory_repetitions'])):
        rows.append(dict(run_id=f'revision896_{fingerprint[:12]}_{method}_{model}_{country}_{language}_s{rep}',
                         model=model, method=method, culture=country, language=language,
                         repetition=rep, seed=11000 + rep, personas=50,
                         protocol_sha256=fingerprint, evidence_type='planned_uncollected'))
    if len(rows) != config['main_networks_target'] or len({r['run_id'] for r in rows}) != len(rows):
        raise ValueError('Planned cells differ from the declared unique study target.')
    return rows


def calibration_cells(config):
    return [cell for cell in cells(config) if
            (cell['model'] == config['calibration']['model'] and cell['repetition'] < config['calibration']['repetitions'])
            or (cell['culture'] == 'us' and cell['language'] == 'english' and cell['repetition'] == 0)]


def ordered_cells(config):
    calibration_ids = {cell['run_id'] for cell in calibration_cells(config)}
    # All 16 model/method base examples first, then inexpensive treatment checks.
    return sorted(cells(config), key=lambda c: (
        c['run_id'] not in calibration_ids,
        not (c['culture'] == 'us' and c['language'] == 'english' and c['repetition'] == 0), c['run_id']))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                     suffix='.tmp', delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n')
        except Exception:
            handle.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare(config):
    roster = prompts.adult_roster()
    path = ROOT / config['persona_file']
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) != roster:
        raise ValueError('Fresh roster was edited; refusing to overwrite it.')
    write_json(path, roster)
    manifest = cells(config)
    write_json(DESTINATION / 'planned_cells.json', manifest)
    write_json(DESTINATION / 'calibration_cells.json', [c for c in ordered_cells(config)
               if c['run_id'] in {r['run_id'] for r in calibration_cells(config)}])
    # A path graph supplies a fixed, explicitly synthetic graph state for the
    # translation review. It does not represent a collected model output.
    graph = nx.path_graph(list(roster))
    catalog, retries = [], []
    for country, language, method in itertools.product(config['cultures'], config['languages'],
            ['global', 'local', 'sequential', 'iterative-add', 'iterative-drop']):
        actor = None if method == 'global' else '0'
        kwargs = dict(curr_pid=actor, culture_context=country, prompt_language=language,
                      num_choices=5 if method in {'local', 'sequential'} else None)
        system = prompts.system_prompt(method, roster, prompts.DEMOS, **kwargs)
        random.seed(224)
        user = prompts.user_prompt(method, roster, list(roster), prompts.DEMOS,
                                  curr_pid=actor, G=graph, prompt_language=language)
        catalog.append(dict(culture=country, language=language, method=method,
                            system=system, user=user, candidate_sha256=hashlib.sha256(user.encode()).hexdigest(),
                            evidence_type='translation_review_fixture'))
        if country == 'us':
            # Capture the real correction path, not a separately translated copy.
            # Retries contain no country wording, so one example per language/action suffices.
            invalid = '0, 1\n1, 0' if method == 'global' else '0'
            valid = {'global': '0, 1', 'local': '1, 2, 3, 4, 5', 'sequential': '1, 2, 3, 4, 5',
                     'iterative-add': '2', 'iterative-drop': '1'}[method]
            parse_args = dict(method=method, G=graph.copy(), curr_pid=actor, num_choices=kwargs['num_choices'])
            with patch.object(shared, 'get_llm_response', side_effect=[invalid, valid]) as call, patch.object(shared.time, 'sleep'):
                _, _, attempts = shared.repeat_prompt_until_parsed('offline-fixture', system, user,
                    generation.update_graph_from_response, parse_args, prompt_language=language)
            retries.append(dict(language=language, method=method, invalid_response=invalid,
                                correction=call.call_args.args[1][-1]['content'], valid_response=valid,
                                attempts=attempts, evidence_type='translation_review_fixture'))
            if method == 'global':
                with patch.object(shared, 'get_llm_response', side_effect=['', valid]) as call, patch.object(shared.time, 'sleep'):
                    shared.repeat_prompt_until_parsed('offline-fixture', system, user,
                        generation.update_graph_from_response, {**parse_args, 'G': graph.copy()}, prompt_language=language)
                retries[-1]['nonresponse_correction'] = call.call_args.args[1][-1]['content']
    write_json(DESTINATION / 'prompt_catalog.json', catalog)
    write_json(DESTINATION / 'retry_catalog.json', retries)
    write_json(DESTINATION / 'preparation.json', dict(status='READY_FOR_OFFLINE_PREFLIGHT',
               protocol_sha256=digest(config), roster_sha256=digest(roster), source_sha256=source_hashes(),
               runtime_versions=runtime_versions(),
               cells=len(manifest), prompt_examples=len(catalog), retry_examples=len(retries), paid_calls=0))
    template = dict(
        protocol_sha256=digest(config), roster_sha256=digest(roster), source_sha256=source_hashes(),
        runtime_versions=runtime_versions(),
        code_review=False, price_review=False, translation_review=False, cost_review=False,
        study_design_review=False, generation_authorized=False, spend_ceiling_usd=None,
        note='Record actual reviews and an explicitly authorized finite ceiling; templates do not authorize execution.')
    write_json(DESTINATION / 'review_template.json', dict(template, scope='main', approved_limit=len(manifest)))
    write_json(DESTINATION / 'calibration_review_template.json', dict(template, scope='calibration',
        approved_limit=config['calibration']['networks'], additional_allowance_usd=5,
        note='Author authorized $5 additional calibration only. Freeze starting ledger total; cumulative ceiling = starting total + 5. Translation/power approval is NOT implied.'))
    return roster, manifest


def verify_graph(graph, events, roster):
    if set(graph) != set(roster) or graph.is_directed() or nx.number_of_selfloops(graph):
        raise ValueError('Graph violates the roster or simple-undirected contract.')
    replay = nx.empty_graph(roster)
    for event in events:
        if event.get('persona') is not None and event['persona'] not in roster:
            raise ValueError('Unknown recorded actor.')
        for a, b in event['removed']:
            if not replay.has_edge(a, b):
                raise ValueError('Trace removed an absent tie.')
            replay.remove_edge(a, b)
        for a, b in event['added']:
            if a == b or a not in roster or b not in roster or replay.has_edge(a, b):
                raise ValueError('Trace added an invalid tie.')
            replay.add_edge(a, b)
    canonical = lambda g: {tuple(sorted(e)) for e in g.edges()}
    if canonical(graph) != canonical(replay):
        raise ValueError('Recorded events do not reproduce the final graph.')
    metrics = clean_numbers(compute_network_metrics(graph))
    homophily = {d: clean_numbers(compute_coleman_homophily(graph, roster, d)) for d in prompts.DEMOS if d != 'age'}
    homophily['age_assortativity'] = clean_numbers(compute_age_assortativity(graph, roster))
    return metrics, homophily


def _preflight(config):
    frozen_sources = source_hashes()
    if frozen_sources != IMPORTED_SOURCE_SHA256:
        raise ValueError('Source changed after import; restart the preflight process.')
    roster, manifest = prepare(config)
    rows, controls = [], []
    real_retry = shared.repeat_prompt_until_parsed

    def fixture_request(model, system, user, parse, parse_args, **kwargs):
        # Deliberately artificial replies test plumbing. They are never exported
        # as model results, priced as API usage or used to answer research questions.
        payload = json.loads(user)
        ids = [r[0] for r in payload['candidates']]
        method = parse_args['method']
        if method == 'global':
            pairs = [sorted((a, b), key=int) for a, b in zip(ids[:-1], ids[1:])]
            response = '\n'.join(', '.join(pair) for pair in pairs)
        else:
            count = parse_args.get('num_choices') or 1
            if len(ids) < count:
                raise ValueError('Prompt has too few eligible candidates.')
            response = ', '.join(ids[:count])
        with patch.object(shared, 'get_llm_response', return_value=response):
            return real_retry(model, system, user, parse, parse_args, **kwargs)

    for cell in manifest:
        random.seed(cell['seed'])
        np.random.seed(cell['seed'])
        events = []
        with patch.object(generation, 'get_system_prompt', prompts.system_prompt), \
                patch.object(generation, 'get_user_prompt', prompts.user_prompt), \
                patch.object(generation, 'repeat_prompt_until_parsed', fixture_request), \
                patch.object(shared, 'OpenAI', side_effect=AssertionError('Offline preflight cannot create an API client')), \
                contextlib.redirect_stdout(io.StringIO()):
            graph, *_ = generation.generate_network(cell['method'], prompts.DEMOS, roster, list(roster),
                cell['model'], mean_choices=5, num_iter=3, culture_context=cell['culture'],
                prompt_language=cell['language'], events=events)
        metrics, homophily = verify_graph(graph, events, roster)
        rows.append(dict(**{**cell, 'evidence_type': 'offline_parser_fixture'}, nodes=len(graph),
                         edges=graph.number_of_edges(), events=len(events), passed=True,
                         metrics=metrics, homophily=homophily))
        if cell['model'] == config['models'][0] and cell['culture'] == 'us' and cell['language'] == 'english' and cell['repetition'] == 0:
            for name, control, complete in matched_baselines(graph, roster, seed=224,
                    categorical=[d for d in prompts.DEMOS if d != 'age']):
                if len(control) != 50 or control.number_of_edges() != graph.number_of_edges():
                    raise ValueError('Matched baseline violates node or edge count.')
                controls.append(dict(method=cell['method'], baseline=name, rewiring_target_reached=complete,
                                     metrics=clean_numbers(compute_network_metrics(control)),
                                     evidence_type='offline_baseline_fixture'))
    write_json(DESTINATION / 'fixture_results.json', rows)
    write_json(DESTINATION / 'baseline_fixtures.json', controls)
    with (DESTINATION / 'verification_summary.csv').open('w', newline='', encoding='utf-8') as handle:
        fields = ['run_id', 'model', 'method', 'culture', 'language', 'repetition', 'nodes', 'edges', 'events', 'passed', 'evidence_type']
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    if source_hashes() != frozen_sources:
        raise ValueError('Source changed during preflight; rerun before reviewing.')
    report = dict(status='READY_FOR_ASTRA_CODE_REVIEW', planned_networks=len(manifest),
                  offline_fixture_cells_checked=len(rows), offline_baseline_fixtures=len(controls),
                  paid_calls=0, dollars_spent=0, generation_authorized=False,
                  protocol_sha256=digest(config), roster_sha256=digest(roster), source_sha256=frozen_sources,
                  runtime_versions=runtime_versions(),
                  remaining=['Astra review', 'Bilingual translation review', 'Fresh token/cost calibration and model access'],
                  limitation='Fixture replies validate code flow only. No new LLM networks have been collected.')
    write_json(DESTINATION / 'report.json', report)
    return report


def preflight(config):
    with workflow_lock():
        return _locked_preflight(config)


def _locked_preflight(config):
    # Invalidate an earlier success BEFORE any preparation or generation. A
    # failed rerun must not leave yesterday's green report beside today's files.
    report = dict(status='RUNNING', protocol_sha256=digest(config),
                  generation_authorized=False, paid_calls=0)
    write_json(DESTINATION / 'report.json', report)
    try:
        return _preflight(config)
    except Exception as error:
        write_json(DESTINATION / 'report.json', {**report, 'status': 'FAILED', 'error_type': type(error).__name__})
        raise


def execution_review(config, limit, scope='main'):
    """A reviewed file version must be the version that actually gets run."""
    report = json.loads((DESTINATION / 'report.json').read_text(encoding='utf-8'))
    if scope not in {'main', 'calibration'}:
        raise ValueError('Unknown execution scope.')
    review = json.loads((DESTINATION / ('calibration_review.json' if scope == 'calibration' else 'review.json')).read_text(encoding='utf-8'))
    expected = dict(protocol_sha256=digest(config), source_sha256=source_hashes(),
                    roster_sha256=digest(prompts.adult_roster()), runtime_versions=runtime_versions())
    if report.get('status') != 'READY_FOR_ASTRA_CODE_REVIEW' or report.get('offline_fixture_cells_checked') != config['main_networks_target']:
        raise ValueError('Complete the fresh offline preflight before execution.')
    for name, value in expected.items():
        if report.get(name) != value or review.get(name) != value:
            raise ValueError(f'Review or preflight is stale at {name}.')
    if review.get('scope') != scope:
        raise ValueError('Review scope does not authorize this execution mode.')
    gates = ['code_review', 'price_review', 'generation_authorized']
    if scope == 'main':
        gates += ['translation_review', 'cost_review', 'study_design_review']
    for gate in gates:
        if review.get(gate) is not True:
            raise ValueError(f'Review gate pending: {gate}.')
    if scope == 'main':
        calibrated = json.loads((DESTINATION / 'calibration_report.json').read_text(encoding='utf-8'))
        if (calibrated.get('status') != 'COMPLETE'
                or calibrated.get('completed_calibration_networks') != config['calibration']['networks']
                or calibrated.get('protocol_sha256') != expected['protocol_sha256']
                or calibrated.get('source_sha256') != expected['source_sha256']
                or calibrated.get('unresolved_attempts') != 0):
            raise ValueError('Main study requires a current complete calibration report with no unresolved charges.')
    ceiling = review.get('spend_ceiling_usd')
    approved_limit = review.get('approved_limit')
    maximum = config['calibration']['networks'] if scope == 'calibration' else config['main_networks_target']
    if type(limit) is not int or type(approved_limit) is not int or not 1 <= limit <= approved_limit <= maximum:
        raise ValueError('Requested cell count exceeds the reviewed execution scope.')
    if type(ceiling) not in {int, float} or not math.isfinite(ceiling) or ceiling <= 0:
        raise ValueError('Review requires an explicit finite cumulative spending ceiling.')
    if scope == 'calibration':
        starting = review.get('starting_ledger_usd')
        if (type(starting) not in {int, float} or not math.isfinite(starting) or starting < 0
                or review.get('additional_allowance_usd') != 5
                or not math.isclose(ceiling, starting + 5, abs_tol=1e-9)):
            raise ValueError('Calibration ceiling must equal its frozen starting ledger total plus the authorized $5.')
    return {**expected, 'generation_source_sha256': generation_source_hashes(config),
            'spend_ceiling_usd': ceiling, 'minimum_ledger_usd': review.get('starting_ledger_usd', 0)}


def execute(config, limit, scope='main'):
    with workflow_lock():
        return _execute(config, limit, scope)


def verify_receipt(path, cell, roster, budget):
    """Verify evidence AND its charges before trusting a completed-run marker."""
    record = json.loads(path.read_text(encoding='utf-8'))
    if record['cell'] != cell:
        raise ValueError('Saved run differs from the frozen review; resume stopped.')
    execution_sources = record.get('execution_source_sha256')
    if execution_sources is not None and execution_sources != source_hashes():
        raise ValueError('Receipt execution provenance differs from reviewed source.')
    if cell['frozen_source_sha256'] != source_hashes():
        if cell['frozen_source_sha256'] != generation_source_hashes(load_config()):
            raise ValueError('Receipt generation contract has no current compatibility review.')
    for suffix in ['.adj', '.png']:
        if hashlib.sha256(path.with_suffix(suffix).read_bytes()).hexdigest() != record['artifact_sha256'][suffix]:
            raise ValueError('Saved artifact changed; resume stopped.')
    graph = nx.read_adjlist(path.with_suffix('.adj'))
    metrics, homophily = verify_graph(graph, record['events'], roster)
    if metrics != record['metrics'] or homophily != record['homophily']:
        raise ValueError('Saved metrics differ from graph; resume stopped.')
    requests = record['requests']
    if not requests or len({r['request_id'] for r in requests}) != len(requests):
        raise ValueError('Missing or duplicate paid request receipts.')
    attempts = [event['attempts'] for event in record['events']]
    retained = record.get('retained_abandoned_request_ids', [])
    if len(set(retained)) != len(retained):
        raise ValueError('Duplicate retained abandoned request receipts.')
    ledger_ids = {row[0] for row in budget.db.execute(
        "SELECT id FROM attempts WHERE phase='main' AND json_extract(request, '$.cell.run_id')=?",
        (cell['run_id'],))}
    if (any(type(count) is not int or not 1 <= count <= 3 for count in attempts)
            or sum(attempts) != len(requests)
            or ledger_ids != {r['request_id'] for r in requests} | set(retained)):
        raise ValueError('Request receipts, decision attempts and ledger rows are incomplete or inconsistent.')
    linked = set()
    for ordinal, receipt in enumerate(requests):
        request_id = receipt['request_id']
        paid = budget.cached(request_id)
        if paid is None or receipt != {'request_id': request_id, **{k: paid.get(k) for k in REQUEST_FIELDS}}:
            raise ValueError('Paid receipt differs from ledger; reconcile before execution.')
        phase, cost, raw = budget.db.execute('SELECT phase,cost,request FROM attempts WHERE id=?', (request_id,)).fetchone()
        request = json.loads(raw)
        actual_sources = request.get('execution_source_sha256')
        if (actual_sources is not None and actual_sources != execution_sources
                or request.get('replaces_request_id') and actual_sources is None):
            raise ValueError('Request execution provenance is missing or differs from its receipt.')
        identity = json.dumps([cell, ordinal, request['model'], request['messages'], request['settings']], sort_keys=True)
        original_id = hashlib.sha256(identity.encode()).hexdigest()
        expected_id = budget.replacement_id(original_id, request)
        if expected_id != original_id:
            if request.get('replaces_request_id') != original_id:
                raise ValueError('Replacement receipt is missing its retained original request.')
            linked.add(original_id)
        if (phase != 'main' or request['cell'] != cell or request['ordinal'] != ordinal
                or request_id != expected_id
                or not math.isclose(cost, paid['conservative_charge_usd'], abs_tol=1e-12)):
            raise ValueError('Paid ledger identity or charge differs from receipt.')
    if linked != set(retained):
        raise ValueError('Replacement receipts do not account for every retained abandoned request.')
    if budget.spent() + 1e-9 < record['cumulative_conservative_charge_usd']:
        raise ValueError('Ledger spending fell below a completed receipt; reconcile before execution.')
    return record, graph


def analyze(config):
    """Offline fresh-only comparisons; never substitute historical graphs."""
    with workflow_lock():
        folder = ROOT / STATS
        report = {'status': 'RUNNING', 'protocol_sha256': digest(config), 'paid_calls': 0}
        write_json(folder / 'analysis_report.json', report)
        try:
            result = _analyze(config, folder)
        except Exception as error:
            write_json(folder / 'analysis_report.json', {**report, 'status': 'FAILED', 'error_type': type(error).__name__})
            raise
        write_json(folder / 'analysis_report.json', {**report, **result})
        return result


def contrast_pairs(rows):
    """Use identical pairing rules for observed results and missing-cell coverage."""
    dimensions = ['model', 'method', 'culture', 'language', 'repetition', 'roster_sha256']
    for left, right in itertools.combinations(rows, 2):
        differences = [key for key in dimensions if left.get(key) != right.get(key)]
        if len(differences) != 1:
            continue
        dimension = differences[0]
        if dimension in {'culture', 'language'}:
            reference = 'us' if dimension == 'culture' else 'english'
            if left[dimension] != reference and right[dimension] != reference:
                continue
            if right[dimension] == reference:
                left, right = right, left
        elif dimension not in {'model', 'method'}:
            continue
        elif right[dimension] < left[dimension]:
            left, right = right, left
        yield left, right, dimension


def contrast_labels(left, right, dimension):
    return {**{key: left[key] for key in ['model', 'method', 'culture', 'language']},
            'dimension': dimension, 'reference_level': left[dimension], 'comparison_level': right[dimension]}


def _analyze(config, folder):
    paths = sorted((ROOT / RESULTS).glob('*.json'))
    if not paths:
        return {'status': 'NOT_COLLECTED', 'analyzed_networks': 0, 'planned_networks': config['main_networks_target']}
    roster = json.loads((ROOT / config['persona_file']).read_text(encoding='utf-8'))
    if roster != prompts.adult_roster():
        raise ValueError('Analysis roster differs from the declared fresh roster.')
    expected = {cell['run_id']: cell for cell in cells(config)}
    records, graphs, source_receipts = [], {}, {}
    budget = Budget(ROOT / 'outputs/revision_budget_v1/budget.sqlite', require_existing=True)
    try:
        contract = None
        for path in paths:
            cell = json.loads(path.read_text(encoding='utf-8'))['cell']
            planned = expected.get(path.stem)
            if planned is None or any(cell.get(k) != v for k, v in planned.items() if k != 'evidence_type'):
                raise ValueError('Analysis contains an unplanned cell.')
            if cell.get('evidence_type') != 'fresh_exploratory_llm_run' or cell.get('roster_sha256') != digest(roster):
                raise ValueError('Analysis accepts only fresh LLM receipts on the declared roster.')
            current = (cell['frozen_source_sha256'], cell['runtime_versions'])
            if contract is not None and contract != current:
                raise ValueError('Cannot pool different generation source/runtime versions.')
            contract = current
            record, graph = verify_receipt(path, cell, roster, budget)
            records.append(record)
            graphs[cell['run_id']] = graph
            source_receipts[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    finally:
        budget.db.close()

    def measures(graph):
        return clean_numbers({**compute_network_metrics(graph),
            **{'coleman_' + d: compute_coleman_homophily(graph, roster, d)[0]
               for d in prompts.DEMOS if d != 'age'},
            'age_assortativity': compute_age_assortativity(graph, roster)})

    rows, groups, controls, contrasts = [], [], [], []
    for record in records:
        cell = record['cell']
        graph = graphs[cell['run_id']]
        meta = {k: cell[k] for k in ['run_id', 'model', 'method', 'culture', 'language', 'repetition', 'seed', 'roster_sha256']}
        values = measures(graph)
        rows.append({**meta, 'nodes': len(graph), 'edges': graph.number_of_edges(), **values})
        for attribute in prompts.DEMOS:
            if attribute != 'age':
                groups.extend({**meta, 'attribute': attribute, **group}
                              for group in record['homophily'][attribute][1])
        for name, control, complete in matched_baselines(graph, roster, seed=cell['seed'],
                categorical=[d for d in prompts.DEMOS if d != 'age']):
            if set(control) != set(roster) or control.number_of_edges() != graph.number_of_edges():
                raise ValueError('Fresh control differs in roster or edge count.')
            control_values = measures(control)
            target = folder / 'baselines' / (cell['run_id'] + '_' + name + '.adj')
            target.parent.mkdir(parents=True, exist_ok=True)
            nx.write_adjlist(control, target)
            controls.append({**meta, 'baseline': name, 'baseline_seed': cell['seed'],
                'evidence_type': 'offline_synthetic_control', 'nodes': len(control),
                'edges': control.number_of_edges(), 'rewiring_target_reached': complete,
                'source_receipt_sha256': source_receipts[cell['run_id'] + '.json'],
                **control_values,
                **{'observed_minus_control_' + key: (values[key] - value if values[key] is not None and value is not None else None)
                   for key, value in control_values.items()}})

    # Each row is one paired repetition, not a p-value or an independent edge.
    # No cross-condition mean can hide a missing partner or an undefined score.
    measure_names = list(measures(graphs[rows[0]['run_id']]))
    for a, b, dimension in contrast_pairs(rows):
        for metric in measure_names:
            contrasts.append({**contrast_labels(a, b, dimension),
                'reference_run': a['run_id'], 'comparison_run': b['run_id'], 'repetition': a['repetition'],
                'metric': metric, 'primary_metric': metric in config['primary_metrics'],
                'reference_value': a[metric], 'comparison_value': b[metric],
                'comparison_minus_reference': b[metric] - a[metric] if a[metric] is not None and b[metric] is not None else None})
    pd.DataFrame(rows).to_csv(folder / 'network_metrics.csv', index=False)
    pd.DataFrame(groups).to_csv(folder / 'group_homophily.csv', index=False)
    pd.DataFrame(controls).to_csv(folder / 'matched_controls.csv', index=False)
    paired = pd.DataFrame(contrasts, columns=['model', 'method', 'culture', 'language', 'dimension', 'reference_run', 'comparison_run', 'reference_level',
        'comparison_level', 'repetition', 'metric', 'primary_metric', 'reference_value', 'comparison_value',
        'comparison_minus_reference'])
    paired.to_csv(folder / 'paired_contrasts.csv', index=False)
    keys = ['model', 'method', 'culture', 'language', 'dimension', 'reference_level', 'comparison_level', 'metric', 'primary_metric']
    paired['comparison_minus_reference'] = pd.to_numeric(paired['comparison_minus_reference'])
    summaries = paired.groupby(keys)['comparison_minus_reference'].agg(
        n_pairs_present='size', n_pairs_defined='count', mean='mean', sample_sd='std', minimum='min', maximum='max').reset_index()
    planned = pd.DataFrame([{**contrast_labels(a, b, dimension), 'metric': metric,
                             'primary_metric': metric in config['primary_metrics']}
        for a, b, dimension in contrast_pairs([c for c in cells(config) if c['repetition'] == 0])
        for metric in measure_names]).drop_duplicates(keys)
    summaries = planned.merge(summaries, on=keys, how='left', validate='one_to_one')
    for column in ['n_pairs_present', 'n_pairs_defined']:
        summaries[column] = summaries[column].fillna(0).astype(int)
    summaries['planned_repetitions'] = config['confirmatory_repetitions']
    summaries.to_csv(folder / 'contrast_summary.csv', index=False)
    return {'status': 'COMPLETE' if len(rows) == config['main_networks_target'] else 'PARTIAL', 'analyzed_networks': len(rows),
        'planned_networks': config['main_networks_target'], 'control_graphs': len(controls), 'source_receipts_sha256': source_receipts,
        'analysis_source_sha256': source_hashes(), 'runtime_versions': runtime_versions(),
        'limitation': 'Eight planned repetitions on one roster do not guarantee precision or population validity. Partial calibration is not a full study. Controls are not empirical realism evidence.'}


def calibration_report(config):
    """Audit calibration usage offline; forecast only after all selected runs exist."""
    with workflow_lock():
        target = DESTINATION / 'calibration_report.json'
        write_json(target, {'status': 'RUNNING', 'generation_authorized': False})
        try:
            result = _calibration_report(config)
        except Exception as error:
            write_json(target, {'status': 'FAILED', 'error_type': type(error).__name__, 'generation_authorized': False})
            raise
        write_json(target, result)
        return result


def _calibration_report(config):
    selected = calibration_cells(config)
    selected_ids = {cell['run_id'] for cell in selected}
    rows, completed = [], set()
    budget = Budget(ROOT / 'outputs/revision_budget_v1/budget.sqlite', require_existing=True)
    try:
        roster = prompts.adult_roster()
        for planned in cells(config):
            path = ROOT / RESULTS / (planned['run_id'] + '.json')
            if not path.exists():
                continue
            cell = {**planned, 'phase': 'main', 'evidence_type': 'fresh_exploratory_llm_run',
                    'frozen_source_sha256': generation_source_hashes(config), 'roster_sha256': digest(roster),
                    'runtime_versions': runtime_versions()}
            record, _ = verify_receipt(path, cell, roster, budget)
            completed.add(cell['run_id'])
            if cell['run_id'] not in selected_ids:
                continue
            requests = record['requests']
            retained_cost = sum(budget.db.execute('SELECT cost FROM attempts WHERE id=?', (rid,)).fetchone()[0]
                                for rid in record.get('retained_abandoned_request_ids', []))
            rows.append({**planned, 'requests': len(requests),
                'prompt_tokens': sum(r['usage']['prompt_tokens'] for r in requests),
                'completion_tokens': sum(r['usage']['completion_tokens'] for r in requests),
                'parse_failures': sum(r['parse'].get('valid') is False for r in requests if r['parse']),
                'abnormal_finishes': sum(r['finish_reason'] != 'stop' for r in requests),
                'retained_reservation_usd': retained_cost,
                'conservative_charge_usd': retained_cost + sum(r['conservative_charge_usd'] for r in requests)})
        unresolved, charged = 0, 0
        for status, cost, raw in budget.db.execute("SELECT status,cost,request FROM attempts WHERE phase='main'"):
            if json.loads(raw)['cell']['run_id'] in selected_ids:
                unresolved += status not in {'received', 'abandoned'}
                charged += cost
        shared_unresolved = budget.unresolved_count()
        abandoned_count, abandoned_cost = budget.db.execute("SELECT COUNT(*),COALESCE(SUM(cost),0) FROM attempts WHERE status='abandoned'").fetchone()
        spent = budget.spent()
    finally:
        budget.db.close()
    report = dict(status='COMPLETE' if len(rows) == len(selected) and not unresolved else 'PARTIAL' if rows or charged else 'NOT_COLLECTED',
        planned_calibration_networks=len(selected), completed_calibration_networks=len(rows),
        protocol_sha256=digest(config), source_sha256=source_hashes(),
        cumulative_ledger_usd=spent, calibration_charged_or_reserved_usd=charged,
        unresolved_attempts=unresolved, networks=rows, full_study_forecast_usd=None,
        shared_ledger_unresolved_attempts=shared_unresolved,
        shared_ledger_abandoned_attempts=abandoned_count, shared_ledger_abandoned_reservation_usd=abandoned_cost,
        generation_authorized=False, api_calls_by_this_report=0,
        limitations=['Two treatment repetitions on one model cannot justify power across four models.',
                     'Other-model multilingual/country execution is not calibrated by their US-English examples.',
                     'Reuse toward 896 is provisional until bilingual review; changed prompts or protocol require fresh generation, not relabeling.',
                     'Forecast transfers Luna treatment cost ratios to other models; this is an assumption, not measured cross-model treatment usage.',
                     'No cache discounts assumed; ledger amounts include conservative input allowance.',
                     'Calibration cannot authorize the main study or certify translation/population validity.'])
    if report['status'] == 'COMPLETE':
        data = pd.DataFrame(rows)
        costs = data.groupby(['model', 'method', 'culture', 'language'])['conservative_charge_usd'].mean()
        luna = config['calibration']['model']
        forecast, remaining = 0, 0
        for model, method, (country, language) in itertools.product(config['models'], config['methods'], config['main_conditions']):
            if model == luna:
                per_graph = costs[(model, method, country, language)]
            else:
                ratio = costs[(luna, method, country, language)] / costs[(luna, method, 'us', 'english')]
                per_graph = costs[(model, method, 'us', 'english')] * ratio
            forecast += per_graph * config['confirmatory_repetitions']
            missing = sum(cell['run_id'] not in completed for cell in cells(config)
                          if (cell['model'], cell['method'], cell['culture'], cell['language']) == (model, method, country, language))
            remaining += per_graph * missing
        report.update(full_study_forecast_usd=float(forecast),
            verified_completed_study_networks=len(completed), forecast_remaining_usd=float(remaining),
            forecast_cumulative_with_20_percent_remaining_contingency_usd=float(spent + remaining * 1.2))
    return clean_numbers(report)


def _execute(config, limit, scope='main'):
    frozen = execution_review(config, limit, scope)
    if source_hashes() != IMPORTED_SOURCE_SHA256:
        raise ValueError('Source changed after import; restart the execution process.')
    roster = json.loads((ROOT / config['persona_file']).read_text(encoding='utf-8'))
    if digest(roster) != frozen['roster_sha256']:
        raise ValueError('Roster differs from the reviewed version.')
    if type(limit) is not int or not 1 <= limit <= config['main_networks_target']:
        raise ValueError('Execution limit exceeds the study target.')
    destination = ROOT / RESULTS
    destination.mkdir(parents=True, exist_ok=True)
    # Reuse the ORIGINAL ledger; removing $50 must not reset historical spending.
    ceiling = frozen['spend_ceiling_usd']
    budget = Budget(ROOT / 'outputs/revision_budget_v1/budget.sqlite', total_cap=ceiling,
                    pilot_cap=min(5, ceiling), require_existing=True, allow_higher_cap=True)
    client = None
    try:
        previous = json.loads((ROOT / 'outputs/revision_budget_v1/pilot_summary.json').read_text(encoding='utf-8'))
        if budget.spent() + 1e-9 < frozen.get('minimum_ledger_usd', 0):
            raise ValueError('Ledger spending is below the calibration authorization starting total.')
        if budget.spent('pilot') + 1e-9 < previous['conservative_charge_or_reservation_usd']:
            raise ValueError('Ledger spending is below the verified historical pilot total; reconcile before execution.')
        manifest = ordered_cells(config)
        manifest = [{**planned, 'phase': 'main', 'evidence_type': 'fresh_exploratory_llm_run',
                     'frozen_source_sha256': frozen.get('generation_source_sha256', frozen['source_sha256']),
                     'roster_sha256': frozen['roster_sha256'],
                     'runtime_versions': frozen['runtime_versions']} for planned in manifest]
        by_id = {cell['run_id']: cell for cell in manifest}
        completed = set()
        # Inspect ALL saved receipts, not just this invocation's selected cells.
        # Otherwise a restored pilot-only ledger can silently erase main costs.
        for path in sorted(destination.glob('*.json')):
            if path.stem not in by_id:
                raise ValueError('Unrecognized fresh receipt; reconcile before execution.')
            verify_receipt(path, by_id[path.stem], roster, budget)
            completed.add(path.stem)
        if scope == 'main' and not {cell['run_id'] for cell in calibration_cells(config)} <= completed:
            raise ValueError('Main study requires every calibration receipt to exist and verify against the ledger.')
        allowed = {cell['run_id'] for cell in calibration_cells(config)} if scope == 'calibration' else set(by_id)
        selected = [cell for cell in manifest if cell['run_id'] in allowed][:limit]
        pending = [cell for cell in selected if cell['run_id'] not in completed]
        if not pending:
            print(f'All {limit} selected runs verified offline; no API client needed.')
            return
        if budget.unresolved_count():
            raise RuntimeError('Unresolved paid attempt in shared ledger; reconcile before any new protocol or API client.')
        client = OpenAI(api_key=credential(), max_retries=0, timeout=60)
        for model in config['models']:
            client.models.retrieve(model)
        for cell in pending:
            if source_hashes() != frozen['source_sha256']:
                raise ValueError('Source changed during execution; restart and review before continuing.')
            path = destination / (cell['run_id'] + '.json')
            # These checkpoints are request cache data, not completed-network
            # markers. A restart reruns the same seeded logic using paid replies.
            random.seed(cell['seed'])
            np.random.seed(cell['seed'])
            events = []
            caller = PaidCaller(client, budget, cell, execution_source_sha256=frozen['source_sha256'])
            with patch.object(generation, 'get_system_prompt', prompts.system_prompt), \
                    patch.object(generation, 'get_user_prompt', prompts.user_prompt), \
                    patch.object(shared, 'get_llm_response', caller), \
                    patch.object(generation, 'update_graph_from_response', caller.parse_response):
                graph, *_ = generation.generate_network(cell['method'], prompts.DEMOS, roster, list(roster),
                    cell['model'], mean_choices=5, num_iter=3, culture_context=cell['culture'],
                    prompt_language=cell['language'], events=events)
            metrics, homophily = verify_graph(graph, events, roster)
            nx.write_adjlist(graph, path.with_suffix('.adj'))
            import matplotlib.pyplot as plt
            figure, axis = plt.subplots(figsize=(6, 6))
            nx.draw_networkx(graph, pos=nx.spring_layout(graph, seed=224), ax=axis,
                with_labels=False, node_size=42, node_color='#496b72', edge_color='#b1b9ba', width=.5)
            axis.set_axis_off()
            figure.savefig(path.with_suffix('.png'), dpi=300, bbox_inches='tight')
            plt.close(figure)
            record = dict(cell=cell, events=events, metrics=metrics, homophily=homophily,
                          execution_source_sha256=frozen['source_sha256'],
                          retained_abandoned_request_ids=caller.retained_abandoned_request_ids,
                          cumulative_conservative_charge_usd=budget.spent(),
                          artifact_sha256={suffix: hashlib.sha256(path.with_suffix(suffix).read_bytes()).hexdigest() for suffix in ['.adj', '.png']})
            record['requests'] = []
            for request_id in caller.request_ids:
                paid = budget.cached(request_id)
                record['requests'].append({'request_id': request_id, **{key: paid.get(key) for key in REQUEST_FIELDS}})
            write_json(path, record)
            print(f'Saved {cell["run_id"]}; cumulative conservative charged/reserved ${budget.spent():.4f}')
    finally:
        budget.db.close()
        if client is not None:
            client.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare', action='store_true')
    action.add_argument('--preflight', action='store_true')
    action.add_argument('--analyze', action='store_true', help='Offline verification, fresh controls and paired RQ contrasts.')
    action.add_argument('--calibration-report', action='store_true', help='Offline actual calibration usage and assumption-labelled completion forecast.')
    action.add_argument('--calibrate', action='store_true', help='Paid 68-cell calibration only, requiring calibration_review.json.')
    action.add_argument('--execute', action='store_true', help='Full paid execution AFTER a separate reviewed main-study authorization.')
    parser.add_argument('--limit', type=int, help='Optional lower cap on selected cells; default 68 calibration or 896 main.')
    args = parser.parse_args()
    try:
        config = load_config()
        if args.calibrate:
            execute(config, config['calibration']['networks'] if args.limit is None else args.limit, 'calibration')
        elif args.execute:
            execute(config, config['main_networks_target'] if args.limit is None else args.limit)
        elif args.analyze:
            print(json.dumps(analyze(config), indent=2))
        elif args.calibration_report:
            print(json.dumps(calibration_report(config), indent=2))
        elif args.preflight:
            print(json.dumps(preflight(config), indent=2))
        else:
            with workflow_lock():
                roster, manifest = prepare(config)
            print(f'Prepared {len(manifest)} planned cells and 80 translated prompt examples. Paid calls: 0.')
    except (RuntimeError, ValueError) as error:
        parser.exit(1, str(error) + '\n')
    except Exception as error:
        # SDK exception strings can echo credential fragments. Preserve the
        # reservation and report only the type for provider/IO failures.
        parser.exit(1, f'{type(error).__name__}: stopped. Inspect local files/ledger; unresolved charges remain reserved.\n')
