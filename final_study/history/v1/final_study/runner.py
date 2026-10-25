"""Serial collection, durable receipts, read-only replay, and explicit spending.

Network failure is not a free retry. The selected policy stops, keeps the full
reservation, and requires reconciliation. Resume reuses received replies. It
never regenerates a completed graph just because its result is inconvenient.
"""
import contextlib
import json
import math
from pathlib import Path
import socket
from unittest.mock import patch

import networkx as nx

from . import protocol as p
from precollection_audit import workflow_lock, write_json


class IncompleteResponse(RuntimeError):
    """Received and charged, but truncated or otherwise not a usable answer."""


class EvidenceMismatch(RuntimeError):
    """Local integrity failure, never a model-output correction request."""


@contextlib.contextmanager
def output_lock():
    p.OUTPUT.mkdir(parents=True, exist_ok=True)
    with workflow_lock(p.OUTPUT):
        yield


def requests_from_ledger(budget, cell):
    rows = budget.db.execute("SELECT id,phase,cost,status,request,response FROM attempts "
        "WHERE json_extract(request,'$.cell.run_id')=?", (cell['run_id'],)).fetchall()
    rows.sort(key=lambda row: json.loads(row[4])['ordinal'])
    result = []
    for ordinal, row in enumerate(rows):
        if row[3] != 'received':
            raise RuntimeError('Unresolved paid attempt; reservation retained. No automatic resend.')
        request, response = json.loads(row[4]), json.loads(row[5])
        if request['ordinal'] != ordinal or request['cell'] != cell:
            raise ValueError('Receipt ordinal or cell differs.')
        result.append(dict(request_id=row[0], request=request, response=response))
    return result


class Caller(p.legacy.Caller):
    """Reuse the reviewed transport, intent-before-send, and tier safeguards."""
    def __call__(self, model, messages, **kwargs):
        p.verify_contract(self.spec)
        p.check_prices(self.spec)
        try:
            return super().__call__(model, messages, **kwargs)
        except ValueError:
            row = p.engine.ledger_row(self.budget, self.last_request_id)
            if row is not None and row[3] == 'received' and json.loads(row[5])['finish_reason'] != 'stop':
                raise IncompleteResponse('Received response did not finish normally.') from None
            raise


class Replay:
    """Check cached prompts and parser decisions without rewriting paid evidence."""
    def __init__(self, items, live=None):
        self.items, self.live, self.ordinal = items, live, 0
        self.last_request_id = None
        self.cached = False

    def __call__(self, model, messages):
        self.cached = self.ordinal < len(self.items)
        if self.cached:
            item = self.items[self.ordinal]
            request, response = item['request'], item['response']
            if (request['model'], request['messages']) != (model, messages):
                raise ValueError('Cached prompt differs; never repurchase it.')
            self.ordinal += 1
            self.last_request_id = item['request_id']
            if response['finish_reason'] != 'stop':
                raise IncompleteResponse('Received response did not finish normally.')
            return response['text']
        if self.live is None:
            raise ValueError('Offline replay lacks a recorded request.')
        self.live.ordinal = self.ordinal
        text = self.live(model, messages)
        self.ordinal += 1
        self.last_request_id = self.live.last_request_id
        return text

    def parse_response(self, **kwargs):
        if not self.cached:
            return self.live.parse_response(**kwargs)
        response = self.items[self.ordinal-1]['response']
        error = None
        try:
            graph = p.engine.revised.parse_response(**kwargs)
        except ValueError as exc:
            error = exc
        saved = response.get('parse')
        if saved is None:
            # Crash after settlement, before parsing: no new request is needed.
            if self.live is None:
                raise EvidenceMismatch('Received reply has no durable parser decision.')
            self.live.last_request_id = self.last_request_id
            self.live.budget.record_parse(self.last_request_id, error)
            self.live.checkpoint()
        elif saved['valid'] != (error is None) or saved.get('error') != (str(error)[:200] if error else None):
            raise EvidenceMismatch('Cached parser decision changed.')
        if error:
            if kwargs['method'] == 'global':
                raise p.legacy.DiagnosticInvalid('Invalid global response.') from error
            raise error
        return graph


def observe(cell, caller):
    try:
        return p.engine.run_cell(cell, caller), None
    except p.legacy.DiagnosticInvalid:
        return None, 'invalid_global_response'
    except IncompleteResponse:
        return None, 'incomplete_provider_response'
    except ValueError as error:
        if str(error) != p.v8.EXHAUSTED:
            raise
        return None, 'local_corrections_exhausted'


def accounting(budget, target, spec, stage):
    allowed = {c['run_id']: c for c in p.cells(spec, stage)}
    rows = {row[0]: row for row in budget.db.execute('SELECT id,phase,cost,status,request,response FROM attempts')}
    journals = {path.stem: p.read(path) for path in (target/'requests').glob('*.json')}
    ids_by_run = {}
    if rows.keys() != journals.keys():
        raise ValueError('Ledger/intent inventory differs; reconcile before sending anything.')
    for rid, row in rows.items():
        request = json.loads(row[4])
        cell = request.get('cell', {})
        ids_by_run.setdefault(cell.get('run_id'), set()).add(rid)
        if allowed.get(cell.get('run_id')) != cell or row[1] != 'main':
            raise ValueError('Unapproved request in this stage ledger.')
        if journals[rid] != dict(request_id=rid, row_sha256=p.legacy.digest(row)):
            raise ValueError('Journal/ledger mismatch; no automatic replacement.')
        if row[3] != 'received':
            raise RuntimeError('Unresolved paid attempt; retain reservation and stop.')
        response = json.loads(row[5])
        validate_request(dict(request_id=rid, request=request, response=response), cell, spec, request['ordinal'])
        if row[2] != response['conservative_charge_usd']:
            raise ValueError('Ledger cost differs from received usage.')
        p.legacy.verify_transport(target, rid, response)
    for path in (target/'transport').glob('*.json'):
        if path.stem not in rows:
            raise ValueError('Orphan transport evidence; do not repurchase.')
    for path in (target/'runs').glob('*.json'):
        record = p.read(path)
        if allowed.get(path.stem) != record.get('cell'):
            raise ValueError('Unplanned terminal result.')
        if {q['request_id'] for q in record['requests']} != ids_by_run.get(path.stem, set()):
            raise ValueError('Terminal receipt omits ledger attempts.')
        for item in record['requests']:
            row = rows.get(item['request_id'])
            if row is None or json.loads(row[4]) != item['request'] or json.loads(row[5]) != item['response']:
                raise ValueError('Orphan/changed terminal receipt; never regenerate.')


def validate_request(item, cell, spec, ordinal):
    request, response, rid = item['request'], item['response'], item['request_id']
    if (request['cell'] != cell or request['ordinal'] != ordinal or request['model'] != cell['model']
            or request['settings'] != spec['settings'][cell['model']][cell['method']]
            or request.get('execution_source_sha256') != spec['sources']
            or rid != p.engine.request_id(cell, ordinal, request['messages'])):
        raise ValueError('Request identity/settings differ from frozen contract.')
    usage = response['usage']
    if any(type(usage.get(k)) is not int or usage[k] < 0 for k in ['prompt_tokens', 'completion_tokens']):
        raise ValueError('Invalid usage counters.')
    a, b = spec['rates'][cell['model']]
    charge = (usage['prompt_tokens']*a*1.25 + usage['completion_tokens']*b)/1e6
    if not math.isclose(charge, response['conservative_charge_usd'], rel_tol=0, abs_tol=1e-12):
        raise ValueError('Usage-derived charge changed.')


def normalization(items, cell):
    if cell['method'] != 'global':
        return []
    result = []
    for item in items:
        try:
            flags = p.candidate.canonical_global(item['response']['text'], p.engine.revised.previous.adult_roster())
        except ValueError:
            flags = None
        result.append(dict(request_id=item['request_id'], normalization=flags))
    return result


def verify_record(path, cell, spec):
    record = p.read(path)
    if record.get('cell') != cell or record.get('evidence_type') != 'received_final_study':
        raise ValueError('Wrong final-study receipt or fixture in paid data.')
    items = record['requests']
    if not items:
        raise ValueError('A terminal result needs received evidence.')
    for ordinal, item in enumerate(items):
        validate_request(item, cell, spec, ordinal)
    replay = Replay(items)
    with p.adapter(cell, spec):
        result, failure = observe(cell, replay)
    if replay.ordinal != len(items) or failure != record.get('failure'):
        raise ValueError('Terminal replay differs.')
    if record.get('normalization') != normalization(items, cell):
        raise ValueError('Global normalization evidence changed.')
    if failure is None:
        graph, events, decisions, schedule = result
        if (p.legacy.paid.clean_numbers(events) != record['events'] or decisions != record['decisions']
                or schedule != record['schedule']
                or p.engine.measures(graph, p.engine.revised.previous.adult_roster()) != record['metrics']):
            raise ValueError('Graph-derived evidence changed.')
        if set(record['artifacts']) != {'.adj', '.png'} or any(
                p.engine.sha(path.with_suffix(ext)) != sha for ext, sha in record['artifacts'].items()):
            raise ValueError('Graph/plot artifact changed.')
        saved = nx.read_adjlist(path.with_suffix('.adj'))
        if set(saved) != set(graph) or {frozenset(e) for e in saved.edges} != {frozenset(e) for e in graph.edges}:
            raise ValueError('Adjacency differs from received decisions.')
    elif record.get('metrics') is not None or record.get('artifacts'):
        raise ValueError('Failed observations cannot carry graph metrics/artifacts.')
    return record


def summarize(records, spec, stage):
    rows = []
    for record in records:
        cell, items = record['cell'], record['requests']
        rows.append(dict(run_id=cell['run_id'], model=cell['model'], method=cell['method'],
            culture=cell['culture'], language=cell['language'], seed=cell['seed'],
            valid_graph=record['failure'] is None, failure=record['failure'], metrics=record.get('metrics'),
            requests=len(items), parse_failures=sum(q['response'].get('parse', {}).get('valid') is False for q in items),
            conservative_usd=sum(q['response']['conservative_charge_usd'] for q in items),
            api_seconds=sum(q['response']['diagnostics']['elapsed_ms']/1000 for q in items)))
    return dict(status='COMPLETE' if len(rows) == len(p.cells(spec, stage)) else 'PARTIAL',
        contract_sha256=p.legacy.digest(spec), stage=stage, attempted=len(rows),
        planned=len(p.cells(spec, stage)), valid_graphs=sum(r['valid_graph'] for r in rows),
        empty_graphs=sum(r['valid_graph'] and r['metrics']['density'] == 0 for r in rows),
        conservative_usd=sum(r['conservative_usd'] for r in rows),
        api_seconds=sum(r['api_seconds'] for r in rows), results=rows)


def verify_stage(spec, stage):
    target = p.folder(spec, stage)
    planned = p.cells(spec, stage)
    expected = {c['run_id'] for c in planned}
    actual = {path.stem for path in (target/'runs').glob('*.json')}
    if actual != expected:
        raise ValueError('Stage is incomplete or contains unplanned results.')
    records = [verify_record(target/'runs'/(c['run_id']+'.json'), c, spec) for c in planned]
    return summarize(records, spec, stage)


def preflight():
    with output_lock():
        write_json(p.OUTPUT/'readiness.json', dict(status='VALIDATION_PENDING', main_authorized=False))
        spec = p.load_contract()
        results = []
        # All model/condition/method combinations, not 3,584 redundant fixture plots.
        selected = [c for c in p.cells(spec, 'main') if c['seed'] == 44000]
        with patch.object(socket.socket, 'connect', side_effect=AssertionError('Offline preflight')), \
                patch.object(p.legacy.paid, 'OpenAI', side_effect=AssertionError('Offline preflight')):
            for cell in selected:
                caller = p.engine.FixtureCaller(cell)
                with p.adapter(cell, spec):
                    graph, events, decisions, schedule = p.engine.run_cell(cell, caller)
                if len(graph) != 50 or not all(d['valid'] for d in decisions):
                    raise ValueError('Engine fixture failed.')
                results.append(dict(run_id=cell['run_id'], nodes=len(graph), edges=len(graph.edges),
                                    requests=len(caller.requests)))
        report = dict(status='OFFLINE_PREFLIGHT_PASSED', contract_sha256=p.legacy.digest(spec),
            cases=len(results), results=results, paid_requests=0)
        write_json(p.OUTPUT/'preflight.json', report)
        return {k: v for k, v in report.items() if k != 'results'}


def require_preflight(spec):
    report = p.read(p.OUTPUT/'preflight.json')
    if report.get('status') != 'OFFLINE_PREFLIGHT_PASSED' or report.get('contract_sha256') != p.legacy.digest(spec) or report.get('cases') != 112:
        raise ValueError('Matching complete offline preflight required.')


def readiness(spec, require_credential=True):
    require_preflight(spec)
    p.check_prices(spec)
    report = verify_stage(spec, 'hindi_check')
    if report['valid_graphs'] != 8:
        raise ValueError('Targeted Hindi live check has invalid observations; main stays blocked.')
    target = p.folder(spec, 'hindi_check')
    approval = p.read(target/'authorization.json')
    budget = open_budget(spec, 'hindi_check', approval)
    try:
        accounting(budget, target, spec, 'hindi_check')
    finally:
        budget.db.close()
    if require_credential:
        p.legacy.paid.credential()
    # Empty graphs are recorded outcomes, never an excuse to resample a model.
    return dict(status='READY_FOR_MAIN_EXECUTION', contract_sha256=p.legacy.digest(spec),
        main_attempts=3584, main_authorized=False, hindi_check=report,
        transport_policy=spec['transport_policy'])


def status():
    """Refresh machine-readable readiness; never leave stale success on failure."""
    with output_lock():
        write_json(p.OUTPUT/'readiness.json', dict(status='VALIDATION_PENDING', main_authorized=False))
        spec = p.load_contract()
        require_preflight(spec)
        p.check_prices(spec)
        target = p.folder(spec, 'hindi_check')
        completed = len(list((target/'runs').glob('*.json')))
        if completed != 8:
            try:
                p.legacy.paid.credential()
                credential_present = True
            except RuntimeError:
                credential_present = False
            result = dict(status='TARGETED_LIVE_CHECK_REQUIRED', main_authorized=False,
                contract_sha256=p.legacy.digest(spec), main_attempts=3584,
                offline_preflight_passed=True, hindi_check_completed=completed,
                hindi_check_planned=8, local_credential_present=credential_present)
        else:
            result = readiness(spec, require_credential=False)
            try:
                p.legacy.paid.credential()
                result['local_credential_present'] = True
            except RuntimeError:
                result.update(status='LOCAL_CREDENTIAL_REQUIRED', local_credential_present=False)
        write_json(p.OUTPUT/'readiness.json', result)
        return result


def authorize(spec, stage, cap, reference):
    if type(cap) not in {int, float} or not math.isfinite(cap) or cap <= 0 or not reference.strip():
        raise ValueError('Explicit positive spending allowance and approval reference required.')
    require_preflight(spec)
    p.check_prices(spec)
    if stage == 'main':
        readiness(spec)
    target = p.folder(spec, stage)
    approval = dict(contract_sha256=p.legacy.digest(spec), stage=stage, additional_usd=cap, reference=reference)
    path, ledger = target/'authorization.json', target/'budget.sqlite'
    if path.exists():
        if p.read(path) != approval or not ledger.exists():
            raise ValueError('Existing authorization differs or its ledger is missing; never reset it.')
        return approval
    if target.exists() and any(target.iterdir()):
        raise ValueError('Prior stage evidence exists; do not create a fresh budget anchor.')
    target.mkdir(parents=True, exist_ok=True)
    with contextlib.closing(p.legacy.paid.Budget(ledger, total_cap=cap, pilot_cap=cap, allow_higher_cap=True).db) as db:
        db.execute('CREATE TABLE authorization (value TEXT NOT NULL)')
        db.execute('INSERT INTO authorization VALUES (?)', (json.dumps(approval, sort_keys=True),))
        db.commit()
    write_json(path, approval)
    return approval


def open_budget(spec, stage, approval):
    target = p.folder(spec, stage)
    cap = approval.get('additional_usd')
    if (approval.get('contract_sha256') != p.legacy.digest(spec) or approval.get('stage') != stage
            or type(cap) not in {int, float} or not math.isfinite(cap) or cap <= 0
            or not isinstance(approval.get('reference'), str) or not approval['reference'].strip()):
        raise ValueError('Authorization identity/allowance differs.')
    if p.read(target/'authorization.json') != approval:
        raise ValueError('Authorization changed.')
    budget = p.legacy.paid.Budget(target/'budget.sqlite', total_cap=approval['additional_usd'],
        pilot_cap=approval['additional_usd'], require_existing=True, allow_higher_cap=True)
    try:
        saved = budget.db.execute('SELECT value FROM authorization').fetchall()
        if len(saved) != 1 or json.loads(saved[0][0]) != approval:
            raise ValueError('Ledger authorization differs; cap cannot be re-anchored.')
    except Exception:
        budget.db.close()
        raise
    return budget


def analyze(records, spec, target):
    rows = []
    for record in records:
        c = record['cell']
        rows.append({k: c[k] for k in p.analysis.IDENTITY if k != 'contract'} |
            dict(contract=spec['analysis']['version'], execution_contract_sha256=p.legacy.digest(spec),
                run_id=c['run_id'], valid_graph=record['failure'] is None,
                **(record.get('metrics') or {})))
    contrasts = p.analysis.primary_contrasts(rows, spec['analysis'])
    write_json(target/'analysis_rows.json', rows)
    write_json(target/'primary_contrasts.json', contrasts)
    fields = sorted(set().union(*(row.keys() for row in rows)))
    p.engine.write_csv(target/'network_metrics.csv', [{k: row.get(k) for k in fields} for row in rows])
    p.engine.write_csv(target/'primary_contrasts.csv', [
        {k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()} for row in contrasts])


def launch(stage, cap, reference):
    with output_lock():
        write_json(p.OUTPUT/'readiness.json', dict(status='VALIDATION_PENDING', main_authorized=False))
        spec = p.load_contract()
        # Credentials must exist before opening a new spending window.
        key = p.legacy.paid.credential()
        approval = authorize(spec, stage, cap, reference)
        target, records = p.folder(spec, stage), []
        budget = open_budget(spec, stage, approval)
        try:
            write_json(target/'summary.json', dict(status='VALIDATION_PENDING', contract_sha256=p.legacy.digest(spec)))
            accounting(budget, target, spec, stage)
            planned = p.cells(spec, stage)
            with contextlib.ExitStack() as stack:
                client = None
                for cell in planned:
                    path = target/'runs'/(cell['run_id']+'.json')
                    if path.exists():
                        record = verify_record(path, cell, spec)
                    else:
                        if client is None:
                            client = stack.enter_context(p.legacy.paid.OpenAI(api_key=key,
                                base_url='https://api.openai.com/v1', max_retries=0, timeout=60))
                            for model in p.analysis.MODELS:
                                if client.models.retrieve(model).id != model:
                                    raise ValueError('Requested model identity not available.')
                        with p.adapter(cell, spec):
                            live = Caller(client, budget, cell, spec, target)
                            caller = Replay(requests_from_ledger(budget, cell), live)
                            result, failure = observe(cell, caller)
                        items = requests_from_ledger(budget, cell)
                        if failure is None:
                            # The first completion marker must already be replayable.
                            # A crash after it is written must not require a graph rerun.
                            def completed_marker(destination, record):
                                write_json(destination, {**record, 'failure': None,
                                    'normalization': normalization(items, cell)})
                            with patch.object(p.engine, 'write_json', completed_marker):
                                p.engine.save_run(path, cell, result, items, 'received_final_study')
                            record = p.read(path)
                        else:
                            record = dict(cell=cell, evidence_type='received_final_study', requests=items,
                                          metrics=None, artifacts={})
                        record.update(failure=failure, normalization=normalization(items, cell))
                        write_json(path, record)
                        record = verify_record(path, cell, spec)
                    records.append(record)
                    print(f'{stage}: {len(records)}/{len(planned)} recorded', flush=True)
            accounting(budget, target, spec, stage)
            summary = summarize(records, spec, stage)
            write_json(target/'summary.json', summary)
            if stage == 'main':
                analyze(records, spec, target)
            else:
                write_json(p.OUTPUT/'readiness.json', readiness(spec))
            return {k: v for k, v in summary.items() if k != 'results'}
        except Exception as error:
            write_json(target/'summary.json', dict(status='STOPPED', error_type=type(error).__name__,
                completed=len(records), contract_sha256=p.legacy.digest(spec),
                conservative_charge_or_reservation_usd=budget.spent()))
            raise
        finally:
            budget.db.close()
