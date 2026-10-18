"""Serial collection, durable receipts, read-only replay, and explicit spending.

Network failure is not a free retry. An uncertain transport attempt stops the
run and keeps its full reservation. `recover` may retire exactly one such
attempt that never received a reply and authorize one separately reserved
replacement for that same request. A replacement that also fails, a received
reply, a client error or more than MAX_RECOVERIES retirements stops for human
review. Resume reuses received replies and never regenerates a completed graph.
"""
import contextlib
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import socket
import time
from unittest.mock import patch

import networkx as nx

from . import protocol as p
from precollection_audit import workflow_lock, write_json

# Unpatched request identity. Replacement ids are derived from it, never stored.
BASE_REQUEST_ID = p.engine.request_id
MAX_RECOVERIES = 30
TRANSPORT_ERRORS = {'APIConnectionError', 'APITimeoutError', 'InternalServerError',
                    'RateLimitError', 'APIStatusError', 'ConflictError'}
RETRYABLE_HTTP = {408, 409, 429}


class RecoveryRefused(RuntimeError):
    """The stop is not a plain lost-transport attempt; a person must review it."""


def replacement_nonce(original_id):
    return 'final-r1-' + original_id


def replacement_of(original_id):
    # Identical to paid_study.Budget.replacement_id for this authorization.
    return hashlib.sha256(json.dumps([original_id, 'replacement', replacement_nonce(original_id)]).encode()).hexdigest()


def retired(row):
    """True only for a retirement made by `recover`; any other abandonment is invalid."""
    if row is None or row[3] != 'abandoned':
        return False
    record = json.loads(row[5])
    if (record.get('replacement_authorization') != replacement_nonce(row[0])
            or record.get('accounting') != 'full_reservation_retained_no_replay'
            or record.get('retained_reservation_usd') != row[2]
            or record.get('previous_status') not in {'reserved', 'uncertain'}):
        raise ValueError('Retired attempt lacks final-study recovery provenance.')
    return True


def mapped_request_id(budget):
    def request_id(cell, ordinal, messages):
        rid = BASE_REQUEST_ID(cell, ordinal, messages)
        return replacement_of(rid) if retired(p.engine.ledger_row(budget, rid)) else rid
    return request_id


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
    # A retired attempt keeps its reservation in the ledger but is not a reply.
    rows = [row for row in rows if not retired(row)]
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
    retirements = 0
    for rid, row in rows.items():
        request = json.loads(row[4])
        cell = request.get('cell', {})
        if allowed.get(cell.get('run_id')) != cell or row[1] != 'main':
            raise ValueError('Unapproved request in this stage ledger.')
        if journals[rid] != dict(request_id=rid, row_sha256=p.legacy.digest(row)):
            raise ValueError('Journal/ledger mismatch; no automatic replacement.')
        if retired(row):
            # Charged in full against the cap; excluded only from replayed replies.
            if (request.get('replaces_request_id') is not None
                    or rid != BASE_REQUEST_ID(cell, request['ordinal'], request['messages'])):
                raise ValueError('Only an original request may be retired, exactly once.')
            if (target/'transport'/(rid+'.json')).exists():
                raise ValueError('A retired attempt has received transport evidence.')
            retirements += 1
            continue
        ids_by_run.setdefault(cell.get('run_id'), set()).add(rid)
        original = request.get('replaces_request_id')
        if original is not None and not retired(rows.get(original)):
            raise ValueError('Replacement without a retired original.')
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
    if retirements > MAX_RECOVERIES:
        raise ValueError('Transport recovery limit exceeded.')
    return retirements


def validate_request(item, cell, spec, ordinal):
    request, response, rid = item['request'], item['response'], item['request_id']
    expected = BASE_REQUEST_ID(cell, ordinal, request['messages'])
    if 'replaces_request_id' in request:
        # Self-verifying provenance: public replay needs no private ledger.
        if request['replaces_request_id'] != expected:
            raise ValueError('Replacement provenance differs from its original request.')
        expected = replacement_of(expected)
    if (request['cell'] != cell or request['ordinal'] != ordinal or request['model'] != cell['model']
            or request['settings'] != spec['settings'][cell['model']][cell['method']]
            or request.get('execution_source_sha256') != spec['sources']
            or rid != expected):
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
    complete = len(rows) == len(p.cells(spec, stage))
    status = ('COMPLETE' if all(r['valid_graph'] for r in rows) else 'COMPLETE_WITH_FAILURES') if complete else 'PARTIAL'
    return dict(status=status,
        contract_sha256=p.legacy.digest(spec), stage=stage, attempted=len(rows),
        planned=len(p.cells(spec, stage)), valid_graphs=sum(r['valid_graph'] for r in rows),
        empty_graphs=sum(r['valid_graph'] and r['metrics']['density'] == 0 for r in rows),
        conservative_usd=sum(r['conservative_usd'] for r in rows),
        api_seconds=sum(r['api_seconds'] for r in rows), results=rows)


def verify_stage(spec, stage):
    spec = p.evidence_spec(spec, stage)
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
        # All model/condition/method combinations, not 896 redundant fixture plots.
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
        fields = ['run_id', 'model', 'method', 'culture', 'language', 'seed', 'wording',
                  'roster_sha256', 'final_contract_sha256']
        p.engine.write_csv(p.OUTPUT/'main_manifest.csv', [
            {k: cell[k] for k in fields} for cell in p.cells(spec, 'main')])
        write_json(p.OUTPUT/'prompt_catalog.json', p.catalog())
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
    historical = p.evidence_spec(spec, 'hindi_check')
    target = p.folder(historical, 'hindi_check')
    approval = p.read(target/'authorization.json')
    budget = open_budget(historical, 'hindi_check', approval)
    try:
        accounting(budget, target, historical, 'hindi_check')
    finally:
        budget.db.close()
    if require_credential:
        p.legacy.paid.credential()
    # Empty graphs are recorded outcomes, never an excuse to resample a model.
    return dict(status='READY_FOR_MAIN_EXECUTION', contract_sha256=p.legacy.digest(spec),
        main_attempts=spec['main_attempts'], main_budget_ceiling_usd=spec['main_budget_ceiling_usd'],
        repetitions_per_condition=spec['schedule_blocks'],
        main_authorized=False, hindi_check=report,
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
                contract_sha256=p.legacy.digest(spec), main_attempts=spec['main_attempts'],
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


def check_allowance(spec, stage, cap, reference):
    if stage != 'main':
        raise ValueError('Historical Hindi checks are read-only; use verify, not paid regeneration.')
    if type(cap) not in {int, float} or not math.isfinite(cap) or cap <= 0 or not isinstance(reference, str) or not reference.strip():
        raise ValueError('Explicit positive spending allowance and approval reference required.')
    if (spec.get('version') != p.VERSION or spec.get('main_attempts') != 896
            or spec.get('schedule_blocks') != 8 or spec.get('main_budget_ceiling_usd') != 120
            or spec['analysis']['schedule_seeds'] != list(range(44000, 44008))):
        raise ValueError('Only the eight-repetition, 896-attempt allocation is approved.')
    if cap > 120:
        raise ValueError('Main-study allowance cannot exceed the owner ceiling of $120.')


def authorize(spec, stage, cap, reference):
    check_allowance(spec, stage, cap, reference)
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
    if stage == 'main':
        check_allowance(spec, stage, cap, approval.get('reference'))
    elif stage != 'hindi_check' or p.legacy.digest(spec) != p.HISTORY_SHA:
        raise ValueError('Only pinned historical Hindi accounting may be opened read-only.')
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
        check_allowance(spec, stage, cap, reference)
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
                # A retired original resolves to its one authorized replacement id.
                stack.enter_context(patch.object(p.engine, 'request_id', mapped_request_id(budget)))
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
            recovered = accounting(budget, target, spec, stage)
            summary = summarize(records, spec, stage)
            retained = budget.db.execute(
                "SELECT COALESCE(SUM(cost),0) FROM attempts WHERE status='abandoned'").fetchone()[0]
            summary.update(transport_recoveries=recovered, retained_retired_reservations_usd=retained,
                ledger_charge_or_reservation_usd=budget.spent())
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


def _unresolved(budget):
    return budget.db.execute("SELECT id,phase,cost,status,request,response FROM attempts "
                             "WHERE status IN ('reserved','uncertain')").fetchall()


def check_recoverable(budget, target, spec, row):
    """Accept only a lost first attempt; everything else needs a person."""
    rid, request = row[0], json.loads(row[4])
    response = json.loads(row[5]) if row[5] else {}
    cell = request.get('cell', {})
    if {c['run_id']: c for c in p.cells(spec, 'main')}.get(cell.get('run_id')) != cell or row[1] != 'main':
        raise RecoveryRefused('Unresolved attempt is not a planned main-study request.')
    if request.get('replaces_request_id') is not None:
        raise RecoveryRefused('The single authorized replacement also failed; human review required.')
    if rid != BASE_REQUEST_ID(cell, request['ordinal'], request['messages']):
        raise RecoveryRefused('Unresolved attempt identity differs.')
    if (target/'transport'/(rid+'.json')).exists():
        raise RecoveryRefused('A provider reply was saved; reconcile it, never resend.')
    if row[3] == 'uncertain':
        diagnostics = response.get('diagnostics') or {}
        status = diagnostics.get('http_status')
        if (response.get('error_type') not in TRANSPORT_ERRORS or diagnostics.get('stage') != 'transport'
                or (status is not None and status < 500 and status not in RETRYABLE_HTTP)):
            raise RecoveryRefused('Not a lost-transport failure (for example 4xx, parse or settlement).')
    journal = p.read(target/'requests'/(rid+'.json'))
    if journal not in (dict(request_id=rid, row_sha256=p.legacy.digest(row)),
                       dict(request_id=rid, status='INTENT_BEFORE_TRANSPORT')):
        raise RecoveryRefused('Request journal differs from the ledger.')
    done = budget.db.execute("SELECT COUNT(*) FROM attempts WHERE status='abandoned'").fetchone()[0]
    if done >= MAX_RECOVERIES:
        raise RecoveryRefused(f'Transport recovery limit ({MAX_RECOVERIES}) reached.')
    return response


def recover(stage, reference, automatic=False):
    """Offline and idempotent: retire one lost attempt; no provider call is made here."""
    if stage != 'main':
        raise RecoveryRefused('Only main-study transport loss is recoverable.')
    with output_lock():
        spec = p.load_contract()
        target = p.folder(spec, stage)
        approval = p.read(target/'authorization.json')
        if approval.get('reference') != reference:
            raise RecoveryRefused('Use the original approval reference; the allowance is unchanged.')
        budget = open_budget(spec, stage, approval)
        try:
            rows = _unresolved(budget)
            if len(rows) != 1:
                raise RecoveryRefused(f'{len(rows)} unresolved attempts; exactly one can be recovered.')
            row = rows[0]
            response = check_recoverable(budget, target, spec, row)
            rid = row[0]
            reason = (f'{"Automatic" if automatic else "Explicit"} final-study transport recovery under '
                      f'{reference}; full reservation retained; one replacement for the same request.')
            budget.abandon(rid, reason, replacement_authorization=replacement_nonce(rid))
            new_row = p.engine.ledger_row(budget, rid)
            retired(new_row)
            write_json(target/'requests'/(rid+'.json'), dict(request_id=rid, row_sha256=p.legacy.digest(new_row)))
            log = target/'transport_recoveries.json'
            entries = p.read(log) if log.exists() else []
            entries.append(dict(original_request_id=rid, replacement_request_id=replacement_of(rid),
                run_id=json.loads(row[4])['cell']['run_id'], ordinal=json.loads(row[4])['ordinal'],
                previous_status=row[3], error_type=response.get('error_type'),
                http_status=(response.get('diagnostics') or {}).get('http_status'),
                retained_reservation_usd=row[2], automatic=automatic,
                date_utc=datetime.now(timezone.utc).isoformat()))
            write_json(log, entries)
            accounting(budget, target, spec, stage)
            return dict(status='ONE_REPLACEMENT_AUTHORIZED', original_request_id=rid,
                replacement_request_id=replacement_of(rid), retained_reservation_usd=row[2],
                recoveries_used=len(entries), recovery_limit=MAX_RECOVERIES, paid_calls=0,
                retry_after=((response.get('diagnostics') or {}).get('headers') or {}).get('retry-after'))
        finally:
            budget.db.close()


FREE_PROBE_ERRORS = ('APIConnectionError', 'APITimeoutError', 'InternalServerError', 'RateLimitError')


def run(stage, cap, reference, auto_recover=False, sleep=time.sleep, max_probe_wait_s=3600):
    """launch(); with auto_recover, retire lost attempts and resume under the same cap."""
    probe_wait = 0
    while True:
        try:
            return launch(stage, cap, reference)
        except Exception as error:
            if not auto_recover or stage != 'main':
                raise
            target = p.folder(p.load_contract(), stage)
            if not (target/'authorization.json').exists():
                raise
            try:
                info = recover(stage, reference, automatic=True)
            except RecoveryRefused as refused:
                # No unresolved paid attempt: a free model-availability check failed.
                if (str(refused).startswith('0 unresolved') and type(error).__name__ in FREE_PROBE_ERRORS
                        and probe_wait < max_probe_wait_s):
                    print(f'{stage}: provider unavailable before any paid request; waiting 60s', flush=True)
                    sleep(60)
                    probe_wait += 60
                    continue
                raise refused from error
            delay = 30
            try:
                delay = min(120, max(30, float(info['retry_after'] or 0)))
            except ValueError:
                pass
            print(f"{stage}: recovered lost attempt {info['recoveries_used']}/{MAX_RECOVERIES}; "
                  f'retained ${info["retained_reservation_usd"]:.6f}; resuming in {delay:.0f}s', flush=True)
            sleep(delay)
