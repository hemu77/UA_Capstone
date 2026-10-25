"""Isolated revised calibration. Preparation/preflight never authorize payment."""
import argparse
import contextlib
import csv
import hashlib
import io
import itertools
import json
import math
import re
import socket
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import networkx as nx
import numpy as np

import revision_next as revised
import paid_study as paid
from analyze_saved_study import measures

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'outputs/calibration_v6'
LEDGER = ROOT / 'outputs/revision_budget_v1/budget.sqlite'
frozen, engine = revised.frozen, revised.engine
write_json = frozen.write_json


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


SOURCE_NAMES = ['revision_next.py', 'study_protocol_next.json', 'analyze_saved_study.py', 'calibration_v6.py']
IMPORTED_SOURCES = {**frozen.source_hashes(), **{n: sha(ROOT / n) for n in SOURCE_NAMES}}
PROBE = ROOT / 'outputs/wording_probe_v6'


def verify_probe(budget=None):
    report = json.loads((PROBE / 'report.json').read_text(encoding='utf-8'))
    if (report.get('status') != 'COMPLETE' or report.get('probe_calls') != 72
            or report.get('engineering_screen') != 'NO_FAILURES_OBSERVED' or report.get('revised_failures') != 0):
        raise ValueError('Completed passing wording screen required.')
    if (report['contract']['candidate_protocol_sha256'] != frozen.digest(revised.config())
            or any(sha(ROOT/name) != value for name,value in report['contract']['source_sha256'].items())):
        raise ValueError('Wording-screen source/protocol no longer matches this candidate.')
    if any(sha(PROBE/'replies'/(rid+'.json')) != value for rid,value in report['replies_sha256'].items()) or len(report['replies_sha256']) != 72:
        raise ValueError('Wording-screen evidence changed.')
    if budget is not None:
        if budget.spent()+1e-12 < report['cumulative_ledger_usd']:
            raise ValueError('Ledger is below verified prior spending.')
        for rid in report['replies_sha256']:
            item = json.loads((PROBE/'replies'/(rid+'.json')).read_text(encoding='utf-8'))
            if budget.cached(item['request_id']) != item['response']:
                raise ValueError('Verified prior probe receipt is missing or changed in ledger.')
    return report


def design():
    cfg = revised.config()
    rows = []
    for model, method, (culture, language), repetition in itertools.product(
            cfg['models'], cfg['methods'], cfg['settings'], range(2)):
        core = model == 'gpt-6-luna' or (culture, language, repetition) == ('us', 'english', 0)
        coverage = culture == 'us' and language != 'english' and repetition == 0
        if core or coverage:
            rows.append(dict(model=model, method=method, culture=culture, language=language,
                repetition=repetition, seed=21000 + repetition, personas=50,
                stage='core68' if core else 'multilingual36'))
    return sorted(rows, key=lambda c: (c['stage'] != 'core68',
        not (c['culture'] == 'us' and c['language'] == 'english' and c['repetition'] == 0),
        cfg['methods'].index(c['method']), c['model'], c['culture'], c['language'], c['repetition']))


def schedule(seed):
    plan = revised.schedule(seed)
    rng = np.random.RandomState(revised.stream_seed(seed, 'actors'))
    first = rng.choice(list(revised.previous.adult_roster()),50,replace=False)
    plan['iterative_actor_orders'] = [rng.choice(first,50,replace=False).tolist() for _ in range(3)]
    return plan


def contract():
    sources = {**frozen.source_hashes(), **{n: sha(ROOT / n) for n in SOURCE_NAMES}}
    if sources != IMPORTED_SOURCES:
        raise ValueError('Source changed after import; restart before preparing a contract.')
    return dict(version='revised-calibration-v6', purpose='exploratory_engineering_calibration_not_confirmation',
        sources=sources, wording_probe_report_sha256=sha(PROBE/'report.json'),
        protocol_sha256=frozen.digest(revised.config()), roster_sha256=frozen.digest(revised.previous.adult_roster()),
        runtime=frozen.runtime_versions(), design=design(), schedules={str(s): schedule(s) for s in [21000, 21001]},
        rates={m: list(paid.RATES[m]) for m in revised.config()['models']},
        settings={m: settings(m, 'local') for m in revised.config()['models']},
        global_output_cap=8192, other_output_cap=512, correction_attempts=3, transport_retries=0,
        global_zero_edges_allowed=True, human_bilingual_validation=False, main_authorized=False)


def settings(model, method):
    value = {'max_completion_tokens': 8192 if method == 'global' else 512}
    value.update({'temperature': .8} if model == 'gpt-4.1' else {'reasoning_effort': 'none'})
    return value


def cells(spec):
    fingerprint = frozen.digest(spec)
    return [dict(**row, run_id=f"calibration_v6_{fingerprint[:12]}_{row['method']}_{row['model']}_{row['culture']}_{row['language']}_r{row['repetition']}",
        phase='main', evidence_type='revised_exploratory_calibration', calibration_contract_sha256=fingerprint,
        frozen_source_sha256=spec['sources'], roster_sha256=spec['roster_sha256'], runtime_versions=spec['runtime'])
        for row in spec['design']]


def folder(spec):
    return BASE / frozen.digest(spec)[:12]


def request_id(cell, ordinal, messages):
    # Keep identical to the immutable PaidCaller identity; tests cover all models.
    value = [cell, ordinal, cell['model'], messages, settings(cell['model'], cell['method'])]
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def ledger_row(budget, rid):
    return budget.db.execute('SELECT id,phase,cost,status,request,response FROM attempts WHERE id=?', (rid,)).fetchone()


class CalibrationCaller(paid.PaidCaller):
    """Write intent before transport so a lost historical cache cannot be rebought."""
    def __init__(self, client, budget, cell, spec, destination):
        super().__init__(client, budget, cell, execution_source_sha256=spec['sources'])
        self.spec, self.destination = spec, destination
        self.parse_method = revised.parse_response

    def checkpoint(self):
        row = ledger_row(self.budget, self.last_request_id)
        if row is not None:
            write_json(self.destination / 'requests' / (self.last_request_id + '.json'),
                dict(request_id=self.last_request_id, row_sha256=frozen.digest(row)))

    def __call__(self, model, messages, **kwargs):
        if self.spec != contract():
            raise ValueError('Calibration contract changed before request.')
        rid = request_id(self.cell, self.ordinal, messages)
        path = self.destination / 'requests' / (rid + '.json')
        row = ledger_row(self.budget, rid)
        if path.exists():
            saved = json.loads(path.read_text(encoding='utf-8'))
            if row is None or saved.get('row_sha256') != frozen.digest(row):
                raise ValueError('Missing or changed journaled request; no replacement permitted.')
            if row[3] != 'received':
                raise RuntimeError('A non-received calibration attempt blocks all replacements.')
        elif row is not None:
            raise ValueError('Unjournaled calibration request; reconcile before continuing.')
        else:
            write_json(path, dict(request_id=rid, status='INTENT_BEFORE_TRANSPORT'))
        try:
            result = super().__call__(model, messages, **kwargs)
            if self.last_request_id != rid:
                raise ValueError('Request identity changed; replacements are not enabled here.')
            return result
        except paid.BudgetExceeded:
            # Reservation failed before transport. Only this known unused intent can go.
            if ledger_row(self.budget, rid) is None:
                path.unlink()
            raise
        finally:
            self.checkpoint()

    def parse_response(self, **kwargs):
        try:
            return super().parse_response(**kwargs)
        finally:
            self.checkpoint()


def run_cell(cell, caller):
    """Same serial engine for fixtures, cached replay, and eventual paid requests."""
    roster, plan = revised.previous.adult_roster(), schedule(cell['seed'])
    actors = np.random.RandomState(revised.stream_seed(cell['seed'], 'actors'))
    counts = np.random.RandomState(revised.stream_seed(cell['seed'], 'counts'))
    local_np = SimpleNamespace(isfinite=np.isfinite, random=SimpleNamespace(choice=actors.choice, exponential=counts.exponential))
    events, decisions = [], []
    active_method = None

    def system(method, personas, demos, curr_pid=None, num_choices=None, **kwargs):
        nonlocal active_method
        active_method = method  # The engine removes prompt_method before calling its retry helper.
        return revised.system_prompt(method, cell['language'], cell['culture'], curr_pid, num_choices)

    def user(method, personas, order, demos, curr_pid=None, G=None, **kwargs):
        return revised.candidate_payload(method, cell['language'], curr_pid,
            G if G is not None else nx.empty_graph(roster), plan['display_order'])

    def ask(model, system_text, user_text, parser, parse_args, **kwargs):
        args = dict(parse_args)
        for attempt in range(1, 4):
            response = caller(model, [{'role': 'system', 'content': system_text}, {'role': 'user', 'content': user_text}])
            decision = dict(step=len(events), attempt=attempt, method=args['method'], prompt_method=active_method,
                actor=args.get('curr_pid'), requested_count=args.get('num_choices'),
                candidate_order=[r[0] for r in json.loads(user_text)['candidates']],
                request_id=caller.last_request_id, valid=False)
            decisions.append(decision)
            try:
                result = caller.parse_response(response=response, **args)
            except ValueError:
                if args['method'] == 'global' and response.strip() and 'required_global_edges' not in args:
                    pairs = [line.replace(',', ' ').split() for line in response.splitlines()]
                    valid = [p for p in pairs if len(p) == 2 and p[0] != p[1] and set(p) <= set(roster)]
                    if not valid:
                        raise ValueError('No recoverable global pairs; stop without choosing a replacement network.')
                    args['required_global_edges'] = valid
                system_text = revised.retry_prompt(active_method, cell['language'], cell['culture'],
                    args.get('curr_pid'), args.get('num_choices'), 'required_global_edges' in args)
                if 'required_global_edges' in args:
                    system_text += '\n' + json.dumps({'required_pairs': sorted({tuple(sorted(p, key=int)) for p in args['required_global_edges']})})
            else:
                decision['valid'] = True
                return result, response, attempt
        raise ValueError('Three correction attempts exhausted; incomplete run remains charged.')

    # ponytail: module adapters require serial execution under the shared workflow lock.
    with patch.object(engine, 'np', local_np), patch.object(engine, 'get_system_prompt', system), \
            patch.object(engine, 'get_user_prompt', user), patch.object(engine, 'repeat_prompt_until_parsed', ask), \
            contextlib.redirect_stdout(io.StringIO()):
        graph, *_ = engine.generate_network(cell['method'], revised.previous.DEMOS, roster, list(roster),
            cell['model'], mean_choices=5, num_iter=3, culture_context=cell['culture'], prompt_language=cell['language'], events=events)
    frozen.verify_graph(graph, events, roster)
    return graph, events, decisions, plan


class FixtureCaller:
    """Deterministic software fixture, with no usage or model-evidence claim."""
    def __init__(self, cell):
        self.plan, self.requests, self.last_request_id = revised.schedule(cell['seed']), [], None

    def __call__(self, model, messages):
        payload = json.loads(messages[1]['content'])
        ids = [r[0] for r in payload['candidates']]
        if payload['actor'] is None:
            text = '\n'.join(', '.join(sorted(pair, key=int)) for pair in zip(ids[:-1], ids[1:]))
        else:
            count = 1 if 'mutual' in payload['fields'] else self.plan['nomination_quotas'][payload['actor'][0]]
            text = ', '.join(ids[:count])
        self.last_request_id = f'fixture-{len(self.requests)}'
        self.requests.append(dict(request_id=self.last_request_id, model=model, messages=messages, text=text))
        return text

    def parse_response(self, **kwargs):
        try:
            result = revised.parse_response(**kwargs)
        except ValueError as error:
            self.requests[-1]['parse'] = dict(valid=False, error=str(error))
            raise
        self.requests[-1]['parse'] = dict(valid=True, error=None)
        return result


class ReplayCaller:
    """Reconstruct graphs from recorded texts and check every submitted prompt."""
    def __init__(self, requests):
        self.requests, self.ordinal, self.last_request_id = requests, 0, None

    def __call__(self, model, messages):
        if self.ordinal >= len(self.requests):
            raise ValueError('Replay lacks a recorded request.')
        item = self.requests[self.ordinal]
        self.ordinal += 1
        if (item['request']['model'], item['request']['messages']) != (model, messages) or item['response']['finish_reason'] != 'stop':
            raise ValueError('Replay prompt or response differs from the calibration contract.')
        self.last_request_id = item['request_id']
        return item['response']['text']

    def parse_response(self, **kwargs):
        saved = self.requests[self.ordinal - 1]['response']['parse']
        try:
            result = revised.parse_response(**kwargs)
        except ValueError as error:
            if saved['valid'] is not False or saved['error'] != str(error)[:200]:
                raise RuntimeError('Replay parse failure differs from saved decision.') from error
            raise
        if saved['valid'] is not True:
            raise RuntimeError('Replay success differs from saved decision.')
        return result


def save_run(path, cell, result, requests, evidence):
    graph, events, decisions, schedule = result
    path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_adjlist(graph, path.with_suffix('.adj'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(7, 7))
    nx.draw_networkx(graph, pos=nx.spring_layout(graph, seed=224), ax=axis, with_labels=False,
        node_size=70, node_color='#46666b', edge_color='#a6b0b2', linewidths=.6, edgecolors='white', width=.6)
    axis.set_axis_off()
    figure.savefig(path.with_suffix('.png'), dpi=220, bbox_inches='tight', metadata={'Description':evidence})
    plt.close(figure)
    record = dict(cell=cell, evidence_type=evidence, events=events, decisions=decisions, schedule=schedule,
        metrics=measures(graph, revised.previous.adult_roster()), requests=requests,
        artifacts={ext: sha(path.with_suffix(ext)) for ext in ['.adj', '.png']})
    write_json(path, record)  # Completed marker is written last, after artifacts.
    return record


def write_csv(path, rows):
    if rows:
        with path.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def verify_receipt(path, cell, budget=None):
    record = json.loads(path.read_text(encoding='utf-8'))
    if record['cell'] != cell or record['evidence_type'] != 'received_revised_calibration':
        raise ValueError('Wrong calibration receipt or fixture data in paid outputs.')
    if any(sha(path.with_suffix(ext)) != value for ext, value in record['artifacts'].items()) or set(record['artifacts']) != {'.adj', '.png'}:
        raise ValueError('Calibration artifact integrity check failed.')
    caller = ReplayCaller(record['requests'])
    graph, events, decisions, schedule = run_cell(cell, caller)
    saved_graph = nx.read_adjlist(path.with_suffix('.adj'))
    canonical = lambda g: {frozenset(e) for e in g.edges()}
    if (set(saved_graph) != set(graph) or canonical(graph) != canonical(saved_graph)
            or paid.clean_numbers(events) != record['events'] or decisions != record['decisions'] or schedule != record['schedule']
            or measures(graph, revised.previous.adult_roster()) != record['metrics'] or caller.ordinal != len(record['requests'])):
        raise ValueError('Saved graph, events, schedule, metrics or requests differ from replay.')
    ids = set()
    for ordinal, item in enumerate(record['requests']):
        request, response, rid = item['request'], item['response'], item['request_id']
        if (rid in ids or request['cell'] != cell or request['ordinal'] != ordinal
                or request['settings'] != settings(cell['model'], cell['method'])
                or rid != request_id(cell, ordinal, request['messages'])):
            raise ValueError('Request identity is invalid.')
        usage = response['usage']
        if any(type(usage[k]) is not int or usage[k] < 0 for k in ['prompt_tokens','completion_tokens']):
            raise ValueError('Invalid recorded usage.')
        input_rate, output_rate = paid.RATES[cell['model']]
        charge = (usage['prompt_tokens']*input_rate*1.25 + usage['completion_tokens']*output_rate)/1e6
        if not math.isclose(response['conservative_charge_usd'],charge,rel_tol=0,abs_tol=1e-12):
            raise ValueError('Recorded charge differs from usage and frozen prices.')
        ids.add(rid)
        if budget is not None:
            row = ledger_row(budget, rid)
            if row is None or row[3] != 'received' or row[2] != response['conservative_charge_usd'] or json.loads(row[4]) != request or json.loads(row[5]) != response:
                raise ValueError('Receipt differs from paid ledger.')
    if budget is not None:
        ledger_ids = {r[0] for r in budget.db.execute("SELECT id FROM attempts WHERE json_extract(request, '$.cell.run_id')=?", (cell['run_id'],))}
        if ids != ledger_ids:
            raise ValueError('Receipt omits paid attempts.')
    return record


def prepare():
    verify_probe()
    spec = contract()
    target = folder(spec)
    write_json(target / 'contract.json', spec)
    write_json(target / 'planned_cells.json', cells(spec))
    write_json(target / 'authorization_template.json', dict(owner_approved=False, main_authorized=False,
        scope=None, additional_usd=None, approval_reference=None, prices_verified_on=None,
        note='Template only. Use --authorize only after explicit owner approval; the $1 probe does not apply.'))
    return spec


def preflight():
    with frozen.workflow_lock():
        spec = prepare()
        target, results, metric_rows = folder(spec) / 'offline', [], []
        with patch.object(paid, 'OpenAI', side_effect=AssertionError('Offline preflight cannot open API client')), \
                patch.object(socket.socket, 'connect', side_effect=AssertionError('Offline preflight cannot use network')):
            for cell in cells(spec):
                caller = FixtureCaller(cell)
                result = run_cell(cell, caller)
                record = save_run(target / (cell['run_id'] + '.json'), cell, result, caller.requests,
                    'OFFLINE_FIXTURE_NOT_MODEL_OUTPUT')
                if any(not d['valid'] for d in record['decisions']):
                    raise ValueError('Nominal fixture did not parse.')
                results.append(dict(run_id=cell['run_id'], nodes=len(result[0]), events=len(result[1]),
                    decisions=len(result[2]), record_sha256=sha(target / (cell['run_id'] + '.json'))))
                metric_rows.append(dict(run_id=cell['run_id'], evidence_type='OFFLINE_FIXTURE_NOT_MODEL_OUTPUT', **record['metrics']))
        write_csv(target/'network_metrics.csv',metric_rows)
        report = dict(status='OFFLINE_PREFLIGHT_PASSED', contract_sha256=frozen.digest(spec),
            fixture_networks=len(results), paid_requests=0, main_authorized=False, calibration_authorized=False, results=results)
        write_json(folder(spec) / 'preflight.json', report)
        return {k: v for k, v in report.items() if k != 'results'}


def approve(additional_usd, scope, approval_reference, prices_verified_on):
    """Explicit operator action, never called by prepare or preflight."""
    if (type(additional_usd) not in {int, float} or not math.isfinite(additional_usd) or additional_usd <= 0
            or scope not in {'core68', 'extended104'} or not re.fullmatch(r'[A-Za-z0-9_.-]{3,100}', approval_reference or '')
            or prices_verified_on != date.today().isoformat()):
        raise ValueError('Need finite owner-approved budget, named scope/reference and current verified pricing date.')
    with frozen.workflow_lock():
        spec = contract()
        target = folder(spec)
        preflight_report = json.loads((target / 'preflight.json').read_text(encoding='utf-8'))
        if preflight_report.get('status') != 'OFFLINE_PREFLIGHT_PASSED' or preflight_report['contract_sha256'] != frozen.digest(spec):
            raise ValueError('Matching offline preflight required.')
        if (target / 'authorization.json').exists():
            raise ValueError('Authorization already exists; never reset its budget anchor.')
        budget = paid.Budget(LEDGER, require_existing=True)
        try:
            verify_probe(budget)
            if budget.unresolved_count():
                raise ValueError('Unresolved ledger entries block authorization.')
            baseline = {row[0]: frozen.digest(row) for row in budget.db.execute('SELECT id,phase,cost,status,request,response FROM attempts')}
            value = dict(contract_sha256=frozen.digest(spec), owner_approved=True, main_authorized=False, scope=scope,
                additional_usd=additional_usd, ledger_start_usd=budget.spent(), cumulative_ceiling_usd=budget.spent()+additional_usd,
                baseline=baseline, approval_reference=approval_reference, prices_verified_on=prices_verified_on)
            write_json(target / 'authorization.json', value)
            return value
        finally:
            budget.db.close()


def review(spec):
    target = folder(spec)
    path = target / 'authorization.json'
    if not path.exists():
        raise ValueError('Paid calibration not authorized. Offline implementation does not grant spending permission.')
    value = json.loads(path.read_text(encoding='utf-8'))
    if json.loads((target/'contract.json').read_text(encoding='utf-8')) != spec or json.loads((target/'planned_cells.json').read_text(encoding='utf-8')) != cells(spec):
        raise ValueError('Prepared contract or manifest changed.')
    if (value.get('contract_sha256') != frozen.digest(spec) or value.get('owner_approved') is not True
            or value.get('main_authorized') is not False or value.get('scope') not in {'core68', 'extended104'}):
        raise ValueError('Invalid calibration authorization.')
    if value.get('prices_verified_on') != date.today().isoformat():
        raise ValueError('Recheck current pricing before resuming; preserve the existing budget anchor.')
    start, extra, cap = (value[k] for k in ['ledger_start_usd', 'additional_usd', 'cumulative_ceiling_usd'])
    if any(type(v) not in {int, float} or not math.isfinite(v) for v in [start, extra, cap]) or start < 0 or extra <= 0 or not math.isclose(start+extra, cap, abs_tol=1e-12):
        raise ValueError('Invalid frozen spending ceiling.')
    return value


def verify_accounting(budget, approved, target, allowed):
    current = {row[0]: frozen.digest(row) for row in budget.db.execute('SELECT id,phase,cost,status,request,response FROM attempts')}
    if any(current.get(rid) != value for rid, value in approved['baseline'].items()) or budget.spent() + 1e-12 < approved['ledger_start_usd']:
        raise ValueError('Historical ledger changed or lost rows.')
    verify_probe(budget)
    journal = {}
    for path in (target / 'requests').glob('*.json'):
        saved = json.loads(path.read_text(encoding='utf-8'))
        rid = path.stem
        if saved.get('request_id') != rid or saved.get('row_sha256') is None or current.get(rid) != saved['row_sha256']:
            raise ValueError('Missing or changed journaled request; reconcile before opening a client.')
        journal[rid] = saved
    relevant = set()
    for rid, status, request in budget.db.execute('SELECT id,status,request FROM attempts'):
        cell = json.loads(request).get('cell', {})
        if cell.get('calibration_contract_sha256') == approved['contract_sha256']:
            if status != 'received':
                raise ValueError('A non-received calibration attempt blocks execution, including abandoned requests.')
            if cell.get('run_id') not in allowed:
                raise ValueError('Unapproved calibration cell in ledger.')
            relevant.add(rid)
    if relevant != set(journal) or budget.unresolved_count():
        raise ValueError('Calibration ledger/journal differs or has uncertain attempts.')


def summarize(target, manifest, budget):
    rows = []
    for cell in manifest:
        path = target / 'runs' / (cell['run_id'] + '.json')
        if not path.exists():
            continue
        record = verify_receipt(path, cell, budget)
        requests = record['requests']
        rows.append({**{k: cell[k] for k in ['run_id', 'model', 'method', 'culture', 'language', 'repetition', 'stage']},
            **record['metrics'], 'requests': len(requests), 'first_response_failures': sum(not d['valid'] and d['attempt']==1 for d in record['decisions']),
            'conservative_usd': sum(r['response']['conservative_charge_usd'] for r in requests),
            'prompt_tokens':sum(r['response']['usage']['prompt_tokens'] for r in requests),
            'completion_tokens':sum(r['response']['usage']['completion_tokens'] for r in requests),
            'api_seconds': sum(r['response']['diagnostics']['elapsed_ms']/1000 for r in requests)})
    write_csv(target/'network_metrics.csv', rows)
    cost_fields = ['run_id','requests','first_response_failures','prompt_tokens','completion_tokens','conservative_usd','api_seconds']
    write_csv(target/'cost_stats.csv', [{key:row[key] for key in cost_fields} for row in rows])
    result = dict(completed_networks=len(rows), planned_networks=len(manifest), main_authorized=False,
        status='COMPLETE_CALIBRATION_NOT_SCIENTIFIC_APPROVAL' if len(rows)==len(manifest) else 'PARTIAL',
        received_network_charge_usd=sum(r['conservative_usd'] for r in rows),
        cumulative_ledger_usd=budget.spent(), unresolved=budget.unresolved_count())
    write_json(target / 'summary.json', result)
    return result


def execute(limit=None):
    with frozen.workflow_lock():
        spec = contract()
        approved = review(spec)  # Must fail before reading a key or opening any client.
        manifest = [c for c in cells(spec) if approved['scope']=='extended104' or c['stage']=='core68']
        if limit is None:
            limit = len(manifest)
        if type(limit) is not int or not 1 <= limit <= len(manifest):
            raise ValueError('Limit exceeds authorized calibration scope.')
        target = folder(spec)
        budget = paid.Budget(LEDGER, total_cap=approved['cumulative_ceiling_usd'], pilot_cap=min(5,approved['cumulative_ceiling_usd']),
            require_existing=True, allow_higher_cap=True)
        try:
            by_id = {c['run_id']: c for c in manifest}
            verify_accounting(budget, approved, target, by_id)
            completed = set()
            for path in (target / 'runs').glob('*.json'):
                if path.stem not in by_id:
                    raise ValueError('Unexpected receipt in isolated calibration output.')
                verify_receipt(path, by_id[path.stem], budget)
                completed.add(path.stem)
            pending = [c for c in manifest[:limit] if c['run_id'] not in completed]
            if pending:
                with paid.OpenAI(api_key=paid.credential(), base_url='https://api.openai.com/v1', max_retries=0, timeout=60) as client:
                    for model in sorted({c['model'] for c in pending}):
                        client.models.retrieve(model)
                    for cell in pending:
                        caller = CalibrationCaller(client, budget, cell, spec, target)
                        result = run_cell(cell, caller)
                        requests = []
                        for rid in caller.request_ids:
                            row = ledger_row(budget, rid)
                            requests.append(dict(request_id=rid, request=json.loads(row[4]), response=json.loads(row[5])))
                        save_run(target / 'runs' / (cell['run_id'] + '.json'), cell, result, requests, 'received_revised_calibration')
                        print('Saved ' + cell['run_id'], flush=True)
            verify_accounting(budget, approved, target, by_id)
            return summarize(target, manifest, budget)
        except Exception as error:
            write_json(target / 'last_stop.json', dict(error_type=type(error).__name__, main_authorized=False,
                cumulative_ledger_usd=budget.spent(), authorization_window_charge_or_reservation_usd=budget.spent()-approved['ledger_start_usd'],
                unresolved=budget.unresolved_count()))
            raise
        finally:
            budget.db.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    for option in ['prepare', 'preflight', 'authorize', 'execute']:
        group.add_argument('--' + option, action='store_true')
    parser.add_argument('--scope', choices=['core68', 'extended104'])
    parser.add_argument('--additional-usd', type=float)
    parser.add_argument('--approval-reference')
    parser.add_argument('--prices-verified-on')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    try:
        if args.prepare:
            with frozen.workflow_lock():
                spec = prepare()
            result = dict(status='PREPARED_NOT_AUTHORIZED', folder=str(folder(spec)), planned_cells=len(cells(spec)), paid_calls=0)
        elif args.preflight:
            result = preflight()
        elif args.authorize:
            value = approve(args.additional_usd, args.scope, args.approval_reference, args.prices_verified_on)
            result = {k:v for k,v in value.items() if k != 'baseline'}
        else:
            result = execute(args.limit)
        print(json.dumps(result, indent=2))
    except Exception as error:
        parser.exit(1, type(error).__name__ + ': stopped; no automatic paid retry. Review local evidence.\n')
