"""Hash-bound V7 diagnostics; shared ledger, immutable history, no main-study command.

Prepare and preflight are offline. Paid execution additionally requires a reviewed
contract, current price evidence, explicit stage permission and a local API key.
"""
import argparse
import contextlib
from datetime import date
import itertools
import json
import math
from pathlib import Path
import random
import re
import socket
from types import SimpleNamespace
from unittest.mock import patch

import calibration_v6 as base
import paid_study as paid
import revision_next as v6
import revision_v7_prompts as prompts

ROOT = Path(__file__).resolve().parent
DESTINATION = ROOT / 'outputs/repair_v7'
LEDGER = ROOT / 'outputs/revision_budget_v1/budget.sqlite'
MODELS = ['gpt-4.1', 'gpt-5.6-luna', 'gpt-6-luna', 'gpt-6-sol']
LANGUAGES = ['english', 'hindi', 'japanese', 'portuguese']
CAPS = {'global_probe': 5., 'wording_probe': 10., 'calibration': 15.}
SOURCES = ['revision_v7_prompts.py', 'repair_v7.py']
IMPORTED_SOURCES = {name: base.sha(ROOT / name) for name in SOURCES}
write_json, digest = base.frozen.write_json, base.frozen.digest


def history_contract():
    historical = json.loads((ROOT / 'outputs/calibration_v6/5fb3715550db/contract.json').read_text(encoding='utf8'))
    for name, expected in historical['sources'].items():
        if base.sha(ROOT / name) != expected:
            raise ValueError('Frozen historical source changed: ' + name)
    return historical


def design(stage, recipe=None):
    if stage not in CAPS:
        raise ValueError('Unknown stage; no main-study execution exists.')
    rows = []
    if stage == 'global_probe':
        for model, arm, rep in itertools.product(MODELS, ['a0_b0', 'a0_b1', 'a1_b0', 'a1_b1'], range(4)):
            rows.append(dict(model=model, method='global', culture='us', language='english',
                repetition=rep, seed=31000 + rep, wording='reference', global_arm=arm))
    elif stage == 'wording_probe':
        for model, language, wording, rep in itertools.product(MODELS, LANGUAGES,
                ['reference', 'alternative_1', 'alternative_2'], range(2)):
            rows.append(dict(model=model, method='local', culture='us', language=language,
                repetition=rep, seed=32000 + rep, wording=wording, global_arm=None))
    else:
        if not recipe or recipe.get('global_arm') not in {'a0_b0', 'a0_b1', 'a1_b0', 'a1_b1'}:
            raise ValueError('Calibration needs an explicitly reviewed global arm.')
        if recipe.get('wording') != 'reference':
            raise ValueError('Alternative local-only wording is not validated for other methods.')
        for row in base.design():
            rows.append({**{k: row[k] for k in ['model', 'method', 'culture', 'language', 'repetition']},
                'seed': 33000 + row['repetition'], 'wording': 'reference',
                'global_arm': recipe['global_arm'] if row['method'] == 'global' else None})
    random.Random({'global_probe': 31000, 'wording_probe': 32000, 'calibration': 33000}[stage]).shuffle(rows)
    return rows


def catalog():
    """Render all countries, languages, operations, cardinalities and retries."""
    rows = []
    for language, country, method in itertools.product(LANGUAGES, ['us', 'india', 'japan', 'brazil'],
            ['global', 'local', 'sequential', 'iterative-add', 'iterative-drop']):
        counts = [1, 2, 8] if method in {'local', 'sequential'} else [None]
        for count in counts:
            actor = None if method == 'global' else '0'
            variants = ['reference', 'alternative_1', 'alternative_2'] if method == 'local' else ['reference']
            arms = ['a0_b0', 'a0_b1', 'a1_b0', 'a1_b1'] if method == 'global' else [None]
            for wording, arm in itertools.product(variants, arms):
                args = (method, language, country, actor, count, wording, arm)
                rows.append(dict(method=method, language=language, country=country, count=count,
                    wording=wording, global_arm=arm, system=prompts.treatment_prompt(*args),
                    retry=None if method == 'global' else prompts.treatment_retry(
                        method, language, country, actor, count, wording=wording)))
    return rows


def contract(stage, recipe=None):
    history = history_contract()
    current_sources = {name: base.sha(ROOT / name) for name in SOURCES}
    if current_sources != IMPORTED_SOURCES:
        raise ValueError('Source changed after import; restart before preparing a contract.')
    return dict(version='repair-v7-diagnostics-2', stage=stage, main_authorized=False,
        requested_service_tier='default',
        sources={**history['sources'], **current_sources},
        history_contract_sha256=digest(history), roster_sha256=history['roster_sha256'],
        runtime=base.frozen.runtime_versions(), rates={model: list(paid.RATES[model]) for model in MODELS},
        catalog_sha256=digest(catalog()), design=design(stage, recipe), recipe=recipe,
        interpretation='Exploratory diagnostics, not confirmatory or human-language validation',
        retries={'sdk': 0, 'global': 0, 'local_corrections': 2}, additional_cap_usd=CAPS[stage])


def folder(spec):
    return DESTINATION / spec['stage'] / digest(spec)[:12]


def cells(spec):
    fingerprint = digest(spec)
    # 'main' is the legacy ledger bucket, not permission for a confirmatory study.
    return [dict(**row, run_id=f"repair_v7_{spec['stage']}_{fingerprint[:12]}_{i:03d}",
        phase='main', repair_contract_sha256=fingerprint, frozen_source_sha256=spec['sources'],
        requested_service_tier=spec['requested_service_tier'],
        roster_sha256=spec['roster_sha256'], evidence_type='repair_diagnostic_not_confirmation')
        for i, row in enumerate(spec['design'])]


class DiagnosticInvalid(RuntimeError):
    """A received invalid global answer is terminal evidence, not a retry trigger."""


@contextlib.contextmanager
def adapter(cell, spec):
    # Serial-only module adapters under the existing workflow lock; V6 files stay intact.
    proxy = SimpleNamespace(**vars(v6))
    proxy.system_prompt = lambda method, language, country, actor=None, count=None: prompts.treatment_prompt(
        method, language, country, actor, count, cell['wording'], cell['global_arm'])
    proxy.retry_prompt = lambda method, language, country, actor=None, count=None, preserving=False: prompts.treatment_retry(
        method, language, country, actor, count, preserving, cell['wording'], cell['global_arm'])
    with patch.object(base, 'revised', proxy), patch.object(base, 'contract', lambda: contract(spec['stage'], spec['recipe'])):
        yield


class Caller(base.CalibrationCaller):
    def __call__(self, model, messages, **kwargs):
        if self.cell.get('requested_service_tier') != 'default':
            raise ValueError('Only Standard processing is budgeted.')
        rid = base.request_id(self.cell, self.ordinal, messages)
        row = base.ledger_row(self.budget, rid)
        if row is not None and row[3] == 'received':
            verify_transport(self.destination, rid, json.loads(row[5]))
        transport = self.client.chat.completions.with_raw_response
        original = transport.create

        def standard_create(**request):
            # Pin actual JSON, not the project's changeable default processing tier.
            request['extra_body'] = {**request.get('extra_body', {}), 'service_tier': 'default'}
            raw = original(**request)
            response = raw.parse()
            write_json(self.destination / 'transport' / (rid + '.json'),
                dict(request_id=rid, requested_service_tier='default', response=response.model_dump()))
            if getattr(response, 'service_tier', None) != 'default':
                raise RuntimeError('Standard tier not confirmed; retain reservation and reconcile the saved reply.')
            return raw

        with patch.object(transport, 'create', standard_create):
            return super().__call__(model, messages, **kwargs)

    def parse_response(self, **kwargs):
        try:
            return super().parse_response(**kwargs)
        except ValueError as error:
            if self.cell['method'] == 'global':
                raise DiagnosticInvalid('Received global answer failed parsing; no replacement.') from error
            raise


def verify_transport(target, rid, recorded):
    audit = json.loads((target / 'transport' / (rid + '.json')).read_text(encoding='utf8'))
    response = audit['response']
    if (audit.get('request_id') != rid or audit.get('requested_service_tier') != 'default'
            or response.get('service_tier') != 'default' or not response.get('choices')
            or response.get('id') != recorded['response_id'] or response.get('model') != recorded['resolved_model']
            or response.get('usage') != recorded['usage']
            or (response['choices'][0]['message'].get('content') or '') != recorded['text']
            or response['choices'][0]['finish_reason'] != recorded['finish_reason']):
        raise ValueError('Transport tier or reply differs from the charged receipt.')


def verify_run(path, cell, spec, budget=None):
    record = json.loads(path.read_text(encoding='utf8'))
    if record.get('cell') != cell or record.get('evidence_type') != 'received_repair_v7':
        raise ValueError('Wrong receipt identity or fixture in paid results.')
    replay = base.ReplayCaller(record['requests'])
    with adapter(cell, spec):
        result = base.run_cell(cell, replay)
    import networkx as nx
    graph, events, decisions, schedule = result
    saved = nx.read_adjlist(path.with_suffix('.adj'))
    if (set(saved) != set(graph) or {frozenset(e) for e in saved.edges} != {frozenset(e) for e in graph.edges}
            or events != record['events'] or decisions != record['decisions'] or schedule != record['schedule']
            or base.measures(graph, v6.previous.adult_roster()) != record['metrics']
            or replay.ordinal != len(record['requests'])):
        raise ValueError('Receipt replay differs from saved output.')
    if set(record['artifacts']) != {'.adj', '.png'} or any(base.sha(path.with_suffix(ext)) != sha for ext, sha in record['artifacts'].items()):
        raise ValueError('Saved artifacts changed.')
    for ordinal, item in enumerate(record['requests']):
        request, response, rid = item['request'], item['response'], item['request_id']
        if (request['cell'] != cell or request['ordinal'] != ordinal
                or request['settings'] != base.settings(cell['model'], cell['method'])
                or rid != base.request_id(cell, ordinal, request['messages'])):
            raise ValueError('Request identity differs.')
        usage = response['usage']
        if any(type(usage[k]) is not int or usage[k] < 0 for k in ['prompt_tokens', 'completion_tokens']):
            raise ValueError('Invalid recorded usage.')
        input_rate, output_rate = spec['rates'][cell['model']]
        charge = (usage['prompt_tokens'] * input_rate * 1.25 + usage['completion_tokens'] * output_rate) / 1e6
        if not math.isclose(response['conservative_charge_usd'], charge, rel_tol=0, abs_tol=1e-12):
            raise ValueError('Recorded charge differs from frozen rates and usage.')
        if budget is not None:
            verify_transport(path.parent.parent, rid, response)
            row = base.ledger_row(budget, rid)
            if row is None or row[3] != 'received' or row[2] != charge or json.loads(row[4]) != request or json.loads(row[5]) != response:
                raise ValueError('Receipt/ledger mismatch.')
    return record


def prepare(stage, recipe=None):
    spec = contract(stage, recipe)
    target = folder(spec)
    write_json(target / 'contract.json', spec)
    write_json(target / 'prompt_catalog.json', catalog())
    write_json(target / 'planned_cells.json', cells(spec))
    return spec


def preflight(stage, recipe=None):
    with base.frozen.workflow_lock():
        spec = prepare(stage, recipe)
        checked = []
        with patch.object(paid, 'OpenAI', side_effect=AssertionError('Offline: API blocked')), \
                patch.object(socket.socket, 'connect', side_effect=AssertionError('Offline: network blocked')):
            for cell in cells(spec):
                fixture = base.FixtureCaller(cell)
                with adapter(cell, spec):
                    graph, events, decisions, schedule = base.run_cell(cell, fixture)
                if set(graph) != set(v6.previous.adult_roster()) or any(a == b for a, b in graph.edges):
                    raise ValueError('Offline engine changed roster membership or produced a self-link.')
                input_rate, output_rate = spec['rates'][cell['model']]
                output_limit = base.settings(cell['model'], cell['method'])['max_completion_tokens']
                # Same byte/framing bound as frozen PaidCaller; this is not expected usage.
                reservation = sum(((sum(len(m['content'].encode('utf8')) + 256 for m in request['messages'])
                    + 4096) * input_rate * 1.25 + output_limit * output_rate) / 1e6 for request in fixture.requests)
                checked.append(dict(run_id=cell['run_id'], nodes=len(graph), edges=graph.number_of_edges(),
                    events=len(events), decisions=len(decisions), request_count=len(fixture.requests),
                    first_attempt_reservation_sum_usd=reservation))
        report = dict(status='OFFLINE_PREFLIGHT_PASSED', contract_sha256=digest(spec),
            fixture_cells=checked, paid_requests=0, main_authorized=False,
            first_attempt_reservation_sum_usd=sum(row['first_attempt_reservation_sum_usd'] for row in checked),
            cost_basis='Historical rates; byte/framing input bounds plus configured output caps; no corrections. '
                'Not an invoice or completion guarantee. Stateful prompts follow fixture, not real, edges.')
        write_json(folder(spec) / 'preflight.json', report)
        return {k: v for k, v in report.items() if k != 'fixture_cells'} | {'fixture_count': len(checked)}


def stage_review(spec):
    target = folder(spec)
    if (json.loads((target / 'contract.json').read_text(encoding='utf8')) != spec
            or json.loads((target / 'planned_cells.json').read_text(encoding='utf8')) != cells(spec)
            or digest(json.loads((target / 'prompt_catalog.json').read_text(encoding='utf8'))) != spec['catalog_sha256']):
        raise ValueError('Prepared contract, cells or prompt catalog changed.')
    report = json.loads((target / 'preflight.json').read_text(encoding='utf8'))
    review = json.loads((target / 'review.json').read_text(encoding='utf8'))
    if report.get('status') != 'OFFLINE_PREFLIGHT_PASSED' or report.get('contract_sha256') != digest(spec):
        raise ValueError('Matching offline preflight required.')
    required = ['prompt_semantics_reviewed', 'diagnostic_design_reviewed', 'spending_safeguards_reviewed']
    if review.get('contract_sha256') != digest(spec) or any(review.get(key) is not True for key in required):
        raise ValueError('Exact-version independent review required.')
    if review.get('human_bilingual_validation') is not False or not review.get('review_reference'):
        raise ValueError('Review provenance must disclose absent human bilingual validation.')
    if review.get('rates') != spec['rates'] or review.get('prices_verified_on') != date.today().isoformat() or not review.get('price_evidence'):
        raise ValueError('Current model-rate evidence required before authorizing payment.')
    if spec['stage'] != 'global_probe':
        upstream = json.loads((target / 'upstream_review.json').read_text(encoding='utf8'))
        if upstream.get('next_contract_sha256') != digest(spec) or upstream.get('accepted') is not True or not upstream.get('rationale'):
            raise ValueError('Previous diagnostic findings must be reviewed, not automatically advanced.')
        required_stages = ['global_probe'] if spec['stage'] == 'wording_probe' else ['global_probe', 'wording_probe']
        for stage in required_stages:
            item = upstream.get('evidence', {}).get(stage, {})
            contract_id = item.get('contract_id', '')
            if not isinstance(contract_id, str) or not re.fullmatch('[0-9a-f]{12}', contract_id):
                raise ValueError('Invalid upstream contract identifier.')
            previous = json.loads((DESTINATION / stage / contract_id / 'summary.json').read_text(encoding='utf8'))
            if (digest(previous) != item.get('summary_sha256') or previous.get('status') != 'COMPLETE_REVIEW_REQUIRED'
                    or previous.get('contract_sha256', '')[:12] != contract_id):
                raise ValueError('Upstream stage evidence missing or changed.')
    return review


def approve(stage, recipe=None):
    with base.frozen.workflow_lock():
        spec = contract(stage, recipe)
        stage_review(spec)
        target = folder(spec)
        # A stage has ONE allowance, even if its source hash later changes.
        authorization = target.parent / 'authorization.json'
        if authorization.exists():
            raise ValueError('Authorization exists; never reset the original budget anchor.')
        budget = paid.Budget(LEDGER, require_existing=True)
        try:
            if budget.unresolved_count():
                raise ValueError('Unresolved historical request blocks spending.')
            if (any(target.parent.glob('*/requests/*.json')) or any(target.parent.glob('*/runs/*.json'))
                    or any(target.parent.glob('*/transport/*.json'))
                    or any(json.loads(row[0]).get('cell', {}).get('run_id', '').startswith('repair_v7_' + stage + '_')
                           for row in budget.db.execute('SELECT request FROM attempts'))):
                raise ValueError('Prior stage attempts exist without authorization; reconcile, never re-anchor.')
            start = budget.spent()
            approved = dict(contract_sha256=digest(spec), ledger_start_usd=start,
                cumulative_ceiling_usd=start + CAPS[stage], additional_usd=CAPS[stage], main_authorized=False,
                basis='Owner approved staged repair plan on 2026-10-12; no main collection.',
                baseline={row[0]: digest(row) for row in budget.db.execute('SELECT id,phase,cost,status,request,response FROM attempts')})
            write_json(authorization, approved)
            return {k: v for k, v in approved.items() if k != 'baseline'}
        finally:
            budget.db.close()


def accounting(budget, approved, spec):
    current = {r[0]: digest(r) for r in budget.db.execute('SELECT id,phase,cost,status,request,response FROM attempts')}
    if budget.unresolved_count() or any(current.get(rid) != sha for rid, sha in approved['baseline'].items()):
        raise ValueError('Historical ledger changed, lost rows, or has unresolved requests.')
    relevant = set()
    allowed = {cell['run_id'] for cell in cells(spec)}
    for rid, status, request in budget.db.execute('SELECT id,status,request FROM attempts'):
        cell = json.loads(request).get('cell', {})
        if cell.get('repair_contract_sha256') == digest(spec):
            if status != 'received' or cell.get('run_id') not in allowed:
                raise ValueError('Unapproved or unresolved stage request.')
            relevant.add(rid)
    journal = {}
    for path in (folder(spec) / 'requests').glob('*.json'):
        item = json.loads(path.read_text(encoding='utf8'))
        journal[path.stem] = item.get('row_sha256')
        if item.get('request_id') != path.stem or journal[path.stem] != current.get(path.stem):
            raise ValueError('Request journal differs; no automatic replacement.')
    if relevant != set(journal):
        raise ValueError('Ledger/journal request set differs.')


def execute(stage, recipe=None):
    with base.frozen.workflow_lock():
        spec = contract(stage, recipe)
        target = folder(spec)
        # Invalidate a prior completion before rechecking any mutable evidence.
        write_json(target / 'summary.json', dict(status='VALIDATION_PENDING',
            contract_sha256=digest(spec), main_authorized=False))
        stage_review(spec)
        approved = json.loads((target.parent / 'authorization.json').read_text(encoding='utf8'))
        if (approved.get('contract_sha256') != digest(spec) or approved.get('main_authorized') is not False
                or approved.get('additional_usd') != CAPS[stage]
                or approved['cumulative_ceiling_usd'] != approved['ledger_start_usd'] + CAPS[stage]):
            raise ValueError('Authorization scope or original cap differs.')
        budget = paid.Budget(LEDGER, total_cap=approved['cumulative_ceiling_usd'],
            pilot_cap=min(5., approved['cumulative_ceiling_usd']), require_existing=True, allow_higher_cap=True)
        results = []
        try:
            accounting(budget, approved, spec)
            manifest = cells(spec)
            with contextlib.ExitStack() as stack:
                client = None
                for cell in manifest:
                    path = target / 'runs' / (cell['run_id'] + '.json')
                    if path.exists():
                        record = verify_run(path, cell, spec, budget)
                    else:
                        if client is None:
                            client = stack.enter_context(paid.OpenAI(api_key=paid.credential(),
                                base_url='https://api.openai.com/v1', max_retries=0, timeout=60))
                            for model in MODELS:
                                client.models.retrieve(model)
                        with adapter(cell, spec):
                            caller = Caller(client, budget, cell, spec, target)
                            result = base.run_cell(cell, caller)
                        requests = []
                        for rid in caller.request_ids:
                            row = base.ledger_row(budget, rid)
                            requests.append(dict(request_id=rid, request=json.loads(row[4]), response=json.loads(row[5])))
                        record = base.save_run(path, cell, result, requests, 'received_repair_v7')
                        verify_run(path, cell, spec, budget)
                    requests = record['requests']
                    results.append(dict(run_id=cell['run_id'], model=cell['model'], method=cell['method'],
                        culture=cell['culture'], language=cell['language'], repetition=cell['repetition'],
                        wording=cell['wording'], global_arm=cell['global_arm'], **record['metrics'],
                        requests=len(requests), decisions=sum(d['attempt'] == 1 for d in record['decisions']),
                        first_response_failures=sum(d['attempt'] == 1 and not d['valid'] for d in record['decisions']),
                        correction_requests=sum(d['attempt'] > 1 for d in record['decisions']),
                        prompt_tokens=sum(r['response']['usage']['prompt_tokens'] for r in requests),
                        completion_tokens=sum(r['response']['usage']['completion_tokens'] for r in requests),
                        conservative_usd=sum(r['response']['conservative_charge_usd'] for r in requests),
                        api_seconds=sum(r['response']['diagnostics']['elapsed_ms'] / 1000 for r in requests)))
                    print(f"Verified {len(results)}/{len(manifest)}; charged/reserved ${budget.spent()-approved['ledger_start_usd']:.5f}", flush=True)
            accounting(budget, approved, spec)
            report = dict(status='COMPLETE_REVIEW_REQUIRED', contract_sha256=digest(spec),
                completed_graphs=len(results), main_authorized=False, results=results,
                stage_charge_usd=sum(row['conservative_usd'] for row in results),
                empty_graphs=sum(row['density'] == 0 for row in results))
            write_json(target / 'summary.json', report)
            return {k: v for k, v in report.items() if k != 'results'}
        except Exception as error:
            write_json(target / 'summary.json', dict(status='STOPPED', error_type=type(error).__name__,
                contract_sha256=digest(spec), verified_partial_results=results,
                completed_graphs=len(results), main_authorized=False,
                authorization_window_charge_or_reservation_usd=budget.spent()-approved['ledger_start_usd']))
            raise
        finally:
            budget.db.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'preflight', 'approve', 'execute'])
    parser.add_argument('stage', choices=list(CAPS))
    parser.add_argument('--recipe', type=Path, help='Reviewed calibration recipe JSON; not a spending authorization.')
    args = parser.parse_args()
    try:
        recipe = json.loads(args.recipe.read_text(encoding='utf8')) if args.recipe else None
        output = globals()[args.action](args.stage, recipe)
        print(json.dumps(output if args.action != 'prepare' else dict(contract=digest(output), destination=str(folder(output))), indent=2))
    except Exception as error:
        parser.exit(1, type(error).__name__ + ': stopped; no automatic paid retry. Inspect local evidence.\n')
