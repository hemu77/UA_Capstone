"""Prospective wording screen and recalibration; frozen V6/V7 remain untouched.

Wording is an independent local-task diagnostic, not conditional on global
density. Recalibration adds explicit construction in each instruction language.
Strict parser failures stay failures; neither duplicates nor NONE are edited.
"""
import argparse
import contextlib
from datetime import date
import json
import math
from pathlib import Path
import socket
from unittest.mock import patch

import repair_v7 as r
import repair_v7_checked as checked

CONSTRUCTION = {
    'english': 'Construct a hypothetical friendship network for a simulation using the fictional people below. Generate a plausible realization of friendships, rather than recovering relationships already recorded in the candidate data.\n\n',
    'portuguese': 'Construa uma rede hipotética de amizades para uma simulação usando as pessoas fictícias abaixo. Gere uma configuração plausível de amizades, em vez de recuperar relações já registradas nos dados dos candidatos.\n\n',
    'hindi': 'नीचे दिए गए काल्पनिक व्यक्तियों का उपयोग करके एक सिमुलेशन के लिए मित्रताओं का एक काल्पनिक नेटवर्क बनाएँ। उम्मीदवारों के डेटा में पहले से दर्ज संबंधों को खोजने के बजाय, मित्रताओं का एक संभावित विन्यास उत्पन्न करें।\n\n',
    'japanese': '以下の架空の人物を使い、シミュレーション用の仮想的な友人関係ネットワークを構築してください。候補者のデータにすでに記録された関係を復元するのではなく、あり得そうな友人関係の一例を生成してください。\n\n',
}
SOURCE = Path(__file__).name
IMPORTED_SHA = r.base.sha(__file__)
ORIGINAL_CONTRACT, ORIGINAL_CATALOG, ORIGINAL_ADAPTER = r.contract, r.catalog, r.adapter
ORIGINAL_ACCOUNTING = r.accounting
RECIPE = {'global_arm': 'a0_b0', 'wording': 'reference'}
EXHAUSTED = 'Three correction attempts exhausted; incomplete run remains charged.'


def catalog():
    return [{**row, 'system': CONSTRUCTION[row['language']] + row['system']}
            if row['method'] == 'global' else row for row in ORIGINAL_CATALOG()]


def contract(stage, recipe=None):
    if stage not in {'wording_probe', 'calibration'} or recipe not in (None, RECIPE):
        raise ValueError('Only the approved wording and reference recalibration are supported.')
    checked.check_source()
    if r.base.sha(__file__) != IMPORTED_SHA:
        raise ValueError('Prospective source changed; restart and review before spending.')
    with patch.object(r, 'catalog', ORIGINAL_CATALOG):
        spec = ORIGINAL_CONTRACT(stage, RECIPE if stage == 'calibration' else None)
    return {**spec, 'version': 'repair-v8-prospective-1',
        'sources': {**spec['sources'], SOURCE: IMPORTED_SHA, 'repair_v7_checked.py': checked.IMPORTED_SHA},
        'catalog_sha256': r.digest(catalog()),
        'dependencies': [] if stage == 'wording_probe' else ['wording_probe'],
        'failure_policy': 'Strict unchanged parser; terminal global failure or exhausted local corrections is missing graph evidence, never replacement or deduplication.',
        'language_validation': 'AI semantic review only; no human/native-speaker certification',
        'selection_rule': 'Reference wording fixed prospectively; alternatives diagnose sensitivity, not outcome optimization.'}


@contextlib.contextmanager
def adapter(cell, spec):
    with ORIGINAL_ADAPTER(cell, spec):
        original = r.base.revised.system_prompt
        if cell['method'] == 'global':
            r.base.revised.system_prompt = lambda *args, **kwargs: CONSTRUCTION[cell['language']] + original(*args, **kwargs)
        yield


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def stage_review(spec):
    target = r.folder(spec)
    if read(target/'contract.json') != spec or read(target/'planned_cells.json') != r.cells(spec):
        raise ValueError('Prepared experiment identity changed.')
    if r.digest(read(target/'prompt_catalog.json')) != spec['catalog_sha256']:
        raise ValueError('Rendered prompt catalog changed.')
    preflight, review = read(target/'preflight.json'), read(target/'review.json')
    if preflight.get('status') != 'OFFLINE_PREFLIGHT_PASSED' or preflight.get('contract_sha256') != r.digest(spec):
        raise ValueError('Matching offline preflight required.')
    if (review.get('contract_sha256') != r.digest(spec)
            or any(review.get(k) is not True for k in ['prompt_semantics_reviewed', 'diagnostic_design_reviewed', 'spending_safeguards_reviewed'])
            or review.get('human_bilingual_validation') is not False or not review.get('review_reference')):
        raise ValueError('Exact-version review with honest language provenance required.')
    if (review.get('rates') != spec['rates'] or review.get('prices_verified_on') != date.today().isoformat()
            or not review.get('price_evidence')):
        raise ValueError('Current rate evidence required.')
    if spec['stage'] == 'calibration':
        upstream = read(target/'upstream_review.json')
        wording = contract('wording_probe')
        summary = read(r.folder(wording)/'summary.json')
        if (upstream.get('next_contract_sha256') != r.digest(spec) or upstream.get('accepted') is not True
                or not upstream.get('rationale') or upstream.get('summary_sha256') != r.digest(summary)
                or summary.get('contract_sha256') != r.digest(wording)
                or summary.get('status') not in {'COMPLETE_REVIEW_REQUIRED', 'COMPLETE_WITH_INVALID_RESPONSES'}
                or summary.get('completed_cells') != 96):
            raise ValueError('Completed wording observations and their explicit interpretation required.')
    return review


def accounting(budget, approved, spec):
    ORIGINAL_ACCOUNTING(budget, approved, spec)
    target = r.folder(spec)
    for audit in (target/'transport').glob('*.json'):
        if r.base.ledger_row(budget, audit.stem) is None:
            raise ValueError('Orphan transport evidence; never repurchase.')
    for directory in ['runs', 'failures']:
        for path in (target/directory).glob('*.json'):
            record = read(path)
            if path.stem != record['cell']['run_id'] or not record.get('requests'):
                raise ValueError('Malformed terminal evidence.')
            for item in record['requests']:
                row = r.base.ledger_row(budget, item['request_id'])
                if row is None or json.loads(row[4]) != item['request'] or json.loads(row[5]) != item['response']:
                    raise ValueError('Orphan or changed terminal evidence; never repurchase.')


def approve(stage):
    # Check every prior contract, not just the current target: failed cells cost money too.
    if any((r.DESTINATION/stage).glob('*/failures/*.json')):
        raise ValueError('Prior failure evidence blocks new authorization; reconcile, never re-anchor.')
    with isolated():
        return r.approve(stage)


@contextlib.contextmanager
def isolated():
    with patch.object(r, 'contract', contract), patch.object(r, 'catalog', catalog), \
            patch.object(r, 'adapter', adapter), patch.object(r, 'stage_review', stage_review), \
            patch.object(r, 'accounting', accounting), patch.object(r, 'verify_run', checked.verify_run):
        yield


def receipts(budget, cell, spec, target):
    rows = budget.db.execute("SELECT id,phase,cost,status,request,response FROM attempts WHERE json_extract(request,'$.cell.run_id')=?",
        (cell['run_id'],)).fetchall()
    rows.sort(key=lambda row: json.loads(row[4])['ordinal'])
    result = []
    for ordinal, row in enumerate(rows):
        rid, _, cost, status, raw_request, raw_response = row
        if status != 'received':
            raise ValueError('Unresolved receipt blocks replacement.')
        request, response = json.loads(raw_request), json.loads(raw_response)
        usage = response['usage']
        if any(type(usage[k]) is not int or usage[k] < 0 for k in ['prompt_tokens', 'completion_tokens']):
            raise ValueError('Invalid token usage.')
        rates = spec['rates'][cell['model']]
        charge = (usage['prompt_tokens']*rates[0]*1.25 + usage['completion_tokens']*rates[1])/1e6
        if (request['cell'] != cell or request['ordinal'] != ordinal
                or request['model'] != cell['model'] or request['settings'] != r.base.settings(cell['model'], cell['method'])
                or rid != r.base.request_id(cell, ordinal, request['messages'])
                or not math.isclose(cost, charge, rel_tol=0, abs_tol=1e-12)
                or response['conservative_charge_usd'] != cost):
            raise ValueError('Receipt identity or accounting differs.')
        r.verify_transport(target, rid, response)
        result.append(dict(request_id=rid, request=request, response=response))
    return result


class ResumingCaller:
    """Replay the paid prefix read-only; send only previously unattempted requests."""
    def __init__(self, cached, live=None):
        self.replay, self.live = r.base.ReplayCaller(cached), live
        self.last_request_id = None
        self.cached_response = False

    def __call__(self, model, messages):
        self.cached_response = self.replay.ordinal < len(self.replay.requests)
        if self.cached_response:
            response = self.replay(model, messages)
            self.last_request_id = self.replay.last_request_id
            return response
        if self.live is None:
            raise ValueError('Replay lacks a recorded request; verify-only cannot purchase one.')
        self.live.ordinal = max(self.live.ordinal, self.replay.ordinal)
        response = self.live(model, messages)
        self.last_request_id = self.live.last_request_id
        return response

    def parse_response(self, **kwargs):
        if not self.cached_response:
            return self.live.parse_response(**kwargs)
        try:
            return self.replay.parse_response(**kwargs)
        except ValueError as error:
            saved = self.replay.requests[self.replay.ordinal-1]['response']['parse']
            if saved.get('duplicate_edges', []) != r.paid.clean_numbers(getattr(error, 'duplicate_edges', [])):
                raise RuntimeError('Recorded duplicate evidence differs.') from error
            if kwargs['method'] == 'global':
                raise r.DiagnosticInvalid('Terminal cached global failure.') from error
            raise


def run_observation(cell, caller):
    try:
        return r.base.run_cell(cell, caller), None
    except r.DiagnosticInvalid:
        return None, 'invalid_global_response'
    except ValueError as error:
        if str(error) != EXHAUSTED:
            raise
        return None, 'local_corrections_exhausted'


def summary_row(cell, record, failure):
    requests = record['requests']
    return dict(run_id=cell['run_id'], model=cell['model'], method=cell['method'], culture=cell['culture'],
        language=cell['language'], repetition=cell['repetition'], wording=cell['wording'], valid_graph=failure is None,
        failure=failure, metrics=record.get('metrics'), requests=len(requests),
        parse_failures=sum(item['response']['parse']['valid'] is False for item in requests),
        first_response_failures=sum(d['attempt'] == 1 and not d['valid'] for d in record.get('decisions', [])) if failure is None else None,
        conservative_usd=sum(item['response']['conservative_charge_usd'] for item in requests),
        api_seconds=sum(item['response']['diagnostics']['elapsed_ms']/1000 for item in requests))


def execute(stage, verify_only=False):
    with r.base.frozen.workflow_lock(), isolated():
        spec, results = contract(stage), []
        target = r.folder(spec)
        r.write_json(target/'summary.json', dict(status='VALIDATION_PENDING', contract_sha256=r.digest(spec), main_authorized=False))
        stage_review(spec)
        approved = read(target.parent/'authorization.json')
        if (approved.get('contract_sha256') != r.digest(spec) or approved.get('main_authorized') is not False
                or approved.get('additional_usd') != r.CAPS[stage]
                or approved['cumulative_ceiling_usd'] != approved['ledger_start_usd']+r.CAPS[stage]):
            raise ValueError('Original stage allowance changed.')
        budget = r.paid.Budget(r.LEDGER, total_cap=approved['cumulative_ceiling_usd'], pilot_cap=5.,
            require_existing=True, allow_higher_cap=True)
        before = {row[0]: r.digest(row) for row in budget.db.execute('SELECT id,phase,cost,status,request,response FROM attempts')}
        try:
            accounting(budget, approved, spec)
            with contextlib.ExitStack() as stack:
                client = None
                if verify_only:
                    stack.enter_context(patch.object(socket.socket, 'connect', side_effect=AssertionError('Offline verification')))
                for cell in r.cells(spec):
                    if contract(stage) != spec:
                        raise ValueError('Source or protocol changed during execution.')
                    path = target/'runs'/(cell['run_id']+'.json')
                    failure_path = target/'failures'/path.name
                    cached = receipts(budget, cell, spec, target)
                    if path.exists():
                        if failure_path.exists():
                            raise ValueError('Conflicting graph and failure evidence.')
                        record = checked.verify_run(path, cell, spec, budget)
                        if record['requests'] != cached:
                            raise ValueError('Saved graph omits paid attempts.')
                        failure = None
                    else:
                        # Reconstruct a complete cached cell before creating any API client.
                        caller = ResumingCaller(cached)
                        with adapter(cell, spec):
                            try:
                                result, failure = run_observation(cell, caller)
                            except ValueError as error:
                                if str(error) != 'Replay lacks a recorded request; verify-only cannot purchase one.' or verify_only or failure_path.exists():
                                    raise
                                if client is None:
                                    client = stack.enter_context(r.paid.OpenAI(api_key=r.paid.credential(),
                                        base_url='https://api.openai.com/v1', max_retries=0, timeout=60))
                                    for model in r.MODELS:
                                        client.models.retrieve(model)
                                caller = ResumingCaller(cached, r.Caller(client, budget, cell, spec, target))
                                result, failure = run_observation(cell, caller)
                        requests = receipts(budget, cell, spec, target)
                        if caller.replay.ordinal != len(cached):
                            raise ValueError('Cached receipt prefix was not fully consumed.')
                        if failure is None:
                            if failure_path.exists():
                                raise ValueError('Historical failure changed to success.')
                            record = r.base.save_run(path, cell, result, requests, 'received_repair_v7')
                            checked.verify_run(path, cell, spec, budget)
                        else:
                            if not requests or requests[-1]['response']['parse']['valid'] is not False:
                                raise ValueError('Failure lacks received parser evidence.')
                            record = dict(cell=cell, evidence_type='received_invalid_v8_observation',
                                requests=requests, failure=failure)
                            if failure_path.exists() and read(failure_path) != record:
                                raise ValueError('Terminal failure evidence changed.')
                            r.write_json(failure_path, record)
                    results.append(summary_row(cell, record, failure))
                    print(f"{stage}: {len(results)}/{len(spec['design'])} observations; ${budget.spent()-approved['ledger_start_usd']:.5f}", flush=True)
            accounting(budget, approved, spec)
            after = {row[0]: r.digest(row) for row in budget.db.execute('SELECT id,phase,cost,status,request,response FROM attempts')}
            if any(after.get(rid) != sha for rid, sha in before.items()) or (verify_only and before != after):
                raise ValueError('Existing paid ledger evidence changed.')
            invalid = sum(not row['valid_graph'] for row in results)
            report = dict(status='COMPLETE_WITH_INVALID_RESPONSES' if invalid else 'COMPLETE_REVIEW_REQUIRED',
                contract_sha256=r.digest(spec), completed_cells=len(results), completed_graphs=len(results)-invalid,
                invalid_cells=invalid, main_authorized=False, results=results,
                stage_charge_usd=sum(row['conservative_usd'] for row in results),
                api_seconds=sum(row['api_seconds'] for row in results), existing_ledger_rows_unchanged=True)
            r.write_json(target/'summary.json', report)
            r.base.write_csv(target/'summary.csv', [{k: json.dumps(val) if k == 'metrics' else val for k,val in row.items()} for row in results])
            return {k: val for k,val in report.items() if k != 'results'}
        except Exception as error:
            r.write_json(target/'summary.json', dict(status='STOPPED', contract_sha256=r.digest(spec),
                error_type=type(error).__name__, completed_cells=len(results), results=results, main_authorized=False,
                authorization_window_charge_or_reservation_usd=budget.spent()-approved['ledger_start_usd']))
            raise
        finally:
            budget.db.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['preflight', 'approve', 'execute', 'verify'])
    parser.add_argument('stage', choices=['wording_probe', 'calibration'])
    args = parser.parse_args()
    if args.action in {'execute', 'verify'}:
        output = execute(args.stage, verify_only=args.action == 'verify')
    elif args.action == 'approve':
        output = approve(args.stage)
    else:
        with isolated():
            output = getattr(r, args.action)(args.stage)
    print(json.dumps(output, indent=2))
